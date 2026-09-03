/**
 * Facial-expression classification in the browser — ports
 * emotion/EmotionPreprocessor.kt, EmotionClassifier.kt and EmotionAggregator.kt
 * onto onnxruntime-web.
 *
 * The preprocessing contract is taken from EmotiEffLib's `facial_analysis.py`:
 *
 *   resize -> /255 -> (x - mean) / std per channel -> transpose to CHW -> batch 1
 *
 * Channel order is **RGB**. EmotiEffLib's own tests do a BGR->RGB conversion
 * before calling preprocess, which makes the function look BGR at a glance; it
 * is not. Canvas `getImageData` already gives RGBA, so no conversion is needed.
 *
 * Every failure mode here is silent: a swapped channel or a missing division by
 * 255 does not crash and does not look wrong — it produces confident, plausible,
 * wrong emotions. That is why the constants are transcribed rather than
 * reasoned about.
 */

// The `/wasm` entry point, not the package root. The root bundles the JSEP
// build, which asks for ort-wasm-simd-threaded.jsep.wasm — a 26MB binary whose
// only purpose is WebGPU, which this does not use. The wasm entry pulls the
// plain 13MB CPU build instead.
import * as ort from 'onnxruntime-web/wasm'

// Resolved through Vite with `?url` rather than copied into public/ and named
// by a path prefix. Vite then fingerprints and rewrites both URLs for whatever
// `base` the site is deployed under, and — importantly — serves the loader as a
// static asset instead of trying to transform it as a module, which is what a
// public/ copy ran into.
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url'
import ortMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url'

import { faceBox } from './faceCropper'
import {
  EMOTION_COUNT,
  EMOTION_LABELS,
  NEUTRAL_INDEX,
  isHighArousal,
  type EmotionLabel,
} from './labels'

/** Input side for the enet_b0 family. The b2 models would want 260. */
const INPUT_SIZE = 224
const PLANE = INPUT_SIZE * INPUT_SIZE
const TENSOR_LENGTH = 3 * PLANE

/** ImageNet statistics, as used by every non-`mbf_` EmotiEffLib model. */
const MEAN = [0.485, 0.456, 0.406]
const STD = [0.229, 0.224, 0.225]

/**
 * Share of probability mass Neutral must hold to be reported.
 *
 * AffectNet-trained models return Neutral for the majority of real video frames,
 * so a plain argmax reports Neutral almost always and the feature does nothing.
 * Requiring a majority lets a weaker but genuine expressive signal through,
 * while still reporting calm when the face really is calm.
 */
const NEUTRAL_DOMINANCE = 0.55

/** Below this the reading is too weak to steer a sentence with. */
export const MIN_CONFIDENCE = 0.35

/** How often the classifier runs. It is far too heavy for the per-frame path. */
const SAMPLE_INTERVAL_MS = 400

/** Readings averaged before a label is reported. */
const WINDOW = 5

export interface EmotionReading {
  descriptor: string
  descriptorMs: string
  confidence: number
  isHighArousal: boolean
  label: EmotionLabel
  styleDirective: string
}

/** Max-shifted softmax. The model emits logits; EmotiEffLib softmaxes outside the graph. */
export function softmax(logits: Float32Array | number[]): Float32Array {
  const out = new Float32Array(logits.length)
  let max = -Infinity
  for (const v of logits) if (v > max) max = v
  let sum = 0
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - max)
    out[i] = e
    sum += e
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum
  return out
}

function argmax(v: ArrayLike<number>): number {
  let best = 0
  for (let i = 1; i < v.length; i++) if (v[i] > v[best]) best = i
  return best
}

function argmaxExcluding(v: ArrayLike<number>, skip: number): number {
  let best = -1
  for (let i = 0; i < v.length; i++) {
    if (i === skip) continue
    if (best < 0 || v[i] > v[best]) best = i
  }
  return best
}

/**
 * Picks the reported class from an averaged distribution.
 *
 * Neutral only wins if it holds a majority; otherwise the runner-up is reported,
 * which is what stops the feature reporting "neutral" forever.
 */
export function resolve(mean: ArrayLike<number>, dominance = NEUTRAL_DOMINANCE): number {
  const top = argmax(mean)
  if (top !== NEUTRAL_INDEX) return top
  if (mean[NEUTRAL_INDEX] >= dominance) return NEUTRAL_INDEX
  const runnerUp = argmaxExcluding(mean, NEUTRAL_INDEX)
  return runnerUp >= 0 ? runnerUp : NEUTRAL_INDEX
}

export class EmotionClassifier {
  private session: ort.InferenceSession | null = null
  private loading: Promise<void> | null = null

  /** Reused across inferences; this is 150,528 floats. */
  private readonly tensorData = new Float32Array(TENSOR_LENGTH)
  private readonly canvas = document.createElement('canvas')
  private readonly ctx: CanvasRenderingContext2D

  private history: Float32Array[] = []
  private lastSampleAt = 0
  private busy = false

  constructor() {
    this.canvas.width = INPUT_SIZE
    this.canvas.height = INPUT_SIZE
    // willReadFrequently: the crop is read back to CPU on every sample.
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true })!
  }

  /** Loads the ONNX graph. Safe to call repeatedly; the work happens once. */
  async load(): Promise<void> {
    if (this.session) return
    if (this.loading) return this.loading

    this.loading = (async () => {
      // Name both artefacts explicitly. Left unset, the runtime guesses a URL
      // that does not exist, the server answers with index.html, and
      // instantiation fails on the HTML's magic bytes ("found 3c 21 64 6f" —
      // that is `<!do`) rather than on anything to do with the model.
      ort.env.wasm.wasmPaths = { wasm: ortWasmUrl, mjs: ortMjsUrl }
      // Single-threaded: the threaded build needs cross-origin isolation
      // headers, which static hosts like GitHub Pages cannot set.
      ort.env.wasm.numThreads = 1

      const url = `${import.meta.env.BASE_URL}models/emotion_enet_b0_8.onnx`
      this.session = await ort.InferenceSession.create(url, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      })
    })()

    try {
      await this.loading
    } finally {
      this.loading = null
    }
  }

  get ready(): boolean {
    return this.session !== null
  }

  /**
   * Samples the face from a video frame and returns a reading when one is due.
   *
   * Returns null when it is not yet time, a face cannot be located, or a
   * previous inference is still running — the caller drops the frame rather than
   * queueing, so a slow machine samples less often instead of falling behind.
   */
  async sample(
    source: CanvasImageSource,
    sourceWidth: number,
    sourceHeight: number,
    keypoints: Float32Array | null,
  ): Promise<EmotionReading | null> {
    if (!this.session || !keypoints || this.busy) return null

    const now = performance.now()
    if (now - this.lastSampleAt < SAMPLE_INTERVAL_MS) return null

    const box = faceBox(keypoints, sourceWidth, sourceHeight)
    if (!box) return null

    this.lastSampleAt = now
    this.busy = true
    try {
      // Crop and resize in one draw.
      this.ctx.drawImage(
        source,
        box.left, box.top, box.width, box.height,
        0, 0, INPUT_SIZE, INPUT_SIZE,
      )
      const { data } = this.ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE)
      this.toTensor(data)

      const input = new ort.Tensor('float32', this.tensorData, [1, 3, INPUT_SIZE, INPUT_SIZE])
      const feeds = { [this.session.inputNames[0]]: input }
      const output = await this.session.run(feeds)
      const logits = output[this.session.outputNames[0]].data as Float32Array

      if (logits.length !== EMOTION_COUNT) {
        // A model whose width does not match must fail loudly rather than
        // silently wrapping onto a wrong label.
        console.error(`[emotion] expected ${EMOTION_COUNT} classes, got ${logits.length}`)
        return null
      }

      return this.accumulate(softmax(logits))
    } catch (e) {
      console.warn('[emotion] inference failed', e)
      return null
    } finally {
      this.busy = false
    }
  }

  /** resize -> /255 -> (x - mean) / std -> CHW. Constants hoisted out of the loop. */
  private toTensor(rgba: Uint8ClampedArray): void {
    const out = this.tensorData
    const rScale = 1 / (255 * STD[0])
    const gScale = 1 / (255 * STD[1])
    const bScale = 1 / (255 * STD[2])
    const rShift = MEAN[0] / STD[0]
    const gShift = MEAN[1] / STD[1]
    const bShift = MEAN[2] / STD[2]

    for (let p = 0; p < PLANE; p++) {
      const i = p * 4
      out[p] = rgba[i] * rScale - rShift
      out[PLANE + p] = rgba[i + 1] * gScale - gShift
      out[2 * PLANE + p] = rgba[i + 2] * bScale - bShift
    }
  }

  /**
   * Averages over a short window before reporting.
   *
   * A single frame is noisy — a blink or a mid-sign mouth shape reads as an
   * expression — so the label only changes when several consecutive samples
   * agree.
   */
  private accumulate(probs: Float32Array): EmotionReading | null {
    this.history.push(probs)
    if (this.history.length > WINDOW) this.history.shift()
    if (this.history.length < WINDOW) return null

    const mean = new Float32Array(EMOTION_COUNT)
    for (const h of this.history) for (let i = 0; i < EMOTION_COUNT; i++) mean[i] += h[i]
    for (let i = 0; i < EMOTION_COUNT; i++) mean[i] /= this.history.length

    const index = resolve(mean)
    const label = EMOTION_LABELS[index]
    return {
      descriptor: label.descriptorEn,
      descriptorMs: label.descriptorMs,
      confidence: mean[index],
      isHighArousal: isHighArousal(label),
      label,
      styleDirective: label.styleDirective,
    }
  }

  reset(): void {
    this.history = []
  }
}
