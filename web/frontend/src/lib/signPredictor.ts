/**
 * SignPredictor — port of ml/SignPredictor.kt.
 *
 * Owns everything between "a frame arrived" and "a sentence is ready": EMA
 * smoothing, the 30-frame window, the prediction cooldown, and the rules for
 * when a recognised gloss joins the sentence.
 *
 * The Kotlin exposes a StateFlow; this exposes subscribe/getState, which is the
 * shape React's useSyncExternalStore wants. The pipeline logic is otherwise
 * unchanged — the thresholds here (0.6 confidence, 10/5-frame cooldown, 8-word
 * buffer, alpha 0.4) were tuned against the model and are not free parameters.
 */

import {
  buildSequenceFeatures,
  normalizeSingleFrame,
  SEQUENCE_LENGTH,
} from './frameNormalizer'
import { IDLE } from './labelMap'
import type { PredictClient } from './predictClient'

/** Matches SignPredictor.kt — a word below this is shown but not committed. */
const CONFIDENCE_THRESHOLD = 0.6

/**
 * How much the newest frame is trusted against the smoothed history.
 * 0.4 means 40% new frame, 60% old — heavy smoothing, which is what makes the
 * model tolerant of MediaPipe's per-frame jitter.
 */
const EMA_ALPHA = 0.4

/** Frames to skip after a prediction, so one sign is not counted repeatedly. */
const COOLDOWN_CONFIDENT = 10
const COOLDOWN_UNSURE = 5

/** Longest sentence held before the oldest word is dropped. */
const MAX_SENTENCE_WORDS = 8

export interface PredictionState {
  currentWord: string
  confidence: number
  isConfident: boolean
  bufferProgress: number
  sentence: string[]
  keypoints: Float32Array | null
  isWaitingForNewSentence: boolean
}

const EMPTY_STATE: PredictionState = {
  currentWord: '',
  confidence: 0,
  isConfident: false,
  bufferProgress: 0,
  sentence: [],
  keypoints: null,
  isWaitingForNewSentence: false,
}

export class SignPredictor {
  private state: PredictionState = EMPTY_STATE
  private listeners = new Set<() => void>()

  private frameBuffer: Float32Array[] = []
  private sentenceWords: string[] = []
  private lastWord = ''
  private previousFrame: Float32Array | null = null
  private cooldownCounter = 0
  private isPredicting = false
  private startNewSentenceNextWord = false

  constructor(private client: PredictClient) {}

  // ── store ──

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  getState = (): PredictionState => this.state

  private update(patch: Partial<PredictionState>): void {
    this.state = { ...this.state, ...patch }
    this.listeners.forEach((l) => l())
  }

  // ── pipeline ──

  /**
   * Feeds one extracted frame through smoothing and the window, firing a
   * prediction when the window is full and the cooldown has elapsed.
   */
  onLandmarksExtracted(rawKeypoints: Float32Array | null): void {
    this.update({ keypoints: rawKeypoints })

    // Nothing detected — clear the smoothing state so the skeleton does not
    // awkwardly morph from a stale pose when hands reappear.
    if (!rawKeypoints) {
      this.previousFrame = null
      return
    }

    const rawNormalized = normalizeSingleFrame(rawKeypoints)
    const smoothed = new Float32Array(rawNormalized.length)
    const prev = this.previousFrame

    if (!prev) {
      // First frame: nothing to smooth against.
      smoothed.set(rawNormalized)
    } else {
      for (let i = 0; i < rawNormalized.length; i++) {
        smoothed[i] = rawNormalized[i] * EMA_ALPHA + prev[i] * (1 - EMA_ALPHA)
      }
    }
    this.previousFrame = smoothed

    this.frameBuffer.push(smoothed)
    if (this.frameBuffer.length > SEQUENCE_LENGTH) this.frameBuffer.shift()

    let snapshot: Float32Array[] | null = null
    if (
      this.frameBuffer.length === SEQUENCE_LENGTH &&
      this.cooldownCounter <= 0 &&
      !this.isPredicting
    ) {
      this.isPredicting = true
      snapshot = [...this.frameBuffer]
    } else if (this.cooldownCounter > 0) {
      this.cooldownCounter--
    }

    this.update({ bufferProgress: this.frameBuffer.length / SEQUENCE_LENGTH })

    if (snapshot) void this.runPrediction(snapshot)
  }

  private async runPrediction(snapshot: Float32Array[]): Promise<void> {
    try {
      const features = buildSequenceFeatures(snapshot)
      const prediction = await this.client.predict(features)
      if (prediction) this.applyPrediction(prediction.label, prediction.confidence)
    } catch (e) {
      console.warn('[predict] window failed', e)
    } finally {
      this.isPredicting = false
    }
  }

  private applyPrediction(word: string, confidence: number): void {
    const isConfident = confidence >= CONFIDENCE_THRESHOLD

    if (isConfident && word !== IDLE) {
      // The hold-then-clear handoff: after a sentence is spoken the old text
      // stays on screen until the signer actually starts the next sign, which
      // is this moment.
      if (this.startNewSentenceNextWord) {
        this.sentenceWords = []
        this.lastWord = ''
        this.startNewSentenceNextWord = false
      }

      // A sign held across several windows predicts repeatedly; only a change
      // of word counts as a new one.
      if (word !== this.lastWord) {
        this.sentenceWords.push(word)
        this.lastWord = word
        if (this.sentenceWords.length > MAX_SENTENCE_WORDS) this.sentenceWords.shift()
      }
    }

    this.update({
      currentWord: isConfident ? word : `${word}?`,
      confidence,
      isConfident,
      sentence: [...this.sentenceWords],
      bufferProgress: 1,
      isWaitingForNewSentence: this.startNewSentenceNextWord,
    })

    this.cooldownCounter = isConfident ? COOLDOWN_CONFIDENT : COOLDOWN_UNSURE
  }

  // ── controls ──

  getSentenceWords(): string[] {
    return [...this.sentenceWords]
  }

  /**
   * Holds the current sentence on screen until the next sign arrives, instead of
   * blanking it the instant translation finishes.
   */
  prepareForNewSentence(): void {
    this.startNewSentenceNextWord = true
    this.update({ isWaitingForNewSentence: true })
  }

  resetAll(): void {
    this.frameBuffer = []
    this.sentenceWords = []
    this.lastWord = ''
    this.previousFrame = null
    this.cooldownCounter = 0
    this.startNewSentenceNextWord = false
    this.state = EMPTY_STATE
    this.listeners.forEach((l) => l())
  }
}
