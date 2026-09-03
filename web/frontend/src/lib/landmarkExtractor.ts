/**
 * LandmarkExtractor — port of ml/LandmarkExtractor.kt onto MediaPipe Tasks
 * Vision for the Web.
 *
 * Produces the same 258-float frame the Android app does:
 *   [0..131]    Pose:       33 landmarks x 4 (x, y, z, visibility)
 *   [132..194]  Left Hand:  21 landmarks x 3 (x, y, z)
 *   [195..257]  Right Hand: 21 landmarks x 3 (x, y, z)
 *
 * Two things about the port are worth knowing:
 *
 * 1. The Android version is built around an async result listener and a
 *    ConcurrentHashMap of in-flight frames, because LIVE_STREAM mode delivers
 *    pose and hand results on separate callbacks that must be joined by
 *    timestamp. The web API's VIDEO mode returns synchronously, so all of that
 *    machinery collapses into two straight-line calls.
 *
 * 2. The frame handed to MediaPipe is ALWAYS mirrored. Android reaches the same
 *    state by two routes — it flips the bitmap for the front camera, and for the
 *    rear camera it instead negates x and swaps the left/right pose indices,
 *    which is the same mirror expressed arithmetically. The model was trained on
 *    mirrored input, so the web build just does the flip once, on the canvas,
 *    for every camera.
 *
 * The dynamic hand crop is ported as-is: it is what extends the usable range
 * from ~50cm to ~1.5m, by using the previous frame's wrist positions to zoom the
 * hand detector in on where the hands actually are.
 */

import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  type HandLandmarkerResult,
  type PoseLandmarkerResult,
} from '@mediapipe/tasks-vision'

import { RAW_FEATURES } from './frameNormalizer'

/**
 * Static assets copied out of app/src/main/assets by scripts/copy-assets.mjs.
 *
 * BASE_URL, not a leading slash: GitHub Pages serves a project site from
 * /<repo>/, so root-absolute paths 404 there. Vite substitutes the configured
 * base at build time, and "/" during local dev.
 */
const BASE = import.meta.env.BASE_URL
const POSE_MODEL = `${BASE}models/pose_landmarker_lite.task`
const HAND_MODEL = `${BASE}models/hand_landmarker.task`
const WASM_PATH = `${BASE}wasm`

/** Matches the ImageAnalysis resolution the Android app requests. */
const FRAME_WIDTH = 480
const FRAME_HEIGHT = 360

export interface ExtractedFrame {
  /** 258 raw keypoints, or null when neither a body nor a hand was found. */
  features: Float32Array | null
}

export class LandmarkExtractor {
  private poseLandmarker: PoseLandmarker
  private handLandmarker: HandLandmarker

  /** Full mirrored frame fed to the pose detector. */
  private frameCanvas: HTMLCanvasElement
  private frameCtx: CanvasRenderingContext2D

  /** Reused crop surface for the hand detector. */
  private cropCanvas: HTMLCanvasElement
  private cropCtx: CanvasRenderingContext2D

  // Cached pose state, used to place the next frame's hand crop.
  private lastLeftWristNormX = -1
  private lastLeftWristNormY = -1
  private lastRightWristNormX = -1
  private lastRightWristNormY = -1
  private lastShoulderWidthPx = 150

  private lastTimestamp = -1

  private constructor(pose: PoseLandmarker, hand: HandLandmarker) {
    this.poseLandmarker = pose
    this.handLandmarker = hand

    this.frameCanvas = document.createElement('canvas')
    this.frameCanvas.width = FRAME_WIDTH
    this.frameCanvas.height = FRAME_HEIGHT
    // willReadFrequently is wrong here: MediaPipe uploads the canvas to the GPU
    // rather than reading pixels back, so the CPU-backed path would be slower.
    this.frameCtx = this.frameCanvas.getContext('2d', { alpha: false })!

    this.cropCanvas = document.createElement('canvas')
    this.cropCtx = this.cropCanvas.getContext('2d', { alpha: false })!
  }

  static async create(): Promise<LandmarkExtractor> {
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH)

    // GPU first, CPU fallback — the same two-attempt strategy SignInterpreter.kt
    // uses for the classifier, for the same reason: a machine without a working
    // WebGL delegate should run slowly, not fail to start.
    const build = async (delegate: 'GPU' | 'CPU') =>
      Promise.all([
        PoseLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: POSE_MODEL, delegate },
          runningMode: 'VIDEO',
          numPoses: 1,
        }),
        HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: HAND_MODEL, delegate },
          runningMode: 'VIDEO',
          numHands: 2,
        }),
      ])

    let pose: PoseLandmarker
    let hand: HandLandmarker
    try {
      ;[pose, hand] = await build('GPU')
    } catch (e) {
      console.warn('[LandmarkExtractor] GPU delegate failed, falling back to CPU', e)
      ;[pose, hand] = await build('CPU')
    }

    return new LandmarkExtractor(pose, hand)
  }

  /**
   * Runs both detectors over one video frame and returns the joined 258 floats.
   *
   * `timestampMs` must strictly increase — MediaPipe's VIDEO mode rejects a
   * repeated or rewound timestamp, and a dropped frame that reuses the previous
   * value would throw rather than merely be ignored.
   */
  extract(video: HTMLVideoElement, timestampMs: number): ExtractedFrame {
    if (timestampMs <= this.lastTimestamp) return { features: null }
    this.lastTimestamp = timestampMs

    // Mirror once, here, for every camera — see the note at the top.
    this.frameCtx.save()
    this.frameCtx.translate(FRAME_WIDTH, 0)
    this.frameCtx.scale(-1, 1)
    this.frameCtx.drawImage(video, 0, 0, FRAME_WIDTH, FRAME_HEIGHT)
    this.frameCtx.restore()

    const features = new Float32Array(RAW_FEATURES)
    let hasData = false

    // 1. Pose always runs on the full frame FIRST — the hand crop depends on it.
    const poseResult = this.poseLandmarker.detectForVideo(this.frameCanvas, timestampMs)
    if (this.applyPose(poseResult, features)) hasData = true

    // 2. Hands run on a crop placed from the wrists cached above.
    const crop = this.planHandCrop()
    const handResult = this.handLandmarker.detectForVideo(
      this.drawCrop(crop),
      timestampMs,
    )
    if (this.applyHands(handResult, crop, features)) hasData = true

    return { features: hasData ? features : null }
  }

  private applyPose(result: PoseLandmarkerResult, features: Float32Array): boolean {
    const landmarks = result.landmarks?.[0]
    if (!landmarks || landmarks.length === 0) {
      // Nobody on screen — reset the crop trackers so the next crop starts cold
      // rather than anchoring on a stale position.
      this.lastLeftWristNormX = -1
      this.lastRightWristNormX = -1
      return false
    }

    for (let i = 0; i < 33; i++) {
      const lm = landmarks[i]
      const idx = i * 4
      features[idx] = lm.x
      features[idx + 1] = lm.y
      features[idx + 2] = lm.z
      features[idx + 3] = lm.visibility ?? 0
    }

    // Cache wrists for the next frame's crop, gated on visibility so a
    // hallucinated wrist does not drag the crop off the signer.
    const leftWrist = landmarks[15]
    const rightWrist = landmarks[16]
    const leftVisible = (leftWrist.visibility ?? 0) > 0.5
    const rightVisible = (rightWrist.visibility ?? 0) > 0.5
    this.lastLeftWristNormX = leftVisible ? leftWrist.x : -1
    this.lastLeftWristNormY = leftVisible ? leftWrist.y : -1
    this.lastRightWristNormX = rightVisible ? rightWrist.x : -1
    this.lastRightWristNormY = rightVisible ? rightWrist.y : -1

    // Dynamic padding based on shoulder width, so the crop scales with how far
    // away the signer is standing.
    const dx = (landmarks[11].x - landmarks[12].x) * FRAME_WIDTH
    const dy = (landmarks[11].y - landmarks[12].y) * FRAME_HEIGHT
    this.lastShoulderWidthPx = Math.max(Math.sqrt(dx * dx + dy * dy), 100)

    return true
  }

  /**
   * Where to crop for the hand detector, in pixels on the mirrored frame.
   *
   * Falls back to the whole frame on a cold start, when both wrists are hidden,
   * or when the computed box is degenerate.
   */
  private planHandCrop(): { x: number; y: number; width: number; height: number } {
    const full = { x: 0, y: 0, width: FRAME_WIDTH, height: FRAME_HEIGHT }

    const xs: number[] = []
    const ys: number[] = []
    if (this.lastLeftWristNormX >= 0) {
      xs.push(this.lastLeftWristNormX)
      ys.push(this.lastLeftWristNormY)
    }
    if (this.lastRightWristNormX >= 0) {
      xs.push(this.lastRightWristNormX)
      ys.push(this.lastRightWristNormY)
    }
    if (xs.length === 0) return full

    const clamp01 = (v: number) => Math.min(Math.max(v, 0), 1)
    const paddingXNorm = (this.lastShoulderWidthPx * 1.5) / FRAME_WIDTH
    const paddingYNorm = (this.lastShoulderWidthPx * 1.5) / FRAME_HEIGHT

    // Shift the crop upwards, because fingers sit above the wrist.
    const yOffsetNorm = paddingYNorm * 0.4

    const startXNorm = clamp01(Math.min(...xs) - paddingXNorm)
    const endXNorm = clamp01(Math.max(...xs) + paddingXNorm)
    const startYNorm = clamp01(Math.min(...ys) - paddingYNorm - yOffsetNorm)
    const endYNorm = clamp01(Math.max(...ys) + paddingYNorm - yOffsetNorm)

    const box = {
      x: Math.round(startXNorm * FRAME_WIDTH),
      y: Math.round(startYNorm * FRAME_HEIGHT),
      width: Math.round((endXNorm - startXNorm) * FRAME_WIDTH),
      height: Math.round((endYNorm - startYNorm) * FRAME_HEIGHT),
    }

    return box.width > 10 && box.height > 10 ? box : full
  }

  private drawCrop(crop: { x: number; y: number; width: number; height: number }): HTMLCanvasElement {
    if (crop.x === 0 && crop.y === 0 && crop.width === FRAME_WIDTH && crop.height === FRAME_HEIGHT) {
      return this.frameCanvas
    }
    if (this.cropCanvas.width !== crop.width || this.cropCanvas.height !== crop.height) {
      this.cropCanvas.width = crop.width
      this.cropCanvas.height = crop.height
    }
    this.cropCtx.drawImage(
      this.frameCanvas,
      crop.x, crop.y, crop.width, crop.height,
      0, 0, crop.width, crop.height,
    )
    return this.cropCanvas
  }

  private applyHands(
    result: HandLandmarkerResult,
    crop: { x: number; y: number; width: number; height: number },
    features: Float32Array,
  ): boolean {
    if (!result.landmarks || result.landmarks.length === 0) return false

    for (let i = 0; i < result.landmarks.length; i++) {
      // The frame is mirrored, so MediaPipe's handedness label already refers to
      // the signer's own hand as a viewer in a mirror would name it — which is
      // the convention the feature layout uses.
      const isLeft = result.handednesses[i]?.[0]?.categoryName === 'Left'
      const offset = isLeft ? 132 : 195

      for (let j = 0; j < 21; j++) {
        const lm = result.landmarks[i][j]
        const idx = offset + j * 3

        // Remap from crop space back to full-frame space, so the classifier
        // sees hand coordinates in the same frame of reference as the pose.
        features[idx] = (crop.x + lm.x * crop.width) / FRAME_WIDTH
        features[idx + 1] = (crop.y + lm.y * crop.height) / FRAME_HEIGHT
        features[idx + 2] = lm.z
      }
    }
    return true
  }

  close(): void {
    this.poseLandmarker.close()
    this.handLandmarker.close()
  }
}
