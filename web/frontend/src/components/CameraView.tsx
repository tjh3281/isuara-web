/**
 * The camera panel — the Box(aspectRatio(3f / 4f)) block of ui/CameraScreen.kt.
 *
 * Geometry, colours and icons are matched to the Compose source rather than
 * chosen: the preview is portrait 3:4 (Compose's aspectRatio takes width/height,
 * so 3f/4f is 0.75 — taller than wide, not landscape), the skeleton uses
 * Compose's fully saturated Color.Green / Color.Magenta / Color.Cyan, and the
 * two HUD buttons are Icons.Default.Settings and Icons.Default.Sync.
 *
 * The preview is mirrored only for a user-facing camera, because that is what
 * looks right to someone signing at their own screen. The frame the model sees
 * is always mirrored regardless — see LandmarkExtractor — so the overlay undoes
 * the difference when the two disagree.
 */

import { useEffect, useRef } from 'react'

import type { FacingMode, PipelineStatus } from '../hooks/useSignPipeline'
import { MIN_CONFIDENCE, type EmotionReading } from '../lib/emotion/classifier'

interface Props {
  videoRef: React.RefObject<HTMLVideoElement>
  keypoints: Float32Array | null
  bufferProgress: number
  fps: number
  facingMode: FacingMode
  showLandmarks: boolean
  status: PipelineStatus
  emotion: EmotionReading | null
  onToggleLandmarks: () => void
  onSwitchCamera: () => void
}

/** Compose's named colours, not softened equivalents. */
const POSE = '#00FF00' // Color.Green
const HAND_LEFT = '#FF00FF' // Color.Magenta
const HAND_RIGHT = '#00FFFF' // Color.Cyan

/**
 * Compose draws at radius = 6f device pixels on a preview roughly 1080px wide.
 * Scaling by width keeps the dots the same relative size here as on the phone
 * instead of the same absolute size, which would look enormous on a narrow
 * canvas.
 */
const RADIUS_RATIO = 6 / 1080

export function CameraView({
  videoRef,
  keypoints,
  bufferProgress,
  fps,
  facingMode,
  showLandmarks,
  status,
  emotion,
  onToggleLandmarks,
  onSwitchCamera,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const mirroredPreview = facingMode === 'user'

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return

    // Match the backing store to the rendered size, or the overlay lands at the
    // wrong scale on high-DPI screens.
    const { width, height } = canvas.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
      canvas.width = width * dpr
      canvas.height = height * dpr
    }

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, width, height)
    if (!showLandmarks || !keypoints) return

    // Landmarks are normalized against the mirrored ML frame. When the preview
    // is not mirrored, flip x back so the dots land on the visible hands.
    const mapX = (xNorm: number) => (mirroredPreview ? xNorm : 1 - xNorm) * width
    const mapY = (yNorm: number) => yNorm * height
    const radius = Math.max(1.5, width * RADIUS_RATIO)

    const dot = (x: number, y: number, color: string) => {
      ctx.fillStyle = color
      ctx.beginPath()
      ctx.arc(x, y, radius, 0, Math.PI * 2)
      ctx.fill()
    }

    // Pose: 33 landmarks x 4, gated on visibility > 0.5 as in the Kotlin.
    for (let i = 0; i < 33; i++) {
      if (keypoints[i * 4 + 3] > 0.5) {
        dot(mapX(keypoints[i * 4]), mapY(keypoints[i * 4 + 1]), POSE)
      }
    }
    // Hands: 21 landmarks x 3 each. A zero x means the hand was not detected.
    for (const [base, color] of [
      [132, HAND_LEFT],
      [195, HAND_RIGHT],
    ] as const) {
      for (let i = 0; i < 21; i++) {
        const idx = base + i * 3
        if (keypoints[idx] > 0) dot(mapX(keypoints[idx]), mapY(keypoints[idx + 1]), color)
      }
    }
  }, [keypoints, showLandmarks, mirroredPreview])

  return (
    <div className="camera">
      <video
        ref={videoRef}
        className="camera__video"
        style={{ transform: mirroredPreview ? 'scaleX(-1)' : 'none' }}
        playsInline
        muted
      />
      <canvas ref={canvasRef} className="camera__overlay" />

      {status.phase !== 'running' && (
        <div className="camera__placeholder">
          {status.phase === 'error' ? (
            <p className="camera__error">{status.message}</p>
          ) : (
            <p>{status.message || 'Starting camera…'}</p>
          )}
        </div>
      )}

      <div className="camera__hud">
        <div className="camera__hud-left">
          <span className="badge">{fps} FPS</span>
          {/*
            The observed expression — emotion/ui/EmotionChip.kt. Only shown once
            a reading clears the confidence floor; below that a hint is worse
            than none, since neither the user nor the model can tell how much to
            discount it.
          */}
          {emotion && emotion.confidence >= MIN_CONFIDENCE && (
            <span
              className={`badge badge--emotion${emotion.isHighArousal ? ' badge--aroused' : ''}`}
              title={`${Math.round(emotion.confidence * 100)}% confident`}
            >
              {emotion.descriptorMs}
            </span>
          )}
        </div>
        <div className="camera__hud-actions">
          {/* Icons.Default.Settings */}
          <button
            type="button"
            className={`icon-button${showLandmarks ? ' icon-button--active' : ''}`}
            onClick={onToggleLandmarks}
            aria-pressed={showLandmarks}
            title="Toggle Landmarks"
          >
            <svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor" aria-hidden="true">
              <path d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.09.63-.09.94s.02.64.07.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z" />
            </svg>
            <span className="sr-only">Toggle Landmarks</span>
          </button>

          {/* Icons.Default.Sync */}
          <button type="button" className="icon-button" onClick={onSwitchCamera} title="Switch Camera">
            <svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor" aria-hidden="true">
              <path d="M12 4V1L8 5l4 4V6c3.31 0 6 2.69 6 6 0 1.01-.25 1.97-.7 2.8l1.46 1.46C19.54 15.03 20 13.57 20 12c0-4.42-3.58-8-8-8zm0 14c-3.31 0-6-2.69-6-6 0-1.01.25-1.97.7-2.8L5.24 7.74C4.46 8.97 4 10.43 4 12c0 4.42 3.58 8 8 8v3l4-4-4-4v3z" />
            </svg>
            <span className="sr-only">Switch Camera</span>
          </button>
        </div>
      </div>

      <div
        className="camera__buffer"
        role="progressbar"
        aria-valuenow={Math.round(bufferProgress * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div className="camera__buffer-fill" style={{ width: `${bufferProgress * 100}%` }} />
      </div>
    </div>
  )
}
