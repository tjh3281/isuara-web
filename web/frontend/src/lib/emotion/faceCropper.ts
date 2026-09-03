/**
 * Locates a square face crop from the pose landmarks — ports
 * emotion/FaceCropper.kt.
 *
 * MediaPipe pose indices 0-10 are nose, eyes, ears and mouth. Sizing is driven
 * by the ear-to-ear span rather than by the bounding box of those points,
 * because they stop at the eyebrows and the mouth, so their box omits the
 * forehead and chin the expression model expects to see.
 */

/** Stride of the 258-float feature vector: x, y, z, visibility per landmark. */
const POSE_STRIDE = 4
const MIN_VISIBILITY = 0.5
/** Below this, too little of the face is showing to size a box from. */
const MIN_VISIBLE_POINTS = 3

/** Face width as a multiple of the ear-to-ear span; ears sit inside the outline. */
const EAR_SPAN_TO_FACE_WIDTH = 1.05
/** Used when the head is turned and one ear is occluded. */
const EYE_SPAN_TO_FACE_WIDTH = 2.2
/** Square side as a multiple of face width; covers hairline to chin with margin. */
const BOX_SCALE = 1.5
/**
 * Upward bias as a fraction of the side. The nose tip sits below the vertical
 * middle of the face, so a box centred on it clips the forehead.
 */
const VERTICAL_BIAS = 0.06
/** Reject a crop that lost more than this to the frame edge. */
const MAX_CLIP_FRACTION = 0.2
const MIN_BOX_PX = 48

const NOSE = 0
const LEFT_EYE = 2
const RIGHT_EYE = 5
const LEFT_EAR = 7
const RIGHT_EAR = 8
const LAST_FACE_POINT = 10

export interface FaceBox {
  left: number
  top: number
  width: number
  height: number
}

const x = (k: Float32Array, i: number) => k[i * POSE_STRIDE]
const y = (k: Float32Array, i: number) => k[i * POSE_STRIDE + 1]
const visible = (k: Float32Array, i: number) => k[i * POSE_STRIDE + 3] > MIN_VISIBILITY

/**
 * The square face box, or null when the face cannot be located reliably.
 *
 * `keypoints` is the raw 258-float vector from LandmarkExtractor, normalised to
 * the full frame.
 */
export function faceBox(
  keypoints: Float32Array,
  imageWidth: number,
  imageHeight: number,
): FaceBox | null {
  if (imageWidth <= 0 || imageHeight <= 0) return null
  if (keypoints.length < (LAST_FACE_POINT + 1) * POSE_STRIDE) return null

  let minX = Number.MAX_VALUE
  let maxX = -Number.MAX_VALUE
  let sumX = 0
  let sumY = 0
  let seen = 0

  for (let i = NOSE; i <= LAST_FACE_POINT; i++) {
    if (!visible(keypoints, i)) continue
    const px = x(keypoints, i)
    minX = Math.min(minX, px)
    maxX = Math.max(maxX, px)
    sumX += px
    sumY += y(keypoints, i)
    seen++
  }
  if (seen < MIN_VISIBLE_POINTS) return null

  // Prefer the ear span; fall back to the eyes when the head is turned; fall
  // back again to the raw landmark spread.
  const earSpan =
    visible(keypoints, LEFT_EAR) && visible(keypoints, RIGHT_EAR)
      ? Math.abs(x(keypoints, LEFT_EAR) - x(keypoints, RIGHT_EAR)) * imageWidth
      : 0
  const eyeSpan =
    visible(keypoints, LEFT_EYE) && visible(keypoints, RIGHT_EYE)
      ? Math.abs(x(keypoints, LEFT_EYE) - x(keypoints, RIGHT_EYE)) * imageWidth
      : 0

  const faceWidth = Math.max(
    earSpan * EAR_SPAN_TO_FACE_WIDTH,
    eyeSpan * EYE_SPAN_TO_FACE_WIDTH,
    (maxX - minX) * imageWidth,
  )
  if (faceWidth <= 0) return null

  const side = faceWidth * BOX_SCALE
  const centerX = visible(keypoints, NOSE)
    ? x(keypoints, NOSE) * imageWidth
    : (sumX / seen) * imageWidth
  const centerY =
    (visible(keypoints, NOSE) ? y(keypoints, NOSE) * imageHeight : (sumY / seen) * imageHeight) -
    side * VERTICAL_BIAS

  const idealLeft = centerX - side / 2
  const idealTop = centerY - side / 2

  const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi)
  const left = clamp(idealLeft, 0, imageWidth - 1)
  const top = clamp(idealTop, 0, imageHeight - 1)
  const right = clamp(idealLeft + side, 0, imageWidth)
  const bottom = clamp(idealTop + side, 0, imageHeight)

  const width = Math.round(right - left)
  const height = Math.round(bottom - top)
  if (width < MIN_BOX_PX || height < MIN_BOX_PX) return null

  // Reject a badly clipped face rather than handing the model a partial one.
  if (1 - Math.min(width, height) / side > MAX_CLIP_FRACTION) return null

  // Keep it square: a stretched face shifts every geometric cue the classifier
  // relies on.
  const squareSide = Math.min(width, height)
  return { left: Math.round(left), top: Math.round(top), width: squareSide, height: squareSide }
}
