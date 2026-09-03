/**
 * Playback and per-frame pose evaluation — ports avatar/engine/MotionPlayer.kt
 * and avatar/engine/AvatarPoseState.kt.
 *
 * Capture runs at the clip's own fps; the renderer runs at the display's. So a
 * pose is evaluated at an arbitrary time by interpolating between the two
 * neighbouring frames, which is what makes playback smooth rather than stepping
 * at the capture rate.
 *
 * Shoulders and torso are fixed anatomy rather than captured — the recording
 * only carries the head, wrists and fingers, so the body is a stable frame the
 * solved arms hang from.
 */

import { solveNaturalElbow, scaleAndClampHandKeypoints } from './kinematics'
import { add, lerp, scale, vec3, type BimFrame, type BimMotion, type Vec3 } from './motion'

export interface AvatarPoseState {
  headPosition: Vec3
  neckStart: Vec3
  neckEnd: Vec3
  torsoPosition: Vec3
  leftShoulder: Vec3
  leftElbow: Vec3
  leftWrist: Vec3
  rightShoulder: Vec3
  rightElbow: Vec3
  rightWrist: Vec3
  leftPalmPos: Vec3 | null
  leftPalmLookAt: Vec3 | null
  rightPalmPos: Vec3 | null
  rightPalmLookAt: Vec3 | null
  leftHandPoints: Vec3[] | null
  rightHandPoints: Vec3[] | null
}

/** Fixed skeleton anatomy, in metres. Mirrors the Kotlin literals exactly. */
const L_SHOULDER = vec3(0.19, 1.15, 0)
const R_SHOULDER = vec3(-0.19, 1.15, 0)
const MID_SHOULDER = vec3(0, 1.15, 0)
const TORSO = vec3(0, 0.93, 0)

/** Where a wrist rests when the capture has no hand for that side. */
const L_WRIST_REST = vec3(0.23, 0.72, 0.12)
const R_WRIST_REST = vec3(-0.23, 0.72, 0.12)

export const REST_POSE: AvatarPoseState = {
  headPosition: vec3(0, 1.34, 0),
  neckStart: vec3(0, 1.15, -0.015),
  neckEnd: vec3(0, 1.22, -0.015),
  torsoPosition: TORSO,
  leftShoulder: L_SHOULDER,
  leftElbow: solveNaturalElbow(L_SHOULDER, L_WRIST_REST, true),
  leftWrist: L_WRIST_REST,
  rightShoulder: R_SHOULDER,
  rightElbow: solveNaturalElbow(R_SHOULDER, R_WRIST_REST, false),
  rightWrist: R_WRIST_REST,
  leftPalmPos: null,
  leftPalmLookAt: null,
  rightPalmPos: null,
  rightPalmLookAt: null,
  leftHandPoints: null,
  rightHandPoints: null,
}

export class MotionPlayer {
  currentMotion: BimMotion | null = null
  isPlaying = false
  playbackSpeed = 1
  currentTimeSec = 0
  /** Signs play once by default; looping would read as a stutter, not a sign. */
  isLooping = false

  private pose: AvatarPoseState = REST_POSE

  setMotion(motion: BimMotion, autoPlay = false): void {
    this.currentMotion = motion
    this.currentTimeSec = 0
    this.isPlaying = autoPlay
    this.pose = this.evaluatePose(0)
  }

  seekTo(timeSec: number): void {
    const duration = this.currentMotion?.duration ?? 0
    this.currentTimeSec = duration > 0 ? Math.min(Math.max(timeSec, 0), duration) : 0
    this.pose = this.evaluatePose(this.currentTimeSec)
  }

  play(): void {
    // Replaying a finished clip restarts it rather than doing nothing.
    const duration = this.currentMotion?.duration ?? 0
    if (duration > 0 && this.currentTimeSec >= duration) this.currentTimeSec = 0
    this.isPlaying = true
  }

  pause(): void {
    this.isPlaying = false
  }

  getCurrentPose(): AvatarPoseState {
    return this.pose
  }

  /** Advances by a wall-clock delta and returns the pose to draw. */
  advanceTime(deltaSec: number): AvatarPoseState {
    const motion = this.currentMotion
    if (!motion || motion.duration <= 0) return this.pose

    if (this.isPlaying) {
      this.currentTimeSec += deltaSec * this.playbackSpeed
      if (this.currentTimeSec >= motion.duration) {
        if (this.isLooping) {
          this.currentTimeSec %= motion.duration
        } else {
          this.currentTimeSec = motion.duration
          this.isPlaying = false
        }
      }
    }

    this.pose = this.evaluatePose(this.currentTimeSec)
    return this.pose
  }

  /** Evaluates the pose at an arbitrary time, interpolating between frames. */
  evaluatePose(timeSec: number): AvatarPoseState {
    const motion = this.currentMotion
    if (!motion || motion.frames.length === 0) return REST_POSE

    const frames = motion.frames
    const fps = motion.fps > 0 ? motion.fps : 50
    const exact = timeSec * fps
    const i0 = Math.min(Math.max(Math.floor(exact), 0), frames.length - 1)
    const i1 = Math.min(i0 + 1, frames.length - 1)
    const alpha = exact - i0

    const f0 = frames[i0]
    const f1 = frames[i1]

    // Head tracks the nose horizontally, damped — the capture's nose moves more
    // than a head should, and the full excursion looks like a wobble.
    const nose0 = f0.pose.nose
    const nose1 = f1.pose.nose ?? nose0
    const nose = nose0 && nose1 ? lerp(nose0, nose1, alpha) : nose0
    const headX = nose ? (nose.x - MID_SHOULDER.x) * 0.35 : 0

    const rightPoints = interpolateHand(f0, f1, alpha, false)
    const leftPoints = interpolateHand(f0, f1, alpha, true)

    const rightWrist = rightPoints?.[0] ?? R_WRIST_REST
    const leftWrist = leftPoints?.[0] ?? L_WRIST_REST

    const leftPalm = palm(leftPoints)
    const rightPalm = palm(rightPoints)

    return {
      headPosition: vec3(headX, 1.34, 0),
      neckStart: vec3(0, 1.15, -0.015),
      neckEnd: vec3(headX, 1.22, -0.015),
      torsoPosition: TORSO,
      leftShoulder: L_SHOULDER,
      leftElbow: solveNaturalElbow(L_SHOULDER, leftWrist, true),
      leftWrist,
      rightShoulder: R_SHOULDER,
      rightElbow: solveNaturalElbow(R_SHOULDER, rightWrist, false),
      rightWrist,
      leftPalmPos: leftPalm.pos,
      leftPalmLookAt: leftPalm.lookAt,
      rightPalmPos: rightPalm.pos,
      rightPalmLookAt: rightPalm.lookAt,
      leftHandPoints: leftPoints,
      rightHandPoints: rightPoints,
    }
  }
}

/**
 * Palm centre and facing, from the wrist and the middle-finger knuckle.
 *
 * Those two points define the plane of the hand, which is what orients the palm;
 * a hand drawn without it reads as a flat sticker.
 */
function palm(points: Vec3[] | null): { pos: Vec3 | null; lookAt: Vec3 | null } {
  if (!points || points.length < 10) return { pos: null, lookAt: null }
  return { pos: scale(add(points[0], points[9]), 0.5), lookAt: points[9] }
}

function interpolateHand(
  f0: BimFrame,
  f1: BimFrame,
  alpha: number,
  isLeft: boolean,
): Vec3[] | null {
  const h0 = isLeft ? f0.leftHand : f0.rightHand
  const h1 = (isLeft ? f1.leftHand : f1.rightHand) ?? h0

  const p0 = h0?.points ?? h1?.points
  const p1 = h1?.points ?? p0
  if (!p0 || !p1 || p0.length !== 21 || p1.length !== 21) return null

  const raw: Vec3[] = []
  for (let i = 0; i < 21; i++) raw.push(lerp(p0[i], p1[i], alpha))
  return scaleAndClampHandKeypoints(raw)
}
