/**
 * Inverse kinematics and hand conditioning — ports avatar/engine/KinematicSolver.kt.
 *
 * The capture gives wrist and finger positions but no elbow, so the elbow is
 * solved rather than recorded. The constants are anatomical and were tuned
 * against the Android build; they are not free parameters.
 */

import { add, normalize, scale, sub, vec3, type Vec3 } from './motion'

/** Humanoid limb lengths, metres. */
export const UPPER_ARM_LENGTH = 0.26
export const FOREARM_LENGTH = 0.24

/** Anatomical constraints matching SignAvatar-Muba. */
export const FINGER_SCALE = 0.72
export const Y_MAX_HEAD = 1.58
export const Y_MIN_WAIST = 0.62

/**
 * Bone connections for the 21 hand landmarks.
 * [0,1..4] thumb, [0,5..8] index, [0,9..12] middle, [0,13..16] ring, [0,17..20] pinky.
 */
export const HAND_BONES: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
]

/**
 * Places the elbow from shoulder and wrist.
 *
 * Takes the midpoint, pushes it outward, down and forward so the arm bends the
 * way a human arm does rather than collapsing into the torso, then constrains
 * the upper arm to its exact length so the limb cannot stretch.
 */
export function solveNaturalElbow(shoulder: Vec3, wrist: Vec3, isLeft: boolean): Vec3 {
  const mid = scale(add(shoulder, wrist), 0.5)
  const sign = isLeft ? 1 : -1
  const flare = vec3(sign * 0.12, -0.05, 0.1)
  const target = add(mid, flare)
  const dir = normalize(sub(target, shoulder))
  return add(shoulder, scale(dir, UPPER_ARM_LENGTH))
}

/**
 * Shrinks finger spread toward the wrist and clamps height to the signing box.
 *
 * The wrist itself is the anchor and is never moved, so the hand stays attached
 * to the arm the IK just solved. The Y clamp keeps a noisy capture frame from
 * flinging a finger above the head or below the waist.
 */
export function scaleAndClampHandKeypoints(raw: Vec3[]): Vec3[] {
  if (raw.length !== 21) return raw

  const wrist = raw[0]
  const out: Vec3[] = [wrist]

  for (let i = 1; i < 21; i++) {
    const p = add(wrist, scale(sub(raw[i], wrist), FINGER_SCALE))
    out.push(vec3(p.x, Math.min(Math.max(p.y, Y_MIN_WAIST), Y_MAX_HEAD), p.z))
  }

  return out
}
