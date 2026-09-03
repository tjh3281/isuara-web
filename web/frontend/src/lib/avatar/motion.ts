/**
 * Motion data model and parser — ports avatar/model/BimMotionModels.kt and
 * avatar/parser/MotionParser.kt.
 *
 * A motion clip is recorded landmark data: per frame, a handful of body joints
 * plus 21 points per hand, in metres. That is the same pose/leftHand/rightHand
 * shape the recognition pipeline works in, which is why these files drop
 * straight into the browser with no conversion step.
 *
 * Vec3 is a plain object rather than a class with operators: these are read in a
 * 60fps render loop, and the allocation-free helpers below matter more there
 * than operator syntax does.
 */

export interface Vec3 {
  x: number
  y: number
  z: number
}

export const ZERO: Vec3 = { x: 0, y: 0, z: 0 }

export const vec3 = (x: number, y: number, z: number): Vec3 => ({ x, y, z })

export const add = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z })
export const sub = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z })
export const scale = (a: Vec3, s: number): Vec3 => ({ x: a.x * s, y: a.y * s, z: a.z * s })

export const length = (a: Vec3): number => Math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z)

/** Matches the Kotlin: a degenerate vector normalises to UP rather than NaN. */
export function normalize(a: Vec3): Vec3 {
  const len = length(a)
  return len > 0.00001 ? scale(a, 1 / len) : { x: 0, y: 1, z: 0 }
}

export function lerp(a: Vec3, b: Vec3, alpha: number): Vec3 {
  const t = Math.min(Math.max(alpha, 0), 1)
  return {
    x: a.x + (b.x - a.x) * t,
    y: a.y + (b.y - a.y) * t,
    z: a.z + (b.z - a.z) * t,
  }
}

/**
 * 21 hand landmarks for one side.
 * 0: wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky.
 */
export interface HandJoints {
  active: boolean
  points: Vec3[]
}

export interface PoseJoints {
  nose: Vec3 | null
  leftShoulder: Vec3 | null
  rightShoulder: Vec3 | null
  leftElbow: Vec3 | null
  rightElbow: Vec3 | null
  leftWrist: Vec3 | null
  rightWrist: Vec3 | null
}

export interface BimFrame {
  frame: number
  time: number
  pose: PoseJoints
  leftHand: HandJoints | null
  rightHand: HandJoints | null
}

export interface BimMotion {
  word: string
  fps: number
  numFrames: number
  duration: number
  frames: BimFrame[]
}

const EMPTY_POSE: PoseJoints = {
  nose: null,
  leftShoulder: null,
  rightShoulder: null,
  leftElbow: null,
  rightElbow: null,
  leftWrist: null,
  rightWrist: null,
}

function parseVec3(arr: unknown): Vec3 | null {
  if (!Array.isArray(arr) || arr.length < 3) return null
  return { x: Number(arr[0]) || 0, y: Number(arr[1]) || 0, z: Number(arr[2]) || 0 }
}

function parsePose(obj: Record<string, unknown> | undefined): PoseJoints {
  if (!obj) return EMPTY_POSE
  return {
    nose: parseVec3(obj.nose),
    leftShoulder: parseVec3(obj.leftShoulder),
    rightShoulder: parseVec3(obj.rightShoulder),
    leftElbow: parseVec3(obj.leftElbow),
    rightElbow: parseVec3(obj.rightElbow),
    leftWrist: parseVec3(obj.leftWrist),
    rightWrist: parseVec3(obj.rightWrist),
  }
}

function parseHand(obj: Record<string, unknown> | undefined): HandJoints | null {
  if (!obj) return null
  const raw = obj.points
  if (!Array.isArray(raw)) return null
  return {
    active: obj.active !== false,
    // A missing point becomes ZERO rather than dropping out, so indices stay
    // aligned with the bone table.
    points: raw.map((p) => parseVec3(p) ?? ZERO),
  }
}

/** Parses a BIM motion JSON document. */
export function parseMotion(json: unknown): BimMotion {
  const root = (json ?? {}) as Record<string, unknown>
  const fps = Number(root.fps) || 50
  const rawFrames = Array.isArray(root.frames) ? root.frames : []

  const frames: BimFrame[] = rawFrames.map((f, i) => {
    const obj = (f ?? {}) as Record<string, unknown>
    return {
      frame: Number(obj.frame ?? i),
      time: obj.time === undefined ? i / fps : Number(obj.time),
      pose: parsePose(obj.pose as Record<string, unknown> | undefined),
      leftHand: parseHand(obj.leftHand as Record<string, unknown> | undefined),
      rightHand: parseHand(obj.rightHand as Record<string, unknown> | undefined),
    }
  })

  const numFrames = Number(root.num_frames) || frames.length
  const duration = Number(root.duration) || frames.length / fps

  return {
    word: typeof root.word === 'string' ? root.word : 'Unknown',
    fps,
    numFrames,
    duration,
    frames,
  }
}
