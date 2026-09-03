/**
 * FrameNormalizer — TypeScript port of ml/FrameNormalizer.kt.
 *
 * Stages 1-5 only. Stage 6 (z-score) is BAKED INTO the model, so nothing here
 * applies it — see the export cell in iSuara_Train_V3_1_5_WithOutput.ipynb,
 * which wraps the trained model in `(x - mean) / std` before conversion.
 *
 * Keypoint layout (258 floats):
 *   [0..131]    Pose:       33 landmarks x 4 (x, y, z, visibility)
 *   [132..194]  Left Hand:  21 landmarks x 3 (x, y, z)
 *   [195..257]  Right Hand: 21 landmarks x 3 (x, y, z)
 *
 * Kept deliberately close to the Kotlin, index arithmetic included, because the
 * two must agree exactly: a transposed offset here would not crash, it would
 * quietly degrade recognition in a way that looks like a bad model.
 */

export const RAW_FEATURES = 258
export const FINAL_FEATURES = 780
export const SEQUENCE_LENGTH = 30

/**
 * Stage 1 (Anchor Subtraction) + Stage 2 (Scale by Shoulder Width) for a
 * single frame.
 *
 * Returns a new array of 258 floats; the input is not modified.
 */
export function normalizeSingleFrame(frame: Float32Array): Float32Array {
  if (frame.length !== RAW_FEATURES) {
    throw new Error(`Expected ${RAW_FEATURES} features, got ${frame.length}`)
  }
  const f = new Float32Array(frame)

  // -- Stage 1: Anchor Subtraction --

  // Shoulder landmarks: left=[44..46](xyz), right=[48..50](xyz)
  const shoulderLx = f[44], shoulderLy = f[45], shoulderLz = f[46]
  const shoulderRx = f[48], shoulderRy = f[49], shoulderRz = f[50]

  // Pose anchor = shoulder midpoint
  const anchorX = (shoulderLx + shoulderRx) / 2
  const anchorY = (shoulderLy + shoulderRy) / 2
  const anchorZ = (shoulderLz + shoulderRz) / 2

  // Subtract anchor from pose xyz (33 landmarks x 4, only xyz not visibility)
  if (anchorX !== 0 || anchorY !== 0 || anchorZ !== 0) {
    for (let i = 0; i < 33; i++) {
      const base = i * 4
      f[base] -= anchorX
      f[base + 1] -= anchorY
      f[base + 2] -= anchorZ
      // f[base + 3] = visibility — unchanged
    }
  }

  // Left hand: anchor to left wrist
  const lhWristX = f[132], lhWristY = f[133], lhWristZ = f[134]
  if (lhWristX !== 0 || lhWristY !== 0 || lhWristZ !== 0) {
    for (let i = 0; i < 21; i++) {
      const base = 132 + i * 3
      f[base] -= lhWristX
      f[base + 1] -= lhWristY
      f[base + 2] -= lhWristZ
    }
  }

  // Right hand: anchor to right wrist
  const rhWristX = f[195], rhWristY = f[196], rhWristZ = f[197]
  if (rhWristX !== 0 || rhWristY !== 0 || rhWristZ !== 0) {
    for (let i = 0; i < 21; i++) {
      const base = 195 + i * 3
      f[base] -= rhWristX
      f[base + 1] -= rhWristY
      f[base + 2] -= rhWristZ
    }
  }

  // -- Stage 2: Scale by Shoulder Width --
  // Uses the ORIGINAL shoulder coordinates, read before Stage 1 shifted them.
  const dx = shoulderLx - shoulderRx
  const dy = shoulderLy - shoulderRy
  const dz = shoulderLz - shoulderRz
  const shoulderWidth = Math.sqrt(dx * dx + dy * dy + dz * dz)

  if (shoulderWidth > 0.01) {
    const invWidth = 1 / shoulderWidth

    // Scale pose xyz (skip visibility at every 4th element)
    for (let i = 0; i < 33; i++) {
      const base = i * 4
      f[base] *= invWidth
      f[base + 1] *= invWidth
      f[base + 2] *= invWidth
    }

    // Scale left hand
    for (let i = 132; i < 195; i++) f[i] *= invWidth

    // Scale right hand
    for (let i = 195; i < 258; i++) f[i] *= invWidth
  }

  return f
}

/**
 * Stages 3 (Velocity) + 4 (Acceleration) + 5 (Engineered Features) for a full
 * 30-frame sequence.
 *
 * Returns 30 x 780 = 23,400 floats, row-major [frame][feature] — the exact
 * buffer the model's input tensor expects.
 */
export function buildSequenceFeatures(sequence: Float32Array[]): Float32Array {
  if (sequence.length !== SEQUENCE_LENGTH) {
    throw new Error(`Expected ${SEQUENCE_LENGTH} frames, got ${sequence.length}`)
  }
  const t = sequence.length
  const n = RAW_FEATURES

  // Stage 3: Velocity (first derivative). Frame 0 stays zero — there is no
  // earlier frame to difference against, matching the training pipeline.
  const velocity: Float32Array[] = Array.from({ length: t }, () => new Float32Array(n))
  for (let i = 1; i < t; i++) {
    const cur = sequence[i], prev = sequence[i - 1], out = velocity[i]
    for (let j = 0; j < n; j++) out[j] = cur[j] - prev[j]
  }

  // Stage 4: Acceleration (second derivative)
  const acceleration: Float32Array[] = Array.from({ length: t }, () => new Float32Array(n))
  for (let i = 1; i < t; i++) {
    const cur = velocity[i], prev = velocity[i - 1], out = acceleration[i]
    for (let j = 0; j < n; j++) out[j] = cur[j] - prev[j]
  }

  // Stage 5: Engineered features (6 per frame)
  // Indices in normalized pose:
  //   left wrist xyz  = [60, 61, 62]  (pose landmark 15: 15*4=60)
  //   right wrist xyz = [64, 65, 66]  (pose landmark 16: 16*4=64)
  //   nose xyz        = [0, 1, 2]     (pose landmark 0)
  const engineered: Float32Array[] = Array.from({ length: t }, () => new Float32Array(6))
  for (let i = 0; i < t; i++) {
    const s = sequence[i]
    const lwX = s[60], lwY = s[61], lwZ = s[62]
    const rwX = s[64], rwY = s[65], rwZ = s[66]
    const noseX = s[0], noseY = s[1], noseZ = s[2]
    const e = engineered[i]

    // [0] Distance between wrists
    const dwX = lwX - rwX, dwY = lwY - rwY, dwZ = lwZ - rwZ
    e[0] = Math.sqrt(dwX * dwX + dwY * dwY + dwZ * dwZ)

    // [1:4] Wrist difference vector
    e[1] = dwX
    e[2] = dwY
    e[3] = dwZ

    // [4] Left wrist to nose distance
    const lnX = lwX - noseX, lnY = lwY - noseY, lnZ = lwZ - noseZ
    e[4] = Math.sqrt(lnX * lnX + lnY * lnY + lnZ * lnZ)

    // [5] Right wrist to nose distance
    const rnX = rwX - noseX, rnY = rwY - noseY, rnZ = rwZ - noseZ
    e[5] = Math.sqrt(rnX * rnX + rnY * rnY + rnZ * rnZ)
  }

  // Concatenate: [position(258) | velocity(258) | acceleration(258) | engineered(6)] = 780
  const result = new Float32Array(t * FINAL_FEATURES)
  for (let i = 0; i < t; i++) {
    const offset = i * FINAL_FEATURES
    result.set(sequence[i], offset)
    result.set(velocity[i], offset + n)
    result.set(acceleration[i], offset + n * 2)
    result.set(engineered[i], offset + n * 3)
  }

  return result
}
