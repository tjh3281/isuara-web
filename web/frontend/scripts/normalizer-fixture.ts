/**
 * Runs the real browser normalizer over a fixture, so a Python reference can
 * check it element by element.
 *
 * This imports src/lib/frameNormalizer.ts directly — the exact module the app
 * ships, not a copy — so the parity test cannot drift away from what actually
 * runs. Node 24 strips the types natively, which is why this needs no build
 * step and no test-runner dependency.
 *
 *   node scripts/normalizer-fixture.ts <input.json> <output.json>
 *
 * Input : { "frames": number[30][258] }
 * Output: { "normalized": number[30][258], "features": number[23400] }
 *
 * Both stages are emitted, not just the final buffer, so a failure points at
 * which stage broke instead of only saying the end result differs.
 */

import { readFileSync, writeFileSync } from 'node:fs'

import {
  buildSequenceFeatures,
  normalizeSingleFrame,
  RAW_FEATURES,
  SEQUENCE_LENGTH,
} from '../src/lib/frameNormalizer.ts'

const [, , inputPath, outputPath] = process.argv
if (!inputPath || !outputPath) {
  console.error('usage: node scripts/normalizer-fixture.ts <input.json> <output.json>')
  process.exit(2)
}

const { frames } = JSON.parse(readFileSync(inputPath, 'utf8')) as { frames: number[][] }

if (frames.length !== SEQUENCE_LENGTH) {
  console.error(`expected ${SEQUENCE_LENGTH} frames, got ${frames.length}`)
  process.exit(2)
}

const normalized = frames.map((frame) => {
  if (frame.length !== RAW_FEATURES) {
    console.error(`expected ${RAW_FEATURES} features per frame, got ${frame.length}`)
    process.exit(2)
  }
  return normalizeSingleFrame(Float32Array.from(frame))
})

const features = buildSequenceFeatures(normalized)

writeFileSync(
  outputPath,
  JSON.stringify({
    normalized: normalized.map((f) => Array.from(f)),
    features: Array.from(features),
  }),
)

console.error(`ok: ${normalized.length} frames normalized, ${features.length} features emitted`)
