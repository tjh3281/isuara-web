/**
 * Copies everything the browser needs that lives outside the frontend tree.
 *
 * The point is that the built site must stand alone. On GitHub Pages there is
 * no backend to serve /models/*, so the MediaPipe bundles and the label map are
 * baked into the static output alongside the WASM runtime and the typeface.
 *
 * Sources stay authoritative — nothing here is duplicated into the repo, so the
 * Android app's assets and the web build can never drift onto different models
 * or a different font.
 *
 * WASM and models go to public/ because they are fetched by URL at runtime.
 * The font goes to src/assets/ instead so Vite hashes it and rewrites the CSS
 * url() with the correct base path — a public/ font referenced from CSS breaks
 * when the site is served from a subdirectory like /iSuara/.
 */

import { copyFileSync, cpSync, existsSync, mkdirSync, rmSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(here, '../../..')
const androidAssets = resolve(repoRoot, 'app/src/main/assets')

const jobs = [
  {
    label: 'mediapipe wasm',
    from: resolve(here, '../node_modules/@mediapipe/tasks-vision/wasm'),
    to: resolve(here, '../public/wasm'),
    dir: true,
    hint: 'run "npm install" first',
  },
  {
    label: 'pose landmarker',
    from: resolve(androidAssets, 'pose_landmarker_lite.task'),
    to: resolve(here, '../public/models/pose_landmarker_lite.task'),
  },
  {
    label: 'hand landmarker',
    from: resolve(androidAssets, 'hand_landmarker.task'),
    to: resolve(here, '../public/models/hand_landmarker.task'),
  },
  {
    label: 'label map',
    from: resolve(androidAssets, 'label_map.json'),
    to: resolve(here, '../public/models/label_map.json'),
  },
  {
    // The AffectNet expression classifier. Runs in the browser via
    // onnxruntime-web, so it never needs the backend.
    label: 'emotion classifier',
    from: resolve(androidAssets, 'emotion_enet_b0_8.onnx'),
    to: resolve(here, '../public/models/emotion_enet_b0_8.onnx'),
  },
  {
    label: 'google sans flex',
    from: resolve(repoRoot, 'app/src/main/res/font/google_sans_flex.ttf'),
    to: resolve(here, '../src/assets/google_sans_flex.ttf'),
  },
  {
    // The avatar's BIM sign clips. Fetched by key at runtime rather than
    // bundled, so a clip is only downloaded when someone actually signs it —
    // all 37 together are several megabytes.
    label: 'avatar motions',
    from: resolve(androidAssets, 'motions'),
    to: resolve(here, '../public/motions'),
    dir: true,
    hint: 'expected app/src/main/assets/motions from the signavatar branch',
  },
]

// onnxruntime's WASM is NOT copied here. It is imported with Vite's `?url` in
// lib/emotion/classifier.ts instead, which fingerprints it, rewrites the URL for
// the deployment base, and serves the loader as a static asset. A public/ copy
// hit a 500 because Vite tried to transform the .mjs loader as a module.

for (const job of jobs) {
  if (!existsSync(job.from)) {
    console.error(`[copy-assets] MISSING ${job.label}: ${job.from}`)
    if (job.hint) console.error(`[copy-assets] ${job.hint}`)
    process.exit(1)
  }

  if (job.dir) {
    rmSync(job.to, { recursive: true, force: true })
    mkdirSync(job.to, { recursive: true })
    cpSync(job.from, job.to, { recursive: true })
  } else {
    mkdirSync(dirname(job.to), { recursive: true })
    copyFileSync(job.from, job.to)
  }

  console.log(`[copy-assets] ${job.label}`)
}
