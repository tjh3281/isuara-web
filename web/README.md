# iSuara — Web

The browser version of the iSuara Android app. Same pipeline, same model, same
prompts; a different set of platform APIs underneath.

```
Browser                                        Backend (FastAPI)
─────────────────────────────────────          ──────────────────────────
getUserMedia                                   TFLite / Keras classifier
  → MediaPipe Tasks Vision (Pose + Hands)        (30×780 → 98 classes)
  → 258 keypoints per frame                    Gemini debate translator
  → FrameNormalizer (EMA, 30-frame window)       (3 personas + judge)
  → 30×780 feature window  ──── WebSocket ───→ /ws/predict
  → gloss buffer           ──── NDJSON ──────→ /api/translate
  → speechSynthesis                            /models/* (Android assets)
```

**Raw video never leaves the browser.** Vision and normalization run on-device,
exactly as on the phone. Only the anonymised feature window and, on translate,
the list of recognised Malay words cross the wire.

## Why there is a backend at all

Two things the browser genuinely cannot do:

1. **Run the classifier.** The model was converted with `SELECT_TF_OPS` — the
   built-in-ops path failed to legalize `tf.TensorListReserve` (see the export
   cell in `iSuara_Train_V3_1_5_WithOutput.ipynb`), so the flatbuffer contains
   `FlexTensorListReserve`, `FlexTensorListSetItem` and `FlexTensorListStack`.
   These need the TFLite Flex delegate, which no browser TFLite runtime ships.
2. **Hold the Gemini keys.** The Android build reads them from `BuildConfig`,
   which is fine on a device the user owns. A browser bundle is public.

Everything else — camera, MediaPipe, all five normalization stages, text-to-speech
— is client-side.

## Setup

### 1. Backend

Requires **Python 3.13** (TensorFlow has no 3.14 wheels).

```bash
cd web/backend
py -3.13 -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt

cp .env.example .env            # then add your Gemini keys
python -m uvicorn app.main:app --reload --port 8000
```

Check it came up cleanly:

```bash
curl http://localhost:8000/api/health
python smoke_test.py            # proves the classifier actually runs
```

### 2. Frontend

```bash
cd web/frontend
npm install
npm run dev                     # http://localhost:5173
```

The dev server proxies `/api`, `/ws` and `/models` to port 8000, so the browser
sees one origin and there is no CORS involved.

## The model file

`/api/health` reports `"status": "degraded"` and `"model": "unavailable"` until a
usable model is loaded. The backend tries two backends in order:

| Backend | File | Works on |
|---|---|---|
| Keras *(preferred)* | `web/backend/models/bim_lstm_v3.keras` | everywhere |
| Keras + scaler | `v3_best.keras` + `scaler_mean_v3.npy` + `scaler_scale_v3.npy` | everywhere |
| TFLite *(fallback)* | `app/src/main/assets/bim_lstm_v3_f32.tflite` | Linux, macOS |

### The TensorFlow version is load-bearing

The model was converted with `SELECT_TF_OPS`, so it carries Flex ops that need
the TFLite **Flex delegate**. Whether you have that delegate depends on the
**TensorFlow version, not the operating system**. Measured against this repo's
own `bim_lstm_v3_f32.tflite`, on both Windows and Linux:

| TensorFlow | Result |
|---|---|
| 2.16.1 | delegate present, but runtime too old — `FULLY_CONNECTED version '12'` |
| 2.17.1 | works |
| 2.18.1 | works |
| **2.19.0** | works — **pinned in `requirements.txt`** |
| 2.20+ | delegate **removed** from the pip package; `allocate_tensors()` raises `Select TensorFlow op(s) ... not supported` |

TF 2.19 supports Python 3.9–3.12, so the venv must use one of those. Python 3.13
and 3.14 have no TensorFlow release that can load this model.

A correct load logs:

```
INFO: Created TensorFlow Lite delegate for select TF ops.
INFO: TfLiteFlexDelegate delegate: 5 nodes delegated out of 105 nodes
```

If you hit the Select-TF-ops error, suspect a TensorFlow upgrade before anything
else. `bash tools/try-tf-versions.sh 2.19.0 2.20.0` re-measures the boundary.

**WSL is not required.** An earlier version of this document said it was; that
was a misdiagnosis of a version problem as a platform problem.

Things that genuinely do **not** work, all for the same reason:

| Attempt | Outcome |
|---|---|
| `ai-edge-litert` | no Flex delegate |
| `tf2onnx` → onnxruntime | `IndexError` parsing the Flex TensorList nodes |
| `tfjs-tflite` / LiteRT.js in the browser | no Flex delegate — see [Google's docs](https://developers.google.com/edge/litert/models/ops_select), which list Android/iOS/C++/Python only |

That last row is why inference runs server-side at all. It would stop being true
if the model were re-exported without `SELECT_TF_OPS` — see below.

### The Keras artifacts (optional)

Get the Keras file from the training notebook's Colab session; it writes both of
these:

- `/content/isuara_model_v3/bim_lstm_v3.keras` — the export model, z-score
  scaler already baked in. Drop this in `web/backend/models/` and nothing else
  is needed.
- `/content/drive/MyDrive/iSuara/v3_best.keras` — the raw trained model. Needs
  `scaler_mean_v3.npy` and `scaler_scale_v3.npy` beside it, which the notebook
  saves to the same Drive folder.

The `models/` directory is gitignored — these files are large and are not part of
the Android build.

Everything except sign recognition works without them: camera, skeleton overlay,
the feature pipeline, translation and speech all run. Only the gloss predictions
are missing.

## Port notes

Where the web build had to differ from the Android one, and why:

| Android | Web | Why |
|---|---|---|
| `LandmarkExtractor` async listener + `pendingFrames` map | two synchronous `detectForVideo` calls | The web API's VIDEO mode returns results inline, so joining pose and hand results by timestamp is unnecessary. |
| Front camera flips the bitmap; rear camera negates x and swaps L/R pose indices | always flip the canvas | Both Android paths produce a mirrored frame — the arithmetic version is the same mirror. The model was trained on mirrored input, so the web build does the flip once. |
| `ImageAnalysis` + `STRATEGY_KEEP_ONLY_LATEST` | `requestAnimationFrame`, processed inline | There is no browser backpressure valve; running synchronously makes a slow machine drop frames rather than build a backlog. |
| `StateFlow<PredictionState>` | `subscribe`/`getState` + `useSyncExternalStore` | Same push-based store, in React's idiom. |
| `Translator.stage` StateFlow | NDJSON stream from `/api/translate` | A debate takes seconds; without the stage labels the UI looks frozen. |
| Android TTS, `ms_MY` → `id_ID` → `en` | Web Speech API, same fallback chain | Desktop Chrome frequently has no Malay voice. |
| `LanguagePreference` (SharedPreferences) | `localStorage` | Same thing, wrapped in try/catch for private windows. |

Unchanged on purpose: `FrameNormalizer` (all five stages, index arithmetic
included), the 0.6 confidence threshold, the 10/5-frame cooldown, the 8-word
sentence cap, EMA α = 0.4, the 2-second auto-translate delay, and every prompt in
`TranslationPrompts.kt`.

## Layout

```
web/
├── backend/
│   ├── app/
│   │   ├── main.py          FastAPI: /ws/predict, /api/translate, /api/health, /models
│   │   ├── inference.py     ← ml/SignInterpreter.kt
│   │   ├── translator.py    ← service/{GeminiClients,GeminiTranslator,DebateTranslator}.kt
│   │   ├── prompts.py       ← service/TranslationPrompts.kt
│   │   ├── parsing.py       ← service/TranslationParsing.kt
│   │   └── config.py
│   ├── tools/               flatbuffer and notebook inspection helpers
│   └── smoke_test.py
└── frontend/
    └── src/
        ├── lib/
        │   ├── frameNormalizer.ts   ← ml/FrameNormalizer.kt
        │   ├── landmarkExtractor.ts ← ml/LandmarkExtractor.kt
        │   ├── signPredictor.ts     ← ml/SignPredictor.kt
        │   ├── tts.ts               ← service/TtsService.kt
        │   ├── language.ts          ← service/{Language,LanguagePreference}.kt
        │   ├── labelMap.ts
        │   ├── predictClient.ts     the one file that knows inference is remote
        │   └── translateClient.ts
        ├── hooks/useSignPipeline.ts ← the CameraX block of ui/CameraScreen.kt
        └── components/              ← the Compose tree of ui/CameraScreen.kt
```

## Browser support

Needs `getUserMedia` (HTTPS or localhost), WebGL2 for the MediaPipe GPU delegate
(it falls back to CPU), and `speechSynthesis` for audio. Chrome and Edge are the
best-tested; Safari works but has a thinner voice list.
