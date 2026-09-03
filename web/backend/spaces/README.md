---
title: iSuara API
emoji: 🤟
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
short_description: Sign recognition and translation API for the iSuara web app
---

# iSuara API

Recognition and translation for the iSuara web app. The browser does the camera
work, the landmark extraction and the feature normalization; this service holds
the classifier and the translator.

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | model status, class count, whether translation is configured |
| `WS /ws/predict` | one 30 x 780 float32 window in, top class out |
| `POST /api/translate` | multi-agent debate, streamed as newline-delimited JSON |

## Configuration

Set these under **Settings → Variables and secrets**. Nothing here belongs in
the code, and the keys are never sent to a browser.

| Name | Kind | Value |
|---|---|---|
| `ALLOWED_ORIGINS` | variable | the site's origin, e.g. `https://isuara.netlify.app` — no trailing slash |
| `GEMINI_API_KEY_1` | secret | one key per debate agent, each from a different Google Cloud project |
| `GEMINI_API_KEY_2` | secret | |
| `GEMINI_API_KEY_3` | secret | |

Without the keys recognition still works; translation falls back to showing and
speaking the raw detected glosses, exactly as the Android build does.

`ALLOWED_ORIGINS` is not optional once a site calls this from another origin —
the browser blocks every response without it.

## Why the image is small

The model is Flex-free, so the container installs the standalone LiteRT
interpreter rather than TensorFlow: 48MB against 1.4GB. Same class, same
timing, outputs agreeing to 1.1e-08.
