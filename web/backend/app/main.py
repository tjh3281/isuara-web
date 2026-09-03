"""
The iSuara web backend.

It does exactly the two things the browser cannot do itself:

  1. Sign inference — the TFLite model needs the Flex delegate (see
     inference.py), which no browser TFLite runtime ships.
  2. Gemini calls — the API keys must not be in a public bundle.

Everything else from the Android app (camera, MediaPipe landmark extraction,
feature normalization, text-to-speech) stays client-side, so raw video never
leaves the browser. Only a normalized 30-frame feature window and, on translate,
a list of Malay glosses cross the wire.
"""

import asyncio
import json
import logging

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, translator
from .inference import SignInterpreter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("isuara")

app = FastAPI(title="iSuara Web API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serves the MediaPipe .task bundles and label_map.json straight from the
# Android app's assets, so the browser and the phone load byte-identical models.
app.mount("/models", StaticFiles(directory=str(config.ASSETS_DIR)), name="models")

# Loaded once at import, and deliberately NOT fatal on failure.
#
# Without the model there is no sign recognition, but the camera, the landmark
# extraction, the feature pipeline and the translator all still work — and on
# Windows a missing Flex delegate is the expected case until someone drops the
# Keras export in. Serving a working site that says exactly what is missing
# beats refusing to start.
interpreter: SignInterpreter | None
try:
    interpreter = SignInterpreter()
except Exception as e:  # noqa: BLE001 — reported through /api/health instead
    interpreter = None
    MODEL_ERROR = str(e)
    log.error("Sign recognition disabled — %s", e)
else:
    MODEL_ERROR = ""


# ─────────────────────────── inference ───────────────────────────

# Bytes in one 30 x 780 float32 window.
_WINDOW_BYTES = config.SEQUENCE_LENGTH * config.NUM_FEATURES * 4


@app.websocket("/ws/predict")
async def predict_socket(websocket: WebSocket) -> None:
    """
    One long-lived socket per camera session.

    Binary in: a Float32Array of 30 x 780, row-major by frame — the same layout
    FrameNormalizer.buildSequenceFeatures produces. JSON out: the top class.

    A socket rather than POST-per-window because the client fires a prediction
    roughly three times a second for as long as the camera runs, and paying TCP
    plus TLS setup on each would dominate the ~2ms of actual inference.
    """
    await websocket.accept()

    if interpreter is None:
        await websocket.send_text(json.dumps({"error": "model unavailable"}))
        await websocket.close(code=1011)
        return

    log.info("predict socket opened")
    try:
        while True:
            data = await websocket.receive_bytes()

            if len(data) != _WINDOW_BYTES:
                await websocket.send_text(
                    json.dumps({"error": f"expected {_WINDOW_BYTES} bytes, got {len(data)}"})
                )
                continue

            features = np.frombuffer(data, dtype="<f4")
            # Off the event loop: inference takes a lock and blocks, and this
            # socket must stay responsive to the next window.
            index, confidence = await asyncio.to_thread(interpreter.predict_top_class, features)

            await websocket.send_text(
                json.dumps(
                    {
                        "label": interpreter.labels[index],
                        "confidence": confidence,
                        "isConfident": confidence >= config.CONFIDENCE_THRESHOLD,
                    }
                )
            )
    except WebSocketDisconnect:
        log.info("predict socket closed")
    except Exception:
        log.exception("predict socket failed")
        await websocket.close(code=1011)


# ─────────────────────────── translation ───────────────────────────


class TranslateRequest(BaseModel):
    words: list[str] = Field(min_length=1, max_length=16)


@app.post("/api/translate")
async def translate(request: TranslateRequest) -> StreamingResponse:
    """
    Runs the debate and streams its progress as newline-delimited JSON.

    NDJSON rather than a single response body because the debate takes seconds
    and the Android UI shows which stage it is in throughout; the browser reads
    these lines to render the same thing. Lines are either
    `{"stage": "JUDGING", "label": "..."}` or a final
    `{"translation": {...}}` / `{"error": "..."}`.

    A failure is reported in-band, with HTTP 200, because the stream has already
    begun by the time an agent can fail — the client treats a body with no
    `translation` line as a failure and falls back to the raw glosses, exactly
    as CameraScreen.kt does when translate() throws.
    """

    async def stream():
        try:
            async for stage, result in translator.debate(request.words):
                if result is None:
                    yield json.dumps(
                        {"stage": stage, "label": translator.STAGE_LABELS[stage]}
                    ) + "\n"
                else:
                    yield json.dumps({"translation": result}) + "\n"
        except Exception as e:
            log.exception("translate failed")
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        # Buffering proxies would defeat the point of streaming the stages.
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────── health ───────────────────────────


@app.get("/api/health")
async def health() -> dict:
    """
    What the client needs to know before it starts.

    `translationEnabled` is false when no key is configured; the client then
    falls back to showing and speaking the raw glosses rather than offering a
    Translate button that cannot work. `modelError` carries the load failure
    verbatim so a broken setup is diagnosable from the browser.
    """
    return {
        "status": "ok" if interpreter else "degraded",
        "model": interpreter.backend if interpreter else "unavailable",
        "modelError": MODEL_ERROR,
        "classes": len(interpreter.labels) if interpreter else 0,
        "translationEnabled": translator.clients.is_configured,
        "keySlots": translator.clients.slot_count,
    }
