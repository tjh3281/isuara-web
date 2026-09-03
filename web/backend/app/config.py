"""
Where the web backend finds its assets and its keys.

Model files are read straight out of the Android app's assets directory rather
than copied: `app/src/main/assets` stays the single source of truth, so the web
build can never drift onto a stale model or an out-of-date label map.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# web/backend/app/config.py -> app -> backend -> web -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
ASSETS_DIR = REPO_ROOT / "app" / "src" / "main" / "assets"

# The SAME file the phone loads — see ml/SignInterpreter.kt, which names
# bim_lstm_v3_int8.tflite. An earlier version of this used the f32 build on the
# reasoning that a server has no size constraint, but that made the web build
# quietly a different model from the app. The notebook measures both variants at
# 90.99% (Keras 91.16%, drift 0.17%), so matching the app costs no accuracy.
#
# f32 stays available for A/B checks; nothing selects it by default.
MODEL_FILE = ASSETS_DIR / "bim_lstm_v3_int8.tflite"
MODEL_FILE_F32 = ASSETS_DIR / "bim_lstm_v3_f32.tflite"
LABEL_MAP_FILE = ASSETS_DIR / "label_map.json"

# Keras artifacts, gitignored because they are large and not in the Android
# build. Drop whichever the training notebook produced into web/backend/models:
#
#   bim_lstm_v3.keras  — the export model, z-score scaler already baked in.
#                        Preferred: nothing else is needed beside it.
#   v3_best.keras      — the raw trained model. Needs the two scaler arrays,
#                        which the notebook saves next to it.
KERAS_DIR = Path(__file__).resolve().parents[1] / "models"
KERAS_MODEL = KERAS_DIR / "bim_lstm_v3.keras"
KERAS_INNER_MODEL = KERAS_DIR / "v3_best.keras"
SCALER_MEAN = KERAS_DIR / "scaler_mean_v3.npy"
SCALER_SCALE = KERAS_DIR / "scaler_scale_v3.npy"

# Must match the tensor shapes baked into the model — see ml/SignInterpreter.kt.
SEQUENCE_LENGTH = 30
NUM_FEATURES = 780
NUM_CLASSES = 98

# Kept identical to SignPredictor.kt so the web build classifies exactly as the
# phone does.
CONFIDENCE_THRESHOLD = 0.6

# Origins allowed to reach the API. The Vite dev server proxies /api and
# /models, so this matters only when the frontend is served from elsewhere.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if o.strip()
]


def gemini_keys() -> list[str]:
    """
    The configured Gemini keys, in slot order.

    Three separate keys because the free tier limits 5 requests/minute per
    Google Cloud *project* and one debate costs four calls — see
    service/GeminiClients.kt. Slot 1 falls back to the original single-key name.
    """
    raw = [
        os.getenv("GEMINI_API_KEY_1") or os.getenv("GEMINI_API_KEY") or "",
        os.getenv("GEMINI_API_KEY_2", ""),
        os.getenv("GEMINI_API_KEY_3", ""),
    ]
    return [k.strip() for k in raw if k.strip()]
