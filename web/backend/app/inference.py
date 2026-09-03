"""
SignInterpreter — the classifier, ported from ml/SignInterpreter.kt.

Input:  (1, 30, 780) float32
Output: (1, 98)      float32 softmax probabilities

Two loading paths, tried in order:

  1. **Keras** (`models/bim_lstm_v3.keras`, or `v3_best.keras` plus the two
     scaler arrays). Used if present; not required.

  2. **TFLite** (`app/src/main/assets/bim_lstm_v3_f32.tflite`) — the file the
     phone ships, already in the repo. This is the normal path.

The model was converted with SELECT_TF_OPS (the built-in-ops path failed to
legalize `tf.TensorListReserve` — see the export cell in
iSuara_Train_V3_1_5_WithOutput.ipynb), so the flatbuffer contains
FlexTensorListReserve/SetItem/Stack and needs the TFLite Flex delegate.

That delegate is a function of the **TensorFlow version, not the operating
system** — a distinction that cost real time to establish, so it is written down
here. Measured against this exact file on both Windows and Linux:

    2.16.1  delegate present, runtime too old for FULLY_CONNECTED v12
    2.17.1  works
    2.18.1  works
    2.19.0  works   <- pinned in requirements.txt
    2.20+   delegate REMOVED from the pip package; allocate_tensors() raises
            "Select TensorFlow op(s) ... not supported by this interpreter"

On the pinned version the load logs `TfLiteFlexDelegate delegate: 5 nodes
delegated out of 105 nodes`. If you ever see the Select-TF-ops error, the cause
is almost certainly a TensorFlow upgrade, not the platform. Re-check with
`tools/try-tf-versions.sh`.

The interpreter is not thread-safe and holds a single set of tensors, so every
call takes a lock. Inference measures ~6ms, so serializing costs far less than
pooling would.

The interpreter is not thread-safe and holds a single set of tensors, so every
call takes a lock. Inference is a few milliseconds, so serializing costs far
less than pooling would.
"""

import json
import logging
import threading

import numpy as np

from . import config

log = logging.getLogger(__name__)


class ModelUnavailable(RuntimeError):
    """Raised when no usable model could be loaded, with actionable guidance."""


class SignInterpreter:
    """Loads whichever model backend is available and classifies feature windows."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._predict_fn = None
        self.backend = "none"

        self.labels: list[str] = json.loads(
            config.LABEL_MAP_FILE.read_text(encoding="utf-8")
        )["actions_ordered"]
        if len(self.labels) != config.NUM_CLASSES:
            raise ValueError(
                f"label map has {len(self.labels)} labels, model has {config.NUM_CLASSES}"
            )

        errors: list[str] = []
        for attempt in (self._load_keras, self._load_tflite):
            try:
                if attempt():
                    log.info(
                        "Model loaded via %s (%d classes)", self.backend, len(self.labels)
                    )
                    return
            except Exception as e:  # noqa: BLE001 — the next backend is the handler
                errors.append(f"{attempt.__name__}: {e}")
                log.warning("%s failed: %s", attempt.__name__, e)

        raise ModelUnavailable(
            "No usable model.\n"
            + "\n".join(f"  - {e}" for e in errors)
            + "\n\nIf the error mentions Select TensorFlow ops, the TensorFlow "
            "version is wrong, not the platform: 2.20+ dropped the TFLite Flex "
            "delegate this model needs. requirements.txt pins 2.19.0 for exactly "
            "this reason — reinstall with `pip install -r requirements.txt`, or "
            "re-check the working range with tools/try-tf-versions.sh."
        )

    # ── loaders ──

    def _load_keras(self) -> bool:
        """
        The exported Keras model, which already has the z-score scaler baked in —
        the same wrapped model the TFLite file was converted from, so it takes
        raw 780-feature input and needs no preprocessing here.

        Also accepts the un-wrapped `v3_best.keras` alongside the two scaler
        arrays, applying the z-score itself, since that is the artifact the
        notebook writes to Drive.
        """
        if not config.KERAS_MODEL.exists() and not config.KERAS_INNER_MODEL.exists():
            return False

        import tensorflow as tf

        if config.KERAS_MODEL.exists():
            model = tf.keras.models.load_model(config.KERAS_MODEL, compile=False)
            mean = scale = None
            self.backend = f"keras ({config.KERAS_MODEL.name})"
        else:
            if not (config.SCALER_MEAN.exists() and config.SCALER_SCALE.exists()):
                raise FileNotFoundError(
                    f"{config.KERAS_INNER_MODEL.name} needs {config.SCALER_MEAN.name} "
                    f"and {config.SCALER_SCALE.name} beside it — it expects "
                    "already-standardised input."
                )
            model = tf.keras.models.load_model(config.KERAS_INNER_MODEL, compile=False)
            mean = np.load(config.SCALER_MEAN).reshape(1, 1, config.NUM_FEATURES)
            scale = np.load(config.SCALER_SCALE).reshape(1, 1, config.NUM_FEATURES)
            self.backend = f"keras ({config.KERAS_INNER_MODEL.name} + scaler)"

        def predict(window: np.ndarray) -> np.ndarray:
            x = window if mean is None else (window - mean) / scale
            return model(x, training=False).numpy()[0]

        self._predict_fn = predict
        return True

    def _load_tflite(self) -> bool:
        """The phone's model. Needs the Flex delegate — Linux and macOS only."""
        if not config.MODEL_FILE.exists():
            return False

        import tensorflow as tf

        interpreter = tf.lite.Interpreter(model_path=str(config.MODEL_FILE))
        interpreter.allocate_tensors()

        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]
        shape = tuple(interpreter.get_input_details()[0]["shape"])
        expected = (1, config.SEQUENCE_LENGTH, config.NUM_FEATURES)
        if shape != expected:
            raise ValueError(f"model expects input {shape}, this build assumes {expected}")

        def predict(window: np.ndarray) -> np.ndarray:
            interpreter.set_tensor(input_index, window)
            interpreter.invoke()
            return interpreter.get_tensor(output_index)[0]

        self._predict_fn = predict
        self.backend = f"tflite ({config.MODEL_FILE.name})"
        return True

    # ── inference ──

    def predict_top_class(self, features: np.ndarray) -> tuple[int, float]:
        """
        Runs one 30-frame window and returns (class index, confidence).

        `features` is the flat 30 x 780 window the browser produced with the same
        normalization stages as ml/FrameNormalizer.kt.
        """
        expected = config.SEQUENCE_LENGTH * config.NUM_FEATURES
        if features.size != expected:
            raise ValueError(f"expected {expected} features, got {features.size}")

        window = np.ascontiguousarray(
            features.reshape(1, config.SEQUENCE_LENGTH, config.NUM_FEATURES),
            dtype=np.float32,
        )

        with self._lock:
            probs = self._predict_fn(window)  # type: ignore[misc]

        index = int(np.argmax(probs))
        return index, float(probs[index])
