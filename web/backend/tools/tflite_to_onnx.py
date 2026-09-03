"""
Try to convert the TFLite model to ONNX, so it can run without the Flex delegate.

Why this is worth attempting: the model's only blocker on Windows is that its
Flex ops (FlexTensorListReserve/SetItem/Stack) need a delegate the TensorFlow
Windows wheels do not ship. tf2onnx reads the TFLite flatbuffer directly and
re-expresses those TensorList operations as ONNX Sequence ops, which
onnxruntime implements natively on every platform.

If it works, two things follow:
  1. The backend runs on Windows with no Keras file and no WSL.
  2. The same .onnx can run in the browser via onnxruntime-web, which would
     remove the inference backend altogether and restore the "everything on the
     edge" story.

If it fails, the fallbacks are WSL2 (Linux ships the Flex delegate) or the
Keras export from the training notebook. See web/README.md.

    python tools/tflite_to_onnx.py
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app import config  # noqa: E402

OUT = config.KERAS_DIR / "bim_lstm_v3.onnx"


def convert() -> bool:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--tflite", str(config.MODEL_FILE),
        "--output", str(OUT),
        "--opset", "17",
    ]
    print("$ " + " ".join(cmd) + "\n")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # tf2onnx is chatty on stderr even when it succeeds, so judge by the
    # artifact and the exit code rather than by whether it printed anything.
    tail = (result.stderr or result.stdout).strip().splitlines()[-25:]
    print("\n".join(tail))
    print(f"\nexit={result.returncode}  output_exists={OUT.exists()}")
    return result.returncode == 0 and OUT.exists()


def verify() -> bool:
    import onnxruntime as ort

    session = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    print(f"\ninput : {inp.name} {inp.shape} {inp.type}")
    print(f"output: {out.name} {out.shape} {out.type}")

    window = np.random.randn(1, config.SEQUENCE_LENGTH, config.NUM_FEATURES).astype(np.float32) * 0.1
    probs = session.run([out.name], {inp.name: window})[0][0]

    print(f"\nprobs shape : {probs.shape}  (want ({config.NUM_CLASSES},))")
    print(f"probs sum   : {probs.sum():.4f}  (want ~1.0 for a softmax)")
    print(f"argmax      : {int(np.argmax(probs))}  conf={float(probs.max()):.4f}")

    ok = probs.shape == (config.NUM_CLASSES,) and 0.9 < float(probs.sum()) < 1.1
    print(f"\nRESULT: {'PASS - ONNX path is viable' if ok else 'FAIL - output does not look like the model softmax'}")
    return ok


if __name__ == "__main__":
    if not convert():
        print("\nRESULT: FAIL - conversion did not produce a model.")
        sys.exit(1)
    sys.exit(0 if verify() else 1)
