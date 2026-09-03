"""
Check whether a .tflite model needs the TFLite Flex delegate.

A model without Flex ops runs on any TensorFlow version AND in the browser via
LiteRT.js / tfjs-tflite — which would remove the inference backend entirely and
put the whole pipeline back on the edge. That makes this a decisive property to
verify rather than assume, so this loads the file and runs a real window instead
of grepping the flatbuffer for op names.

    python tools/check_model.py <path-to.tflite>
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"not found: {path}")
        return 2

    raw = path.read_bytes()
    flex = sorted({m for m in _find_flex(raw)})
    print(f"file      : {path.name}  ({len(raw) / 1024 / 1024:.2f} MB)")
    print(f"flex ops  : {flex if flex else 'none'}")

    try:
        interpreter = tf.lite.Interpreter(model_path=str(path))
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"\nLOAD FAILED: {str(e)[:200]}")
        return 1

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    print(f"input     : {tuple(inp['shape'])} {np.dtype(inp['dtype']).name}")
    print(f"output    : {tuple(out['shape'])} {np.dtype(out['dtype']).name}")

    window = (np.random.randn(*inp["shape"]) * 0.1).astype(inp["dtype"])
    interpreter.set_tensor(inp["index"], window)
    interpreter.invoke()
    probs = interpreter.get_tensor(out["index"])[0]

    print(f"softmax   : shape={probs.shape} sum={probs.sum():.4f} argmax={int(np.argmax(probs))}")

    ok = 0.9 < float(probs.sum()) < 1.1
    verdict = "runs, and needs NO Flex delegate" if (ok and not flex) else (
        "runs, but carries Flex ops" if ok else "output is not a softmax"
    )
    print(f"\nRESULT    : {verdict}")
    return 0 if ok else 1


def _find_flex(raw: bytes):
    text = raw.decode("ascii", errors="ignore")
    import re

    return re.findall(r"Flex[A-Za-z]+", text)


if __name__ == "__main__":
    sys.exit(main())
