"""
Does ai-edge-litert (TFLite's successor runtime) carry the Flex delegate on
Windows, where the tensorflow wheel does not?

If yes, the whole blocker disappears: the .tflite already in the repo runs as-is,
with no Keras export and no WSL.

    python tools/probe_litert.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app import config  # noqa: E402


def main() -> int:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except Exception as e:
        print(f"import failed: {e}")
        return 1

    print(f"model: {config.MODEL_FILE.name}")
    try:
        interpreter = Interpreter(model_path=str(config.MODEL_FILE))
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"\nallocate_tensors FAILED:\n  {e}")
        print("\nRESULT: FAIL - litert has no Flex delegate on Windows either.")
        return 1

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    print(f"input : {tuple(inp['shape'])} {inp['dtype']}")
    print(f"output: {tuple(out['shape'])} {out['dtype']}")

    window = np.random.randn(1, config.SEQUENCE_LENGTH, config.NUM_FEATURES).astype(np.float32) * 0.1
    interpreter.set_tensor(inp["index"], window)
    interpreter.invoke()
    probs = interpreter.get_tensor(out["index"])[0]

    print(f"\nprobs shape: {probs.shape}  (want ({config.NUM_CLASSES},))")
    print(f"probs sum  : {probs.sum():.4f}  (want ~1.0)")
    print(f"argmax     : {int(np.argmax(probs))} conf={float(probs.max()):.4f}")

    ok = probs.shape == (config.NUM_CLASSES,) and 0.9 < float(probs.sum()) < 1.1
    print(f"\nRESULT: {'PASS - litert runs the Flex model on Windows' if ok else 'FAIL - output is not the model softmax'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
