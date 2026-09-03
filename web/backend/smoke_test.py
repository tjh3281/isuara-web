"""
Startup smoke test for the inference path.

The one assumption worth proving before anything is built on top of it: the
model was converted with SELECT_TF_OPS, so it carries Flex ops that only the
full tensorflow package can execute. Run this after `pip install -r
requirements.txt` to confirm the environment can actually serve predictions.

    python smoke_test.py
"""

import time

import numpy as np

from app import config
from app.inference import SignInterpreter


def main() -> None:
    print(f"model : {config.MODEL_FILE}")
    interpreter = SignInterpreter()
    print(f"labels: {len(interpreter.labels)} (first 5: {interpreter.labels[:5]})")

    window = np.random.randn(config.SEQUENCE_LENGTH * config.NUM_FEATURES).astype(np.float32) * 0.1

    start = time.perf_counter()
    index, confidence = interpreter.predict_top_class(window)
    cold_ms = (time.perf_counter() - start) * 1000
    print(f"cold  : idx={index} label={interpreter.labels[index]!r} conf={confidence:.4f} ({cold_ms:.1f} ms)")

    runs = 20
    start = time.perf_counter()
    for _ in range(runs):
        interpreter.predict_top_class(window)
    print(f"warm  : {(time.perf_counter() - start) / runs * 1000:.2f} ms/inference")

    probs_sum_ok = 0.0 <= confidence <= 1.0
    print(f"result: {'PASS' if probs_sum_ok else 'FAIL — confidence outside [0,1], check output tensor'}")


if __name__ == "__main__":
    main()
