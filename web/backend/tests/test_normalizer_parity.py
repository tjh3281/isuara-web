"""
Proves the browser's TypeScript FrameNormalizer matches ml/FrameNormalizer.kt.

This is the highest-value test in the web build. The normalizer produces the
exact 30x780 buffer the model consumes, and the model was trained on buffers
built by the Kotlin. A transposed offset here does not crash and does not look
wrong on screen — it produces confident, incorrect glosses. No other test in the
project would notice.

How it works: a fixed-seed fixture is fed to BOTH the real TypeScript module the
browser ships (run through Node, which strips types natively) and to
reference_normalizer.py, transcribed independently from the Kotlin. Every one of
the 23,400 output values is compared.

Requires Node on PATH. Skips rather than fails if Node is absent, so the rest of
the suite still runs in a Python-only environment.

    python -m unittest tests.test_normalizer_parity -v
"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.reference_normalizer import (  # noqa: E402
    FINAL_FEATURES,
    RAW_FEATURES,
    SEQUENCE_LENGTH,
    build_sequence_features,
    normalize_single_frame,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_SCRIPT = REPO_ROOT / "web" / "frontend" / "scripts" / "normalizer-fixture.ts"

# Loose enough to absorb float32-vs-float64 rounding, tight enough that any
# structural mistake -- a wrong offset, a skipped guard, a misordered stage --
# blows straight through it.
TOLERANCE = 1e-5


def _prng(seed: int):
    """
    A tiny deterministic LCG.

    Not `random`: the fixture must be reproducible across Python versions, and
    the exact stream matters if this test ever needs debugging against a
    recorded failure.
    """
    state = seed

    def nxt() -> float:
        nonlocal state
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        return state / 0x7FFFFFFF

    return nxt


def make_fixture() -> list[list[float]]:
    """
    30 frames of plausible landmark data, with the awkward cases mixed in.

    The edge cases are the point. Ordinary random frames exercise the happy path
    only; the guards in Stage 1 and Stage 2 are exactly where a port silently
    diverges.
    """
    rnd = _prng(20260902)
    frames: list[list[float]] = []

    for i in range(SEQUENCE_LENGTH):
        frame = [(rnd() - 0.5) * 2.0 for _ in range(RAW_FEATURES)]

        if i == 5:
            # Nothing detected at all: anchor is zero and shoulder width is
            # zero, so BOTH guards must skip.
            frame = [0.0] * RAW_FEATURES
        elif i == 11:
            # Left hand missing -- its wrist is (0,0,0), so the left-hand anchor
            # block must be skipped while the right-hand one still runs.
            frame[132:195] = [0.0] * 63
        elif i == 17:
            # Shoulders almost coincident: shoulder_width falls under the 0.01
            # guard, so Stage 2 must not scale.
            frame[44], frame[45], frame[46] = 0.5, 0.5, 0.5
            frame[48], frame[49], frame[50] = 0.5000001, 0.5000001, 0.5000001
        elif i == 23:
            # Right hand missing, mirror of the i == 11 case.
            frame[195:258] = [0.0] * 63

        frames.append(frame)

    return frames


class TestNormalizerParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.node = shutil.which("node")
        if not cls.node:
            raise unittest.SkipTest("node not on PATH")
        if not FIXTURE_SCRIPT.exists():
            raise unittest.SkipTest(f"missing {FIXTURE_SCRIPT}")

        cls.frames = make_fixture()

        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / "input.json"
            out_path = Path(tmp) / "output.json"
            in_path.write_text(json.dumps({"frames": cls.frames}), encoding="utf-8")

            result = subprocess.run(
                [cls.node, str(FIXTURE_SCRIPT), str(in_path), str(out_path)],
                capture_output=True,
                text=True,
                cwd=str(FIXTURE_SCRIPT.parent.parent),
            )
            if result.returncode != 0 or not out_path.exists():
                raise AssertionError(
                    f"fixture script failed (exit {result.returncode}):\n"
                    f"{result.stdout}\n{result.stderr}"
                )
            cls.ts = json.loads(out_path.read_text(encoding="utf-8"))

    def _max_diff(self, a, b) -> tuple[float, int]:
        """Largest absolute difference and where it is, for a useful failure."""
        worst, index = 0.0, -1
        for i, (x, y) in enumerate(zip(a, b)):
            d = abs(x - y)
            if d > worst:
                worst, index = d, i
        return worst, index

    def test_stage_1_and_2_match_per_frame(self):
        """normalizeSingleFrame: anchor subtraction and shoulder-width scaling."""
        for i, frame in enumerate(self.frames):
            expected = normalize_single_frame(frame)
            actual = self.ts["normalized"][i]
            self.assertEqual(len(actual), RAW_FEATURES, f"frame {i} wrong length")
            worst, idx = self._max_diff(expected, actual)
            self.assertLess(
                worst, TOLERANCE,
                f"frame {i} diverges at feature {idx}: "
                f"kotlin-reference={expected[idx]!r} typescript={actual[idx]!r} (diff {worst})",
            )

    def test_stage_3_4_5_match_over_the_sequence(self):
        """buildSequenceFeatures: velocity, acceleration and engineered features."""
        normalized = [normalize_single_frame(f) for f in self.frames]
        expected = build_sequence_features(normalized)
        actual = self.ts["features"]

        self.assertEqual(len(actual), SEQUENCE_LENGTH * FINAL_FEATURES)
        worst, idx = self._max_diff(expected, actual)

        if idx >= 0 and worst >= TOLERANCE:
            frame_i, within = divmod(idx, FINAL_FEATURES)
            if within < 258:
                block = f"position[{within}]"
            elif within < 516:
                block = f"velocity[{within - 258}]"
            elif within < 774:
                block = f"acceleration[{within - 516}]"
            else:
                block = f"engineered[{within - 774}]"
            self.fail(
                f"diverges at frame {frame_i} {block}: "
                f"kotlin-reference={expected[idx]!r} typescript={actual[idx]!r} (diff {worst})"
            )

    def test_engineered_block_is_where_the_kotlin_puts_it(self):
        """
        Guards the layout itself, not just the values.

        The engineered features sit in the last 6 slots of each 780-wide row.
        Everything upstream could be correct while this block landed at the
        wrong offset, and the model would still return a confident answer.
        """
        normalized = [normalize_single_frame(f) for f in self.frames]
        expected = build_sequence_features(normalized)

        for i in range(SEQUENCE_LENGTH):
            base = i * FINAL_FEATURES + 774
            got = self.ts["features"][base : base + 6]
            want = expected[base : base + 6]
            worst, idx = self._max_diff(want, got)
            self.assertLess(
                worst, TOLERANCE,
                f"frame {i} engineered[{idx}] differs: {want[idx]!r} vs {got[idx]!r}",
            )


if __name__ == "__main__":
    unittest.main()
