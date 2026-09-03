"""
Proves the backend asks the model for exactly what the Android app asks for.

The web build and the phone are supposed to be the same product. If a prompt,
a threshold or the model file drifts between them, nothing crashes — the two
just quietly start behaving differently, and comparing their output becomes
meaningless. This suite reads the Kotlin source directly and asserts the Python
matches it, so drift fails a test instead of going unnoticed.

Kotlin is parsed rather than duplicated on purpose: a hand-copied expected value
would drift alongside the code it is meant to guard.

    python -m unittest tests.test_android_parity -v
"""

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, prompts, translator  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
KOTLIN = REPO_ROOT / "app" / "src" / "main" / "java" / "com" / "isuara" / "app"

PROMPTS_KT = KOTLIN / "service" / "TranslationPrompts.kt"
PREDICTOR_KT = KOTLIN / "ml" / "SignPredictor.kt"
INTERPRETER_KT = KOTLIN / "ml" / "SignInterpreter.kt"
NORMALIZER_KT = KOTLIN / "ml" / "FrameNormalizer.kt"
DEBATE_KT = KOTLIN / "service" / "DebateTranslator.kt"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def kotlin_raw_string(source: str, name: str) -> str:
    """
    Extracts `val NAME = \"\"\"...\"\"\"`, applying Kotlin's trimIndent().

    trimIndent removes the common leading indentation and drops blank first and
    last lines, which is what makes the Kotlin literal and the Python literal
    comparable at all.
    """
    match = re.search(rf'val\s+{name}\s*=\s*"""(.*?)"""', source, re.DOTALL)
    assert match, f"could not find val {name} in the Kotlin source"
    body = match.group(1)

    lines = body.split("\n")
    if lines and not lines[0].strip():
        lines.pop(0)
    if lines and not lines[-1].strip():
        lines.pop()

    indents = [len(ln) - len(ln.lstrip()) for ln in lines if ln.strip()]
    pad = min(indents) if indents else 0
    return "\n".join(ln[pad:] if ln.strip() else "" for ln in lines)


def kotlin_personas(source: str) -> list[str]:
    """
    Extracts the PERSONAS list, joining Kotlin's `"..." + "..."` concatenations.

    Every persona begins with the same marker, so splitting the joined text on
    it recovers the individual entries without parsing Kotlin expressions.
    """
    block = re.search(r"val\s+PERSONAS\s*=\s*listOf\((.*?)\n    \)", source, re.DOTALL)
    assert block, "could not find val PERSONAS"
    joined = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', block.group(1)))
    joined = joined.replace('\\"', '"')

    marker = "INTERPRETIVE STANCE:"
    parts = [p.strip() for p in joined.split(marker) if p.strip()]
    return [f"{marker} {p}" for p in parts]


class TestPromptParity(unittest.TestCase):
    """The prompts must be identical, not merely similar."""

    def test_system_prompt_is_byte_identical(self):
        self.assertEqual(kotlin_raw_string(read(PROMPTS_KT), "SYSTEM"), prompts.SYSTEM)

    def test_judge_prompt_is_byte_identical(self):
        self.assertEqual(kotlin_raw_string(read(PROMPTS_KT), "JUDGE"), prompts.JUDGE)

    def test_personas_are_identical(self):
        kotlin = kotlin_personas(read(PROMPTS_KT))
        self.assertEqual(len(kotlin), 3, "expected 3 interpretive stances")
        self.assertEqual(kotlin, prompts.PERSONAS)


class TestConstantParity(unittest.TestCase):
    """Tuning values the model was validated against."""

    def _int(self, source: str, pattern: str) -> int:
        m = re.search(pattern, source)
        assert m, f"pattern not found: {pattern}"
        return int(m.group(1))

    def _float(self, source: str, pattern: str) -> float:
        m = re.search(pattern, source)
        assert m, f"pattern not found: {pattern}"
        return float(m.group(1))

    def test_confidence_threshold(self):
        kt = self._float(read(PREDICTOR_KT), r"confidence\s*>=\s*([\d.]+)f")
        self.assertEqual(kt, config.CONFIDENCE_THRESHOLD)

    def test_tensor_shapes(self):
        kt = read(NORMALIZER_KT)
        self.assertEqual(self._int(kt, r"SEQUENCE_LENGTH\s*=\s*(\d+)"), config.SEQUENCE_LENGTH)
        self.assertEqual(self._int(kt, r"FINAL_FEATURES\s*=\s*(\d+)"), config.NUM_FEATURES)
        self.assertEqual(self._int(kt, r"RAW_FEATURES\s*=\s*(\d+)"), 258)

    def test_class_count(self):
        kt = self._int(read(INTERPRETER_KT), r"NUM_CLASSES\s*=\s*(\d+)")
        self.assertEqual(kt, config.NUM_CLASSES)

    def test_stage_hold_matches_the_debate_pacing(self):
        kt_ms = self._int(read(DEBATE_KT), r"STAGE_HOLD_MS\s*=\s*(\d+)L")
        self.assertAlmostEqual(kt_ms / 1000, translator._STAGE_HOLD_S, places=6)


class TestModelParity(unittest.TestCase):
    """
    The web build must load the SAME weights as the phone.

    Both TFLite variants score 90.99% in the notebook, so this is not about
    accuracy — it is about the two builds not being different models.
    """

    def test_backend_loads_the_same_file_the_app_does(self):
        m = re.search(r'MODEL_FILE\s*=\s*"([^"]+)"', read(INTERPRETER_KT))
        assert m, "could not find MODEL_FILE in SignInterpreter.kt"
        self.assertEqual(m.group(1), config.MODEL_FILE.name)

    def test_that_file_exists(self):
        self.assertTrue(config.MODEL_FILE.exists(), f"missing {config.MODEL_FILE}")

    def test_label_map_is_the_shared_android_asset(self):
        # Not a copy under web/ — the same file the app ships, so retraining
        # cannot leave the two builds on different gloss sets.
        self.assertIn("app", config.LABEL_MAP_FILE.parts)
        self.assertIn("assets", config.LABEL_MAP_FILE.parts)
        self.assertTrue(config.LABEL_MAP_FILE.exists())


if __name__ == "__main__":
    unittest.main()
