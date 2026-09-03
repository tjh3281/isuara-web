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

    def test_emotion_confidence_floor_matches(self):
        m = re.search(r"MIN_EMOTION_CONFIDENCE\s*=\s*([\d.]+)f", read(PROMPTS_KT))
        assert m, "could not find MIN_EMOTION_CONFIDENCE"
        self.assertEqual(float(m.group(1)), prompts.MIN_EMOTION_CONFIDENCE)

    def test_user_turn_renders_the_expression_line(self):
        # The wording is what rule 8 keys on, so it is worth pinning.
        turn = prompts.user_turn(
            ["Tolong", "Polis"],
            {"descriptor": "fear", "confidence": 0.9, "isHighArousal": True},
        )
        self.assertIn("Input: [Tolong, Polis]", turn)
        self.assertIn("Observed facial expression: fear (high arousal)", turn)

    def test_weak_readings_are_dropped_entirely(self):
        # A hedged hint is worse than no hint: the model cannot tell how much to
        # discount it, so below the floor the line is omitted rather than softened.
        turn = prompts.user_turn(
            ["Tolong"], {"descriptor": "fear", "confidence": 0.2, "isHighArousal": True}
        )
        self.assertNotIn("Observed facial expression", turn)


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

    def test_model_divergence_from_the_app_is_the_documented_one(self):
        """
        The web build deliberately runs a NEWER model than this branch's app.

        `feature/signavatar` loads bim_lstm_v3_int8.tflite, which carries Flex
        ops. The web build loads bim_lstm_v312_fp16.tflite — the export from the
        `gonka` branch — because it has none, verified with tools/check_model.py.
        That is what frees the backend from the TensorFlow 2.17-2.19 window and
        what would let the model run in the browser instead of here at all.

        Both are (1, 30, 780) -> (1, 98) over the same 98-gloss label map, so
        this is a swap, not a different pipeline.

        If the app later adopts the fp16 export, this test should collapse back
        into a plain equality check.
        """
        m = re.search(r'MODEL_FILE\s*=\s*"([^"]+)"', read(INTERPRETER_KT))
        assert m, "could not find MODEL_FILE in SignInterpreter.kt"
        android, web = m.group(1), config.MODEL_FILE.name

        if android == web:
            return  # converged; nothing to justify

        self.assertEqual(
            web,
            "bim_lstm_v312_fp16.tflite",
            "the web build diverges from the app on the model file, and the only "
            "sanctioned divergence is the Flex-free fp16 export",
        )
        self.assertEqual(android, "bim_lstm_v3_int8.tflite")

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
