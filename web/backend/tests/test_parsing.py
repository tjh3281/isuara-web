"""
Port of TranslationParsingTest.kt and JudgeChoiceParsingTest.kt.

Same cases as the Android suite, against the Python implementation, so the two
parsers cannot drift: a reply shape one tolerates and the other rejects would
mean the web and phone builds behave differently on identical model output.

Pure parsing — no network, no API cost.

    python -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import prompts  # noqa: E402
from app.parsing import extract_choice, extract_translation  # noqa: E402


class TestExtractTranslation(unittest.TestCase):
    def test_parses_a_clean_json_object(self):
        t = extract_translation(
            '{"ms": "Polis datang ke rumah kami.", "en": "The police came to our house.", '
            '"zh": "警察来到了我们家。", "ta": "காவல்துறை எங்கள் வீட்டிற்கு வந்தது."}'
        )
        self.assertEqual("Polis datang ke rumah kami.", t["ms"])
        self.assertEqual("The police came to our house.", t["en"])
        self.assertEqual("警察来到了我们家。", t["zh"])

    def test_strips_markdown_code_fences(self):
        t = extract_translation(
            '```json\n'
            '{"ms": "Saya lapar.", "en": "I am hungry.", "zh": "我饿了。", "ta": "எனக்கு பசிக்கிறது."}\n'
            '```'
        )
        self.assertEqual("Saya lapar.", t["ms"])
        self.assertEqual("我饿了。", t["zh"])

    def test_ignores_a_leading_reasoning_monologue(self):
        """MiniMax leaked a <think> monologue on 10/10 benchmark runs."""
        t = extract_translation(
            "<think>\n"
            "The glosses are Polis, Rumah, Kami. In BIM the order is topic-comment,\n"
            "so this probably means the police came to our house.\n"
            "</think>\n"
            '{"ms": "Polis ke rumah kami.", "en": "Police to our house.", '
            '"zh": "警察来我们家。", "ta": "காவல்துறை எங்கள் வீட்டிற்கு வந்தது."}'
        )
        self.assertEqual("Polis ke rumah kami.", t["ms"])

    def test_ignores_trailing_commentary_after_the_object(self):
        t = extract_translation(
            '{"ms": "Saya faham.", "en": "I understand.", "zh": "我明白。", '
            '"ta": "எனக்கு புரிகிறது."} Hope this helps!'
        )
        self.assertEqual("Saya faham.", t["ms"])

    def test_keeps_commas_and_escaped_quotes_inside_a_sentence(self):
        t = extract_translation(
            r'{"ms": "Ya, saya \"faham\" sekarang.", "en": "Yes, I \"understand\" now.", '
            r'"zh": "是的，我现在“明白”了。", "ta": "ஆம், இப்போது எனக்கு புரிகிறது."}'
        )
        self.assertEqual('Ya, saya "faham" sekarang.', t["ms"])
        self.assertIn(",", t["en"])

    def test_rejects_a_missing_language(self):
        with self.assertRaises(ValueError):
            extract_translation('{"ms": "Saya lapar.", "en": "I am hungry."}')

    def test_rejects_a_reply_missing_tamil(self):
        with self.assertRaises(ValueError):
            extract_translation('{"ms": "Saya lapar.", "en": "I am hungry.", "zh": "我饿了。"}')

    def test_rejects_a_blank_language(self):
        with self.assertRaises(ValueError):
            extract_translation(
                '{"ms": "Saya lapar.", "en": "", "zh": "我饿了。", "ta": "எனக்கு பசி."}'
            )

    def test_rejects_prose_with_no_json_at_all(self):
        with self.assertRaises(ValueError):
            extract_translation("Polis means police, rumah means house.")

    def test_rejects_malformed_json(self):
        with self.assertRaises(ValueError):
            extract_translation('{"ms": "Saya lapar." "en": broken}')


class TestExtractChoice(unittest.TestCase):
    def test_parses_a_clean_choice(self):
        choice, reason = extract_choice(
            '{"choice": 1, "reason": "Most natural Malay without adding detail."}', 3
        )
        self.assertEqual(1, choice)
        self.assertIn("natural", reason)

    def test_strips_markdown_code_fences(self):
        choice, _ = extract_choice(
            '```json\n{"choice": 2, "reason": "Best reflects urgency."}\n```', 3
        )
        self.assertEqual(2, choice)

    def test_ignores_a_leading_reasoning_monologue(self):
        choice, _ = extract_choice(
            "<think>\nCandidate 0 is too literal. Candidate 1 reads naturally.\n</think>\n"
            '{"choice": 1, "reason": "Faithful and fluent."}',
            3,
        )
        self.assertEqual(1, choice)

    def test_ignores_trailing_commentary(self):
        choice, _ = extract_choice(
            '{"choice": 0, "reason": "Adds nothing."} Let me know if you need more.', 3
        )
        self.assertEqual(0, choice)

    def test_accepts_the_first_and_last_valid_index(self):
        self.assertEqual(0, extract_choice('{"choice": 0, "reason": "x"}', 3)[0])
        self.assertEqual(2, extract_choice('{"choice": 2, "reason": "x"}', 3)[0])

    def test_rejects_an_index_past_the_end(self):
        with self.assertRaises(ValueError):
            extract_choice('{"choice": 3, "reason": "x"}', 3)

    def test_rejects_a_negative_index(self):
        with self.assertRaises(ValueError):
            extract_choice('{"choice": -1, "reason": "x"}', 3)

    def test_bounds_the_index_to_the_candidates_actually_offered(self):
        """With two surviving agents, index 2 must not be accepted."""
        with self.assertRaises(ValueError):
            extract_choice('{"choice": 2, "reason": "x"}', 2)

    def test_rejects_a_missing_choice(self):
        with self.assertRaises(ValueError):
            extract_choice('{"reason": "I like the second one."}', 3)

    def test_rejects_a_non_integer_choice(self):
        with self.assertRaises(ValueError):
            extract_choice('{"choice": "the second one", "reason": "x"}', 3)

    def test_rejects_prose_with_no_json(self):
        with self.assertRaises(ValueError):
            extract_choice("I think candidate 1 is best.", 3)

    def test_tolerates_a_missing_reason(self):
        choice, reason = extract_choice('{"choice": 1}', 3)
        self.assertEqual(1, choice)
        self.assertEqual("", reason)


class TestPrompts(unittest.TestCase):
    """
    The prompt text is what the two builds must share; a difference here would
    silently change what the model is asked for.
    """

    def test_user_turn_matches_kotlin_list_rendering(self):
        # Kotlin interpolates List.toString(), which renders as "[A, B, C]".
        self.assertEqual(
            "Input: [Polis, Siapa, Salah]\nOutput:",
            prompts.user_turn(["Polis", "Siapa", "Salah"]),
        )

    def test_judge_turn_numbers_candidates_from_zero(self):
        turn = prompts.judge_turn(
            ["Polis", "Rumah"],
            [{"ms": "Polis ke rumah."}, {"ms": "Polis datang ke rumah kami."}],
        )
        self.assertIn("Glosses: [Polis, Rumah]", turn)
        self.assertIn("0. Polis ke rumah.", turn)
        self.assertIn("1. Polis datang ke rumah kami.", turn)

    def test_the_system_prompt_asks_for_emotion_and_style(self):
        # Personas were removed upstream: diversity now comes from using three
        # different models on one shared prompt, so the prompt must stay a
        # constant across agents. What it gained instead is the expression
        # rules and the two extra response fields.
        self.assertIn('"emotion"', prompts.SYSTEM)
        self.assertIn('"style"', prompts.SYSTEM)
        self.assertIn("Observed facial expression", prompts.SYSTEM)
        self.assertFalse(hasattr(prompts, "PERSONAS"))


if __name__ == "__main__":
    unittest.main()
