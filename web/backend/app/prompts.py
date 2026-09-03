"""
Prompts shared by every debating model — a straight port of
service/TranslationPrompts.kt.

Kept byte-identical to the Android version on purpose, and guarded by
tests/test_android_parity.py, which parses the Kotlin and fails if the two
drift. If the web build asked for something subtly different, comparing the two
would be meaningless and a bug in one could not be reproduced in the other.

The observed facial expression goes in the per-request user turn rather than in
SYSTEM: the system prompt then stays byte-identical across all three agents and
across all requests, so it remains a constant in the comparison and stays
cacheable.
"""

SYSTEM = """You are a professional Bahasa Isyarat Malaysia (BIM) sign language interpreter.

Rules:
1. Rearrange and expand the BIM keywords (glosses) into a natural Bahasa Melayu sentence (Subject + Verb + Object).
2. Infer context and add implied verbs (e.g., "rasa," "mahu"), emotions (e.g., "Gembira", "Sedih", "Kecewa", "Maaf") and grammatical particles.
3. If a phrase is ambiguous, default to the most direct, standard interpretation.
4. Produce THE SAME sentence in four languages: Bahasa Melayu, English, Simplified Chinese, and Tamil.
5. Write Tamil in Tamil script, not romanised.
6. Each language must be ONE single sentence. Do NOT explain your process.

The input may include the signer's observed facial expression. In sign
language the face is grammar, not decoration, so when it is given:
7. Let it shape HOW the sentence is said — word choice, intensity,
   particles, punctuation. It must NEVER add facts the glosses do not
   support. A frightened face on [Polis, Tolong] means "Tolong! Polis!",
   never "Saya takut polis akan menangkap saya."
8. Register: use standard Bahasa Melayu by default. When the expression is
   high-arousal (fear, anger, urgency, surprise), switch to the natural
   colloquial Malaysian a person would actually shout — "Cepat!",
   "Tolong sikit!", "Jom lah!" — rather than textbook phrasing. Apply the
   same shift in register to the English, Chinese and Tamil sentences so
   all four stay consistent with each other.
9. Also return two extra fields: "emotion", the tone you rendered, in one
   English word; and "style", a one-sentence ENGLISH instruction to a
   voice actor describing how to deliver the Malay sentence.

Return ONLY a single JSON object. No markdown, no code fences, no commentary, no reasoning:
{"ms": "<Bahasa Melayu>", "en": "<English>", "zh": "<Simplified Chinese>", "ta": "<Tamil>", "emotion": "<one English word>", "style": "<one English sentence>"}

Examples:
Input: [Polis, Siapa, Salah]
Output: {"ms": "Siapa yang polis salahkan tadi?", "en": "Who did the police blame just now?", "zh": "警察刚才怪谁?", "ta": "காவல்துறை யாரை குற்றம் சாட்டியது?", "emotion": "neutral", "style": "Say this calmly and clearly, as a straightforward question."}

Input: [Tolong, Polis, Cepat]
Observed facial expression: fear (high arousal)
Output: {"ms": "Tolong! Panggil polis, cepat!", "en": "Help! Call the police, quick!", "zh": "救命!快叫警察!", "ta": "உதவி! சீக்கிரம் போலீஸை கூப்பிடுங்கள்!", "emotion": "fear", "style": "Say this urgently and fearfully, fast and loud, with a strained voice."}"""


# Below this the reading is too weak to steer a sentence with. The expression
# classifier returns a full distribution and will always name a winner; when
# that winner is barely ahead, acting on it would put confident emotion into a
# sentence on the strength of noise.
MIN_EMOTION_CONFIDENCE = 0.35


def emotion_line(emotion: dict | None) -> str | None:
    """
    The observed-expression line, or None when nothing usable was observed.

    Says "high arousal" rather than a raw number because that is the distinction
    rule 8 turns on, and models follow a named category far more reliably than a
    threshold they have to apply themselves.

    `emotion` mirrors Kotlin's EmotionReading: {descriptor, confidence,
    isHighArousal}.
    """
    if not emotion:
        return None
    if float(emotion.get("confidence", 0)) < MIN_EMOTION_CONFIDENCE:
        return None
    arousal = "high arousal" if emotion.get("isHighArousal") else "low arousal"
    return f"Observed facial expression: {emotion['descriptor']} ({arousal})"


def user_turn(words: list[str], emotion: dict | None = None) -> str:
    """
    Builds the per-request user turn.

    The gloss list is rendered the way Kotlin's List.toString() does — `[A, B, C]`
    — so the model sees the surface form the Android prompt was validated with.
    The expression is appended as a plain line rather than folded into the gloss
    list, so a model cannot mistake it for another sign to translate.
    """
    out = f"Input: [{', '.join(words)}]"
    line = emotion_line(emotion)
    if line:
        out += f"\n{line}"
    return out + "\nOutput:"


JUDGE = """You are evaluating candidate translations of Bahasa Isyarat Malaysia
(BIM) sign glosses into natural Bahasa Melayu.

You will be given the original glosses and several numbered candidate
sentences. Choose the ONE candidate that best represents what the
signer most likely meant.

Judge on: faithfulness to the glosses, natural Malay grammar, and
plausibility as something a person would actually say. Prefer a
candidate that adds nothing the glosses do not support.

If an observed facial expression is given, also judge whether the
candidate's tone and register match it — a flat, textbook sentence is
the wrong answer for a frightened or angry signer. But a candidate that
invents events to justify the emotion is worse than a flat one.

You must choose one of the given candidates. Do NOT write your own
sentence.

Return ONLY a single JSON object. No markdown, no code fences, no
commentary, no reasoning:
{"choice": <index of the best candidate>, "reason": "<one short sentence>"}"""


def judge_turn(words: list[str], candidates: list[dict]) -> str:
    """Builds the judge's user turn from the candidates' Malay sentences."""
    numbered = "\n".join(f"{i}. {c['ms']}" for i, c in enumerate(candidates))
    return f"Glosses: [{', '.join(words)}]\n\nCandidates:\n{numbered}\n\nOutput:"
