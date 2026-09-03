"""
Provider-neutral parsing of LLM replies — a port of TranslationParsing.kt.

Lives apart from any one translator because every provider needs the same
tolerance: models wrap objects in markdown fences and some prepend a visible
reasoning monologue. Both are handled by taking the outermost brace-delimited
span.

Failures raise ValueError, matching the Kotlin contract where an unparseable
reply is an IllegalArgumentException rather than a returned error string — the
caller distinguishes a real translation from a failure by catching, never by
inspecting the text.
"""

import json
from typing import Any


def _json_span(raw: str) -> str | None:
    """The outermost `{...}` span, or None when there isn't one."""
    start = raw.find("{")
    end = raw.rfind("}")
    return raw[start : end + 1] if start >= 0 and end > start else None


def _parse(raw: str, what: str) -> dict[str, Any]:
    span = _json_span(raw)
    if span is None:
        raise ValueError(f"no JSON object in {what}: {raw[:200]}")
    try:
        obj = json.loads(span)
    except json.JSONDecodeError as e:
        raise ValueError(f"malformed JSON in {what}: {raw[:200]}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"JSON in {what} is not an object: {raw[:200]}")
    return obj


def extract_translation(raw: str) -> dict[str, str | None]:
    """
    Pulls the translation out of a raw model reply.

    The four languages are required. `emotion` and `style` are optional and
    default to None, mirroring Translator.kt: a model that ignores rule 9 must
    still produce a usable translation rather than failing the whole request.
    `style` is the delivery directive the voice engine consumes.

    Raises ValueError if no complete object is present or any of the four
    languages is missing or blank.
    """
    obj = _parse(raw, "reply")
    out: dict[str, str | None] = {
        k: str(obj.get(k, "")).strip() for k in ("ms", "en", "zh", "ta")
    }
    if not all(out.values()):
        present = " ".join(f"{k}={bool(v)}" for k, v in out.items())
        raise ValueError(f"missing language in reply: {present}")

    for k in ("emotion", "style"):
        value = str(obj.get(k, "")).strip()
        out[k] = value or None

    return out


def extract_choice(raw: str, candidate_count: int) -> tuple[int, str]:
    """
    Pulls the judge's choice out of a raw reply.

    `candidate_count` bounds the index — a judge that names a candidate we never
    sent must not select one.
    """
    obj = _parse(raw, "judge reply")
    if "choice" not in obj:
        raise ValueError(f"judge reply has no choice: {raw[:200]}")
    try:
        choice = int(obj["choice"])
    except (TypeError, ValueError) as e:
        raise ValueError(f"judge choice is not an integer: {raw[:200]}") from e
    if not 0 <= choice < candidate_count:
        raise ValueError(f"judge chose {choice}, outside 0..{candidate_count - 1}")
    return choice, str(obj.get("reason", "")).strip()
