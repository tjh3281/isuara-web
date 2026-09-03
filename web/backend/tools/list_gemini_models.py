"""
List the Gemini models these API keys can actually reach.

The debate upstream moved from three prompt-personas on one model to three
DIFFERENT models sharing one prompt, so any disagreement is attributable to the
model rather than the stance. Reproducing that on Gemini needs three model ids
that genuinely exist for this account — guessing them would fail at runtime, on
the demo.

    python tools/list_gemini_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config  # noqa: E402


def main() -> int:
    keys = config.gemini_keys()
    if not keys:
        print("no Gemini key configured (web/backend/.env)")
        return 2

    from google import genai

    client = genai.Client(api_key=keys[0])

    rows = []
    for m in client.models.list():
        actions = list(getattr(m, "supported_actions", None) or [])
        if actions and "generateContent" not in actions:
            continue
        rows.append(m.name.removeprefix("models/"))

    rows.sort()
    print(f"{len(rows)} models support generateContent:\n")
    for name in rows:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
