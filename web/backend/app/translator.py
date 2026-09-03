"""
The multi-agent debate translator — a port of GeminiClients.kt,
GeminiTranslator.kt and DebateTranslator.kt.

BIM glosses arrive loosely ordered and the mapping to a sentence is genuinely
ambiguous, so a single model just commits to one reading. Three stances disagree
in useful ways and a judge resolves them. On unambiguous glosses the three
converge, which is correct but means the cost buys nothing there.

The judge returns an INDEX, not a sentence. That way the result is always
something an agent actually proposed — the judge cannot invent a fourth answer —
and the four languages stay mutually consistent because they come from one
agent's single response.

Costs four calls per translation. Wall clock is max(agents) + judge.

This is the one part of the app that had to move server-side rather than merely
being ported: the Android build reads its keys from BuildConfig, which is fine
on a device the user owns, but a browser bundle is public and shipping the keys
there would publish them.
"""

import asyncio
import itertools
import logging
from typing import AsyncIterator

from google import genai
from google.genai import types

from . import config, parsing, prompts

log = logging.getLogger(__name__)

# gemini-3.7-flash returned 503 "experiencing high demand" when this was first
# wired up on Android; this is the model the pipeline was validated against.
DEFAULT_MODEL = "gemini-3.1-flash-lite"

# Where the pipeline currently is, for display while the user waits.
#
# Deliberately vague — a port of TranslationStage.kt. The label tells the user
# the multi-agent machinery is working without putting three competing sentences
# or the judge's reasoning on screen. Those go to the server log instead.
STAGE_LABELS = {
    "IDLE": "",
    "CONSULTING": "Consulting interpreters…",
    "COLLECTED": "Interpretations received",
    "JUDGING": "Weighing interpretations…",
    "DECIDING": "Choosing best translation…",
}

# Long enough for each stage label to actually be read.
_STAGE_HOLD_S = 0.4


class GeminiClients:
    """
    The pool of Gemini clients — one per configured API key.

    A key is bound at client construction, so giving each debate agent its own
    key means giving each its own client. That is the point: the free-tier limit
    is 5 requests/minute *per Google Cloud project*, and the debate costs 4 calls
    per translation, so a single key allows barely one translation a minute.
    Three keys from three separate projects give three independent buckets. Keys
    issued from the *same* project share one bucket and buy nothing.

    Each client owns its own connection pool, so the pool is built once and
    shared. Never construct a client per request.
    """

    def __init__(self) -> None:
        self._keys = config.gemini_keys()
        self._clients: list[genai.Client] | None = None

        if not self._keys:
            log.warning("No Gemini API key configured — translation disabled")
        elif len(self._keys) == 3:
            log.info("3 Gemini keys configured (~15 req/min if separate projects)")
        else:
            # Still works, just with proportionally less quota, so say so loudly
            # rather than letting it surface later as mysterious 429s.
            log.warning(
                "Only %d of 3 Gemini keys configured — agents will share keys and "
                "quota. Add GEMINI_API_KEY_2/_3 to web/backend/.env.",
                len(self._keys),
            )

    @property
    def is_configured(self) -> bool:
        return bool(self._keys)

    @property
    def slot_count(self) -> int:
        return len(self._keys)

    def for_slot(self, index: int) -> genai.Client:
        """The client for agent/judge slot `index`, wrapping when fewer keys are set."""
        if not self._keys:
            raise RuntimeError("no Gemini API key configured")
        if self._clients is None:
            self._clients = [genai.Client(api_key=k) for k in self._keys]
        return self._clients[index % len(self._clients)]

    def slot_of(self, index: int) -> int:
        return -1 if not self._keys else index % len(self._keys)


clients = GeminiClients()


async def _complete(system: str, user: str, client: genai.Client, model_id: str) -> str:
    """
    One raw Gemini completion on a specific client, and therefore a specific key.

    Shared by the agents and the judge, which needs the same transport but not
    the translation contract.
    """
    response = await client.aio.models.generate_content(
        model=model_id,
        contents=user,
        config=types.GenerateContentConfig(system_instruction=system),
    )
    return response.text or ""


async def _agent(words: list[str], persona: str, slot: int, model_id: str) -> dict[str, str]:
    """
    One interpreter: gloss list in, four-language translation out.

    One retry, because a model that rambles once usually complies on a re-ask and
    a second round trip is cheaper than falling back to raw glosses. This catches
    parse failures only — a 429 or 503 propagates immediately, which is what we
    want since the debate tolerates a missing agent and retrying an exhausted
    quota would only stall.
    """
    system = prompts.with_persona(persona)
    last_error: Exception | None = None

    for attempt in range(2):
        try:
            raw = await _complete(system, prompts.user_turn(words), clients.for_slot(slot), model_id)
            translation = parsing.extract_translation(raw)
            log.info("key[%d] %s %s -> %s", slot, model_id, words, translation["ms"])
            return translation
        except ValueError as e:
            last_error = e
            log.warning("key[%d] unparseable reply, attempt %d: %s", slot, attempt + 1, e)

    raise last_error or RuntimeError("translation failed")


_judge_slot = itertools.count()


async def _judge(words: list[str], candidates: list[dict[str, str]], model_id: str) -> int:
    """
    Returns the index of the winning candidate.

    The judge rotates round-robin across the keys: there are four calls per
    translation but only three keys, so pinning the judge to one would make that
    key carry two calls per translation and hit its 5/min ceiling first.
    """
    slot = clients.slot_of(next(_judge_slot))
    log.info("judging on key[%d]", slot)
    raw = await _complete(
        prompts.JUDGE, prompts.judge_turn(words, candidates), clients.for_slot(slot), model_id
    )
    choice, reason = parsing.extract_choice(raw, len(candidates))
    log.info('judge chose [%d] "%s" — %s', choice, candidates[choice]["ms"], reason)
    return choice


async def debate(
    words: list[str], model_id: str = DEFAULT_MODEL
) -> AsyncIterator[tuple[str, dict[str, str] | None]]:
    """
    Runs the debate, yielding `(stage, result)` as it goes.

    A generator rather than a plain coroutine because the Android UI collects a
    StateFlow of the pipeline stage and shows it while the user waits; streaming
    the stages to the browser is how that same feedback survives the port. Every
    yield before the last carries `None` — only the final one carries the
    translation.

    Agent i draws on key slot i, so the three agents consume three separate
    free-tier quota buckets rather than competing for one.
    """
    if not words:
        raise ValueError("no glosses to translate")

    yield "CONSULTING", None

    settled = await asyncio.gather(
        *(
            _agent(words, persona, clients.slot_of(i), model_id)
            for i, persona in enumerate(prompts.PERSONAS)
        ),
        return_exceptions=True,
    )

    candidates: list[dict[str, str]] = []
    for i, outcome in enumerate(settled):
        if isinstance(outcome, BaseException):
            log.warning("interpreter[%d] failed: %s", i, outcome)
        else:
            candidates.append(outcome)
            log.info("candidate[%d]: %s", i, outcome["ms"])

    if not candidates:
        raise RuntimeError(f"all {len(prompts.PERSONAS)} interpreters failed")

    # Nothing to choose between — skip the judge and its latency.
    if len(candidates) == 1:
        log.info("only one interpreter succeeded, returning it unjudged")
        yield "IDLE", candidates[0]
        return

    yield "COLLECTED", None
    await asyncio.sleep(_STAGE_HOLD_S)
    yield "JUDGING", None
    await asyncio.sleep(_STAGE_HOLD_S)

    # A failed judge must not discard candidates we already hold; an arbitrary
    # but valid answer beats falling back to raw glosses.
    try:
        chosen = await _judge(words, candidates, model_id)
    except Exception as e:
        log.warning("judge failed, using first candidate: %s", e)
        chosen = 0

    yield "DECIDING", None
    await asyncio.sleep(_STAGE_HOLD_S)
    yield "IDLE", candidates[chosen]
