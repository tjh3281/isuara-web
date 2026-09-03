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

# The model the pipeline was originally validated against, and the judge.
DEFAULT_MODEL = "gemini-3.1-flash-lite"

# The three models that debate.
#
# Upstream (service/GonkaTranslator.kt) moved from three prompt-personas on one
# model to three DIFFERENT models sharing one prompt: diversity then comes from
# the models themselves, so a disagreement between candidates is attributable to
# the model rather than to a differing stance, and the system prompt stays a
# constant across agents.
#
# This build stays on Gemini rather than GonkaRouter, which is a deliberate,
# user-directed divergence from the app — so the same idea is reproduced with
# three Gemini models spanning two generations and two size tiers. All three are
# confirmed present for these keys via tools/list_gemini_models.py; guessing ids
# would fail at runtime, in front of an audience.
AGENT_MODELS = [
    DEFAULT_MODEL,
    "gemini-2.5-flash",
    "gemini-3.5-flash",
]

# The judge. Mirrors the Android choice of reusing the default agent model:
# it sits on the critical path once the agents finish, so it wants to be the
# fast, clean one rather than the most capable.
JUDGE_MODEL = DEFAULT_MODEL

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


async def _agent(
    words: list[str],
    slot: int,
    model_id: str,
    emotion: dict | None = None,
) -> dict[str, str | None]:
    """
    One interpreter: gloss list in, translation out.

    Every agent gets the identical system prompt — see AGENT_MODELS. The only
    per-request variation is the observed expression line, which prompts.user_turn
    adds and which is the same for all three agents within a request.

    One retry, because a model that rambles once usually complies on a re-ask and
    a second round trip is cheaper than falling back to raw glosses. This catches
    parse failures only — a 429 or 503 propagates immediately, which is what we
    want since the debate tolerates a missing agent and retrying an exhausted
    quota would only stall.
    """
    user = prompts.user_turn(words, emotion)
    last_error: Exception | None = None

    for attempt in range(2):
        try:
            raw = await _complete(prompts.SYSTEM, user, clients.for_slot(slot), model_id)
            translation = parsing.extract_translation(raw)
            log.info("key[%d] %s %s -> %s", slot, model_id, words, translation["ms"])
            return translation
        except ValueError as e:
            last_error = e
            log.warning("key[%d] unparseable reply, attempt %d: %s", slot, attempt + 1, e)

    raise last_error or RuntimeError("translation failed")


_judge_slot = itertools.count()


async def _judge(
    words: list[str], candidates: list[dict[str, str | None]]
) -> tuple[int, str]:
    """
    Returns the winning candidate's index and the judge's stated reason.

    The reason is surfaced in the UI rather than only logged — it is what makes
    the accordion's verdict row meaningful instead of an unexplained pick.

    The judge rotates round-robin across the keys: there are four calls per
    translation but only three keys, so pinning the judge to one would make that
    key carry two calls per translation and hit its 5/min ceiling first.
    """
    slot = clients.slot_of(next(_judge_slot))
    log.info("judging on key[%d]", slot)
    raw = await _complete(
        prompts.JUDGE, prompts.judge_turn(words, candidates), clients.for_slot(slot), JUDGE_MODEL
    )
    choice, reason = parsing.extract_choice(raw, len(candidates))
    log.info('judge chose [%d] "%s" — %s', choice, candidates[choice]["ms"], reason)
    return choice, reason


async def debate(words: list[str], emotion: dict | None = None) -> AsyncIterator[dict]:
    """
    Runs the debate, streaming events as they happen.

    Events, mirroring DebateProgress.kt:
      {"stage": ...}                     pipeline stage changed
      {"candidate": {index, model, sentence?, failed}}   one agent resolved
      {"verdict": {choice, reason}}      the judge decided
      {"translation": {...}}             the final answer

    Agents are revealed **as they arrive** rather than after the slowest
    finishes. The Kotlin notes a measured spread of ~9s to ~126s across its three
    models, so batching the reveal would mean minutes of dead air — which is the
    whole reason DebateProgress exists alongside the flat stage enum.

    Agent i draws on key slot i, so the three consume three separate free-tier
    quota buckets rather than competing for one.
    """
    if not words:
        raise ValueError("no glosses to translate")

    yield {"stage": "CONSULTING"}

    # Announce the slots up front so the UI can lay out all three rows as
    # pending, then fill each in when its model answers.
    yield {
        "candidates": [
            {"index": i, "model": model, "sentence": None, "failed": False}
            for i, model in enumerate(AGENT_MODELS)
        ]
    }

    async def run(index: int) -> tuple[int, dict[str, str | None] | None, Exception | None]:
        """
        Wraps an agent so completion carries its own index.

        asyncio.as_completed yields results in finishing order and gives no way
        back to the task that produced them, so the index rides along rather
        than being reconstructed afterwards.
        """
        try:
            return index, await _agent(words, clients.slot_of(index), AGENT_MODELS[index], emotion), None
        except Exception as e:  # noqa: BLE001 — reported per-agent, never fatal
            return index, None, e

    tasks = [asyncio.create_task(run(i)) for i in range(len(AGENT_MODELS))]

    # Keyed by index so the judge sees candidates in a stable order regardless
    # of which model happened to answer first.
    resolved: dict[int, dict[str, str | None]] = {}

    for finished in asyncio.as_completed(tasks):
        index, translation, error = await finished
        model = AGENT_MODELS[index]

        if error is not None or translation is None:
            log.warning("interpreter[%d] (%s) failed: %s", index, model, error)
            yield {
                "candidate": {"index": index, "model": model, "sentence": None, "failed": True}
            }
            continue

        resolved[index] = translation
        log.info("candidate[%d] (%s): %s", index, model, translation["ms"])
        yield {
            "candidate": {
                "index": index,
                "model": model,
                "sentence": translation["ms"],
                "failed": False,
            }
        }

    candidates = [resolved[i] for i in sorted(resolved)]

    if not candidates:
        raise RuntimeError(f"all {len(AGENT_MODELS)} interpreters failed")

    # Nothing to choose between — skip the judge and its latency.
    if len(candidates) == 1:
        log.info("only one interpreter succeeded, returning it unjudged")
        yield {"stage": "IDLE"}
        yield {"translation": candidates[0]}
        return

    yield {"stage": "COLLECTED"}
    await asyncio.sleep(_STAGE_HOLD_S)
    yield {"stage": "JUDGING"}

    # A failed judge must not discard candidates we already hold; an arbitrary
    # but valid answer beats falling back to raw glosses.
    reason = ""
    try:
        chosen, reason = await _judge(words, candidates)
    except Exception as e:
        log.warning("judge failed, using first candidate: %s", e)
        chosen = 0

    yield {"verdict": {"choice": chosen, "reason": reason}}
    yield {"stage": "DECIDING"}
    await asyncio.sleep(_STAGE_HOLD_S)
    yield {"stage": "IDLE"}
    yield {"translation": candidates[chosen]}
