"""Central configuration for the Guarded Agent Ensemble.

Per CLAUDE.md §2 ("Config, not hardcoding"), every model ID, path, rate-limit
number and GAI weight lives here rather than being scattered through the code.
Swapping the backbone model (the architecture diagram's Hosted/Local mode
toggle) should be a one-line change in this file.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(_ENV_FILE)
_DOTENV = dotenv_values(_ENV_FILE)


def _setting(name: str, default: str = "") -> str:
    """Read a setting, preferring a non-empty value in .env.

    Plain `load_dotenv` never overrides a variable already exported in the
    shell, which makes a stale exported key impossible to fix by editing .env
    - the old value keeps winning and the failure looks like a bad key. Here
    .env wins whenever it actually has a value, and the ambient environment is
    the fallback rather than the other way round.
    """
    value = (_DOTENV.get(name) or "").strip()
    return value or os.getenv(name, default).strip()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SANDBOX_DIR = PROJECT_ROOT / "sandbox"
RESULTS_DIR = PROJECT_ROOT / "results"
TESTSUITES_DIR = PROJECT_ROOT / "src" / "eval" / "testsuites"

LLM_CACHE_DIR = PROJECT_ROOT / "src" / "llm" / "cache"
LLM_BUDGET_FILE = PROJECT_ROOT / "src" / "llm" / ".budget.json"


# ---------------------------------------------------------------------------
# OpenRouter (CLAUDE.md §5)
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = _setting("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# OpenRouter uses these two headers for its public rankings. Harmless, free.
HTTP_REFERER = "https://github.com/nishant/guarded-agent-ensemble"
X_TITLE = "Guarded Agent Ensemble"

# Ordered fallback chain (§5.2 point 6). Verified live against
# GET https://openrouter.ai/api/v1/models on 2026-09-09: all three exist, are
# zero-priced on prompt *and* completion, and advertise `tools` + `temperature`.
#
# Chosen for protocol reliability first (§5.3 is a prompted-text protocol, so
# instruction-following matters more than native tool support), then for
# provider diversity — a chain of three models from one org is no protection
# against that org's free tier being retired.
#
#   1. nex-agi/nex-n2.5-pro       agentic-tuned, 262K ctx, structured outputs,
#                                 dialable reasoning_effort. §7's nominated
#                                 primary, re-confirmed live.
#   2. google/gemma-4-31b-it      dense 31B; steadier at plain-text protocol
#                                 compliance than low-active-param MoEs, and
#                                 supports `seed` for reproducible demos.
#   3. nvidia/nemotron-3.5-lightning  1M ctx, `seed`, built for high-throughput
#                                 agentic loops; fastest, so it degrades
#                                 gracefully under budget pressure.
#
# Deliberate deviation from §7's example chain: liquid/lfm-2.5-2.6b:free is
# live but dropped. At 2.6B with an 8K max output — and with Liquid's own model
# card advising against agentic use — it will not reliably emit the Tool
# Dependency Graph that Phase 3 needs, and a fallback that cannot speak the
# protocol is not a fallback.
FREE_MODEL_CHAIN = [
    "nex-agi/nex-n2.5-pro:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3.5-lightning:free",
]

# §5.1: 50/day with no purchased credits, 1000/day after a one-time $10 top-up.
DAILY_REQUEST_CAP = int(_setting("DAILY_REQUEST_CAP", "50") or 50)


# §5.2 point 2: real cap is 20/min on :free models; target 15 for retry headroom.
RATE_LIMIT_PER_MINUTE = 15

# Warn loudly once this fraction of the daily budget is spent.
BUDGET_WARN_THRESHOLD = 0.8

# §5.2 point 5: retry policy for 429 / 5xx.
MAX_RETRIES = 5
RETRY_BASE_DELAY_S = 1.0
RETRY_MAX_DELAY_S = 30.0

# §5.2 point 4: cache ON for demos/dev, OFF for scored eval runs where N=3
# repeats are meant to capture real stochastic variance.
CACHE_ENABLED = True

REQUEST_TIMEOUT_S = 120.0
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024


# ---------------------------------------------------------------------------
# Google Gemini (second backend, same client.chat() interface)
# ---------------------------------------------------------------------------

# OpenRouter's 50/day free tier cannot cover a full A/B evaluation run (§5.4
# does the arithmetic). Gemini's free tier is several hundred per day, so it
# is the overflow backend: when OpenRouter's daily budget is spent, the client
# continues on Gemini rather than raising.
GEMINI_API_KEY = _setting("GEMINI_API_KEY")

# Verified live against GET https://generativelanguage.googleapis.com/v1beta/models
# on 2026-09-10, and probed with a real generateContent call: gemini-2.5-flash
# emitted the §5.3 protocol ("Final: 4") correctly on the first attempt.
#
# Deliberately pinned, non-preview, non-alias names. A `-latest` alias would
# silently change model underneath a scored run and destroy its
# reproducibility, which is the same volatility risk §5.1 warns about for
# OpenRouter's free catalogue.
GEMINI_MODEL_CHAIN = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

# MEASURED, not estimated. A run on 2026-09-10 hit HTTP 429 and the API named
# its own limit in the error body:
#
#   quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier
#   model:   gemini-2.5-flash
#   value:   20
#
# So the free tier is 20 requests per day PER MODEL - not the several hundred
# that secondary sources report. Two consequences:
#
#   - a cap of 500 here is not a guard at all; the local counter passed 73
#     while the real quota had already been refusing requests for a while.
#   - because the quota is per model, each entry in GEMINI_MODEL_CHAIN carries
#     its own separate 20, which is why budgets are scoped per model for this
#     provider (see PROVIDER_LIMITS below).
GEMINI_DAILY_REQUEST_CAP = int(_setting("GEMINI_DAILY_REQUEST_CAP", "20") or 20)
GEMINI_RATE_LIMIT_PER_MINUTE = 10


# Ordered (provider, model) pairs the client walks on failure. OpenRouter
# first because it is what the project was specified against; Gemini after it
# as the overflow.
#
# Crossing a provider boundary mid-run changes the backbone mid-experiment,
# which §9.1's A/B comparison cannot tolerate. The client logs every switch
# loudly and each run result records the model that served it - but a scored
# run should pin one model explicitly rather than rely on the chain.
PROVIDER_CHAIN: list[tuple[str, str]] = [
    *(("openrouter", model) for model in FREE_MODEL_CHAIN),
    *(("gemini", model) for model in GEMINI_MODEL_CHAIN),
]

# Per-provider limits, looked up by provider name.
#
# `budget_scope` records what the daily cap actually applies to, because the
# two backends differ and getting it wrong makes the counter meaningless:
#   - OpenRouter bills 50/day against the whole account, shared across models.
#   - Gemini allows 20/day per model, so each model has independent headroom.
PROVIDER_LIMITS: dict[str, dict] = {
    "openrouter": {
        "daily_cap": DAILY_REQUEST_CAP,
        "rate_limit_per_minute": RATE_LIMIT_PER_MINUTE,
        "budget_scope": "account",
    },
    "gemini": {
        "daily_cap": GEMINI_DAILY_REQUEST_CAP,
        "rate_limit_per_minute": GEMINI_RATE_LIMIT_PER_MINUTE,
        "budget_scope": "model",
    },
}


def budget_key(provider: str, model: str) -> str:
    """Which counter a call is charged to.

    Per model where the provider's quota is per model, per account otherwise.
    """
    scope = PROVIDER_LIMITS.get(provider, {}).get("budget_scope", "account")
    return f"{provider}:{model}" if scope == "model" else provider


def api_key_for(provider: str) -> str:
    """The configured credential for one backend."""
    return {
        "openrouter": OPENROUTER_API_KEY,
        "gemini": GEMINI_API_KEY,
    }.get(provider, "")


# ---------------------------------------------------------------------------
# Agent loop (CLAUDE.md §5.3)
# ---------------------------------------------------------------------------

# Hard ceiling on ReAct iterations, so a confused model cannot drain the budget.
AGENT_MAX_STEPS = 6

# How many times to hand a malformed Action back to the model before giving up
# on that step (§5.3: "let it retry once or twice").
AGENT_MAX_PARSE_RETRIES = 2


# ---------------------------------------------------------------------------
# GAI weights (CLAUDE.md §9) — consumed by eval/scorer.py from Phase 2 onward
# ---------------------------------------------------------------------------

GAI_WEIGHTS_DEFAULT = {
    "ASR_inj": 0.20,
    "DIV_ASR": 0.20,
    "HS": 0.15,
    "UA": 0.15,
    "BU": 0.10,
    "MF1": 0.10,
    "LAT": 0.10,
}
