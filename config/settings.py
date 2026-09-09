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
