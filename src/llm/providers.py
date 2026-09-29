"""Backend adapters behind the single `LLMClient.chat()` entry point.

CLAUDE.md 5.2 requires one wrapper that every module calls through, and 11
anticipates swapping the backbone "behind the same client.chat() interface so
nothing else in the codebase needs to change". This module is what makes that
true for a second provider: adapters translate between our OpenAI-style
message list and each backend's wire format, and nothing else.

Providers are deliberately *pure translation*. They do not retry, count
budget, rate-limit or decide fallbacks - all of that is policy and lives in
`client.py`, so the two backends cannot drift apart in how they are governed.

Adding Google's Gemini alongside OpenRouter matters for one practical reason:
OpenRouter's free tier allows 50 requests/day, which is not enough to run a
full A/B evaluation. Gemini's free tier allows several hundred.

  METHODOLOGICAL WARNING. Falling back across *providers* mid-run changes the
  backbone mid-experiment. 9.1 requires "same backbone, same test suite,
  same day" for the A/B comparison to mean anything - a Condition A on one
  model and a Condition B on another measures the models, not the defenses.
  The client logs every switch loudly and every RunResult records the model
  that produced it; scored runs should pin a single model. See
  `LLMClient.chat(model=...)`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from config import settings

logger = logging.getLogger(__name__)


class Provider(ABC):
    """Translates a chat request to and from one backend's wire format."""

    name: str
    base_url: str

    @abstractmethod
    def endpoint(self, model: str) -> str:
        """URL path (relative to base_url) for a completion on `model`."""

    @abstractmethod
    def headers(self, api_key: str) -> dict[str, str]:
        """Auth and content headers."""

    @abstractmethod
    def build_body(
        self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Render our message list into the backend's request shape."""

    @abstractmethod
    def extract_content(self, payload: dict[str, Any]) -> str:
        """Pull the assistant text out. Raises ValueError when there is none."""

    def error_in_body(self, payload: dict[str, Any]) -> str | None:
        """Some backends return HTTP 200 with an error body. Detect that."""
        error = payload.get("error")
        if error:
            return str(error)
        return None

    def model_reported(self, payload: dict[str, Any], fallback: str) -> str:
        """Which model the backend says actually served the request."""
        return payload.get("model") or fallback

    def finish_signal(self, payload: dict[str, Any]) -> tuple[str, bool]:
        """Return (finish_reason, provider_filter_fired).

        This exists to separate two things our grader would otherwise score
        identically: the *model* declining a request, and the *provider*
        blocking it at the edge before or after the model saw it.

        It matters for the measurement. HS is meant to be a property of the
        backbone; if a hosted safety layer intercepts a request, the harness
        would record a refusal the model never produced, and Condition B's
        defenses would be measured against a baseline that is partly somebody
        else's filter. Recording the distinction turns that confound into a
        number the final report can state instead of a caveat it has to
        hand-wave. See docs/PROGRESS.md.
        """
        return "", False


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------


class OpenAICompatibleProvider(Provider):
    """Shared behaviour for backends that speak the OpenAI chat shape.

    OpenRouter and Groq differ only in base URL and auth headers, so the
    request/response handling lives here once rather than being copied and
    left to drift apart.
    """

    def endpoint(self, model: str) -> str:
        return "/chat/completions"

    def headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def build_body(
        self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]
    ) -> dict[str, Any]:
        return {"model": model, "messages": messages, **params}

    def extract_content(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            raise ValueError("response contained no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):  # some providers return content parts
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if not content:
            # Reasoning models sometimes leave `content` empty and put
            # everything in `reasoning`, which would otherwise look like a
            # refusal. Falling back keeps the call usable, but the text is
            # chain-of-thought rather than an answer, so the agent's protocol
            # parser will probably reject it - warn rather than fail silently.
            # openai/gpt-oss-* on Groq does this whenever max_tokens is small
            # enough that reasoning consumes the whole budget.
            content = message.get("reasoning") or ""
            if content.strip():
                logger.warning(
                    "Model returned empty content; falling back to its reasoning "
                    "field. Raise max_tokens or choose a non-reasoning model."
                )
        if not isinstance(content, str) or not content.strip():
            raise ValueError("response contained no usable text")
        return content


    # finish_reason values that mean a provider-side safety layer intervened
    # rather than the model finishing normally.
    _FILTER_REASONS = {"content_filter", "safety", "blocked", "moderation"}

    def finish_signal(self, payload: dict[str, Any]) -> tuple[str, bool]:
        choices = payload.get("choices") or []
        if not choices:
            return "", False
        choice = choices[0]
        reason = str(
            choice.get("finish_reason") or choice.get("native_finish_reason") or ""
        )
        return reason, reason.lower() in self._FILTER_REASONS


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter's aggregator API (CLAUDE.md 5.2 point 1)."""

    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"

    def __init__(self, referer: str = "", title: str = "") -> None:
        self.referer = referer
        self.title = title

    def headers(self, api_key: str) -> dict[str, str]:
        return {
            **super().headers(api_key),
            # OpenRouter uses these for its public rankings; free, harmless.
            "HTTP-Referer": self.referer,
            "X-Title": self.title,
        }


class GroqProvider(OpenAICompatibleProvider):
    """Groq's inference API - OpenAI-compatible, so only the endpoint differs.

    Measured on 2026-09-11 from its own response headers, which are the
    authoritative source rather than the published ranges: 1000 requests/day
    and 8000 tokens/minute per model on the free tier, with each model
    carrying its own independent bucket.

    The token ceiling binds before the request ceiling for this project - a
    ReAct turn carries the tool catalogue plus a growing transcript - which is
    why the configured per-minute rate is well below what the request quota
    alone would allow.
    """

    name = "groq"
    base_url = "https://api.groq.com/openai/v1"


class OllamaProvider(OpenAICompatibleProvider):
    """A model served locally by Ollama (CLAUDE.md §11, "Local/Colab GPU mode").

    Ollama exposes an OpenAI-compatible endpoint at `/v1`, so this is a base
    URL and auth and nothing else. That is the point of §11's promise that the
    backbone can be swapped "behind the same client.chat() interface so nothing
    else in the codebase needs to change": no defense module, no pipeline and
    no test knows this exists.

    WHY A LOCAL BACKBONE IS A MEASUREMENT DECISION, NOT A CONVENIENCE
    -----------------------------------------------------------------
    Two constraints have shaped every number this project has produced, and a
    local model removes both:

      - **The rate limit, not the daily cap, is what costs time.** Groq's 1000
        OTPM ceiling works out at 2 requests/minute, which makes the frozen
        ablation a ~9-hour job, forces N=1, and injects ~30s of sleep into
        every latency measurement (see `scorer.added_latency`). Locally there
        is no limiter at all.
      - **The pinned backbone resists the attacks.** 0/30 against AgentDojo's
        own five templates, and `ASR_inj` 0.07 at baseline. At that rate a
        significant result needs n=65 (`stats.required_n`); at a baseline of
        0.40 it needs n=9. A smaller model failing more often is *headroom*,
        which is the scarce resource here - not a downgrade.

    THE FLOOR, WHICH IS MEASURED
    ----------------------------
    Small does not mean arbitrarily small. `liquid/lfm-2.5-2.6b` was dropped
    from `FREE_MODEL_CHAIN` because at 2.6B it could not reliably emit the Tool
    Dependency Graph Phase 3 needs (settings.py, and CLAUDE.md §5.1). A
    backbone that cannot speak the protocol does not give a weak Condition B -
    it gives **no** Condition B, and the ablation disappears rather than
    gaining sensitivity. So a candidate is probed before it is trusted:
    `python demos/local_probe.py`.

    THINKING MODE CANNOT BE SWITCHED OFF, AND THAT IS MEASURED
    ----------------------------------------------------------
    qwen3:4b is a hybrid-reasoning model. Three ways of disabling that were
    tried against Ollama 0.34.4 on 2026-09-29, prompt "Reply with exactly:
    Final: hello", max_tokens=200:

        parameter                              reasoning   content
        (baseline)                                 660ch      12ch  "Final: hello"
        think: false                               736ch      12ch  no effect
        chat_template_kwargs.enable_thinking       736ch      12ch  no effect
        reasoning_effort: "none"                     0ch     759ch  WORSE

    The first two are no-ops through the `/v1` shim. The third is actively
    harmful: it does not stop the thinking, it relocates it *into* `content`,
    where the `Action:`/`Final:` parser sees prose. Code that did the first
    two shipped briefly and was removed - a setting that silently does
    nothing is exactly what this project keeps finding and deleting.

    What the baseline row shows is that none of it is necessary: Ollama
    returns the thinking in a separate `reasoning` field and `content` holds a
    clean answer. The only real constraint is that the answer comes *after*
    the thinking, so the reply budget has to cover both. Measured on the real
    ReAct prompt:

        max_tokens=400    finish=length  reasoning=1771ch  content=0ch     unparseable
        max_tokens=1200   finish=stop    reasoning=3524ch  content=144ch   parses
        max_tokens=2500   finish=stop    reasoning=3524ch  content=144ch   identical

    Hence `OLLAMA_MIN_MAX_TOKENS` rather than a thinking switch.

    No API key. Ollama ignores the Authorization header, but one is sent
    anyway so the request path stays identical to the hosted providers - a
    second code path here is a second place for the two to drift.
    """

    name = "ollama"
    base_url = settings.OLLAMA_BASE_URL

    def __init__(self, base_url: str = "") -> None:
        # Instance-level override, so a test or a machine serving on another
        # port does not have to mutate the class.
        if base_url:
            self.base_url = base_url

    def headers(self, api_key: str) -> dict[str, str]:
        # Ollama does not authenticate. Sending a placeholder rather than
        # omitting the header keeps this on the same code path as every other
        # OpenAI-compatible backend.
        return super().headers(api_key or "ollama")

    def build_body(
        self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]
    ) -> dict[str, Any]:
        body = super().build_body(model, messages, params)
        # RAISE THE REPLY BUDGET. This is the whole adaptation a local
        # reasoning model needs, and it replaces three things that were tried
        # first and measured to be useless - see the class docstring.
        #
        # `DEFAULT_MAX_TOKENS = 400` is a *Groq* constraint: that provider
        # charges the requested max_tokens against a 1000 output-tokens-per-
        # minute ceiling, so 400 is what keeps two calls a minute possible. No
        # such ceiling exists locally, and qwen3:4b spends ~900 tokens
        # thinking before it emits a single `Thought:/Action:` line. At 400 it
        # is cut off mid-thought (finish_reason "length", content empty), the
        # provider falls back to the reasoning field, and the agent loop sees
        # prose it cannot parse.
        #
        # Only ever raises, never lowers: a caller that deliberately asked for
        # a bigger budget keeps it.
        floor = settings.OLLAMA_MIN_MAX_TOKENS
        if body.get("max_tokens") is None or body["max_tokens"] < floor:
            body["max_tokens"] = floor
        return body


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------


class GeminiProvider(Provider):
    """Google's generativelanguage API.

    Three differences from the OpenAI shape have to be bridged here, and each
    one silently corrupts a request if missed:

      - the assistant role is called "model", not "assistant";
      - system prompts are a separate `systemInstruction` field, not a message;
      - consecutive same-role turns are merged, because our ReAct loop emits
        two user turns in a row whenever it feeds back a parse-repair prompt.
    """

    name = "gemini"
    base_url = "https://generativelanguage.googleapis.com/v1beta"

    # Our parameter names -> Gemini's generationConfig names.
    _PARAM_MAP = {
        "temperature": "temperature",
        "max_tokens": "maxOutputTokens",
        "top_p": "topP",
        "top_k": "topK",
        "stop": "stopSequences",
        "seed": "seed",
    }

    def endpoint(self, model: str) -> str:
        return f"/models/{model}:generateContent"

    def headers(self, api_key: str) -> dict[str, str]:
        # Header auth rather than the ?key= query parameter, so the credential
        # never lands in a URL, a log line or an error message.
        return {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    def build_body(
        self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]
    ) -> dict[str, Any]:
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []

        for message in messages:
            role = message.get("role", "user")
            text = str(message.get("content", ""))
            if role == "system":
                system_parts.append(text)
                continue
            gemini_role = "model" if role == "assistant" else "user"
            if contents and contents[-1]["role"] == gemini_role:
                contents[-1]["parts"].append({"text": text})
            else:
                contents.append({"role": gemini_role, "parts": [{"text": text}]})

        generation_config = {
            gemini_name: params[ours]
            for ours, gemini_name in self._PARAM_MAP.items()
            if params.get(ours) is not None
        }

        body: dict[str, Any] = {"contents": contents}
        if system_parts:
            body["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_parts)}]
            }
        if generation_config:
            body["generationConfig"] = generation_config
        return body

    def extract_content(self, payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates") or []
        if not candidates:
            raise ValueError("response contained no candidates")
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        if not text.strip():
            # A thinking model can spend its whole output budget on reasoning
            # and return no text at all. Say which, so the cause is obvious.
            reason = candidate.get("finishReason", "unknown")
            raise ValueError(
                f"response contained no usable text (finishReason={reason})"
            )
        return text

    def model_reported(self, payload: dict[str, Any], fallback: str) -> str:
        return payload.get("modelVersion") or fallback

    # Gemini signals a blocked generation through finishReason, and a blocked
    # *prompt* through promptFeedback.blockReason before the model runs.
    _FILTER_REASONS = {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}

    def finish_signal(self, payload: dict[str, Any]) -> tuple[str, bool]:
        if (payload.get("promptFeedback") or {}).get("blockReason"):
            return f"PROMPT_BLOCKED:{payload['promptFeedback']['blockReason']}", True
        candidates = payload.get("candidates") or []
        if not candidates:
            return "", False
        reason = str(candidates[0].get("finishReason") or "")
        return reason, reason.upper() in self._FILTER_REASONS


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def build_providers(referer: str = "", title: str = "") -> dict[str, Provider]:
    """Every backend the client knows how to talk to, by name."""
    return {
        OpenRouterProvider.name: OpenRouterProvider(referer=referer, title=title),
        GeminiProvider.name: GeminiProvider(),
        GroqProvider.name: GroqProvider(),
        OllamaProvider.name: OllamaProvider(),
    }
