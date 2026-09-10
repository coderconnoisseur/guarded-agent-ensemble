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


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------


class OpenRouterProvider(Provider):
    """OpenAI-compatible chat completions (CLAUDE.md 5.2 point 1)."""

    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"

    def __init__(self, referer: str = "", title: str = "") -> None:
        self.referer = referer
        self.title = title

    def endpoint(self, model: str) -> str:
        return "/chat/completions"

    def headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            # OpenRouter uses these for its public rankings; free, harmless.
            "HTTP-Referer": self.referer,
            "X-Title": self.title,
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
            # everything in `reasoning`, which would look like a refusal.
            content = message.get("reasoning") or ""
        if not isinstance(content, str) or not content.strip():
            raise ValueError("response contained no usable text")
        return content


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


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def build_providers(referer: str = "", title: str = "") -> dict[str, Provider]:
    """Every backend the client knows how to talk to, by name."""
    return {
        OpenRouterProvider.name: OpenRouterProvider(referer=referer, title=title),
        GeminiProvider.name: GeminiProvider(),
    }
