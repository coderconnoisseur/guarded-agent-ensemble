"""OpenRouter chat client - the single choke point for every LLM call.

Implements all six requirements of CLAUDE.md 5.2:

  1. Base config: OpenRouter base URL, bearer auth, HTTP-Referer / X-Title.
  2. Rate limiter: sleep-based, capped safely under the real 20 req/min.
  3. Daily budget tracker: persisted to disk, raises BudgetExceededError.
  4. Disk response cache: keyed on (model, messages, params), with force_refresh.
  5. Retry with backoff: 429 / 5xx, exponential + jitter, honours Retry-After.
  6. Model fallback chain: on persistent *errors* (not rate limits), switch model.

Every defense module, the agent loop, and the eval harness call `LLMClient.chat`
rather than touching HTTP directly - that is what makes each call countable
against the budget and inspectable in a transcript (CLAUDE.md 2).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from collections import deque
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import httpx
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMError(RuntimeError):
    """Base class for every failure raised by this module."""


class BudgetExceededError(LLMError):
    """The configured daily request cap has been reached (5.2 point 3)."""


class RateLimitExhaustedError(LLMError):
    """Retried through the backoff schedule and the API is still rate-limiting.

    Deliberately *not* a fallback trigger: OpenRouter's free-tier limits are
    account-wide, so trying the next model in the chain would only burn more
    quota against the same wall.
    """


class AllModelsFailedError(LLMError):
    """Every model in FREE_MODEL_CHAIN errored out (5.2 point 6)."""


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class LLMResponse(BaseModel):
    """One completion, plus the provenance the eval harness needs.

    `from_cache` is what the Phase 0 demo checks to prove the disk cache is
    live; `model_used` is what proves whether the fallback chain fired.
    """

    content: str
    model_used: str
    latency_ms: int
    from_cache: bool
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def usage(self) -> dict[str, Any]:
        """Token counts as reported by OpenRouter, or {} if absent."""
        return self.raw.get("usage") or {}


# ---------------------------------------------------------------------------
# Rate limiter (5.2 point 2)
# ---------------------------------------------------------------------------


class RateLimiter:
    """Sliding-window limiter capping outbound calls to N per minute.

    Sliding window rather than a fixed one so a burst at 0:59 followed by
    another at 1:01 cannot smuggle 2N requests past a 20/min ceiling.
    """

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Block until a slot is free. Returns seconds actually slept."""
        slept = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] >= 60.0:
                    self._times.popleft()
                if len(self._times) < self.per_minute:
                    self._times.append(now)
                    return slept
                wait = 60.0 - (now - self._times[0]) + 0.05
            logger.info(
                "Rate limiter: sleeping %.1fs to stay under %d/min", wait, self.per_minute
            )
            time.sleep(wait)
            slept += wait


# ---------------------------------------------------------------------------
# Daily budget tracker (5.2 point 3)
# ---------------------------------------------------------------------------


class BudgetTracker:
    """Persists {date, count} so the daily cap survives process restarts.

    Cache hits are never charged - only calls that actually leave the machine.
    """

    def __init__(self, path: Path, daily_cap: int, warn_threshold: float = 0.8) -> None:
        self.path = path
        self.daily_cap = daily_cap
        self.warn_threshold = warn_threshold
        self._lock = threading.Lock()

    def _load(self) -> dict[str, Any]:
        today = date.today().isoformat()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"date": today, "count": 0}
        if data.get("date") != today:
            return {"date": today, "count": 0}
        return {"date": today, "count": int(data.get("count", 0))}

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @property
    def used_today(self) -> int:
        return self._load()["count"]

    @property
    def remaining(self) -> int:
        return max(0, self.daily_cap - self.used_today)

    def check(self) -> None:
        """Raise BudgetExceededError if today's cap is spent."""
        used = self.used_today
        if used >= self.daily_cap:
            raise BudgetExceededError(
                f"Daily OpenRouter request cap reached: {used}/{self.daily_cap} used today. "
                f"Wait for the day rollover, raise DAILY_REQUEST_CAP in config/settings.py "
                f"if the account has purchased credits, or rely on the disk cache."
            )

    def charge(self) -> int:
        """Record one real network call. Returns the new count."""
        with self._lock:
            data = self._load()
            data["count"] += 1
            self._save(data)
            count = data["count"]
        if count >= self.daily_cap * self.warn_threshold:
            logger.warning(
                "OpenRouter daily budget: %d/%d used (%.0f%%) - approaching the cap.",
                count,
                self.daily_cap,
                100.0 * count / self.daily_cap,
            )
        else:
            logger.debug("OpenRouter daily budget: %d/%d used.", count, self.daily_cap)
        return count


# ---------------------------------------------------------------------------
# Disk cache (5.2 point 4)
# ---------------------------------------------------------------------------

# Only params that change the model's output belong in the cache key. Adding
# anything volatile (timestamps, request ids) here would silently disable
# caching by making every key unique.
_CACHE_KEY_PARAMS = (
    "temperature",
    "max_tokens",
    "top_p",
    "seed",
    "stop",
    "response_format",
    "reasoning",
    "reasoning_effort",
)


class DiskCache:
    """Content-addressed JSON cache of raw OpenRouter responses.

    Without this, re-running a demo during development burns real quota for
    zero new information - which is why 5.2 calls it "not optional".
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @staticmethod
    def make_key(
        model: str, messages: list[dict[str, Any]], params: dict[str, Any]
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "params": {k: params[k] for k in sorted(params) if k in _CACHE_KEY_PARAMS},
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        try:
            return json.loads(self._path(key).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, key: str, response: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            self._path(key).write_text(
                json.dumps(response, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            logger.warning("Could not write cache entry %s", key, exc_info=True)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LLMClient:
    """Wrapper around OpenRouter's /chat/completions endpoint.

    Usage:
        client = LLMClient()
        resp = client.chat([{"role": "user", "content": "hello"}])
        print(resp.content, resp.model_used, resp.from_cache)
    """

    def __init__(
        self,
        model_chain: Iterable[str] | None = None,
        api_key: str | None = None,
        cache_enabled: bool | None = None,
        daily_cap: int | None = None,
    ) -> None:
        self.model_chain = list(model_chain or settings.FREE_MODEL_CHAIN)
        if not self.model_chain:
            raise LLMError("FREE_MODEL_CHAIN is empty - nothing to call.")

        self.api_key = api_key if api_key is not None else settings.OPENROUTER_API_KEY
        self.cache_enabled = (
            settings.CACHE_ENABLED if cache_enabled is None else cache_enabled
        )

        self.cache = DiskCache(settings.LLM_CACHE_DIR)
        self.limiter = RateLimiter(settings.RATE_LIMIT_PER_MINUTE)
        self.budget = BudgetTracker(
            settings.LLM_BUDGET_FILE,
            daily_cap if daily_cap is not None else settings.DAILY_REQUEST_CAP,
            settings.BUDGET_WARN_THRESHOLD,
        )

        # Index into model_chain. Advances permanently once a model is judged
        # dead, so one bad model is diagnosed once per process, not per call.
        self._model_index = 0
        self._client = httpx.Client(
            base_url=settings.OPENROUTER_BASE_URL,
            timeout=settings.REQUEST_TIMEOUT_S,
        )

    # -- plumbing ----------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": settings.HTTP_REFERER,
            "X-Title": settings.X_TITLE,
            "Content-Type": "application/json",
        }

    @property
    def current_model(self) -> str:
        return self.model_chain[self._model_index]

    @staticmethod
    def _extract_content(payload: dict[str, Any]) -> str:
        """Pull assistant text out of an OpenRouter response.

        Reasoning models put their chain-of-thought in a sibling `reasoning`
        field, so `content` stays clean - but some return an empty string and
        put everything in `reasoning`, which would silently look like a refusal
        to the ReAct parser. Fall back to `reasoning` in exactly that case.
        """
        choices = payload.get("choices") or []
        if not choices:
            raise LLMError(f"Response contained no choices: {json.dumps(payload)[:400]}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):  # some providers return content parts
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if not content:
            content = message.get("reasoning") or ""
        if not isinstance(content, str) or not content.strip():
            raise LLMError(
                f"Response contained no usable text: {json.dumps(payload)[:400]}"
            )
        return content

    def _post_with_retries(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST one completion, retrying 429/5xx. Raises on give-up.

        Distinguishes the two give-up modes the fallback logic depends on:
        RateLimitExhaustedError (account-wide, do not switch model) versus
        LLMError (this model is broken, do switch).
        """
        last_error: str = ""
        for attempt in range(settings.MAX_RETRIES):
            self.limiter.acquire()
            self.budget.check()
            try:
                response = self._client.post(
                    "/chat/completions", headers=self._headers(), json=body
                )
            except httpx.RequestError as exc:  # network-level, worth retrying
                last_error = f"network error: {exc!r}"
                self._sleep_backoff(attempt, None)
                continue

            # Only charge for requests OpenRouter actually attributed to the
            # account: a served completion, or a 429 (which means the account
            # was identified and its quota consulted). A 401/403 never reached
            # an account, and a 5xx was never served - counting either would
            # spend the local daily budget on requests that cost nothing.
            if response.status_code == 200 or response.status_code == 429:
                self.budget.charge()

            if response.status_code == 200:
                payload = response.json()
                # OpenRouter can return HTTP 200 with an error body.
                if isinstance(payload, dict) and payload.get("error"):
                    raise LLMError(f"OpenRouter error: {payload['error']}")
                return payload

            if response.status_code == 429:
                last_error = f"429 rate limited: {response.text[:200]}"
                logger.warning("Rate limited by OpenRouter (attempt %d)", attempt + 1)
                self._sleep_backoff(attempt, response.headers.get("Retry-After"))
                continue

            if response.status_code >= 500:
                last_error = f"{response.status_code} server error: {response.text[:200]}"
                logger.warning(
                    "OpenRouter %d (attempt %d)", response.status_code, attempt + 1
                )
                self._sleep_backoff(attempt, response.headers.get("Retry-After"))
                continue

            # 4xx other than 429: retrying will not help. Auth failures are the
            # user's problem; anything else means this model is unusable.
            detail = f"HTTP {response.status_code}: {response.text[:300]}"
            if response.status_code in (401, 403):
                raise LLMError(
                    f"OpenRouter rejected the API key ({detail}). "
                    f"Check OPENROUTER_API_KEY in .env."
                )
            raise LLMError(detail)

        if last_error.startswith("429"):
            raise RateLimitExhaustedError(
                f"Still rate limited after {settings.MAX_RETRIES} attempts: {last_error}"
            )
        raise LLMError(f"Gave up after {settings.MAX_RETRIES} attempts: {last_error}")

    @staticmethod
    def _sleep_backoff(attempt: int, retry_after: str | None) -> None:
        """Exponential backoff with jitter, honouring Retry-After when sane."""
        if retry_after:
            try:
                delay = min(float(retry_after), settings.RETRY_MAX_DELAY_S)
                logger.info("Honouring Retry-After: sleeping %.1fs", delay)
                time.sleep(delay)
                return
            except ValueError:
                pass  # Retry-After can be an HTTP date; fall through to backoff
        delay = min(
            settings.RETRY_BASE_DELAY_S * (2**attempt), settings.RETRY_MAX_DELAY_S
        )
        delay += random.uniform(0, delay * 0.25)  # jitter, avoids retry convoys
        logger.info("Backing off %.1fs before retry %d", delay, attempt + 2)
        time.sleep(delay)

    # -- public API --------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        force_refresh: bool = False,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion and return the parsed result.

        Args:
            messages: OpenAI-style [{"role": ..., "content": ...}] list.
            model: Pin a specific model, bypassing the fallback chain entirely.
            force_refresh: Skip the cache read (still writes the fresh result).
                Use when a call is meant to sample new stochastic output.
            use_cache: Per-call override of settings.CACHE_ENABLED. Eval runs
                that measure variance should pass False.
            **kwargs: Passed through to OpenRouter (temperature, max_tokens...).

        Raises:
            BudgetExceededError: daily cap spent.
            RateLimitExhaustedError: still 429 after the full retry schedule.
            AllModelsFailedError: every model in the chain errored.
        """
        if not self.api_key:
            raise LLMError(
                "OPENROUTER_API_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )

        params: dict[str, Any] = {
            "temperature": settings.DEFAULT_TEMPERATURE,
            "max_tokens": settings.DEFAULT_MAX_TOKENS,
            **kwargs,
        }
        caching_on = self.cache_enabled if use_cache is None else use_cache

        # A pinned model tries once; otherwise walk the chain from where we are.
        candidates = [model] if model else self.model_chain[self._model_index :]
        failures: list[str] = []

        for candidate in candidates:
            key = DiskCache.make_key(candidate, messages, params)

            if caching_on and not force_refresh:
                cached = self.cache.get(key)
                if cached is not None:
                    logger.debug("Cache hit for %s (%s)", candidate, key[:12])
                    return LLMResponse(
                        content=self._extract_content(cached),
                        model_used=cached.get("model", candidate),
                        latency_ms=0,
                        from_cache=True,
                        raw=cached,
                    )

            body = {"model": candidate, "messages": messages, **params}
            started = time.perf_counter()
            try:
                payload = self._post_with_retries(body)
            except (BudgetExceededError, RateLimitExhaustedError):
                raise  # account-wide; the next model would hit the same wall
            except LLMError as exc:
                failures.append(f"{candidate}: {exc}")
                logger.error("Model %s failed, falling back. Reason: %s", candidate, exc)
                if model is None and self._model_index < len(self.model_chain) - 1:
                    self._model_index += 1
                    logger.warning("FALLBACK: now using %s", self.current_model)
                continue

            latency_ms = int((time.perf_counter() - started) * 1000)
            content = self._extract_content(payload)
            if caching_on:
                self.cache.put(key, payload)
            return LLMResponse(
                content=content,
                model_used=payload.get("model", candidate),
                latency_ms=latency_ms,
                from_cache=False,
                raw=payload,
            )

        raise AllModelsFailedError(
            "Every model in the chain failed:\n  " + "\n  ".join(failures)
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
