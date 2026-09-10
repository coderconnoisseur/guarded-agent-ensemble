"""Multi-provider chat client - the single choke point for every LLM call.

Implements all six requirements of CLAUDE.md 5.2:

  1. Base config: per-provider base URL, auth and headers.
  2. Rate limiter: sleep-based, one sliding window per provider.
  3. Daily budget tracker: persisted to disk, one counter per provider.
  4. Disk response cache: keyed on (model, messages, params), with force_refresh.
  5. Retry with backoff: 429 / 5xx, exponential + jitter, honours Retry-After.
  6. Model fallback chain: on persistent errors, or on a provider running out
     of daily budget, continue down the chain.

Every defense module, the agent loop, and the eval harness call
`LLMClient.chat` rather than touching HTTP directly - that is what makes each
call countable against a budget and inspectable in a transcript (2).

Two backends are wired in (see `providers.py`): OpenRouter, which the project
was specified against, and Google Gemini, whose free tier is large enough to
run a full A/B evaluation that OpenRouter's 50/day cannot.

  METHODOLOGICAL WARNING. When a provider's budget runs out mid-run the client
  continues on the next provider, which changes the backbone mid-experiment.
  9.1 requires the same backbone across a comparison, so every switch is
  logged loudly and every result records the model that served it. Pin a model
  explicitly for scored runs.
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
from src.llm.providers import Provider, build_providers

logger = logging.getLogger(__name__)

# Human-facing name of the credential each backend needs, for error messages.
_KEY_NAMES = {
    "openrouter": "OPENROUTER_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMError(RuntimeError):
    """Base class for every failure raised by this module."""


class BudgetExceededError(LLMError):
    """Every usable provider has spent its daily request cap (5.2 point 3)."""


class RateLimitExhaustedError(LLMError):
    """Retried through the backoff schedule and still being rate-limited."""


class AllModelsFailedError(LLMError):
    """Every entry in the provider chain failed (5.2 point 6)."""


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class LLMResponse(BaseModel):
    """One completion, plus the provenance the eval harness needs.

    `from_cache` is what the Phase 0 demo checks to prove the disk cache is
    live; `model_used` and `provider` are what prove whether the fallback
    chain fired, and whether a run stayed on one backbone.
    """

    content: str
    model_used: str
    latency_ms: int
    from_cache: bool
    raw: dict[str, Any] = Field(default_factory=dict)
    provider: str = ""
    # Why generation stopped, and whether a provider-side safety layer caused
    # it. Recorded so a hosted filter cannot be mistaken for the model itself
    # refusing - see Provider.finish_signal.
    finish_reason: str = ""
    provider_filtered: bool = False

    @property
    def usage(self) -> dict[str, Any]:
        """Token counts as reported by the backend, or {} if absent."""
        return self.raw.get("usage") or self.raw.get("usageMetadata") or {}


# ---------------------------------------------------------------------------
# Rate limiter (5.2 point 2)
# ---------------------------------------------------------------------------


class RateLimiter:
    """Sliding-window limiter capping outbound calls to N per minute.

    Sliding window rather than a fixed one so a burst at 0:59 followed by
    another at 1:01 cannot smuggle 2N requests past the ceiling.
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
    """Persists per-provider request counts so caps survive restarts.

    One file holds a counter per backend, because the caps differ by an order
    of magnitude (OpenRouter 50/day, Gemini several hundred) and spending one
    must not consume the other's headroom.

    Cache hits are never charged - only calls that actually leave the machine.
    """

    def __init__(
        self,
        path: Path,
        daily_cap: int,
        provider: str = "openrouter",
        warn_threshold: float = 0.8,
    ) -> None:
        self.path = path
        self.daily_cap = daily_cap
        self.provider = provider
        self.warn_threshold = warn_threshold
        self._lock = threading.Lock()

    def _load_document(self) -> dict[str, Any]:
        today = date.today().isoformat()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"date": today, "counts": {}}
        if data.get("date") != today:
            return {"date": today, "counts": {}}
        counts = data.get("counts")
        if not isinstance(counts, dict):
            # Migrate the single-provider format this file used before a
            # second backend existed.
            counts = {"openrouter": int(data.get("count", 0))}
        return {"date": today, "counts": {k: int(v) for k, v in counts.items()}}

    def _save_document(self, document: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(document, indent=2), encoding="utf-8")

    @property
    def used_today(self) -> int:
        return self._load_document()["counts"].get(self.provider, 0)

    @property
    def remaining(self) -> int:
        return max(0, self.daily_cap - self.used_today)

    @property
    def exhausted(self) -> bool:
        return self.used_today >= self.daily_cap

    def check(self) -> None:
        """Raise BudgetExceededError if this provider's cap is spent."""
        used = self.used_today
        if used >= self.daily_cap:
            raise BudgetExceededError(
                f"Daily {self.provider} request cap reached: {used}/{self.daily_cap} "
                f"used today. Wait for the day rollover, raise the cap in "
                f"config/settings.py if the account allows more, or rely on the cache."
            )

    def charge(self) -> int:
        """Record one real network call. Returns the new count."""
        with self._lock:
            document = self._load_document()
            count = document["counts"].get(self.provider, 0) + 1
            document["counts"][self.provider] = count
            self._save_document(document)
        if count >= self.daily_cap * self.warn_threshold:
            logger.warning(
                "%s daily budget: %d/%d used (%.0f%%) - approaching the cap.",
                self.provider, count, self.daily_cap, 100.0 * count / self.daily_cap,
            )
        else:
            logger.debug(
                "%s daily budget: %d/%d used.", self.provider, count, self.daily_cap
            )
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
    """Content-addressed JSON cache of raw backend responses.

    Keyed on the model id, which is globally unique across our backends, so
    entries written before a second provider existed remain valid.
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
    """Wrapper around every chat backend the project can use.

    Usage:
        client = LLMClient()
        resp = client.chat([{"role": "user", "content": "hello"}])
        print(resp.content, resp.model_used, resp.provider, resp.from_cache)
    """

    def __init__(
        self,
        model_chain: Iterable[str] | Iterable[tuple[str, str]] | None = None,
        api_key: str | None = None,
        cache_enabled: bool | None = None,
        daily_cap: int | None = None,
    ) -> None:
        self.chain: list[tuple[str, str]] = self._normalise_chain(model_chain)
        if not self.chain:
            raise LLMError("The provider chain is empty - nothing to call.")

        self.providers: dict[str, Provider] = build_providers(
            referer=settings.HTTP_REFERER, title=settings.X_TITLE
        )
        # A single explicit key overrides every backend's configured key. Used
        # by tests, and by anyone pinning one backend deliberately.
        self._api_key_override = api_key
        self.cache_enabled = (
            settings.CACHE_ENABLED if cache_enabled is None else cache_enabled
        )
        self.cache = DiskCache(settings.LLM_CACHE_DIR)

        self.limiters: dict[str, RateLimiter] = {}
        self.budgets: dict[str, BudgetTracker] = {}
        for provider_name, model_name in self.chain:
            limits = settings.PROVIDER_LIMITS.get(
                provider_name,
                {
                    "daily_cap": settings.DAILY_REQUEST_CAP,
                    "rate_limit_per_minute": settings.RATE_LIMIT_PER_MINUTE,
                },
            )
            self.limiters.setdefault(
                provider_name, RateLimiter(limits["rate_limit_per_minute"])
            )
            # Keyed by whatever the provider's quota actually applies to:
            # OpenRouter's 50/day covers the account, Gemini's 20/day covers
            # one model, so they cannot share a counter.
            key = settings.budget_key(provider_name, model_name)
            self.budgets.setdefault(
                key,
                BudgetTracker(
                    settings.LLM_BUDGET_FILE,
                    daily_cap if daily_cap is not None else limits["daily_cap"],
                    provider=key,
                    warn_threshold=settings.BUDGET_WARN_THRESHOLD,
                ),
            )

        # Index into the chain. Advances permanently once an entry is judged
        # dead, so one bad model is diagnosed once per process, not per call.
        self._index = 0
        self._client = httpx.Client(timeout=settings.REQUEST_TIMEOUT_S)

    @staticmethod
    def _normalise_chain(
        model_chain: Iterable[str] | Iterable[tuple[str, str]] | None,
    ) -> list[tuple[str, str]]:
        """Accept plain model ids or explicit (provider, model) pairs.

        Bare strings are assumed to be OpenRouter models, which keeps every
        existing caller and test working unchanged.
        """
        if model_chain is None:
            return list(settings.PROVIDER_CHAIN)
        chain: list[tuple[str, str]] = []
        for entry in model_chain:
            if isinstance(entry, (tuple, list)):
                chain.append((str(entry[0]), str(entry[1])))
            else:
                chain.append(("openrouter", str(entry)))
        return chain

    # -- plumbing ----------------------------------------------------------

    def api_key(self, provider: str) -> str:
        if self._api_key_override is not None:
            return self._api_key_override
        return settings.api_key_for(provider)

    @property
    def current(self) -> tuple[str, str]:
        return self.chain[self._index]

    @property
    def current_model(self) -> str:
        return self.chain[self._index][1]

    @property
    def current_provider(self) -> str:
        return self.chain[self._index][0]

    def budget_for(self, provider: str, model: str) -> BudgetTracker:
        """The counter a (provider, model) pair is charged against."""
        return self.budgets[settings.budget_key(provider, model)]

    @property
    def budget(self) -> BudgetTracker:
        """The current chain entry's budget, for callers that want just one."""
        return self.budget_for(*self.current)

    def budget_summary(self) -> str:
        """One line per backend, for demo output."""
        return "; ".join(
            f"{name} {tracker.used_today}/{tracker.daily_cap}"
            for name, tracker in sorted(self.budgets.items())
        )

    def _post_with_retries(
        self, provider: Provider, model: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """POST one completion, retrying 429/5xx. Raises on give-up.

        Distinguishes the give-up modes the fallback logic depends on:
        RateLimitExhaustedError and BudgetExceededError are provider-wide, so
        the caller skips that whole backend; a plain LLMError means this model
        is broken and the caller moves to the next entry.
        """
        name = provider.name
        url = provider.base_url + provider.endpoint(model)
        headers = provider.headers(self.api_key(name))
        budget = self.budget_for(name, model)
        last_error = ""

        for attempt in range(settings.MAX_RETRIES):
            self.limiters[name].acquire()
            budget.check()
            try:
                response = self._client.post(url, headers=headers, json=body)
            except httpx.RequestError as exc:  # network-level, worth retrying
                last_error = f"network error: {exc!r}"
                self._sleep_backoff(attempt, None)
                continue

            # Only charge for requests the backend actually attributed to the
            # account: a served completion, or a 429 (identified, quota
            # consulted). A 401/403 never reached an account and a 5xx was
            # never served - counting either spends budget on nothing.
            if response.status_code in (200, 429):
                budget.charge()

            if response.status_code == 200:
                payload = response.json()
                error = provider.error_in_body(payload)
                if error:
                    raise LLMError(f"{name} error: {error}")
                return payload

            if response.status_code == 429:
                if self._is_daily_quota_error(response):
                    raise BudgetExceededError(
                        f"{name}/{model} has exhausted its daily quota "
                        f"(the API reported a per-day limit). Retrying cannot "
                        f"help until the quota resets."
                    )
                last_error = f"429 rate limited: {response.text[:200]}"
                logger.warning("Rate limited by %s (attempt %d)", name, attempt + 1)
                self._sleep_backoff(attempt, response.headers.get("Retry-After"))
                continue

            if response.status_code >= 500:
                last_error = f"{response.status_code} server error: {response.text[:200]}"
                logger.warning("%s %d (attempt %d)", name, response.status_code, attempt + 1)
                self._sleep_backoff(attempt, response.headers.get("Retry-After"))
                continue

            detail = f"HTTP {response.status_code}: {response.text[:300]}"
            if response.status_code in (401, 403):
                raise LLMError(
                    f"{name} rejected the API key ({detail}). "
                    f"Check {_KEY_NAMES.get(name, 'the API key')} in .env."
                )
            raise LLMError(detail)

        if last_error.startswith("429"):
            raise RateLimitExhaustedError(
                f"{name} still rate limited after {settings.MAX_RETRIES} attempts: "
                f"{last_error}"
            )
        raise LLMError(f"Gave up after {settings.MAX_RETRIES} attempts: {last_error}")

    @staticmethod
    def _is_daily_quota_error(response: httpx.Response) -> bool:
        """Does this 429 mean "done for today" rather than "slow down"?

        Backing off against a per-minute limit is correct; backing off against
        a per-day quota just burns the retry schedule and wall-clock time for
        a request that cannot succeed until tomorrow. Gemini names the limit
        it hit in the error body, so the two are distinguishable.
        """
        try:
            details = (response.json().get("error") or {}).get("details") or []
        except (ValueError, AttributeError):
            return False
        for detail in details:
            for violation in detail.get("violations", []) or []:
                if "PerDay" in str(violation.get("quotaId", "")):
                    return True
        return False

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
        provider: str | None = None,
        force_refresh: bool = False,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion and return the parsed result.

        Args:
            messages: OpenAI-style [{"role": ..., "content": ...}] list.
            model: Pin a specific model, bypassing the chain. Scored runs
                should do this so the whole run shares one backbone.
            provider: Which backend the pinned model belongs to. Inferred from
                the chain when omitted.
            force_refresh: Skip the cache read (still writes the fresh result).
            use_cache: Per-call override of settings.CACHE_ENABLED. Eval runs
                measuring variance should pass False.
            **kwargs: Passed through as generation parameters.

        Raises:
            BudgetExceededError: every usable backend has spent its cap.
            RateLimitExhaustedError: still throttled after the retry schedule.
            AllModelsFailedError: every chain entry errored.
        """
        params: dict[str, Any] = {
            "temperature": settings.DEFAULT_TEMPERATURE,
            "max_tokens": settings.DEFAULT_MAX_TOKENS,
            **kwargs,
        }
        caching_on = self.cache_enabled if use_cache is None else use_cache
        candidates = self._candidates(model, provider)

        failures: list[str] = []
        # Tracked as a count rather than by matching error text: deciding which
        # exception to raise by grepping a message breaks the moment the
        # wording changes.
        budget_failures = 0
        blocked: set[str] = set()  # providers out of budget or throttled

        for provider_name, candidate in candidates:
            key = DiskCache.make_key(candidate, messages, params)

            if caching_on and not force_refresh:
                cached = self.cache.get(key)
                if cached is not None:
                    logger.debug("Cache hit for %s (%s)", candidate, key[:12])
                    backend = self.providers[provider_name]
                    reason, filtered = backend.finish_signal(cached)
                    return LLMResponse(
                        content=backend.extract_content(cached),
                        model_used=backend.model_reported(cached, candidate),
                        latency_ms=0,
                        from_cache=True,
                        raw=cached,
                        provider=provider_name,
                        finish_reason=reason,
                        provider_filtered=filtered,
                    )

            if provider_name in blocked:
                continue
            if not self.api_key(provider_name):
                failures.append(
                    f"{provider_name}: {_KEY_NAMES.get(provider_name, 'API key')} "
                    f"is not set"
                )
                blocked.add(provider_name)
                continue

            backend = self.providers[provider_name]
            body = backend.build_body(candidate, messages, params)
            started = time.perf_counter()
            try:
                payload = self._post_with_retries(backend, candidate, body)
            except (BudgetExceededError, RateLimitExhaustedError) as exc:
                if isinstance(exc, BudgetExceededError):
                    budget_failures += 1
                # Provider-wide, not model-specific: skip this whole backend
                # and let the chain carry on with the next one. This is what
                # lets a run continue on Gemini when OpenRouter is spent.
                failures.append(f"{provider_name}/{candidate}: {exc}")
                blocked.add(provider_name)
                logger.warning(
                    "Provider %s is unavailable (%s). Trying the next backend.",
                    provider_name, type(exc).__name__,
                )
                self._advance_past(provider_name)
                continue
            except LLMError as exc:
                failures.append(f"{provider_name}/{candidate}: {exc}")
                logger.error("Model %s failed, falling back. Reason: %s", candidate, exc)
                if model is None and self._index < len(self.chain) - 1:
                    self._index += 1
                    logger.warning(
                        "FALLBACK: now using %s/%s", *self.current
                    )
                continue

            latency_ms = int((time.perf_counter() - started) * 1000)
            try:
                content = backend.extract_content(payload)
            except ValueError as exc:
                failures.append(f"{provider_name}/{candidate}: {exc}")
                logger.error("Model %s returned nothing usable: %s", candidate, exc)
                continue

            if caching_on:
                self.cache.put(key, payload)
            reason, filtered = backend.finish_signal(payload)
            if filtered:
                logger.warning(
                    "Provider-side safety filter fired on %s/%s (%s). This is "
                    "the provider, not the model - do not score it as a model "
                    "refusal.", provider_name, candidate, reason,
                )
            return LLMResponse(
                content=content,
                model_used=backend.model_reported(payload, candidate),
                latency_ms=latency_ms,
                from_cache=False,
                raw=payload,
                provider=provider_name,
                finish_reason=reason,
                provider_filtered=filtered,
            )

        if failures and budget_failures == len(failures):
            raise BudgetExceededError(
                "Every backend has spent its daily budget:\n  " + "\n  ".join(failures)
            )
        raise AllModelsFailedError(
            "Every entry in the provider chain failed:\n  " + "\n  ".join(failures)
        )

    def _candidates(
        self, model: str | None, provider: str | None
    ) -> list[tuple[str, str]]:
        """Which (provider, model) pairs this call may use, in order."""
        if model is None:
            return self.chain[self._index :]
        if provider:
            return [(provider, model)]
        for provider_name, candidate in self.chain:
            if candidate == model:
                return [(provider_name, model)]
        return [(self.current_provider, model)]

    def _advance_past(self, provider_name: str) -> None:
        """Move the chain pointer to the first entry on a different backend."""
        while (
            self._index < len(self.chain) - 1
            and self.chain[self._index][0] == provider_name
        ):
            self._index += 1

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
