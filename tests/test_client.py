"""Unit tests for the LLM client's caching, budget and fallback logic.

CLAUDE.md 12 calls these out specifically: a silent bug in caching or in the
fallback chain would quietly change which model produced a number without
being visible in any demo's output.
"""

from __future__ import annotations

import json

import httpx
import pytest

from config import settings
from src.llm.client import (
    AllModelsFailedError,
    BudgetExceededError,
    BudgetTracker,
    DiskCache,
    LLMClient,
    LLMError,
    RateLimiter,
)


def completion(text: str, model: str = "test/model") -> dict:
    return {
        "id": "gen-test",
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point the cache and budget file at a temp dir, not the real repo."""
    monkeypatch.setattr(settings, "LLM_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(settings, "LLM_BUDGET_FILE", tmp_path / "budget.json")
    return tmp_path


def make_client(handler, **kwargs) -> LLMClient:
    """An LLMClient whose HTTP layer is a MockTransport."""
    client = LLMClient(api_key="test-key", **kwargs)
    client._client = httpx.Client(
        base_url=settings.OPENROUTER_BASE_URL,
        transport=httpx.MockTransport(handler),
    )
    return client


# ---------------------------------------------------------------------------
# Cache keys
# ---------------------------------------------------------------------------


class TestCacheKey:
    def test_identical_inputs_give_identical_keys(self):
        messages = [{"role": "user", "content": "hello"}]
        params = {"temperature": 0.0, "max_tokens": 100}
        assert DiskCache.make_key("m", messages, params) == DiskCache.make_key(
            "m", messages, params
        )

    def test_key_changes_with_model(self):
        messages = [{"role": "user", "content": "hello"}]
        params = {"temperature": 0.0}
        assert DiskCache.make_key("a", messages, params) != DiskCache.make_key(
            "b", messages, params
        )

    def test_key_changes_with_messages(self):
        params = {"temperature": 0.0}
        a = DiskCache.make_key("m", [{"role": "user", "content": "x"}], params)
        b = DiskCache.make_key("m", [{"role": "user", "content": "y"}], params)
        assert a != b

    def test_key_changes_with_temperature(self):
        messages = [{"role": "user", "content": "hello"}]
        a = DiskCache.make_key("m", messages, {"temperature": 0.0})
        b = DiskCache.make_key("m", messages, {"temperature": 0.7})
        assert a != b

    def test_key_ignores_params_that_do_not_affect_output(self):
        """A volatile param in the key would silently disable caching."""
        messages = [{"role": "user", "content": "hello"}]
        base = {"temperature": 0.0}
        noisy = {"temperature": 0.0, "user": "req-12345", "metadata": {"ts": 99}}
        assert DiskCache.make_key("m", messages, base) == DiskCache.make_key(
            "m", messages, noisy
        )

    def test_key_is_order_independent_for_params(self):
        messages = [{"role": "user", "content": "hello"}]
        a = DiskCache.make_key("m", messages, {"temperature": 0.0, "max_tokens": 50})
        b = DiskCache.make_key("m", messages, {"max_tokens": 50, "temperature": 0.0})
        assert a == b


class TestDiskCache:
    def test_round_trip(self, tmp_path):
        cache = DiskCache(tmp_path / "cache")
        payload = completion("hi")
        cache.put("abc123", payload)
        assert cache.get("abc123") == payload

    def test_miss_returns_none(self, tmp_path):
        assert DiskCache(tmp_path / "cache").get("nope") is None

    def test_corrupt_entry_is_a_miss_not_a_crash(self, tmp_path):
        directory = tmp_path / "cache"
        directory.mkdir()
        (directory / "bad.json").write_text("{not json", encoding="utf-8")
        assert DiskCache(directory).get("bad") is None


# ---------------------------------------------------------------------------
# Budget tracker
# ---------------------------------------------------------------------------


class TestBudgetTracker:
    def test_starts_empty(self, tmp_path):
        assert BudgetTracker(tmp_path / "b.json", 50).used_today == 0

    def test_charge_persists_across_instances(self, tmp_path):
        path = tmp_path / "b.json"
        BudgetTracker(path, 50).charge()
        BudgetTracker(path, 50).charge()
        assert BudgetTracker(path, 50).used_today == 2

    def test_check_raises_at_cap(self, tmp_path):
        tracker = BudgetTracker(tmp_path / "b.json", 2)
        tracker.charge()
        tracker.check()  # 1/2 - fine
        tracker.charge()
        with pytest.raises(BudgetExceededError):
            tracker.check()

    def test_counter_resets_on_a_new_day(self, tmp_path):
        path = tmp_path / "b.json"
        path.write_text(json.dumps({"date": "1999-01-01", "count": 999}), encoding="utf-8")
        assert BudgetTracker(path, 50).used_today == 0

    def test_remaining_never_goes_negative(self, tmp_path):
        tracker = BudgetTracker(tmp_path / "b.json", 1)
        tracker.charge()
        tracker.charge()
        assert tracker.remaining == 0


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_allows_up_to_the_cap_without_sleeping(self):
        limiter = RateLimiter(per_minute=5)
        assert all(limiter.acquire() == 0.0 for _ in range(5))

    def test_blocks_past_the_cap_until_the_window_slides(self, monkeypatch):
        """A fixed window would let 2N requests through across a boundary."""
        clock = {"t": 1000.0}
        slept: list[float] = []

        def fake_sleep(seconds: float) -> None:
            slept.append(seconds)
            clock["t"] += seconds

        monkeypatch.setattr("src.llm.client.time.monotonic", lambda: clock["t"])
        monkeypatch.setattr("src.llm.client.time.sleep", fake_sleep)

        limiter = RateLimiter(per_minute=2)
        limiter.acquire()
        limiter.acquire()
        waited = limiter.acquire()  # third call must wait the window out

        assert slept, "the third call should have slept"
        assert waited == pytest.approx(sum(slept))
        assert 59.0 < sum(slept) <= 61.0


# ---------------------------------------------------------------------------
# Client behaviour
# ---------------------------------------------------------------------------


class TestChat:
    def test_returns_content_and_marks_fresh(self, isolated):
        client = make_client(lambda r: httpx.Response(200, json=completion("hello")))
        response = client.chat([{"role": "user", "content": "hi"}])
        assert response.content == "hello"
        assert response.from_cache is False

    def test_second_identical_call_is_served_from_cache(self, isolated):
        """The Phase 0 Definition of Done depends on exactly this."""
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(200, json=completion("hello"))

        client = make_client(handler)
        messages = [{"role": "user", "content": "hi"}]
        first = client.chat(messages)
        second = client.chat(messages)

        assert calls["n"] == 1, "the second call should not have hit the network"
        assert first.from_cache is False
        assert second.from_cache is True
        assert second.content == first.content

    def test_cache_hits_do_not_spend_budget(self, isolated):
        client = make_client(lambda r: httpx.Response(200, json=completion("hello")))
        messages = [{"role": "user", "content": "hi"}]
        client.chat(messages)
        spent_after_first = client.budget.used_today
        client.chat(messages)
        assert client.budget.used_today == spent_after_first

    def test_force_refresh_bypasses_the_cache(self, isolated):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(200, json=completion("hello"))

        client = make_client(handler)
        messages = [{"role": "user", "content": "hi"}]
        client.chat(messages)
        response = client.chat(messages, force_refresh=True)
        assert calls["n"] == 2
        assert response.from_cache is False

    def test_use_cache_false_disables_caching_for_eval_runs(self, isolated):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(200, json=completion("hello"))

        client = make_client(handler)
        messages = [{"role": "user", "content": "hi"}]
        client.chat(messages, use_cache=False)
        client.chat(messages, use_cache=False)
        assert calls["n"] == 2

    def test_missing_api_key_fails_loudly(self, isolated):
        client = LLMClient(api_key="")
        with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
            client.chat([{"role": "user", "content": "hi"}])

    def test_reasoning_only_response_is_not_treated_as_empty(self, isolated):
        """Some reasoning models leave `content` empty and fill `reasoning`."""
        payload = {
            "model": "test/model",
            "choices": [{"message": {"content": "", "reasoning": "Final: 42"}}],
        }
        client = make_client(lambda r: httpx.Response(200, json=payload))
        assert client.chat([{"role": "user", "content": "hi"}]).content == "Final: 42"

    def test_http_200_with_error_body_is_an_error(self, isolated):
        payload = {"error": {"message": "model unavailable", "code": 503}}
        client = make_client(lambda r: httpx.Response(200, json=payload))
        with pytest.raises(AllModelsFailedError):
            client.chat([{"role": "user", "content": "hi"}])


class TestFallbackChain:
    def test_falls_back_when_the_primary_model_errors(self, isolated):
        seen: list[str] = []

        def handler(request):
            model = json.loads(request.content)["model"]
            seen.append(model)
            if model == "a":
                return httpx.Response(404, text="No endpoints found for a")
            return httpx.Response(200, json=completion("hi", model=model))

        client = make_client(handler, model_chain=["a", "b"])
        response = client.chat([{"role": "user", "content": "x"}])
        assert seen == ["a", "b"]
        assert response.model_used == "b"

    def test_fallback_is_sticky_within_a_process(self, isolated):
        """A dead model should be diagnosed once, not re-tried on every call."""
        seen: list[str] = []

        def handler(request):
            model = json.loads(request.content)["model"]
            seen.append(model)
            if model == "a":
                return httpx.Response(404, text="gone")
            return httpx.Response(200, json=completion("hi", model=model))

        client = make_client(handler, model_chain=["a", "b"])
        client.chat([{"role": "user", "content": "x"}])
        client.chat([{"role": "user", "content": "y"}])
        assert seen.count("a") == 1
        assert client.current_model == "b"

    def test_all_models_failing_raises(self, isolated):
        client = make_client(
            lambda r: httpx.Response(404, text="gone"), model_chain=["a", "b"]
        )
        with pytest.raises(AllModelsFailedError):
            client.chat([{"role": "user", "content": "x"}])

    def test_auth_failure_does_not_silently_swap_models(self, isolated):
        """A bad key breaks every model; the message must say so."""
        client = make_client(
            lambda r: httpx.Response(401, text="User not found"), model_chain=["a", "b"]
        )
        with pytest.raises(AllModelsFailedError, match="OPENROUTER_API_KEY"):
            client.chat([{"role": "user", "content": "x"}])

    def test_auth_failures_do_not_spend_the_daily_budget(self, isolated):
        """A 401 never reached an account, so it cost nothing upstream."""
        client = make_client(
            lambda r: httpx.Response(401, text="User not found"), model_chain=["a", "b"]
        )
        with pytest.raises(AllModelsFailedError):
            client.chat([{"role": "user", "content": "x"}])
        assert client.budget.used_today == 0

    def test_pinned_model_ignores_the_chain(self, isolated):
        seen: list[str] = []

        def handler(request):
            seen.append(json.loads(request.content)["model"])
            return httpx.Response(200, json=completion("hi"))

        client = make_client(handler, model_chain=["a", "b"])
        client.chat([{"role": "user", "content": "x"}], model="pinned")
        assert seen == ["pinned"]


class TestRetries:
    def test_a_server_error_is_not_charged_to_the_budget(self, isolated, monkeypatch):
        monkeypatch.setattr("src.llm.client.time.sleep", lambda _s: None)
        state = {"n": 0}

        def handler(request):
            state["n"] += 1
            if state["n"] == 1:
                return httpx.Response(500, text="upstream boom")
            return httpx.Response(200, json=completion("recovered"))

        client = make_client(handler)
        client.chat([{"role": "user", "content": "x"}])
        assert client.budget.used_today == 1, "only the served completion counts"

    def test_retries_a_500_then_succeeds(self, isolated, monkeypatch):
        monkeypatch.setattr("src.llm.client.time.sleep", lambda _s: None)
        state = {"n": 0}

        def handler(request):
            state["n"] += 1
            if state["n"] == 1:
                return httpx.Response(500, text="upstream boom")
            return httpx.Response(200, json=completion("recovered"))

        client = make_client(handler)
        assert client.chat([{"role": "user", "content": "x"}]).content == "recovered"
        assert state["n"] == 2

    def test_rate_limit_does_not_trigger_model_fallback(self, isolated, monkeypatch):
        """Free-tier limits are account-wide - switching models just burns quota."""
        monkeypatch.setattr("src.llm.client.time.sleep", lambda _s: None)
        seen: list[str] = []

        def handler(request):
            seen.append(json.loads(request.content)["model"])
            return httpx.Response(429, text="rate limited")

        client = make_client(handler, model_chain=["a", "b"])
        with pytest.raises(LLMError):
            client.chat([{"role": "user", "content": "x"}])
        assert set(seen) == {"a"}, "should never have tried model b"
