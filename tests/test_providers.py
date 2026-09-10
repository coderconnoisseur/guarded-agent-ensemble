"""Tests for the provider adapters and cross-provider fallback.

The message translation is the risky part: Gemini renames the assistant role,
moves the system prompt out of the message list, and rejects some shapes our
ReAct loop produces naturally. A translation bug would not crash - it would
quietly send the model a different conversation than the one we recorded in
the transcript, and every number derived from that run would be wrong.

No network calls in this file.
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
)
from src.llm.providers import (
    GeminiProvider,
    GroqProvider,
    OpenRouterProvider,
    build_providers,
)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "LLM_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(settings, "LLM_BUDGET_FILE", tmp_path / "budget.json")
    return tmp_path


def gemini_reply(text: str) -> dict:
    return {
        "candidates": [
            {"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}
        ],
        "modelVersion": "gemini-2.5-flash",
        "usageMetadata": {"totalTokenCount": 42},
    }


def openrouter_reply(text: str, model: str = "test/model") -> dict:
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": text}}],
    }


# ---------------------------------------------------------------------------
# Gemini request translation
# ---------------------------------------------------------------------------


class TestGeminiRequestShape:
    def setup_method(self):
        self.provider = GeminiProvider()

    def test_system_prompt_moves_out_of_the_message_list(self):
        body = self.provider.build_body(
            "gemini-2.5-flash",
            [
                {"role": "system", "content": "You are a tool-using assistant."},
                {"role": "user", "content": "hello"},
            ],
            {},
        )
        assert body["systemInstruction"]["parts"][0]["text"].startswith("You are")
        assert [c["role"] for c in body["contents"]] == ["user"]

    def test_assistant_role_is_renamed_to_model(self):
        body = self.provider.build_body(
            "m",
            [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
            ],
            {},
        )
        assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]

    def test_consecutive_same_role_turns_are_merged(self):
        """The ReAct loop emits two user turns whenever it repairs a parse."""
        body = self.provider.build_body(
            "m",
            [
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "bad output"},
                {"role": "user", "content": "Observation: ..."},
                {"role": "user", "content": "Your last reply could not be parsed"},
            ],
            {},
        )
        roles = [c["role"] for c in body["contents"]]
        assert roles == ["user", "model", "user"]
        assert len(body["contents"][-1]["parts"]) == 2

    def test_multiple_system_messages_are_concatenated(self):
        body = self.provider.build_body(
            "m",
            [
                {"role": "system", "content": "one"},
                {"role": "system", "content": "two"},
                {"role": "user", "content": "hi"},
            ],
            {},
        )
        assert body["systemInstruction"]["parts"][0]["text"] == "one\n\ntwo"

    def test_parameters_are_renamed_to_generation_config(self):
        body = self.provider.build_body(
            "m", [{"role": "user", "content": "hi"}],
            {"temperature": 0.0, "max_tokens": 512, "top_p": 0.9},
        )
        config = body["generationConfig"]
        assert config["maxOutputTokens"] == 512
        assert config["temperature"] == 0.0
        assert config["topP"] == 0.9
        assert "max_tokens" not in config

    def test_unknown_parameters_are_dropped_not_forwarded(self):
        """OpenRouter-only params would be rejected by Gemini."""
        body = self.provider.build_body(
            "m", [{"role": "user", "content": "hi"}],
            {"temperature": 0.0, "reasoning_effort": "low", "response_format": {}},
        )
        assert "reasoning_effort" not in body.get("generationConfig", {})
        assert "response_format" not in body.get("generationConfig", {})

    def test_no_system_message_means_no_system_instruction_field(self):
        body = self.provider.build_body("m", [{"role": "user", "content": "hi"}], {})
        assert "systemInstruction" not in body

    def test_api_key_goes_in_a_header_not_a_url(self):
        """A key in a query string leaks into logs and error messages."""
        assert self.provider.headers("secret")["x-goog-api-key"] == "secret"
        assert "secret" not in self.provider.endpoint("gemini-2.5-flash")


class TestGeminiResponseParsing:
    def setup_method(self):
        self.provider = GeminiProvider()

    def test_extracts_text(self):
        assert self.provider.extract_content(gemini_reply("Final: 4")) == "Final: 4"

    def test_joins_multiple_parts(self):
        payload = {
            "candidates": [
                {"content": {"parts": [{"text": "Thought: a\n"}, {"text": "Final: b"}]}}
            ]
        }
        assert "Thought: a" in self.provider.extract_content(payload)
        assert "Final: b" in self.provider.extract_content(payload)

    def test_empty_output_names_the_finish_reason(self):
        """A thinking model can spend its whole budget before emitting text."""
        payload = {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]}
        with pytest.raises(ValueError, match="MAX_TOKENS"):
            self.provider.extract_content(payload)

    def test_no_candidates_is_an_error(self):
        with pytest.raises(ValueError, match="no candidates"):
            self.provider.extract_content({"candidates": []})

    def test_reports_the_served_model_version(self):
        assert self.provider.model_reported(gemini_reply("x"), "fallback") == (
            "gemini-2.5-flash"
        )


class TestOpenRouterUnchanged:
    def test_still_parses_the_openai_shape(self):
        provider = OpenRouterProvider()
        assert provider.extract_content(openrouter_reply("hi")) == "hi"

    def test_reasoning_only_response_still_recovered(self):
        payload = {"choices": [{"message": {"content": "", "reasoning": "Final: 42"}}]}
        assert OpenRouterProvider().extract_content(payload) == "Final: 42"

    def test_registry_exposes_every_backend(self):
        assert set(build_providers()) == {"openrouter", "gemini", "groq"}

    def test_groq_shares_the_openai_wire_format(self):
        """Groq is OpenAI-compatible, so only the endpoint should differ."""
        groq, openrouter = GroqProvider(), OpenRouterProvider()
        messages = [{"role": "user", "content": "hi"}]
        assert groq.build_body("m", messages, {"temperature": 0}) == (
            openrouter.build_body("m", messages, {"temperature": 0})
        )
        assert groq.base_url != openrouter.base_url
        assert "groq.com" in groq.base_url

    def test_groq_does_not_send_openrouter_ranking_headers(self):
        headers = GroqProvider().headers("k")
        assert headers["Authorization"] == "Bearer k"
        assert "HTTP-Referer" not in headers
        assert "X-Title" not in headers

    def test_reasoning_only_response_warns(self, caplog):
        """gpt-oss returns empty content with the budget spent on reasoning.

        Falling back silently would feed chain-of-thought to the ReAct parser.
        """
        payload = {"choices": [{"message": {"content": "", "reasoning": "thinking..."}}]}
        with caplog.at_level("WARNING"):
            assert GroqProvider().extract_content(payload) == "thinking..."
        assert any("reasoning" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Per-provider budgets
# ---------------------------------------------------------------------------


class TestPerProviderBudget:
    def test_providers_have_independent_counters(self, tmp_path):
        path = tmp_path / "b.json"
        openrouter = BudgetTracker(path, 50, provider="openrouter")
        gemini = BudgetTracker(path, 500, provider="gemini")

        openrouter.charge()
        openrouter.charge()
        gemini.charge()

        assert openrouter.used_today == 2
        assert gemini.used_today == 1
        assert gemini.remaining == 499

    def test_old_single_provider_file_is_migrated(self, tmp_path):
        """The budget file predates the second backend."""
        from datetime import date

        path = tmp_path / "b.json"
        path.write_text(
            json.dumps({"date": date.today().isoformat(), "count": 37}),
            encoding="utf-8",
        )
        assert BudgetTracker(path, 50, provider="openrouter").used_today == 37
        assert BudgetTracker(path, 500, provider="gemini").used_today == 0

    def test_exhausted_reports_correctly(self, tmp_path):
        tracker = BudgetTracker(tmp_path / "b.json", 1, provider="gemini")
        assert tracker.exhausted is False
        tracker.charge()
        assert tracker.exhausted is True


# ---------------------------------------------------------------------------
# Cross-provider fallback
# ---------------------------------------------------------------------------


class TestCrossProviderFallback:
    def test_spent_openrouter_budget_continues_on_gemini(self, isolated, monkeypatch):
        """The whole reason a second backend exists (CLAUDE.md 5.4)."""
        monkeypatch.setattr("src.llm.client.time.sleep", lambda _s: None)
        seen: list[str] = []

        def handler(request):
            seen.append(request.url.host)
            return httpx.Response(200, json=gemini_reply("Final: from gemini"))

        client = LLMClient(
            model_chain=[("openrouter", "or-model"), ("gemini", "gemini-2.5-flash")],
            api_key="test-key",
        )
        client._client = httpx.Client(transport=httpx.MockTransport(handler))
        # Spend OpenRouter's budget before the call.
        client.budgets["openrouter"].daily_cap = 1
        client.budgets["openrouter"].charge()

        response = client.chat([{"role": "user", "content": "hi"}])

        assert response.provider == "gemini"
        assert response.content == "Final: from gemini"
        assert all("googleapis" in host for host in seen), seen

    def test_openrouter_budget_is_not_charged_when_skipped(self, isolated):
        def handler(request):
            return httpx.Response(200, json=gemini_reply("ok"))

        client = LLMClient(
            model_chain=[("openrouter", "or"), ("gemini", "g")], api_key="k"
        )
        client._client = httpx.Client(transport=httpx.MockTransport(handler))
        client.budgets["openrouter"].daily_cap = 1
        client.budgets["openrouter"].charge()

        client.chat([{"role": "user", "content": "hi"}])
        assert client.budgets["openrouter"].used_today == 1
        # Gemini's quota is per model, so its counter is scoped that way.
        assert client.budget_for("gemini", "g").used_today == 1

    def test_a_provider_without_a_key_is_skipped_not_fatal(self, isolated, monkeypatch):
        monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")
        monkeypatch.setattr(settings, "GEMINI_API_KEY", "gem-key")

        def handler(request):
            return httpx.Response(200, json=gemini_reply("ok"))

        client = LLMClient(model_chain=[("openrouter", "or"), ("gemini", "g")])
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        assert client.chat([{"role": "user", "content": "hi"}]).provider == "gemini"

    def test_all_backends_out_of_budget_raises_budget_error(self, isolated):
        client = LLMClient(
            model_chain=[("openrouter", "or"), ("gemini", "g")], api_key="k"
        )
        client._client = httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))
        )
        for tracker in client.budgets.values():
            tracker.daily_cap = 1
            tracker.charge()

        with pytest.raises(BudgetExceededError):
            client.chat([{"role": "user", "content": "hi"}])

    def test_gemini_auth_error_names_the_right_key(self, isolated):
        client = LLMClient(model_chain=[("gemini", "g")], api_key="bad")
        client._client = httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(403, text="permission denied")
            )
        )
        with pytest.raises(AllModelsFailedError, match="GEMINI_API_KEY"):
            client.chat([{"role": "user", "content": "hi"}])

    def test_response_records_which_backend_served_it(self, isolated):
        """9.1 needs to know a comparison stayed on one backbone."""
        client = LLMClient(model_chain=[("gemini", "gemini-2.5-flash")], api_key="k")
        client._client = httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json=gemini_reply("hi"))
            )
        )
        response = client.chat([{"role": "user", "content": "x"}])
        assert response.provider == "gemini"
        assert response.model_used == "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Cache compatibility
# ---------------------------------------------------------------------------


class TestCacheCompatibility:
    def test_cache_key_ignores_the_provider(self):
        """Entries cached before a second backend existed must stay valid."""
        messages = [{"role": "user", "content": "hi"}]
        params = {"temperature": 0.0, "max_tokens": 1024}
        assert DiskCache.make_key(
            "nex-agi/nex-n2.5-pro:free", messages, params
        ) == DiskCache.make_key("nex-agi/nex-n2.5-pro:free", messages, params)

    def test_different_backends_do_not_collide(self):
        messages = [{"role": "user", "content": "hi"}]
        params = {"temperature": 0.0}
        assert DiskCache.make_key("gemini-2.5-flash", messages, params) != (
            DiskCache.make_key("nex-agi/nex-n2.5-pro:free", messages, params)
        )

    def test_a_cached_gemini_response_is_replayed_without_a_call(self, isolated):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(200, json=gemini_reply("cached me"))

        client = LLMClient(model_chain=[("gemini", "gemini-2.5-flash")], api_key="k")
        client._client = httpx.Client(transport=httpx.MockTransport(handler))
        messages = [{"role": "user", "content": "hi"}]

        first = client.chat(messages)
        second = client.chat(messages)

        assert calls["n"] == 1
        assert second.from_cache is True
        assert second.content == first.content
        assert second.provider == "gemini"


class TestQuotaSemantics:
    """A per-day 429 is not a per-minute 429 (measured against the real API)."""

    def test_daily_quota_429_is_not_retried(self, isolated, monkeypatch):
        """Backing off five times against a daily quota wastes the schedule."""
        slept: list[float] = []
        monkeypatch.setattr("src.llm.client.time.sleep", slept.append)
        attempts = {"n": 0}

        def handler(request):
            attempts["n"] += 1
            return httpx.Response(429, json={
                "error": {
                    "code": 429,
                    "message": "You exceeded your current quota",
                    "details": [{
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [{
                            "quotaId": (
                                "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
                            ),
                            "quotaValue": "20",
                        }],
                    }],
                }
            })

        client = LLMClient(model_chain=[("gemini", "gemini-2.5-flash")], api_key="k")
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        with pytest.raises(BudgetExceededError):
            client.chat([{"role": "user", "content": "hi"}])

        assert attempts["n"] == 1, "a per-day quota must not be retried"
        assert not slept, "no backoff for a limit that resets tomorrow"

    def test_per_minute_429_is_still_retried(self, isolated, monkeypatch):
        monkeypatch.setattr("src.llm.client.time.sleep", lambda _s: None)
        attempts = {"n": 0}

        def handler(request):
            attempts["n"] += 1
            if attempts["n"] < 3:
                return httpx.Response(429, text="slow down")
            return httpx.Response(200, json=gemini_reply("recovered"))

        client = LLMClient(model_chain=[("gemini", "g")], api_key="k")
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        assert client.chat([{"role": "user", "content": "x"}]).content == "recovered"
        assert attempts["n"] == 3

    def test_gemini_models_have_independent_counters(self, isolated):
        """The quota is per model, so one model's spend is not the other's."""
        client = LLMClient(
            model_chain=[("gemini", "gemini-2.5-flash"),
                         ("gemini", "gemini-2.5-flash-lite")],
            api_key="k",
        )
        client.budget_for("gemini", "gemini-2.5-flash").charge()
        assert client.budget_for("gemini", "gemini-2.5-flash").used_today == 1
        assert client.budget_for("gemini", "gemini-2.5-flash-lite").used_today == 0

    def test_openrouter_models_share_one_account_counter(self, isolated):
        """OpenRouter's 50/day covers the account, not each model."""
        client = LLMClient(
            model_chain=[("openrouter", "a"), ("openrouter", "b")], api_key="k"
        )
        client.budget_for("openrouter", "a").charge()
        assert client.budget_for("openrouter", "b").used_today == 1
