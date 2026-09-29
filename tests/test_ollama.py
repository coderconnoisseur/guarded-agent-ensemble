"""The local backbone arm: Ollama provider and the capability probe (§11).

Two things are worth guarding here, and neither is "does it call the API".

**The hosted arm must not move.** Every number in `results/` was measured
against the pinned Groq backbone, and §9.1 needs one pinned model per run.
Adding a provider that could quietly insert itself into the chain would
invalidate the whole ablation without failing anything, so the default
chain is asserted explicitly.

**The probe must gate on the right thing.** Its job is to decide whether a
model can speak the protocol at all - `liquid/lfm-2.5-2.6b` was retired for
failing exactly this - so a probe that passes a model whose planner degraded
is worse than no probe.
"""

from __future__ import annotations

import httpx
import pytest

from config import settings
from src.llm.providers import (
    GroqProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    build_providers,
)


class TestProviderWiring:
    def test_registered_under_its_own_name(self):
        assert isinstance(build_providers()["ollama"], OllamaProvider)

    def test_speaks_the_openai_wire_format(self):
        """Ollama's /v1 shim is OpenAI-compatible, so only the base URL and
        auth should differ. A second response-handling path here would be a
        second place for the backends to drift."""
        assert isinstance(OllamaProvider(), OpenAICompatibleProvider)
        assert OllamaProvider().endpoint("qwen3:4b") == GroqProvider().endpoint("q")

    def test_base_url_keeps_the_v1_suffix(self):
        """The bare root serves Ollama's *native* API, which has a different
        response shape - it would fail in extract_content rather than at
        connect time, which is a much more confusing failure."""
        assert OllamaProvider().base_url.endswith("/v1")

    def test_base_url_is_overridable_per_instance(self):
        assert OllamaProvider("http://box:9999/v1").base_url == "http://box:9999/v1"
        assert OllamaProvider().base_url == settings.OLLAMA_BASE_URL

    def test_sends_an_auth_header_even_though_ollama_ignores_it(self):
        """Same request path as every hosted backend, deliberately."""
        assert OllamaProvider().headers("")["Authorization"] == "Bearer ollama"

    def test_extracts_content_exactly_as_the_hosted_backends_do(self):
        payload = {"choices": [{"message": {"content": "Final: 42"}}]}
        assert OllamaProvider().extract_content(payload) == "Final: 42"


class TestReplyBudget:
    """Thinking cannot be switched off through Ollama's /v1 shim - measured,
    with the table in OllamaProvider's docstring - so the reply budget has to
    cover the thinking AND the answer that follows it.

    An earlier version of this provider sent `think: false` and
    `chat_template_kwargs.enable_thinking: false`. Both were measured to be
    exact no-ops and were removed: a setting that silently does nothing is the
    failure mode this project keeps finding.
    """

    def _body(self, params=None):
        return OllamaProvider().build_body(
            "qwen3:4b", [{"role": "user", "content": "hi"}], params or {}
        )

    def test_raises_the_hosted_default_to_the_local_floor(self):
        """400 is a Groq OTPM constraint. Locally it truncates qwen3:4b
        mid-thought: finish_reason 'length', content empty, unparseable."""
        body = self._body({"max_tokens": settings.DEFAULT_MAX_TOKENS})
        assert body["max_tokens"] == settings.OLLAMA_MIN_MAX_TOKENS
        assert settings.OLLAMA_MIN_MAX_TOKENS > settings.DEFAULT_MAX_TOKENS

    def test_covers_the_measured_requirement(self):
        """1200 was measured sufficient on the real ReAct prompt; 2500 gave a
        byte-identical reply, so the floor only needs to clear 1200."""
        assert settings.OLLAMA_MIN_MAX_TOKENS >= 1200

    def test_applies_when_the_caller_sets_nothing(self):
        assert self._body()["max_tokens"] == settings.OLLAMA_MIN_MAX_TOKENS

    def test_never_lowers_a_larger_caller_budget(self):
        big = settings.OLLAMA_MIN_MAX_TOKENS * 3
        assert self._body({"max_tokens": big})["max_tokens"] == big

    def test_the_ordinary_fields_are_untouched(self):
        body = self._body({"temperature": 0})
        assert body["model"] == "qwen3:4b"
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert body["temperature"] == 0

    def test_the_removed_no_ops_are_not_sent(self):
        """Guards the deletion. Sending these back would look like a fix and
        measurably do nothing."""
        body = self._body()
        assert "think" not in body
        assert "chat_template_kwargs" not in body

    def test_the_hosted_providers_are_unaffected(self):
        """Only the local arm gets the bigger budget - raising it on Groq
        would exceed that provider's entire per-minute output allowance."""
        hosted = GroqProvider().build_body(
            "q", [], {"max_tokens": settings.DEFAULT_MAX_TOKENS}
        )
        assert hosted["max_tokens"] == settings.DEFAULT_MAX_TOKENS


class TestTheHostedArmIsUnaffected:
    """The pinned backbone is what every saved result was measured against."""

    def test_ollama_is_absent_from_the_default_chain(self):
        assert settings.USE_LOCAL_BACKBONE is False
        assert not [p for p, _ in settings.PROVIDER_CHAIN if p == "ollama"]

    def test_the_pinned_backbone_still_leads(self):
        assert settings.PROVIDER_CHAIN[0] == ("groq", settings.BACKBONE_MODEL)

    def test_local_candidates_respect_the_measured_size_floor(self):
        """`liquid/lfm-2.5-2.6b` was dropped for being unable to emit a TDG.
        Anything advertised as sub-3B here would reintroduce that failure as a
        *silent* one: the planner degrades rather than raising, so Condition B
        would look like it ran."""
        banned = ("0.5b", "1b", "1.5b", "1.7b", "2b", "2.6b")
        for model in settings.OLLAMA_MODEL_CHAIN:
            assert not any(model.lower().endswith(s) for s in banned), model

    def test_first_candidate_matches_the_pinned_model_family(self):
        """Holding family and generation constant and varying only scale is
        what makes a two-backbone grid attributable to capability rather than
        to a different training recipe."""
        family = settings.BACKBONE_MODEL.split("/")[-1][:5]  # "qwen3"
        assert settings.OLLAMA_MODEL_CHAIN[0].startswith(family)


class TestLocalTimeout:
    def test_local_gets_longer_than_hosted(self):
        """A 4B on a 4GB card runs at tens of tokens/second, and a first call
        also pays model load time. The 120s hosted default can expire on a
        perfectly healthy local server and read as the model being broken."""
        from src.llm.client import LLMClient

        assert LLMClient._timeout_for("ollama") > LLMClient._timeout_for("groq")
        assert LLMClient._timeout_for("groq") == float(settings.REQUEST_TIMEOUT_S)


class TestProbeReachability:
    """The cheapest gate, and the one with failure modes that are not about
    the model at all: 'start Ollama' and 'pull the model' must not be
    reported as 'this model cannot be the backbone'."""

    @staticmethod
    def _reply(models):
        # raise_for_status() needs a request attached, or it raises a
        # RuntimeError that has nothing to do with what is under test.
        return httpx.Response(
            200, json={"models": models},
            request=httpx.Request("GET", "http://localhost:11434/api/tags"),
        )

    def _patch(self, monkeypatch, handler):
        monkeypatch.setattr(httpx, "get", handler)
        from demos import local_probe

        return local_probe.check_reachable

    def test_no_server_is_reported_as_no_server(self, monkeypatch):
        def boom(*a, **k):
            raise httpx.ConnectError("refused")

        ok, detail = self._patch(monkeypatch, boom)("qwen3:4b")
        assert ok is False
        assert "no Ollama server" in detail

    def test_missing_model_names_the_pull_command(self, monkeypatch):
        reply = self._reply([])
        ok, detail = self._patch(monkeypatch, lambda *a, **k: reply)("qwen3:4b")
        assert ok is False
        assert "ollama pull qwen3:4b" in detail

    def test_present_model_passes(self, monkeypatch):
        reply = self._reply([{"name": "qwen3:4b"}])
        ok, detail = self._patch(monkeypatch, lambda *a, **k: reply)("qwen3:4b")
        assert ok is True

    def test_same_family_different_tag_is_reported_not_rejected(self, monkeypatch):
        """Ollama often resolves a bare family name to a pulled tag, so this
        is a warning rather than a hard stop."""
        reply = self._reply([{"name": "qwen3:4b-q4_K_M"}])
        ok, detail = self._patch(monkeypatch, lambda *a, **k: reply)("qwen3:4b")
        assert ok is True
        assert "not pulled exactly" in detail

    def test_probes_the_native_root_not_the_v1_shim(self, monkeypatch):
        """/api/tags lives on the server root; /v1 only serves the OpenAI
        shim, which has no model-listing endpoint."""
        seen: list[str] = []

        def record(url, *a, **k):
            seen.append(url)
            return self._reply([{"name": "qwen3:4b"}])

        self._patch(monkeypatch, record)("qwen3:4b")
        assert seen and seen[0].endswith("/api/tags")
        assert "/v1/" not in seen[0]


class TestProbeGatesOnPlanning:
    """The check that retired a 2.6B model. `build_plan` never raises - it
    degrades, so Condition B stays usable - which means `degraded` is the
    signal and an exception is not."""

    class _Graph:
        def __init__(self, nodes, degraded):
            self.nodes = nodes
            self.degraded = degraded

    class _Node:
        def __init__(self, tool):
            self.tool = tool
            self.depends_on = []

    def _run(self, monkeypatch, graph):
        from demos import local_probe

        monkeypatch.setattr(
            local_probe, "Planner",
            lambda *a, **k: type("P", (), {"build_plan": lambda s, t, r: graph})(),
        )
        return local_probe.check_tdg(object(), "m")

    def test_a_degraded_plan_fails_the_gate(self, monkeypatch):
        ok, detail, _ = self._run(
            monkeypatch, self._Graph([self._Node("files.read")], degraded=True)
        )
        assert ok is False
        assert "DEGRADED" in detail

    def test_an_empty_plan_fails_the_gate(self, monkeypatch):
        ok, detail, _ = self._run(monkeypatch, self._Graph([], degraded=False))
        assert ok is False
        assert "empty" in detail

    def test_a_real_plan_passes(self, monkeypatch):
        ok, detail, rendered = self._run(
            monkeypatch,
            self._Graph([self._Node("files.read"), self._Node("comms.send_email")],
                        degraded=False),
        )
        assert ok is True
        assert "2-node" in detail
        assert "files.read" in rendered


class TestProbeUsesTheRealParser:
    """A probe that accepts a reply `parse_agent_reply` would reject is worse
    than no probe - it would green-light a model the agent loop cannot drive."""

    def _check(self, monkeypatch, content):
        from demos import local_probe

        class FakeClient:
            def chat(self, messages, **kw):
                return type("R", (), {"content": content})()

        return local_probe.check_react_protocol(FakeClient(), "m")

    def test_prose_fails(self, monkeypatch):
        ok, detail, _ = self._check(monkeypatch, "Sure! I would read the file.")
        assert ok is False
        assert "did not parse" in detail

    def test_a_well_formed_action_passes(self, monkeypatch):
        ok, detail, _ = self._check(
            monkeypatch,
            'Thought: read it\nAction: {"tool": "files.read", '
            '"args": {"path": "welcome.txt"}}',
        )
        assert ok is True
        assert "files.read" in detail

    def test_a_final_answer_passes(self, monkeypatch):
        ok, detail, _ = self._check(monkeypatch, "Final: done")
        assert ok is True
        assert "Final" in detail
