"""Tests for Vertex AI Context Caching of full 10-K sections (`agents/tenk_context_cache.py`)."""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from edgar_capstone.agents import (
    calculate_period_variances,
    get_context_cache_runtime_stats,
    get_context_latency_tracker,
    get_conversation_memory_manager,
)
from edgar_capstone.agents.chat_engine import run_conversational_rag_turn
from edgar_capstone.agents.tenk_context_cache import (
    TenKContextCacheManager,
    apply_tenk_cache_decision,
    build_tenk_section_bundle,
    configure_tenk_context_cache_manager,
    is_cached_content_error,
)
from edgar_capstone.agents.tools import _BoundedSectionCache

NVDA_SECTION_KEYS = ["NVDA_2026_item_7", "NVDA_2026_item_1a"]


# ---------------------------------------------------------------------------
# Fake google.genai async caches API (shared "remote" store across managers)
# ---------------------------------------------------------------------------


class _FakePager:
    def __init__(self, items: list[genai_types.CachedContent]) -> None:
        self._items = items

    def __aiter__(self) -> AsyncGenerator[genai_types.CachedContent, None]:
        async def _gen() -> AsyncGenerator[genai_types.CachedContent, None]:
            for item in self._items:
                yield item

        return _gen()


class _FakeAsyncCaches:
    def __init__(self) -> None:
        self.store: dict[str, genai_types.CachedContent] = {}
        self.create_calls: list[dict[str, Any]] = []
        self.update_calls: list[dict[str, Any]] = []
        self.list_calls = 0
        self.fail_create: Exception | None = None

    @staticmethod
    def _ttl_seconds(ttl: str | None) -> int:
        return int(str(ttl or "1800s").rstrip("s"))

    async def create(
        self, *, model: str, config: genai_types.CreateCachedContentConfig
    ) -> genai_types.CachedContent:
        if self.fail_create is not None:
            raise self.fail_create
        self.create_calls.append({"model": model, "config": config})
        name = f"projects/p/locations/us-central1/cachedContents/{len(self.store) + 1}"
        cached = genai_types.CachedContent(
            name=name,
            display_name=config.display_name,
            model=f"projects/p/locations/us-central1/publishers/google/models/{model}",
            expire_time=dt.datetime.now(dt.UTC)
            + dt.timedelta(seconds=self._ttl_seconds(config.ttl)),
        )
        self.store[name] = cached
        return cached

    async def list(self, *, config: Any = None) -> _FakePager:
        self.list_calls += 1
        return _FakePager(list(self.store.values()))

    async def update(
        self, *, name: str, config: genai_types.UpdateCachedContentConfig
    ) -> genai_types.CachedContent:
        self.update_calls.append({"name": name, "ttl": config.ttl})
        cached = self.store[name]
        cached.expire_time = dt.datetime.now(dt.UTC) + dt.timedelta(
            seconds=self._ttl_seconds(config.ttl)
        )
        return cached


def _fake_client() -> SimpleNamespace:
    return SimpleNamespace(aio=SimpleNamespace(caches=_FakeAsyncCaches()))


def _acquire(manager: TenKContextCacheManager, client: Any, bundle: Any, **overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "client": client,
        "model": "gemini-2.5-pro",
        "system_instruction": "You are FinancialAnalystAgent.",
        "tools": [
            genai_types.Tool(
                function_declarations=[genai_types.FunctionDeclaration(name="calc_variances")]
            )
        ],
        "tool_config": None,
        "bundle": bundle,
    }
    kwargs.update(overrides)
    return asyncio.run(manager.acquire(**kwargs))


@pytest.fixture(autouse=True)
def _reset_manager() -> Any:
    configure_tenk_context_cache_manager(None)
    yield
    configure_tenk_context_cache_manager(None)


# ---------------------------------------------------------------------------
# Process-local bounded section cache
# ---------------------------------------------------------------------------


def test_bounded_section_cache_evicts_lru_and_expires_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = _BoundedSectionCache(max_entries=2, ttl_seconds=10.0)
    cache["A_2024_item_7"] = {"full_text": "a"}
    cache["B_2024_item_7"] = {"full_text": "b"}
    assert cache["A_2024_item_7"]["full_text"] == "a"  # touch A -> B becomes LRU
    cache["C_2024_item_7"] = {"full_text": "c"}
    assert "B_2024_item_7" not in cache
    assert set(cache.keys()) == {"A_2024_item_7", "C_2024_item_7"}
    assert cache.evictions == 1

    import edgar_capstone.agents.tools as tools_mod

    real_monotonic = tools_mod.time.monotonic
    monkeypatch.setattr(tools_mod.time, "monotonic", lambda: real_monotonic() + 60.0)
    assert "A_2024_item_7" not in cache
    assert cache.get("C_2024_item_7") is None
    assert len(cache) == 0


# ---------------------------------------------------------------------------
# Section bundles
# ---------------------------------------------------------------------------


def test_section_bundle_is_deterministic_and_respects_token_budget() -> None:
    bundle_a = build_tenk_section_bundle(NVDA_SECTION_KEYS)
    bundle_b = build_tenk_section_bundle(list(reversed(NVDA_SECTION_KEYS)))
    assert bundle_a is not None and bundle_b is not None
    # Same set of sections -> identical sorted bundle & fingerprint regardless of request order
    assert bundle_a.section_keys == ("NVDA_2026_item_1a", "NVDA_2026_item_7")
    assert bundle_a.content_sha256 == bundle_b.content_sha256
    assert '<10k_section cache_key="NVDA_2026_item_7"' in bundle_a.text
    assert "Item 1A. Risk Factors" in bundle_a.text
    assert bundle_a.to_content().role == "user"

    # Budget admits sections in priority order and reports the rest as dropped
    first_only = build_tenk_section_bundle(
        NVDA_SECTION_KEYS + ["NOPE_2026_item_7"],
        max_tokens=bundle_a.token_estimate - 1,
    )
    assert first_only is not None
    assert len(first_only.section_keys) == 1
    assert "NOPE_2026_item_7" in first_only.dropped_section_keys
    assert build_tenk_section_bundle(["NOPE_2026_item_7"]) is None


# ---------------------------------------------------------------------------
# Manager: create -> L1 hit -> cross-instance discovery -> refresh / failures
# ---------------------------------------------------------------------------


def test_manager_creates_cache_then_reuses_it_locally_and_across_instances() -> None:
    client = _fake_client()
    bundle = build_tenk_section_bundle(NVDA_SECTION_KEYS)
    instance_a = TenKContextCacheManager(min_tokens_override=1)

    created = _acquire(instance_a, client, bundle)
    assert created.mode == "cached"
    assert created.source == "created"
    assert created.cache_name.startswith("projects/p/locations/us-central1/cachedContents/")
    assert len(client.aio.caches.create_calls) == 1
    cfg = client.aio.caches.create_calls[0]["config"]
    assert client.aio.caches.create_calls[0]["model"] == "gemini-2.5-pro"
    assert cfg.display_name == f"edgar10k-{created.cache_key[:48]}"
    assert cfg.ttl == "1800s"
    assert cfg.system_instruction == "You are FinancialAnalystAgent."
    assert cfg.tools[0].function_declarations[0].name == "calc_variances"
    assert cfg.contents[0].parts[0].text == bundle.text

    # Same process: L1 hit, no remote calls
    again = _acquire(instance_a, client, bundle)
    assert (again.mode, again.source, again.cache_name) == ("cached", "l1", created.cache_name)
    assert len(client.aio.caches.create_calls) == 1

    # Another Cloud Run instance (empty L1) discovers the same CachedContent by display name
    instance_b = TenKContextCacheManager(min_tokens_override=1)
    remote = _acquire(instance_b, client, bundle)
    assert (remote.mode, remote.source, remote.cache_name) == (
        "cached",
        "remote",
        created.cache_name,
    )
    assert len(client.aio.caches.create_calls) == 1
    assert client.aio.caches.list_calls >= 1

    # A different system instruction (prompt version) must not reuse the cache
    other = _acquire(instance_b, client, bundle, system_instruction="v2 instructions")
    assert other.source == "created"
    assert other.cache_name != created.cache_name

    stats_a = instance_a.stats()
    assert stats_a["creates"] == 1 and stats_a["l1_hits"] == 1
    assert stats_a["reuse_rate"] == 0.5
    assert stats_a["cached_tokens_registered"] == bundle.token_estimate
    assert instance_b.stats()["remote_hits"] == 1


def test_manager_refreshes_ttl_and_ignores_expired_remote_caches() -> None:
    client = _fake_client()
    bundle = build_tenk_section_bundle(NVDA_SECTION_KEYS)
    manager = TenKContextCacheManager(min_tokens_override=1, ttl_seconds=600)
    first = _acquire(manager, client, bundle)

    # Less than half the TTL left -> caches.update extends it on the next hit
    stored = client.aio.caches.store[first.cache_name]
    stored.expire_time = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=200)
    manager._handles[first.cache_key].expire_time = stored.expire_time
    hit = _acquire(manager, client, bundle)
    assert hit.source == "l1"
    assert client.aio.caches.update_calls == [{"name": first.cache_name, "ttl": "600s"}]
    assert manager.stats()["refreshes"] == 1

    # Within the safety margin of expiry -> never used; a fresh cache is created
    stored.expire_time = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=30)
    manager._handles[first.cache_key].expire_time = stored.expire_time
    recreated = _acquire(manager, client, bundle)
    assert recreated.source == "created"
    assert recreated.cache_name != first.cache_name


def test_manager_fallbacks_inline_skip_and_failures() -> None:
    bundle = build_tenk_section_bundle(NVDA_SECTION_KEYS)

    # Below the model's minimum cacheable size -> inline (no client needed)
    small = _acquire(TenKContextCacheManager(min_tokens_override=10_000_000), None, bundle)
    assert (small.mode, small.reason) == ("inline", "below_minimum_cacheable_tokens")

    # No genai client (e.g. custom BaseLlm) -> passages-only grounding
    no_client = _acquire(TenKContextCacheManager(min_tokens_override=1), None, bundle)
    assert (no_client.mode, no_client.reason) == ("skipped", "no_genai_client")

    # Disabled via env
    disabled = _acquire(TenKContextCacheManager(enabled=False), _fake_client(), bundle)
    assert (disabled.mode, disabled.reason) == ("skipped", "disabled")

    # API failure -> skipped with reason, never raises into the model call
    client = _fake_client()
    client.aio.caches.fail_create = RuntimeError("quota exhausted")
    mgr = TenKContextCacheManager(min_tokens_override=1)
    failed = _acquire(mgr, client, bundle)
    assert (failed.mode, failed.reason) == ("skipped", "create_failed:RuntimeError")
    assert mgr.stats()["create_failures"] == 1
    assert mgr.stats()["skip_reasons"] == {"create_failed": 1}

    # Server-side minimum token rejection -> inline
    client.aio.caches.fail_create = genai_errors.ClientError(
        400,
        {
            "error": {
                "message": "Cached content is too small. total_token_count=1000, min_total_token_count=2048"
            }
        },
    )
    too_small = _acquire(TenKContextCacheManager(min_tokens_override=1), client, bundle)
    assert (too_small.mode, too_small.reason) == ("inline", "below_minimum_cacheable_tokens_server")


def test_circuit_breaker_and_create_timeout_protect_ttft() -> None:
    bundle = build_tenk_section_bundle(NVDA_SECTION_KEYS)

    # A failing cache API (e.g. tenant-project provisioning errors) trips the breaker:
    # subsequent calls skip immediately without touching the API.
    client = _fake_client()
    client.aio.caches.fail_create = genai_errors.ServerError(
        500, {"error": {"message": "Failed to create the tenant project", "status": "INTERNAL"}}
    )
    mgr = TenKContextCacheManager(min_tokens_override=1, failure_cooldown_seconds=600)
    first = _acquire(mgr, client, bundle)
    assert (first.mode, first.reason) == ("skipped", "create_failed:ServerError")
    client.aio.caches.fail_create = None
    list_calls_before = client.aio.caches.list_calls
    second = _acquire(mgr, client, bundle)
    assert (second.mode, second.reason) == ("skipped", "circuit_open")
    assert client.aio.caches.create_calls == []
    assert client.aio.caches.list_calls == list_calls_before
    stats = mgr.stats()
    assert stats["circuit_open"] is True and stats["circuit_opens"] == 1
    assert "tenant project" in stats["last_failure"]

    # After the cooldown the manager tries again and recovers
    mgr._circuit_open_until = 0.0
    recovered = _acquire(mgr, client, bundle)
    assert (recovered.mode, recovered.source) == ("cached", "created")

    # A slow create is abandoned at the timeout instead of blocking the model call
    slow_client = _fake_client()
    real_create = slow_client.aio.caches.create

    async def _slow_create(**kwargs: Any) -> Any:
        await asyncio.sleep(5)
        return await real_create(**kwargs)

    slow_client.aio.caches.create = _slow_create
    slow_mgr = TenKContextCacheManager(min_tokens_override=1, create_timeout_seconds=0.05)
    timed_out = _acquire(slow_mgr, slow_client, bundle)
    assert (timed_out.mode, timed_out.reason) == ("skipped", "create_failed:TimeoutError")
    assert timed_out.latency_ms < 2000
    assert slow_mgr.stats()["timeouts"] == 1
    assert slow_mgr.circuit_open()


def test_apply_decision_moves_system_instruction_and_tools_into_cache() -> None:
    bundle = build_tenk_section_bundle(NVDA_SECTION_KEYS)
    client = _fake_client()
    decision = _acquire(TenKContextCacheManager(min_tokens_override=1), client, bundle)

    req = LlmRequest(
        model="gemini-2.5-pro",
        contents=[genai_types.Content(role="user", parts=[genai_types.Part(text="prompt")])],
        config=genai_types.GenerateContentConfig(
            system_instruction="sys",
            tools=[genai_types.Tool(function_declarations=[])],
            temperature=0.1,
        ),
    )
    apply_tenk_cache_decision(req, decision, bundle)
    assert req.config.cached_content == decision.cache_name
    assert req.config.system_instruction is None
    assert req.config.tools is None
    assert req.config.tool_config is None
    assert req.config.temperature == 0.1
    assert req.cache_config is None
    assert [c.parts[0].text for c in req.contents] == ["prompt"]

    inline_decision = _acquire(
        TenKContextCacheManager(min_tokens_override=10_000_000), None, bundle
    )
    req2 = LlmRequest(
        model="gemini-2.5-pro",
        contents=[genai_types.Content(role="user", parts=[genai_types.Part(text="prompt")])],
        config=genai_types.GenerateContentConfig(system_instruction="sys"),
    )
    apply_tenk_cache_decision(req2, inline_decision, bundle)
    assert req2.config.cached_content is None
    assert req2.config.system_instruction == "sys"
    assert req2.contents[0].parts[0].text == bundle.text
    assert req2.contents[1].parts[0].text == "prompt"


def test_is_cached_content_error_classification() -> None:
    assert is_cached_content_error(
        genai_errors.ClientError(
            404, {"error": {"message": "CachedContent projects/p/cachedContents/1 not found"}}
        )
    )
    assert is_cached_content_error(
        genai_errors.ClientError(400, {"error": {"message": "Cached content has expired."}})
    )
    assert not is_cached_content_error(
        genai_errors.ClientError(429, {"error": {"message": "Resource exhausted"}})
    )
    assert not is_cached_content_error(RuntimeError("boom"))


# ---------------------------------------------------------------------------
# End-to-end: ADK Runner + FinancialAnalystAgent callback on /api/v1/chat engine
# ---------------------------------------------------------------------------


class _RecordingAnalystLlm(BaseLlm):
    """Streams a grounded NVDA synthesis and records the outgoing request config."""

    model: str = "gemini-2.5-pro"
    requests: list[dict[str, Any]] = []
    fail_first_with: Exception | None = None

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        cfg = llm_request.config
        self.requests.append(
            {
                "cached_content": cfg.cached_content if cfg else None,
                "system_instruction": cfg.system_instruction if cfg else None,
                "tools": cfg.tools if cfg else None,
                "contents_count": len(llm_request.contents or []),
            }
        )
        if self.fail_first_with is not None:
            exc, self.fail_first_with = self.fail_first_with, None
            raise exc

        rep = calculate_period_variances(tickers="NVDA", start_year=2024, end_year=2026)[
            "variance_reports"
        ][0]
        final_md = (
            "### NVIDIA Corp. (NVDA) — Cached 10-K Grounded Synthesis\n\n"
            f"- Revenue expanded **+{rep['revenue_variance']['percentage_delta']:.2f}%** while operating "
            f"income increased **+{rep['operating_income_variance']['percentage_delta']:.2f}%** on Data "
            "Center demand [NVDA FY2026 | Item 7 MD&A | NVDA_2026_item_7_0000]"
            "(https://www.sec.gov/Archives/edgar/data/1045810)."
        )
        yield LlmResponse(
            model_version=self.model,
            partial=False,
            content=genai_types.Content(role="model", parts=[genai_types.Part(text=final_md)]),
            usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
                prompt_token_count=36000,
                cached_content_token_count=34000 if cfg and cfg.cached_content else 0,
                candidates_token_count=120,
            ),
        )


class _FlashJudgeLlm(BaseLlm):
    model: str = "gemini-3.5-flash"

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        yield LlmResponse(
            model_version=self.model,
            partial=False,
            content=genai_types.Content(
                role="model",
                parts=[
                    genai_types.Part(
                        text=(
                            '{"evaluator_model": "gemini-3.5-flash", "task_category": "evaluation", '
                            '"grade": "pass", "overall_score": 0.96, "numeric_fidelity_score": 1.0, '
                            '"citation_grounding_score": 0.95, "explanation_quality_score": 0.94, '
                            '"thematic_consistency_score": 0.94, "rationale": "Grounded."}'
                        )
                    )
                ],
            ),
        )


def _run_nvda_turn(session_id: str, llm: _RecordingAnalystLlm) -> dict[str, Any]:
    return run_conversational_rag_turn(
        query="Analyze NVIDIA FY2024 to FY2026 revenue and operating income drivers",
        tickers="NVDA",
        start_year=2024,
        end_year=2026,
        session_id=session_id,
        user_id="analyst@fde-workspace.corp",
        reasoning_llm=llm,
        judge_llm=_FlashJudgeLlm(),
    )


def test_chat_turn_attaches_shared_cached_content_across_instances() -> None:
    get_context_latency_tracker().reset()
    get_conversation_memory_manager().reset()
    client = _fake_client()
    configure_tenk_context_cache_manager(
        TenKContextCacheManager(min_tokens_override=1, client=client)
    )

    llm = _RecordingAnalystLlm(requests=[])
    turn = _run_nvda_turn("tenk-cache-sess-1", llm)
    assert turn["execution_mode"] == "live_adk"
    cache_meta = turn["adk_runtime"]["vertex_context_cache"]
    decision = cache_meta["last_decision"]
    assert decision["mode"] == "cached"
    assert decision["source"] == "created"
    assert set(decision["section_keys"]) >= {"NVDA_2026_item_7"}
    assert cache_meta["cached_calls"] == cache_meta["model_calls"] >= 1

    # The model call referenced the CachedContent and carried no system instruction / tools
    sent = llm.requests[-1]
    assert sent["cached_content"] == decision["cache_name"]
    assert sent["system_instruction"] is None
    assert sent["tools"] is None
    assert len(client.aio.caches.create_calls) == 1
    cached_cfg = client.aio.caches.create_calls[0]["config"]
    assert cached_cfg.system_instruction  # agent instruction moved into the cache
    assert cached_cfg.tools  # FunctionTool declarations moved into the cache
    assert "NVDA" in cached_cfg.contents[0].parts[0].text

    # Cached-token savings flow into context latency telemetry
    pro_records = [
        r
        for r in turn["context_and_memory"]["model_call_records"]
        if r["model_name"] == "gemini-2.5-pro"
    ]
    assert pro_records and pro_records[-1]["context_breakdown"]["cached_context_tokens"] == 34000

    # A second Cloud Run instance answering the same scope reuses the same remote cache
    configure_tenk_context_cache_manager(
        TenKContextCacheManager(min_tokens_override=1, client=client)
    )
    llm_b = _RecordingAnalystLlm(requests=[])
    turn_b = _run_nvda_turn("tenk-cache-sess-2", llm_b)
    decision_b = turn_b["adk_runtime"]["vertex_context_cache"]["last_decision"]
    assert (decision_b["mode"], decision_b["source"]) == ("cached", "remote")
    assert decision_b["cache_name"] == decision["cache_name"]
    assert len(client.aio.caches.create_calls) == 1

    stats = get_context_cache_runtime_stats()["vertex_context_cache"]
    assert stats["remote_hits"] >= 1
    assert stats["active_handles"][0]["cache_name"] == decision["cache_name"]


def test_chat_turn_retries_once_when_cached_content_expired_at_generate_time() -> None:
    get_context_latency_tracker().reset()
    get_conversation_memory_manager().reset()
    client = _fake_client()
    manager = TenKContextCacheManager(min_tokens_override=1, client=client)
    configure_tenk_context_cache_manager(manager)

    llm = _RecordingAnalystLlm(
        requests=[],
        fail_first_with=genai_errors.ClientError(
            404, {"error": {"message": "CachedContent not found (expired)."}}
        ),
    )
    turn = _run_nvda_turn("tenk-cache-retry-1", llm)
    assert turn["execution_mode"] == "live_adk"
    assert turn["adk_runtime"]["grounding_fallback_applied"] is False
    assert len(llm.requests) == 2
    assert all(r["cached_content"] for r in llm.requests)
    assert manager.stats()["invalidations"] >= 1
