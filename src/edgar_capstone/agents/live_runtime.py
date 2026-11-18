"""Live Google ADK `Runner` & Gemini (`gemini-3.8-flash` / `gemini-3.5-flash`) Streaming Execution Engine.

Bridges the active ADK specialist agents (`FinancialAnalystAgent` and `NarrativeEvaluatorAgent`)
to live Vertex AI / Gemini execution via `google.adk.runners.Runner` with Server-Sent Events
(`StreamingMode.SSE`) to capture real wall-clock Time-to-First-Thought / Time-to-First-Token
(`observed_ttft_ms`), running synchronous tools on a thread pool (`ToolThreadPoolConfig`) so
`_ADK_LOOP` is never blocked, and cleaning up ephemeral ADK sessions after completion.

Every answer comes from live Gemini. The backend is Vertex AI (Application Default Credentials;
the default), the Gemini API when `GEMINI_API_KEY` / `GOOGLE_API_KEY` is set and
`GOOGLE_GENAI_USE_VERTEXAI` is not, or an explicitly injected ADK `BaseLlm`. There is no offline or
template-synthesis mode: upstream failures surface as errors after retry backoff.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import contextlib
import contextvars
import json
import os
import threading
import time
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from google.adk.agents import BaseAgent
from google.adk.agents.run_config import RunConfig, StreamingMode, ToolThreadPoolConfig
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.models.base_llm import BaseLlm
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import Runner
from google.genai import types as genai_types

from edgar_capstone.agents.callbacks import (
    _DEFER_STRUCTURED_VERIFICATION_KEY,
    ModelTelemetryPlugin,
)
from edgar_capstone.agents.context_memory import (
    create_edgar_adk_app,
    get_context_latency_tracker,
    get_conversation_memory_manager,
    is_ephemeral_session_id,
    resolve_model_for_task,
)
from edgar_capstone.agents.factories import (
    create_financial_analyst_agent,
    create_narrative_evaluator_agent,
)
from edgar_capstone.agents.prompt_builders import (
    _build_financial_reasoning_prompt,
    _compact_evidence_leads,
    _compact_variance_reports_for_prompt,
    build_financial_reasoning_prompt,
    build_narrative_evaluation_prompt,
    select_citations_for_judge,
)
from edgar_capstone.agents.schemas import (
    ModelContextLatencyRecord,
    ModelTaskCategory,
    NarrativeEvaluationReport,
)
from edgar_capstone.agents.tool_runtime import ToolOutputs, turn_scoped_state_seed
from edgar_capstone.common.domain import sha256_text
from edgar_capstone.common.env import TRUTHY_ENV_VALUES
from edgar_capstone.common.formatting import parse_model_output
from edgar_capstone.gcp.defaults import resolve_gcp_project_id
from edgar_capstone.observability.tracing import set_span_attributes, traced
from edgar_capstone.security.model_armor import ModelArmorSafetyPlugin

LiveBackendType = Literal["vertex_ai", "gemini_api", "custom_llm"]

_PLACEHOLDER_SECRET_VALUES = frozenset({"", "unset", "none", "null", "placeholder", "changeme"})
_DEFAULT_TOOL_THREAD_POOL_WORKERS = 8

_DEFAULT_GEMINI_RETRY_OPTIONS = genai_types.HttpRetryOptions(
    attempts=3,
    initial_delay=1.0,
    max_delay=8.0,
    exp_base=2.0,
    http_status_codes=[429, 500, 502, 503, 504],
)

_T = TypeVar("_T")


@dataclass(frozen=True)
class LiveRuntimeResolution:
    """Resolved live ADK / Gemini execution configuration (with secrets redacted from repr)."""

    backend: LiveBackendType
    project_id: str
    location: str
    api_key_configured: bool = False
    _api_key: str | None = field(default=None, repr=False, compare=False)

    def gemini_api_key(self) -> str | None:
        """Return the configured Gemini API key when `backend == 'gemini_api'`, else `None`."""
        return self._api_key

    def to_public_dict(self) -> dict[str, Any]:
        """Return safe, secret-free runtime status metadata for `/health` and telemetry responses."""
        return {
            "backend": self.backend,
            "vertex_project_id": self.project_id,
            "vertex_location": self.location,
            "gemini_api_key_configured": self.api_key_configured,
            "streaming_mode": StreamingMode.SSE.value,
        }


@dataclass
class AdkStreamResult:
    """Structured result captured from an ADK `Runner.run_async(..., StreamingMode.SSE)` invocation."""

    agent_name: str
    app_name: str
    session_id: str
    backend: LiveBackendType
    final_text: str
    thoughts: list[str] = field(default_factory=list)
    function_calls: list[dict[str, Any]] = field(default_factory=list)
    function_responses: list[dict[str, Any]] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    partial_chunks_count: int = 0
    total_events_count: int = 0
    observed_ttft_ms: float | None = None
    total_stream_latency_ms: float = 0.0
    session_state: dict[str, Any] = field(default_factory=dict)
    latency_records: list[ModelContextLatencyRecord] = field(default_factory=list)


def _clean_api_key(raw: str | None) -> str | None:
    if raw is None:
        return None
    cleaned = raw.strip()
    if cleaned.lower() in _PLACEHOLDER_SECRET_VALUES:
        return None
    return cleaned


def resolve_live_adk_runtime(*, custom_llm: BaseLlm | None = None) -> LiveRuntimeResolution:
    """Resolve which live Gemini backend to target (Vertex AI unless an API key is configured)."""
    project_id = resolve_gcp_project_id()
    # Gemini models are served from the `global` endpoint in this project while BigQuery, GCS,
    # Model Armor, and embeddings stay regional, so the Gemini location is set separately.
    location = os.environ.get("EDGAR_GEMINI_LOCATION", "").strip() or "global"
    api_key = _clean_api_key(os.environ.get("GOOGLE_API_KEY")) or _clean_api_key(
        os.environ.get("GEMINI_API_KEY")
    )
    use_vertexai = (
        os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() in TRUTHY_ENV_VALUES
    )

    if custom_llm is not None:
        backend: LiveBackendType = "custom_llm"
    elif api_key is not None and not use_vertexai:
        backend = "gemini_api"
    else:
        backend = "vertex_ai"
    return LiveRuntimeResolution(
        backend=backend,
        project_id=project_id,
        location=location,
        api_key_configured=api_key is not None,
        _api_key=api_key if backend == "gemini_api" else None,
    )


def get_live_runtime_status_metadata() -> dict[str, Any]:
    """Return secret-redacted metadata describing active ADK `Runner` / Gemini execution configuration."""
    return resolve_live_adk_runtime().to_public_dict()


class _PersistentAdkLoopManager:
    """Manages a single persistent background asyncio loop for ADK Gemini calls.

    ADK's `Gemini.api_client` is cached per event loop (`@PerLoopCachedProperty`). Reusing
    one persistent loop preserves the underlying `genai.Client` HTTP/2 keepalive pool and
    OAuth2 credential cache across turns, cutting ~1.0-1.4s of cold TLS/ADC latency off
    every request's TTFT.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def get_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is not None and self._loop.is_running():
                return self._loop
            ready = threading.Event()
            loop = asyncio.new_event_loop()

            def _runner() -> None:
                asyncio.set_event_loop(loop)
                loop.call_soon(ready.set)
                loop.run_forever()

            thread = threading.Thread(
                target=_runner,
                name="edgar-adk-async-loop",
                daemon=True,
            )
            thread.start()
            ready.wait(timeout=5.0)
            self._loop = loop
            self._thread = thread
            return loop

    def is_loop_thread(self) -> bool:
        return self._thread is not None and threading.current_thread() is self._thread

    def stop(self) -> None:
        with self._lock:
            loop = self._loop
            thread = self._thread
            self._loop = None
            self._thread = None
        if loop is not None and loop.is_running():
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)


_ADK_LOOP_MANAGER = _PersistentAdkLoopManager()
atexit.register(_ADK_LOOP_MANAGER.stop)

_GEMINI_CACHE_LOCK = threading.Lock()
_CACHED_GEMINI_LLMS: dict[tuple[str, str, str, str, str], Gemini] = {}


def build_adk_llm_for_task(
    task_category: ModelTaskCategory,
    *,
    resolution: LiveRuntimeResolution | None = None,
    custom_llm: BaseLlm | None = None,
    model_override: str | None = None,
) -> BaseLlm:
    """Return a cached ADK `Gemini` (or custom `BaseLlm`) configured with Vertex AI / API key & retries."""
    if custom_llm is not None:
        return custom_llm

    resolved = resolution or resolve_live_adk_runtime()
    model_name = (model_override or "").strip() or resolve_model_for_task(task_category)

    if resolved.backend == "gemini_api":
        api_key = resolved.gemini_api_key()
        if not api_key:
            raise ValueError(
                "LiveRuntimeResolution selected 'gemini_api' backend without a configured API key."
            )
        key_hash = sha256_text(api_key)[:16]
        cache_key = ("gemini_api", "", "", model_name, key_hash)
        with _GEMINI_CACHE_LOCK:
            cached = _CACHED_GEMINI_LLMS.get(cache_key)
            if cached is None:
                cached = Gemini(
                    model=model_name,
                    client_kwargs={"api_key": api_key},
                    retry_options=_DEFAULT_GEMINI_RETRY_OPTIONS,
                )
                _CACHED_GEMINI_LLMS[cache_key] = cached
            return cached

    if resolved.backend == "custom_llm":
        raise ValueError("backend='custom_llm' requires an explicit `custom_llm` BaseLlm instance.")

    cache_key = ("vertex_ai", resolved.project_id, resolved.location, model_name, "")
    with _GEMINI_CACHE_LOCK:
        cached = _CACHED_GEMINI_LLMS.get(cache_key)
        if cached is None:
            cached = Gemini(
                model=model_name,
                client_kwargs={
                    "vertexai": True,
                    "project": resolved.project_id,
                    "location": resolved.location,
                },
                retry_options=_DEFAULT_GEMINI_RETRY_OPTIONS,
            )
            _CACHED_GEMINI_LLMS[cache_key] = cached
        return cached


def _run_coro_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Execute an async coroutine on the persistent ADK loop preserving `contextvars` (OTel spans)."""
    loop = _ADK_LOOP_MANAGER.get_loop()
    ctx = contextvars.copy_context()
    # Guard against re-entrant deadlock if a tool invoked on the ADK loop thread calls us synchronously.
    if _ADK_LOOP_MANAGER.is_loop_thread():
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="adk-live-reentrant"
        ) as pool:
            return pool.submit(ctx.run, asyncio.run, coro).result()

    fut: concurrent.futures.Future[_T] = concurrent.futures.Future()

    def _schedule() -> None:
        if fut.cancelled():
            coro.close()
            return
        try:
            task = loop.create_task(coro, context=ctx)
        except Exception as exc:
            coro.close()
            if not fut.done():
                fut.set_exception(exc)
            return

        def _on_done(done_task: asyncio.Task[_T]) -> None:
            if fut.done():
                return
            if done_task.cancelled():
                fut.cancel()
            else:
                exc = done_task.exception()
                if exc is not None:
                    fut.set_exception(exc)
                else:
                    fut.set_result(done_task.result())

        task.add_done_callback(_on_done)

    loop.call_soon_threadsafe(_schedule)
    return fut.result()


def _resolve_runner_plugins(plugins: list[BasePlugin] | None) -> list[BasePlugin]:
    """Ensure `ModelTelemetryPlugin` runs ahead of `ModelArmorSafetyPlugin` when guardrails are active."""
    if plugins is None:
        return [ModelTelemetryPlugin(), ModelArmorSafetyPlugin()]
    resolved = list(plugins)
    has_armor = any(isinstance(p, ModelArmorSafetyPlugin) for p in resolved)
    has_telemetry = any(isinstance(p, ModelTelemetryPlugin) for p in resolved)
    if has_armor and not has_telemetry:
        return [ModelTelemetryPlugin(), *resolved]
    return resolved


def _extract_raw_json_dict(raw_text: str) -> dict[str, Any] | None:
    """Fallback dict parser when structured JSON fails strict Pydantic range constraints."""
    cleaned = raw_text.strip()
    if not cleaned:
        return None
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            body = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
            cleaned = "\n".join(body).strip()
    for candidate in (cleaned,):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


@dataclass
class _StreamAccumulator:
    stream_start: float
    client_ttft_ms: float | None = None
    partial_chunks: int = 0
    total_events: int = 0
    thoughts: list[str] = field(default_factory=list)
    fn_calls: list[dict[str, Any]] = field(default_factory=list)
    fn_responses: list[dict[str, Any]] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    latest_non_partial_text: str = ""
    partial_text_buffer: list[str] = field(default_factory=list)

    def ingest(self, event: Any, default_author: str) -> None:
        self.total_events += 1
        is_partial = bool(getattr(event, "partial", False))
        if is_partial:
            self.partial_chunks += 1

        author = str(getattr(event, "author", "") or default_author)
        if author and author not in self.authors:
            self.authors.append(author)

        content = getattr(event, "content", None)
        parts = list(getattr(content, "parts", None) or []) if content is not None else []

        if self.client_ttft_ms is None and parts:
            has_first_signal = any(
                bool(getattr(p, "text", None)) or getattr(p, "function_call", None) is not None
                for p in parts
            )
            if has_first_signal:
                self.client_ttft_ms = round((time.perf_counter() - self.stream_start) * 1000.0, 3)

        if is_partial:
            for part in parts:
                txt = getattr(part, "text", None)
                if txt and not getattr(part, "thought", False):
                    self.partial_text_buffer.append(txt)
            return

        turn_text_parts: list[str] = []
        for part in parts:
            fc = getattr(part, "function_call", None)
            if fc is not None:
                self.fn_calls.append(
                    {
                        "author": author,
                        "name": str(getattr(fc, "name", "") or ""),
                        "args": dict(getattr(fc, "args", {}) or {}),
                    }
                )
            fr = getattr(part, "function_response", None)
            if fr is not None:
                self.fn_responses.append(
                    {
                        "author": author,
                        "name": str(getattr(fr, "name", "") or ""),
                    }
                )
            txt = getattr(part, "text", None)
            if txt:
                if getattr(part, "thought", False):
                    self.thoughts.append(txt.strip())
                else:
                    turn_text_parts.append(txt)

        if turn_text_parts:
            self.latest_non_partial_text = "".join(turn_text_parts).strip()

    @property
    def final_text(self) -> str:
        return self.latest_non_partial_text or "".join(self.partial_text_buffer).strip()


@traced("adk.run_agent_stream", record_args=("app_name", "backend", "session_id", "max_llm_calls"))
async def run_adk_agent_stream_async(
    *,
    agent: BaseAgent,
    prompt_text: str,
    user_id: str = "analyst",
    session_id: str | None = None,
    app_name: str = "edgar_adk_runtime",
    state_delta: dict[str, Any] | None = None,
    use_adk_app: bool = True,
    summarizer_llm: BaseLlm | None = None,
    backend: LiveBackendType = "vertex_ai",
    max_llm_calls: int = 8,
    plugins: list[BasePlugin] | None = None,
    ephemeral: bool = False,
) -> AdkStreamResult:
    """Drive an ADK `Agent` via `Runner.run_async` with `StreamingMode.SSE` and capture real TTFT."""
    set_span_attributes(**{"adk.agent_name": getattr(agent, "name", type(agent).__name__)})
    memory_mgr = get_conversation_memory_manager()
    session_svc = memory_mgr.adk_session_service
    memory_svc = memory_mgr.adk_memory_service
    raw_session_id = (session_id or "").strip()
    session_was_generated = not raw_session_id
    eff_session_id = raw_session_id or f"adk-sess-{uuid.uuid4().hex[:12]}"
    should_cleanup = ephemeral or session_was_generated or is_ephemeral_session_id(eff_session_id)

    prior_session = await session_svc.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=eff_session_id,
    )
    prior_call_ids: set[str] = set()
    if prior_session is not None and isinstance(prior_session.state.get("latency_metrics"), list):
        prior_call_ids = {
            str(item.get("call_id"))
            for item in prior_session.state["latency_metrics"]
            if isinstance(item, dict) and item.get("call_id")
        }

    resolved_plugins = _resolve_runner_plugins(plugins)
    if use_adk_app:
        adk_app = create_edgar_adk_app(
            agent,
            app_name=app_name,
            summarizer_llm=summarizer_llm,
            enable_context_cache=backend in ("vertex_ai", "custom_llm"),
            plugins=resolved_plugins,
        )
    else:
        adk_app = App(
            name=app_name,
            root_agent=agent,
            plugins=resolved_plugins,
        )
    runner = Runner(
        app=adk_app,
        session_service=session_svc,
        memory_service=memory_svc,
        auto_create_session=True,
    )

    run_config = RunConfig(
        streaming_mode=StreamingMode.SSE,
        max_llm_calls=max_llm_calls,
        tool_thread_pool_config=ToolThreadPoolConfig(max_workers=_DEFAULT_TOOL_THREAD_POOL_WORKERS),
    )
    user_content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=prompt_text)],
    )

    acc = _StreamAccumulator(stream_start=time.perf_counter())
    session_state: dict[str, Any] = {}
    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=eff_session_id,
            new_message=user_content,
            state_delta=state_delta,
            run_config=run_config,
        ):
            acc.ingest(event, agent.name)

        updated_session = await session_svc.get_session(
            app_name=app_name,
            user_id=user_id,
            session_id=eff_session_id,
        )
        session_state = dict(updated_session.state) if updated_session is not None else {}
    finally:
        if should_cleanup:
            with contextlib.suppress(Exception):
                await session_svc.delete_session(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=eff_session_id,
                )

    total_stream_ms = round((time.perf_counter() - acc.stream_start) * 1000.0, 3)

    new_call_ids: list[str] = []
    callback_ttft_ms: float | None = None
    for item in session_state.get("latency_metrics") or []:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("call_id") or "")
        if cid and cid not in prior_call_ids:
            new_call_ids.append(cid)
            if callback_ttft_ms is None and item.get("observed_ttft_ms") is not None:
                callback_ttft_ms = float(item["observed_ttft_ms"])

    latency_records = get_context_latency_tracker().get_records_by_ids(new_call_ids)
    effective_ttft = callback_ttft_ms if callback_ttft_ms is not None else acc.client_ttft_ms

    return AdkStreamResult(
        agent_name=agent.name,
        app_name=app_name,
        session_id=eff_session_id,
        backend=backend,
        final_text=acc.final_text,
        thoughts=[t for t in acc.thoughts if t],
        function_calls=acc.fn_calls,
        function_responses=acc.fn_responses,
        authors=acc.authors,
        partial_chunks_count=acc.partial_chunks,
        total_events_count=acc.total_events,
        observed_ttft_ms=effective_ttft,
        total_stream_latency_ms=total_stream_ms,
        session_state=session_state,
        latency_records=latency_records,
    )


def run_adk_agent_stream(
    *,
    agent: BaseAgent,
    prompt_text: str,
    user_id: str = "analyst",
    session_id: str | None = None,
    app_name: str = "edgar_adk_runtime",
    state_delta: dict[str, Any] | None = None,
    use_adk_app: bool = True,
    summarizer_llm: BaseLlm | None = None,
    backend: LiveBackendType = "vertex_ai",
    max_llm_calls: int = 8,
    plugins: list[BasePlugin] | None = None,
    ephemeral: bool = False,
) -> AdkStreamResult:
    """Synchronous wrapper around `run_adk_agent_stream_async`."""
    return _run_coro_sync(
        run_adk_agent_stream_async(
            agent=agent,
            prompt_text=prompt_text,
            user_id=user_id,
            session_id=session_id,
            app_name=app_name,
            state_delta=state_delta,
            use_adk_app=use_adk_app,
            summarizer_llm=summarizer_llm,
            backend=backend,
            max_llm_calls=max_llm_calls,
            plugins=plugins,
            ephemeral=ephemeral,
        )
    )


async def run_live_adk_financial_reasoning_async(
    *,
    query: str,
    intent: str,
    scope: dict[str, Any],
    variance_reports: list[dict[str, Any]],
    variance_evaluations: list[dict[str, Any]],
    cached_explanations: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    sector_benchmarks: list[dict[str, Any]],
    chunk_health: dict[str, Any] | None,
    memory_context: dict[str, Any] | None = None,
    session_id: str | None = None,
    user_id: str = "analyst",
    resolution: LiveRuntimeResolution | None = None,
    custom_llm: BaseLlm | None = None,
    summarizer_llm: BaseLlm | None = None,
    material_variance_flags: list[dict[str, Any]] | None = None,
    disclosure_deltas: dict[str, Any] | None = None,
    coverage_notices: list[dict[str, Any]] | None = None,
    sector_map_summaries: dict[str, Any] | None = None,
    prompt_already_verified: bool = False,
    verification_feedback: str | None = None,
    defer_callback_fact_check: bool = False,
    ephemeral: bool = False,
) -> AdkStreamResult:
    """Execute `FinancialAnalystAgent` (`EDGAR_ANALYST_MODEL`) via ADK `Runner` with SSE streaming TTFT."""
    resolved = resolution or resolve_live_adk_runtime(custom_llm=custom_llm)
    analyst_llm = build_adk_llm_for_task(
        "financial_reasoning",
        resolution=resolved,
        custom_llm=custom_llm,
    )
    flash_llm = summarizer_llm or (
        custom_llm
        if custom_llm is not None
        else build_adk_llm_for_task("context_compaction", resolution=resolved)
    )
    analyst_agent = create_financial_analyst_agent(
        name="FinancialAnalystAgent",
        model=analyst_llm,
    )

    cache_hits = sum(1 for c in cached_explanations if c.get("cache_hit"))
    cache_misses = max(0, len(cached_explanations) - cache_hits)
    cached_tokens = sum(
        int(c.get("token_estimate") or 0) for c in cached_explanations if c.get("cache_hit")
    )
    prompt_text = build_financial_reasoning_prompt(
        query=query,
        intent=intent,
        scope=scope,
        variance_reports=variance_reports,
        variance_evaluations=variance_evaluations,
        cached_explanations=cached_explanations,
        citations=citations,
        sector_benchmarks=sector_benchmarks,
        chunk_health=chunk_health,
        memory_summary_text=str((memory_context or {}).get("memory_summary_text") or ""),
        material_variance_flags=material_variance_flags,
        disclosure_deltas=disclosure_deltas,
        coverage_notices=coverage_notices,
        sector_map_summaries=sector_map_summaries,
        verification_feedback=verification_feedback,
    )

    # Every tool-output key is reset for this run (chat sessions persist across turns) and
    # carries only outputs the agent already produced this turn (the regeneration pass).
    prior_outputs = ToolOutputs(
        variance_reports=list(variance_reports),
        material_variance_flags=list(material_variance_flags or []),
        disclosure_deltas=disclosure_deltas,
        sector_coverage=(sector_map_summaries or {}).get("sector_coverage"),
        sector_map_summaries=sector_map_summaries,
        sector_benchmarks=list(sector_benchmarks),
        chunk_health=chunk_health,
    )
    state_seed: dict[str, Any] = {
        **turn_scoped_state_seed(citations=list(citations), prior=prior_outputs),
        "memory_context": dict(memory_context or {}),
        "context_cache_stats": {
            "hits": cache_hits,
            "misses": cache_misses,
            "cached_tokens_saved": cached_tokens,
        },
    }
    if defer_callback_fact_check:
        state_seed[_DEFER_STRUCTURED_VERIFICATION_KEY] = True

    plugins_override: list[BasePlugin] | None = None
    if prompt_already_verified:
        prompt_hash = sha256_text(prompt_text)
        state_seed["_model_armor_verified_prompt_sha256"] = prompt_hash
        plugins_override = [
            ModelTelemetryPlugin(),
            ModelArmorSafetyPlugin(pre_verified_prompt_hashes={prompt_hash}),
        ]

    return await run_adk_agent_stream_async(
        agent=analyst_agent,
        prompt_text=prompt_text,
        user_id=user_id,
        session_id=session_id,
        app_name="edgar_chat_app",
        state_delta=state_seed,
        use_adk_app=True,
        summarizer_llm=flash_llm,
        backend=resolved.backend,
        # Retrieval is capped at 6 tool calls (retrieval_budget_before_tool_callback); the
        # remaining calls leave room for parallel tool rounds plus the final AnalystReport.
        max_llm_calls=10,
        plugins=plugins_override,
        ephemeral=ephemeral,
    )


def run_live_adk_financial_reasoning(
    *,
    query: str,
    intent: str,
    scope: dict[str, Any],
    variance_reports: list[dict[str, Any]],
    variance_evaluations: list[dict[str, Any]],
    cached_explanations: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    sector_benchmarks: list[dict[str, Any]],
    chunk_health: dict[str, Any] | None,
    memory_context: dict[str, Any] | None = None,
    session_id: str | None = None,
    user_id: str = "analyst",
    resolution: LiveRuntimeResolution | None = None,
    custom_llm: BaseLlm | None = None,
    summarizer_llm: BaseLlm | None = None,
    material_variance_flags: list[dict[str, Any]] | None = None,
    disclosure_deltas: dict[str, Any] | None = None,
    coverage_notices: list[dict[str, Any]] | None = None,
    sector_map_summaries: dict[str, Any] | None = None,
    prompt_already_verified: bool = False,
    verification_feedback: str | None = None,
    defer_callback_fact_check: bool = False,
    ephemeral: bool = False,
) -> AdkStreamResult:
    """Synchronous wrapper around `run_live_adk_financial_reasoning_async`."""
    return _run_coro_sync(
        run_live_adk_financial_reasoning_async(
            query=query,
            intent=intent,
            scope=scope,
            variance_reports=variance_reports,
            variance_evaluations=variance_evaluations,
            cached_explanations=cached_explanations,
            citations=citations,
            sector_benchmarks=sector_benchmarks,
            chunk_health=chunk_health,
            memory_context=memory_context,
            session_id=session_id,
            user_id=user_id,
            resolution=resolution,
            custom_llm=custom_llm,
            summarizer_llm=summarizer_llm,
            material_variance_flags=material_variance_flags,
            disclosure_deltas=disclosure_deltas,
            coverage_notices=coverage_notices,
            sector_map_summaries=sector_map_summaries,
            prompt_already_verified=prompt_already_verified,
            verification_feedback=verification_feedback,
            defer_callback_fact_check=defer_callback_fact_check,
            ephemeral=ephemeral,
        )
    )


async def run_live_adk_narrative_evaluation_async(
    *,
    query: str,
    response_markdown: str,
    variance_reports: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    cached_explanations_count: int = 0,
    resolution: LiveRuntimeResolution | None = None,
    custom_llm: BaseLlm | None = None,
    reference_explanation: str | None = None,
) -> tuple[dict[str, Any], AdkStreamResult]:
    """Execute `NarrativeEvaluatorAgent` (`gemini-3.5-flash`) via ADK `Runner` with SSE streaming.

    When `reference_explanation` is supplied (golden eval runs), it is appended as an
    analyst-authored reference so the judge can score thematic consistency and explanation
    quality against the expected drivers rather than in isolation.
    """
    resolved = resolution or resolve_live_adk_runtime(custom_llm=custom_llm)
    flash_llm = build_adk_llm_for_task(
        "evaluation",
        resolution=resolved,
        custom_llm=custom_llm,
    )
    evaluator_agent = create_narrative_evaluator_agent(
        name="NarrativeEvaluatorAgent",
        model=flash_llm,
    )

    calc_map = {
        f"{r.get('ticker', 'UNK')}_{r.get('start_year', 0)}_{r.get('end_year', 0)}": r
        for r in variance_reports
    }
    judge_citations = select_citations_for_judge(response_markdown, citations)
    eval_prompt = build_narrative_evaluation_prompt(
        query=query,
        response_markdown=response_markdown,
        variance_reports=variance_reports,
        judge_citations=judge_citations,
        cached_explanations_count=cached_explanations_count,
        reference_explanation=reference_explanation,
    )

    eval_prompt_hash = sha256_text(eval_prompt)
    stream_res = await run_adk_agent_stream_async(
        agent=evaluator_agent,
        prompt_text=eval_prompt,
        user_id="evaluator",
        session_id=f"eval-{uuid.uuid4().hex[:12]}",
        app_name="edgar_eval_app",
        state_delta={
            "calculated_variances": calc_map,
            "retrieved_citations": judge_citations,
            "_model_armor_verified_prompt_sha256": eval_prompt_hash,
        },
        use_adk_app=False,
        backend=resolved.backend,
        max_llm_calls=3,
        plugins=[],
        ephemeral=True,
    )

    raw_state_obj = stream_res.session_state.get("narrative_evaluation_report")
    validated = parse_model_output(
        raw_state_obj,
        NarrativeEvaluationReport,
        fallback_text=stream_res.final_text,
    )
    if validated is not None:
        parsed = validated.model_dump(mode="json")
    elif isinstance(raw_state_obj, dict):
        parsed = dict(raw_state_obj)
    else:
        parsed = _extract_raw_json_dict(stream_res.final_text) or {}

    return parsed, stream_res


def run_live_adk_narrative_evaluation(
    *,
    query: str,
    response_markdown: str,
    variance_reports: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    cached_explanations_count: int = 0,
    resolution: LiveRuntimeResolution | None = None,
    custom_llm: BaseLlm | None = None,
    reference_explanation: str | None = None,
) -> tuple[dict[str, Any], AdkStreamResult]:
    """Synchronous wrapper around `run_live_adk_narrative_evaluation_async`."""
    return _run_coro_sync(
        run_live_adk_narrative_evaluation_async(
            query=query,
            response_markdown=response_markdown,
            variance_reports=variance_reports,
            citations=citations,
            cached_explanations_count=cached_explanations_count,
            resolution=resolution,
            custom_llm=custom_llm,
            reference_explanation=reference_explanation,
        )
    )


__all__ = [
    "AdkStreamResult",
    "LiveBackendType",
    "LiveRuntimeResolution",
    "_build_financial_reasoning_prompt",
    "_compact_evidence_leads",
    "_compact_variance_reports_for_prompt",
    "build_adk_llm_for_task",
    "build_financial_reasoning_prompt",
    "build_narrative_evaluation_prompt",
    "get_live_runtime_status_metadata",
    "resolve_live_adk_runtime",
    "run_adk_agent_stream",
    "run_adk_agent_stream_async",
    "run_live_adk_financial_reasoning",
    "run_live_adk_financial_reasoning_async",
    "run_live_adk_narrative_evaluation",
    "run_live_adk_narrative_evaluation_async",
    "select_citations_for_judge",
]
