"""Live Google ADK `Runner` & Gemini (`gemini-3.8-flash` / `gemini-3.5-flash`) Streaming Execution Engine.

Bridges the ADK multi-agent hierarchy (`edgar_adk_app`, `RootOrchestrator`, `FinancialAnalystAgent`,
`ContextCacheAgent`, `NarrativeEvaluatorAgent`, and `CodingAssistantAgent`) to live Vertex AI / Gemini
execution via `google.adk.runners.Runner` with Server-Sent Events (`StreamingMode.SSE`) to capture real
wall-clock Time-to-First-Thought / Time-to-First-Token (`observed_ttft_ms`).

Every answer comes from live Gemini. The backend is Vertex AI (Application Default Credentials;
the default), the Gemini API when `GEMINI_API_KEY` / `GOOGLE_API_KEY` is set and
`GOOGLE_GENAI_USE_VERTEXAI` is not, or an explicitly injected ADK `BaseLlm`. There is no offline or
template-synthesis mode: upstream failures surface as errors after retry backoff.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import contextvars
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from google.adk.agents import BaseAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.models.base_llm import BaseLlm
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import Runner
from google.genai import types as genai_types

from edgar_capstone.agents.context_memory import (
    create_edgar_adk_app,
    get_context_latency_tracker,
    get_conversation_memory_manager,
    resolve_model_for_task,
)
from edgar_capstone.agents.schemas import (
    CodingAssistanceReport,
    ModelContextLatencyRecord,
    ModelTaskCategory,
    NarrativeEvaluationReport,
)
from edgar_capstone.observability.tracing import set_span_attributes, traced
from edgar_capstone.security.model_armor import ModelArmorSafetyPlugin
from edgar_capstone.security.secrets import DEFAULT_PROJECT_ID

logger = logging.getLogger(__name__)

LiveBackendType = Literal["vertex_ai", "gemini_api", "custom_llm"]

_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
_PLACEHOLDER_SECRET_VALUES = frozenset({"", "unset", "none", "null", "placeholder", "changeme"})

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
    project_id = (
        os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
        or os.environ.get("GCP_PROJECT_ID", "").strip()
        or DEFAULT_PROJECT_ID
    )
    # Gemini models are served from the `global` endpoint in this project while BigQuery, GCS,
    # Model Armor, and embeddings stay regional, so the Gemini location is set separately.
    location = os.environ.get("EDGAR_GEMINI_LOCATION", "").strip() or "global"
    api_key = _clean_api_key(os.environ.get("GOOGLE_API_KEY")) or _clean_api_key(
        os.environ.get("GEMINI_API_KEY")
    )
    use_vertexai = (
        os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() in _TRUTHY_ENV_VALUES
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


_GEMINI_CACHE_LOCK = threading.Lock()
_CACHED_GEMINI_LLMS: dict[tuple[str, str, str, str, str], Gemini] = {}

_ADK_LOOP_LOCK = threading.Lock()
_ADK_LOOP: asyncio.AbstractEventLoop | None = None
_ADK_LOOP_THREAD: threading.Thread | None = None


def _get_persistent_adk_loop() -> asyncio.AbstractEventLoop:
    """Return a single persistent background asyncio loop for ADK Gemini calls.

    ADK's `Gemini.api_client` is cached per event loop (`@PerLoopCachedProperty`). Reusing
    one persistent loop preserves the underlying `genai.Client` HTTP/2 keepalive pool and
    OAuth2 credential cache across turns, cutting ~1.0-1.4s of cold TLS/ADC latency off
    every request's TTFT.
    """
    global _ADK_LOOP, _ADK_LOOP_THREAD
    with _ADK_LOOP_LOCK:
        if _ADK_LOOP is not None and _ADK_LOOP.is_running():
            return _ADK_LOOP
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
        _ADK_LOOP = loop
        _ADK_LOOP_THREAD = thread
        return loop


def _stop_persistent_adk_loop() -> None:
    global _ADK_LOOP, _ADK_LOOP_THREAD
    with _ADK_LOOP_LOCK:
        loop = _ADK_LOOP
        thread = _ADK_LOOP_THREAD
        _ADK_LOOP = None
        _ADK_LOOP_THREAD = None
    if loop is not None and loop.is_running():
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            pass
    if thread is not None and thread.is_alive():
        thread.join(timeout=2.0)


atexit.register(_stop_persistent_adk_loop)


def build_adk_llm_for_task(
    task_category: ModelTaskCategory,
    *,
    resolution: LiveRuntimeResolution | None = None,
    custom_llm: BaseLlm | None = None,
    model_override: str | None = None,
) -> BaseLlm | str:
    """Return a cached ADK `Gemini` (or custom `BaseLlm`) configured with Vertex AI / API key & retries."""
    if custom_llm is not None:
        return custom_llm

    resolved = resolution or resolve_live_adk_runtime()
    model_name = (model_override or "").strip() or resolve_model_for_task(task_category)

    if resolved.backend == "vertex_ai":
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

    if resolved.backend == "gemini_api" and resolved._api_key:
        key_hash = hashlib.sha256(resolved._api_key.encode("utf-8")).hexdigest()[:16]
        cache_key = ("gemini_api", "", "", model_name, key_hash)
        with _GEMINI_CACHE_LOCK:
            cached = _CACHED_GEMINI_LLMS.get(cache_key)
            if cached is None:
                cached = Gemini(
                    model=model_name,
                    client_kwargs={"api_key": resolved._api_key},
                    retry_options=_DEFAULT_GEMINI_RETRY_OPTIONS,
                )
                _CACHED_GEMINI_LLMS[cache_key] = cached
            return cached

    return model_name


def _run_coro_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Execute an async coroutine on the persistent ADK loop preserving `contextvars` (OTel spans)."""
    loop = _get_persistent_adk_loop()
    ctx = contextvars.copy_context()
    # Guard against re-entrant deadlock if a tool invoked on the ADK loop thread calls us synchronously.
    if threading.current_thread() is _ADK_LOOP_THREAD:
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


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    """Parse a JSON object from raw or fenced Markdown model output."""
    cleaned = raw_text.strip()
    if not cleaned:
        return None
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            body_lines = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
            cleaned = "\n".join(body_lines).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start : end + 1])
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
    return None


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
) -> AdkStreamResult:
    """Drive an ADK `Agent` via `Runner.run_async` with `StreamingMode.SSE` and capture real TTFT."""
    set_span_attributes(**{"adk.agent_name": getattr(agent, "name", type(agent).__name__)})
    memory_mgr = get_conversation_memory_manager()
    session_svc = memory_mgr.adk_session_service
    memory_svc = memory_mgr.adk_memory_service
    eff_session_id = (session_id or "").strip() or f"adk-sess-{uuid.uuid4().hex[:12]}"

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

    resolved_plugins: list[BasePlugin] = (
        list(plugins) if plugins is not None else [ModelArmorSafetyPlugin()]
    )
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
    )
    user_content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=prompt_text)],
    )

    stream_start = time.perf_counter()
    client_ttft_ms: float | None = None
    partial_chunks = 0
    total_events = 0
    thoughts: list[str] = []
    fn_calls: list[dict[str, Any]] = []
    fn_responses: list[dict[str, Any]] = []
    authors: list[str] = []
    latest_non_partial_text = ""
    partial_text_buffer: list[str] = []

    async for event in runner.run_async(
        user_id=user_id,
        session_id=eff_session_id,
        new_message=user_content,
        state_delta=state_delta,
        run_config=run_config,
    ):
        total_events += 1
        is_partial = bool(getattr(event, "partial", False))
        if is_partial:
            partial_chunks += 1

        author = str(getattr(event, "author", "") or agent.name)
        if author and author not in authors:
            authors.append(author)

        content = getattr(event, "content", None)
        parts = list(getattr(content, "parts", None) or []) if content is not None else []

        if client_ttft_ms is None and parts:
            has_first_signal = any(
                bool(getattr(p, "text", None)) or getattr(p, "function_call", None) is not None
                for p in parts
            )
            if has_first_signal:
                client_ttft_ms = round((time.perf_counter() - stream_start) * 1000.0, 3)

        if is_partial:
            for part in parts:
                txt = getattr(part, "text", None)
                if txt and not getattr(part, "thought", False):
                    partial_text_buffer.append(txt)
            continue

        turn_text_parts: list[str] = []
        for part in parts:
            fc = getattr(part, "function_call", None)
            if fc is not None:
                fn_calls.append(
                    {
                        "author": author,
                        "name": str(getattr(fc, "name", "") or ""),
                        "args": dict(getattr(fc, "args", {}) or {}),
                    }
                )
            fr = getattr(part, "function_response", None)
            if fr is not None:
                fn_responses.append(
                    {
                        "author": author,
                        "name": str(getattr(fr, "name", "") or ""),
                    }
                )
            txt = getattr(part, "text", None)
            if txt:
                if getattr(part, "thought", False):
                    thoughts.append(txt.strip())
                else:
                    turn_text_parts.append(txt)

        if turn_text_parts:
            latest_non_partial_text = "".join(turn_text_parts).strip()

    total_stream_ms = round((time.perf_counter() - stream_start) * 1000.0, 3)
    final_text = latest_non_partial_text or "".join(partial_text_buffer).strip()

    updated_session = await session_svc.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=eff_session_id,
    )
    session_state = dict(updated_session.state) if updated_session is not None else {}

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
    effective_ttft = callback_ttft_ms if callback_ttft_ms is not None else client_ttft_ms

    return AdkStreamResult(
        agent_name=agent.name,
        app_name=app_name,
        session_id=eff_session_id,
        backend=backend,
        final_text=final_text,
        thoughts=[t for t in thoughts if t],
        function_calls=fn_calls,
        function_responses=fn_responses,
        authors=authors,
        partial_chunks_count=partial_chunks,
        total_events_count=total_events,
        observed_ttft_ms=effective_ttft,
        total_stream_latency_ms=total_stream_ms,
        session_state=session_state,
        latency_records=latency_records,
    )


def run_adk_agent_stream(**kwargs: Any) -> AdkStreamResult:
    """Synchronous wrapper around `run_adk_agent_stream_async`."""
    return _run_coro_sync(run_adk_agent_stream_async(**kwargs))


# Evidence lead budget: generous enough that the analyst can extract verbatim quotes directly
# without burning extra ReAct tool round-trips for passages already retrieved, while keeping
# prompt prefill lean for sub-3s TTFT.
_EVIDENCE_LEAD_EXCERPT_CHARS = 650
_MAX_EVIDENCE_LEADS = 12
_MAX_PROMPT_FLAGS = 8
_MAX_PROMPT_DELTAS_PER_PAIR = 4


def _compact_json(obj: Any) -> str:
    """Serialize JSON compactly (dropping indentation whitespace to cut prefill latency)."""
    return json.dumps(obj, separators=(", ", ": "), default=str)


def _compact_variance_reports_for_prompt(
    variance_reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Strip repetitive nested XBRL citation metadata from the LLM prompt copy.

    Full `NumericCitation` structures (`start_citation`, `end_citation`, `derived_from`,
    `derived_margin_inputs`, `metric_provenance`, `source_urls`) are ~18 KB/company and are
    rendered deterministically by `render_analyst_report_markdown()`; keeping only compact
    numbers, dates, and margins cuts prompt prefill tokens by >10x without losing any fact.
    """
    _DROP_MV_KEYS = frozenset({"start_citation", "end_citation", "derived_from", "currency"})
    compact_list: list[dict[str, Any]] = []
    for rep in variance_reports:
        if not isinstance(rep, dict):
            continue
        slim: dict[str, Any] = {
            k: v
            for k, v in rep.items()
            if k
            not in {
                "derived_margin_inputs",
                "metric_provenance",
                "source_urls",
                "filing_dates",
                "cik",
                "currency",
            }
        }
        for mv_key in (
            "revenue_variance",
            "operating_income_variance",
            "net_income_variance",
            "research_and_development_variance",
        ):
            mv = slim.get(mv_key)
            if isinstance(mv, dict):
                slim[mv_key] = {
                    mk: mv_val
                    for mk, mv_val in mv.items()
                    if mk not in _DROP_MV_KEYS and mv_val is not None
                }
        series = slim.get("annual_series")
        if isinstance(series, list):
            slim["annual_series"] = [
                {
                    sk: sv
                    for sk, sv in step.items()
                    if sk not in {"citations", "sec_filing_url", "filing_date"} and sv is not None
                }
                for step in series
                if isinstance(step, dict)
            ]
        compact_list.append(slim)
    return compact_list


def _compact_evidence_leads(
    cached_explanations: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    *,
    scope: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Citable passages already retrieved during the deterministic pre-pass.

    Uses boundary-first round-robin selection so every `(ticker, start_year)` and
    `(ticker, end_year)` present in `citations` is guaranteed representation before filling
    remaining lead slots in retrieval score order.
    """
    unique: list[dict[str, Any]] = []
    seen_all: set[str] = set()
    for c in citations:
        cid = str(c.get("chunk_id") or "").strip()
        if not cid or cid in seen_all:
            continue
        seen_all.add(cid)
        unique.append(c)

    start_yr = None
    end_yr = None
    if scope:
        try:
            if scope.get("start_year") is not None:
                start_yr = int(scope["start_year"])
            if scope.get("end_year") is not None:
                end_yr = int(scope["end_year"])
        except (TypeError, ValueError):
            start_yr = end_yr = None

    # Group citations by (ticker, fiscal_year) while preserving within-group relevance order
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    ticker_order: list[str] = []
    for c in unique:
        tk = str(c.get("ticker") or "").upper()
        try:
            fy = int(c.get("fiscal_year") or 0)
        except (TypeError, ValueError):
            fy = 0
        if tk and tk not in ticker_order:
            ticker_order.append(tk)
        grouped.setdefault((tk, fy), []).append(c)

    ordered: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()

    def _pick(c: dict[str, Any]) -> None:
        cid = str(c.get("chunk_id") or "").strip()
        if cid and cid not in chosen_ids and len(ordered) < _MAX_EVIDENCE_LEADS:
            chosen_ids.add(cid)
            ordered.append(c)

    # Pass 1: Guarantee at least 1 boundary-year passage per (ticker, start_yr) & (ticker, end_yr)
    boundary_years = [y for y in (start_yr, end_yr) if y is not None]
    for tk in ticker_order:
        for by in dict.fromkeys(boundary_years):
            bucket = grouped.get((tk, by)) or []
            if bucket:
                # Prefer Item 7 (MD&A) when section filter isn't strictly item_1a
                sec_pref = next((b for b in bucket if b.get("section_id") == "item_7"), bucket[0])
                _pick(sec_pref)

    # Pass 2: One passage from any remaining (ticker, fiscal_year) bucket (round-robin)
    for bucket in grouped.values():
        if len(ordered) >= _MAX_EVIDENCE_LEADS:
            break
        for c in bucket:
            if str(c.get("chunk_id") or "").strip() not in chosen_ids:
                _pick(c)
                break

    # Pass 3: Fill remaining slots up to _MAX_EVIDENCE_LEADS in original retrieval order
    for c in unique:
        if len(ordered) >= _MAX_EVIDENCE_LEADS:
            break
        _pick(c)

    leads: list[dict[str, Any]] = [
        {
            "chunk_id": str(c.get("chunk_id") or "").strip(),
            "ticker": c.get("ticker"),
            "fiscal_year": c.get("fiscal_year"),
            "period_of_report": c.get("period_of_report"),
            "section_id": c.get("section_id"),
            "section_title": c.get("section_title"),
            "excerpt": " ".join(str(c.get("excerpt") or "").split())[:_EVIDENCE_LEAD_EXCERPT_CHARS],
        }
        for c in ordered
    ]
    sections_queried = sorted(
        {str(e.get("cache_key")) for e in cached_explanations if e.get("cache_key")}
    )
    if sections_queried:
        leads.append({"pre_queried_sections": sections_queried})
    return leads


def _retrieval_instructions(
    scope: dict[str, Any],
    variance_evaluations: list[dict[str, Any]],
) -> str:
    suggested: list[dict[str, Any]] = []
    for ev in variance_evaluations:
        for tq in ev.get("targeted_10k_queries") or []:
            suggested.append(
                {
                    "ticker": tq.get("ticker"),
                    "fiscal_year": tq.get("fiscal_year"),
                    "section_id": tq.get("section_id"),
                    "driver_focus": tq.get("driver_focus"),
                }
            )
    lines = [
        "## 10-K Retrieval Guidance (Synthesize First When Leads Suffice)",
        "Full 10-K sections are NOT in this request. The **10-K Evidence Leads**, "
        "**Material Variance Flag excerpts**, **Disclosure Deltas**, and **Sector Map-Reduce "
        "Summaries** above already include verbatim retrieved filing passages with canonical "
        "`chunk_id`s. If they already cover every ticker and fiscal year needed to answer the "
        "user's question accurately, synthesize your `AnalystReport` directly WITHOUT calling "
        "extra tools.",
        "Call retrieval tools ONLY to fill a specific missing gap (e.g. a missing start/end fiscal "
        "year or an unretrieved ticker/section):",
        "- `query_cached_10k_context(ticker, fiscal_year, section_id, variance_query, max_passages)`: "
        "ranked passages (with `chunk_id`) from one company-year's `item_7` or `item_1a`, for "
        "explaining a specific variance driver.",
        "- `search_10k_chunks_rag(query, tickers, gics_sector, section_id, start_year, end_year, "
        "top_k)`: hybrid search across filings, for peer comparisons, thematic questions, and "
        "changes across fiscal years.",
        "- `compare_disclosures_across_years(theme, tickers, gics_sector, section_id, start_year, "
        "end_year)`: deterministic year-over-year wording comparison (added / removed / modified "
        "passages with both-year chunk_ids). Use only if disclosure deltas are not already above.",
        "- `flag_material_variances(tickers, start_year, end_year)`: deterministic anomaly flags "
        "for companies or years the payload does not cover.",
        f"- Scope: tickers `{scope.get('tickers_csv', 'ALL')}`, sector "
        f"`{scope.get('gics_sector', 'ALL')}`, sections `{scope.get('section_id', 'ALL')}`, "
        f"fiscal years `FY{scope.get('start_year')}`–`FY{scope.get('end_year')}`.",
        "- If you do call tools, keep result sets small (`max_passages` <= 4, `top_k` <= 8), "
        "issue independent calls in parallel in one step, and never re-fetch evidence already "
        "present in the prompt (at most 6 retrieval calls).",
    ]
    if suggested:
        lines.append(
            "- Suggested retrieval targets from the variance evaluation:\n"
            f"```json\n{_compact_json(suggested[:8])}\n```"
        )
    return "\n".join(lines)


def _material_flags_prompt_section(flags: list[dict[str, Any]] | None) -> str:
    """Compact, authoritative anomaly flags for the analyst (including XBRL provenance & excerpt)."""
    if not flags:
        return ""
    compact = []
    for f in flags[:_MAX_PROMPT_FLAGS]:
        src = f.get("source") or {}
        exp = f.get("explanation") or {}
        compact.append(
            {
                "flag_id": f.get("flag_id"),
                "ticker": f.get("ticker"),
                "fiscal_year": f.get("fiscal_year"),
                "prior_fiscal_year": f.get("prior_fiscal_year"),
                "metric": f.get("metric"),
                "rule": f.get("rule"),
                "severity": f.get("severity"),
                "value": f.get("value"),
                "unit": f.get("unit"),
                "description": f.get("description"),
                "period_of_report": src.get("period_of_report"),
                "prior_period_of_report": src.get("prior_period_of_report"),
                "accession_number": src.get("accession_number"),
                "xbrl_concept": src.get("xbrl_concept"),
                "explanation_query": f.get("explanation_query"),
                "explanation_chunk_id": exp.get("chunk_id"),
                "explanation_excerpt": " ".join(str(exp.get("excerpt") or "").split())[:600]
                if exp.get("excerpt")
                else None,
            }
        )
    return (
        "## Authoritative Material Variance Flags (deterministic anomaly detection)\n"
        "(These anomalies were detected by fixed rules; never recompute or re-threshold them. "
        "Explain the high-severity flags first with MD&A evidence: use `explanation_chunk_id` and "
        "`explanation_excerpt` when present, otherwise call `query_cached_10k_context` with the "
        "flag's `explanation_query`.)\n"
        f"```json\n{_compact_json(compact)}\n```"
    )


def _disclosure_deltas_prompt_section(result: dict[str, Any] | None) -> str:
    """Compact year-over-year disclosure deltas (theme wording changes) for the analyst."""
    if not result or result.get("status") != "success" or not result.get("companies"):
        return ""
    compact_companies: list[dict[str, Any]] = []
    for comp in result["companies"]:
        pairs = []
        for pair in comp.get("year_pairs") or []:
            pairs.append(
                {
                    "years": f"FY{pair['prior_fiscal_year']}->FY{pair['current_fiscal_year']}",
                    "counts": pair.get("counts"),
                    "deltas": [
                        {
                            "change_type": d["change_type"],
                            "prior_chunk_id": d.get("prior_chunk_id"),
                            "current_chunk_id": d.get("current_chunk_id"),
                            "added_phrases": d.get("added_phrases") or [],
                            "removed_phrases": d.get("removed_phrases") or [],
                            "excerpt": " ".join(
                                str(
                                    d.get("current_excerpt") or d.get("prior_excerpt") or ""
                                ).split()
                            )[:_EVIDENCE_LEAD_EXCERPT_CHARS],
                        }
                        for d in pair.get("deltas") or []
                        if d["change_type"] != "unchanged"
                    ][:_MAX_PROMPT_DELTAS_PER_PAIR],
                }
            )
        compact_companies.append(
            {
                "ticker": comp["ticker"],
                "theme_mentions": comp.get("theme_mentions") or {},
                "missing_fiscal_years": comp.get("missing_fiscal_years") or [],
                "year_pairs": pairs,
            }
        )
    payload = {
        "theme": result.get("theme"),
        "summary": result.get("summary"),
        "companies": compact_companies,
    }
    return (
        "## Authoritative Year-over-Year Disclosure Deltas (deterministic wording comparison)\n"
        "(Each delta compares a theme passage with the adjacent year's filing. Summarize the trend "
        "from these; for a `modified` delta cite BOTH `prior_chunk_id` and `current_chunk_id` in the "
        "finding's chunk_ids and quote the current-year chunk; for `removed` quote the prior-year "
        "chunk. Report missing fiscal years as data gaps.)\n"
        f"```json\n{_compact_json(payload)}\n```"
    )


def _build_financial_reasoning_prompt(
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
    memory_summary_text: str = "",
    material_variance_flags: list[dict[str, Any]] | None = None,
    disclosure_deltas: dict[str, Any] | None = None,
    coverage_notices: list[dict[str, Any]] | None = None,
    sector_map_summaries: dict[str, Any] | None = None,
    verification_feedback: str | None = None,
) -> str:
    """Assemble the deterministic facts and retrieval brief for `FinancialAnalystAgent`.

    The request carries authoritative numbers (variances, benchmarks, fiscal period-end dates),
    anomaly flags, disclosure deltas, optional sector map-reduce summaries, and boundary-balanced
    verbatim evidence leads.
    """
    sections: list[str] = [
        f"## Analyst Query\n{query.strip()}",
        (
            f"## Resolved Scope & Intent\n"
            f"- Intent: `{intent}`\n"
            f"- Tickers: `{scope.get('tickers_csv', 'ALL')}`\n"
            f"- GICS Sector: `{scope.get('gics_sector', 'ALL')}`\n"
            f"- Section Filter: `{scope.get('section_id', 'ALL')}`\n"
            f"- Fiscal Period: `FY{scope.get('start_year')}–FY{scope.get('end_year')}`"
        ),
    ]
    if coverage_notices:
        notice_lines = [
            f"- [{n.get('severity', 'info').upper()}] {n.get('message')}"
            for n in coverage_notices
            if n.get("message")
        ]
        if notice_lines:
            sections.append(
                "## System Coverage & Fiscal Calendar Notices (already rendered verbatim)\n"
                + "\n".join(notice_lines)
            )
    if memory_summary_text.strip():
        sections.append(f"## Recalled Conversational Session Memory\n{memory_summary_text.strip()}")
    if variance_reports:
        compact_vr = _compact_variance_reports_for_prompt(variance_reports)
        sections.append(
            "## Authoritative Deterministic Period-over-Period Variance Calculations\n"
            "(Use these exact USD figures, percentage deltas, CAGRs, margin bps shifts, and "
            "`period_end_dates` — never alter or recompute them. When comparing peers whose fiscal "
            "years end in different months, acknowledge the fiscal period-end dates clearly.)\n"
            f"```json\n{_compact_json(compact_vr)}\n```"
        )
    if variance_evaluations:
        compact_evals = [
            {k: v for k, v in ev.items() if k != "targeted_10k_queries"}
            for ev in variance_evaluations
        ]
        sections.append(
            "## Pre-Evaluated ReAct Variance Movers & Operating Leverage Diagnostics\n"
            f"```json\n{_compact_json(compact_evals)}\n```"
        )
    if sector_benchmarks:
        sections.append(
            "## Authoritative Sector Quantitative Benchmarks\n"
            f"```json\n{_compact_json(sector_benchmarks[:5])}\n```"
        )
    if chunk_health:
        sections.append(
            "## Vector Corpus & Chunk Health Diagnostics\n"
            f"```json\n{_compact_json(chunk_health)}\n```"
        )
    flags_section = _material_flags_prompt_section(material_variance_flags)
    if flags_section:
        sections.append(flags_section)
    has_sector_map = bool(sector_map_summaries and sector_map_summaries.get("batch_summaries"))
    if not has_sector_map:
        deltas_section = _disclosure_deltas_prompt_section(disclosure_deltas)
        if deltas_section:
            sections.append(deltas_section)
    if has_sector_map and sector_map_summaries is not None:
        compact_batches = [
            {
                "tickers_covered": b.get("tickers_covered"),
                "dominant_themes": b.get("dominant_themes"),
                "wording_shifts": b.get("wording_shifts"),
                "evidence": [
                    {
                        "ticker": ev.get("ticker"),
                        "fiscal_year": ev.get("fiscal_year"),
                        "chunk_id": ev.get("chunk_id"),
                        "verbatim_quote": ev.get("verbatim_quote"),
                        "takeaway": ev.get("takeaway"),
                    }
                    for ev in (b.get("evidence") or [])
                ],
            }
            for b in sector_map_summaries.get("batch_summaries") or []
        ]
        compact_sector_payload = {
            "sector_coverage": sector_map_summaries.get("sector_coverage"),
            "delta_summary": (disclosure_deltas or {}).get("summary"),
            "batch_summaries": compact_batches,
        }
        sections.append(
            "## Sector-Wide Map-Reduce Batch Summaries (covering all matching issuers in the sector)\n"
            "(Synthesize sector-wide trends from these verified batch summaries. Every claim in "
            "`findings` MUST cite one of the real company `chunk_id`s provided below or in evidence leads "
            "with a verbatim `evidence_quote`.)\n"
            f"```json\n{_compact_json(compact_sector_payload)}\n```"
        )
    leads = _compact_evidence_leads(cached_explanations, citations, scope=scope)
    if leads:
        sections.append(
            "## 10-K Evidence Leads (pre-retrieved passages — cite directly or extend with tools)\n"
            f"```json\n{_compact_json(leads)}\n```"
        )
    sections.append(_retrieval_instructions(scope, variance_evaluations))
    if verification_feedback:
        sections.append(
            "## Verification Correction Required (Prior Draft Rejected)\n"
            f"{verification_feedback.strip()}\n"
            "Fix every finding so each `evidence_quote` is copied verbatim from its cited "
            "`chunk_id`, and every dollar/percentage figure in `statement` matches the exact "
            "figure in `evidence_quote` or the deterministic variance payload above (avoid scaling "
            "or rewriting table numbers if units are ambiguous)."
        )
    sections.append(
        "## Answer Format\n"
        "Answer with a concise `AnalystReport` JSON (`headline`, `key_metrics`, `findings`). "
        "There is NO `notes` field: every qualitative claim must be a verified finding in `findings` "
        "with `chunk_ids` from your tools or the evidence leads plus a verbatim `evidence_quote` "
        "(10-40 words) copied exactly from one of those chunks. For multi-year or peer queries, "
        "include at least one verified finding citing EACH in-scope company's start fiscal year "
        "(`FY"
        + str(scope.get("start_year"))
        + "`) AND end fiscal year (`FY"
        + str(scope.get("end_year"))
        + "`) when passages for those years appear in the evidence leads. Figures in a finding "
        "must appear in its quote or in the deterministic payload / flags. Findings that fail "
        "verification are dropped. `key_metrics` are references only (`ticker` + `metric`); exact "
        "values, `period_of_report` dates, and XBRL provenance citations are rendered automatically "
        "from the engine."
    )
    return "\n\n".join(sections)


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
) -> AdkStreamResult:
    """Execute `FinancialAnalystAgent` (`EDGAR_ANALYST_MODEL`) via ADK `Runner` with SSE streaming TTFT."""
    from edgar_capstone.agents.financial_analyst import create_financial_analyst_agent

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

    calc_map = {
        f"{r.get('ticker', 'UNK')}_{r.get('start_year', 0)}_{r.get('end_year', 0)}": r
        for r in variance_reports
    }
    cache_hits = sum(1 for c in cached_explanations if c.get("cache_hit"))
    cache_misses = max(0, len(cached_explanations) - cache_hits)
    cached_tokens = sum(
        int(c.get("token_estimate") or 0) for c in cached_explanations if c.get("cache_hit")
    )
    prompt_text = _build_financial_reasoning_prompt(
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

    state_seed: dict[str, Any] = {
        "calculated_variances": calc_map,
        "retrieved_citations": list(citations),
        "memory_context": dict(memory_context or {}),
        "material_variance_flags": list(material_variance_flags or []),
        "context_cache_stats": {
            "hits": cache_hits,
            "misses": cache_misses,
            "cached_tokens_saved": cached_tokens,
        },
    }
    plugins_override: list[BasePlugin] | None = None
    if prompt_already_verified:
        prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        state_seed["_model_armor_verified_prompt_sha256"] = prompt_hash
        plugins_override = [ModelArmorSafetyPlugin(pre_verified_prompt_hashes={prompt_hash})]

    return await run_adk_agent_stream_async(
        agent=analyst_agent,
        prompt_text=prompt_text,
        user_id=user_id,
        session_id=session_id,
        app_name="edgar_chat_app",
        state_delta=state_seed,
        use_adk_app=True,
        summarizer_llm=flash_llm if isinstance(flash_llm, BaseLlm) else None,
        backend=resolved.backend,
        # Retrieval is capped at 6 tool calls (retrieval_budget_before_tool_callback); the
        # remaining calls leave room for parallel tool rounds plus the final AnalystReport.
        max_llm_calls=10,
        plugins=plugins_override,
    )


def run_live_adk_financial_reasoning(**kwargs: Any) -> AdkStreamResult:
    """Synchronous wrapper around `run_live_adk_financial_reasoning_async`."""
    return _run_coro_sync(run_live_adk_financial_reasoning_async(**kwargs))


_JUDGE_MIN_CITATIONS = 16
_JUDGE_MAX_CITATIONS = 30
_JUDGE_EXCERPT_CHARS = 4000


def select_citations_for_judge(
    response_markdown: str,
    citations: list[dict[str, Any]],
    *,
    min_citations: int = _JUDGE_MIN_CITATIONS,
    max_citations: int = _JUDGE_MAX_CITATIONS,
) -> list[dict[str, Any]]:
    """Give the judge every passage the report references (full excerpts), then other evidence.

    With agent-driven retrieval, the passages the analyst cites are usually ones it fetched
    itself, which sit at the end of the turn's citation list. Taking the first N citations, or
    truncating their excerpts, hid the supporting text from the judge, so valid claims and
    citations were scored as hallucinated. A chunk counts as referenced when its chunk_id
    appears anywhere in the report: inline tags or the closing citation table.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for c in citations:
        cid = str(c.get("chunk_id") or "")
        if cid:
            by_id.setdefault(cid, c)
    referenced = sorted(
        (cid for cid in by_id if cid in response_markdown),
        key=response_markdown.index,
    )
    limit = min(max_citations, max(min_citations, len(referenced)))
    ordered = [by_id[cid] for cid in referenced]
    ordered += [c for cid, c in by_id.items() if cid not in set(referenced)]
    selected: list[dict[str, Any]] = []
    for c in ordered[:limit]:
        trimmed = dict(c)
        trimmed["excerpt"] = str(c.get("excerpt") or "")[:_JUDGE_EXCERPT_CHARS]
        selected.append(trimmed)
    return selected


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
    from edgar_capstone.agents.evaluation_and_coding import create_narrative_evaluator_agent

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
    eval_prompt = (
        f"Evaluate the following synthesized SEC 10-K variance report against the deterministic "
        f"calculation outputs and retrieved citations. Return JSON matching NarrativeEvaluationReport, "
        f"including `citation_precision_score` (fraction of inline citations whose cited passage "
        f"actually supports the exact claim alongside it, 0.0 to 1.0) and `unsupported_claims_count` "
        f"(integer count of narrative claims not supported by the cited passage or deterministic payload).\n\n"
        f"### Analyst Query\n{query}\n\n"
        f"### Deterministic Variance Calculations\n"
        f"```json\n{json.dumps(variance_reports, default=str)}\n```\n\n"
        f"### Retrieved 10-K Citations (cached_explanations_count={cached_explanations_count})\n"
        f"```json\n{json.dumps(judge_citations, default=str)}\n```\n\n"
        f"### Synthesized Narrative Report Under Evaluation\n{response_markdown}\n"
    )
    if reference_explanation and reference_explanation.strip():
        eval_prompt += (
            "\n### Golden Reference Explanation (analyst-authored ground truth)\n"
            f"{reference_explanation.strip()}\n\n"
            "Score `explanation_quality_score` and `thematic_consistency_score` by how well the "
            "report recovers the drivers and themes in this reference. Do not penalize the report "
            "for omitting reference claims that its retrieved citations do not support, and never "
            "reward claims that appear only in the reference but not in the citations.\n"
        )

    eval_prompt_hash = hashlib.sha256(eval_prompt.encode("utf-8")).hexdigest()
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
    )

    raw_state_obj = stream_res.session_state.get("narrative_evaluation_report")
    if isinstance(raw_state_obj, NarrativeEvaluationReport):
        parsed = raw_state_obj.model_dump(mode="json")
    elif isinstance(raw_state_obj, dict):
        parsed = dict(raw_state_obj)
    else:
        parsed = _extract_json_object(stream_res.final_text) or {}

    return parsed, stream_res


def run_live_adk_narrative_evaluation(
    **kwargs: Any,
) -> tuple[dict[str, Any], AdkStreamResult]:
    """Synchronous wrapper around `run_live_adk_narrative_evaluation_async`."""
    return _run_coro_sync(run_live_adk_narrative_evaluation_async(**kwargs))


async def run_live_adk_coding_assistance_async(
    *,
    prompt: str,
    language: Literal["sql", "python", "lookml"],
    parsed_tickers: list[str],
    start_year: int,
    end_year: int,
    dataset_id: str,
    resolution: LiveRuntimeResolution | None = None,
    custom_llm: BaseLlm | None = None,
) -> tuple[dict[str, Any], AdkStreamResult]:
    """Execute `CodingAssistantAgent` (`gemini-3.5-flash`) via ADK `Runner` with SSE streaming."""
    from edgar_capstone.agents.evaluation_and_coding import create_coding_assistant_agent

    resolved = resolution or resolve_live_adk_runtime(custom_llm=custom_llm)
    flash_llm = build_adk_llm_for_task(
        "coding_assistance",
        resolution=resolved,
        custom_llm=custom_llm,
    )
    coder_agent = create_coding_assistant_agent(
        name="CodingAssistantAgent",
        model=flash_llm,
    )

    coding_prompt = (
        f"Generate a production-grade `{language}` analytical artifact for SEC 10-K financial analysis.\n"
        f"- Request: {prompt}\n"
        f"- Target Language: `{language}`\n"
        f"- Tickers: `{', '.join(parsed_tickers)}`\n"
        f"- Fiscal Year Window: `FY{start_year}` to `FY{end_year}`\n"
        f"- BigQuery Dataset ID: `{dataset_id}` (views/tables: `{dataset_id}.v_financial_variances_yoy`, "
        f"`{dataset_id}.financial_metrics`, `{dataset_id}.filing_chunks`)\n"
    )

    stream_res = await run_adk_agent_stream_async(
        agent=coder_agent,
        prompt_text=coding_prompt,
        user_id="coder",
        session_id=f"code-{uuid.uuid4().hex[:12]}",
        app_name="edgar_coding_app",
        use_adk_app=False,
        backend=resolved.backend,
        max_llm_calls=3,
    )

    raw_state_obj = stream_res.session_state.get("coding_assistance_report")
    if isinstance(raw_state_obj, CodingAssistanceReport):
        parsed = raw_state_obj.model_dump(mode="json")
    elif isinstance(raw_state_obj, dict):
        parsed = dict(raw_state_obj)
    else:
        parsed = _extract_json_object(stream_res.final_text) or {}

    return parsed, stream_res


def run_live_adk_coding_assistance(
    **kwargs: Any,
) -> tuple[dict[str, Any], AdkStreamResult]:
    """Synchronous wrapper around `run_live_adk_coding_assistance_async`."""
    return _run_coro_sync(run_live_adk_coding_assistance_async(**kwargs))


async def run_live_root_orchestrator_turn_async(
    *,
    query: str,
    user_id: str = "analyst",
    session_id: str | None = None,
    orchestrator_llm: BaseLlm | None = None,
    flash_llm: BaseLlm | None = None,
) -> AdkStreamResult:
    """Execute the full multi-agent `RootOrchestrator` ADK `App` via `Runner.run_async` (SSE)."""
    from edgar_capstone.agents.root_orchestrator import create_root_orchestrator

    resolved = resolve_live_adk_runtime(custom_llm=orchestrator_llm)

    pro_model = build_adk_llm_for_task(
        "orchestration",
        resolution=resolved,
        custom_llm=orchestrator_llm,
    )
    flash_model = build_adk_llm_for_task(
        "evaluation",
        resolution=resolved,
        custom_llm=flash_llm or orchestrator_llm,
    )
    root = create_root_orchestrator(
        name="RootOrchestrator",
        model=pro_model,
        analyst_model=pro_model,
        flash_model=flash_model,
    )
    return await run_adk_agent_stream_async(
        agent=root,
        prompt_text=query,
        user_id=user_id,
        session_id=session_id,
        app_name="agents",
        use_adk_app=True,
        summarizer_llm=flash_model if isinstance(flash_model, BaseLlm) else None,
        backend=resolved.backend,
        max_llm_calls=12,
    )


def run_live_root_orchestrator_turn(**kwargs: Any) -> AdkStreamResult:
    """Synchronous wrapper around `run_live_root_orchestrator_turn_async`."""
    return _run_coro_sync(run_live_root_orchestrator_turn_async(**kwargs))
