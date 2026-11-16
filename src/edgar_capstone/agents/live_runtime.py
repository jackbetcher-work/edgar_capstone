"""Live Google ADK `Runner` & Gemini (`gemini-3.8-flash` / `gemini-3.5-flash`) Streaming Execution Engine.

Bridges the ADK multi-agent hierarchy (`edgar_adk_app`, `RootOrchestrator`, `FinancialAnalystAgent`,
`ContextCacheAgent`, `NarrativeEvaluatorAgent`, and `CodingAssistantAgent`) to live Vertex AI / Gemini
execution via `google.adk.runners.Runner` with Server-Sent Events (`StreamingMode.SSE`) to capture real
wall-clock Time-to-First-Thought / Time-to-First-Token (`observed_ttft_ms`).

Execution Modes (`EDGAR_EXECUTION_MODE`):
- `"auto"` (default): Automatically activates live ADK `Runner` execution when Vertex AI
  (`GOOGLE_GENAI_USE_VERTEXAI=true` + project/location), Gemini API (`GEMINI_API_KEY` / `GOOGLE_API_KEY`),
  or an explicit `BaseLlm` instance is configured, and gracefully falls back to deterministic offline
  synthesis in local/CI environments when no live credentials are present.
- `"live"`: Strictly requires live ADK `Runner` execution and raises immediately if credentials are missing
  or if an upstream Gemini call fails after retry backoff.
- `"deterministic"`: Forces offline deterministic Python orchestration and template synthesis.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import time
import uuid
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
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

ExecutionMode = Literal["auto", "live", "deterministic"]
LiveBackendType = Literal[
    "vertex_ai",
    "gemini_api",
    "custom_llm",
    "deterministic_fallback",
]

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

    requested_mode: ExecutionMode
    live_adk_enabled: bool
    backend: LiveBackendType
    project_id: str
    location: str
    api_key_configured: bool = False
    _api_key: str | None = field(default=None, repr=False, compare=False)

    def to_public_dict(self) -> dict[str, Any]:
        """Return safe, secret-free runtime status metadata for `/health` and telemetry responses."""
        return {
            "requested_mode": self.requested_mode,
            "live_adk_enabled": self.live_adk_enabled,
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


def resolve_live_adk_runtime(
    *,
    execution_mode: str | None = None,
    custom_llm: BaseLlm | None = None,
) -> LiveRuntimeResolution:
    """Resolve whether live ADK `Runner` execution is active and which Gemini backend to target."""
    raw_mode = (
        execution_mode
        if execution_mode is not None
        else os.environ.get("EDGAR_EXECUTION_MODE", "auto")
    )
    norm_mode = raw_mode.strip().lower() or "auto"
    if norm_mode not in ("auto", "live", "deterministic"):
        raise ValueError(
            f"Unsupported EDGAR_EXECUTION_MODE={raw_mode!r}; expected 'auto', 'live', or 'deterministic'."
        )
    mode: ExecutionMode = norm_mode  # type: ignore[assignment]

    project_id = (
        os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
        or os.environ.get("GCP_PROJECT_ID", "").strip()
        or DEFAULT_PROJECT_ID
    )
    # Gemini models may only be served from the `global` endpoint while BigQuery, GCS,
    # Model Armor, and embeddings stay regional, so the Gemini location is set separately.
    location = (
        os.environ.get("EDGAR_GEMINI_LOCATION", "").strip()
        or os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip()
        or os.environ.get("GCP_REGION", "").strip()
        or "us-central1"
    )
    api_key = _clean_api_key(os.environ.get("GOOGLE_API_KEY")) or _clean_api_key(
        os.environ.get("GEMINI_API_KEY")
    )
    use_vertexai = (
        os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() in _TRUTHY_ENV_VALUES
    )

    if custom_llm is not None:
        return LiveRuntimeResolution(
            requested_mode=mode,
            live_adk_enabled=True,
            backend="custom_llm",
            project_id=project_id,
            location=location,
            api_key_configured=api_key is not None,
            _api_key=api_key,
        )

    if mode == "deterministic":
        return LiveRuntimeResolution(
            requested_mode="deterministic",
            live_adk_enabled=False,
            backend="deterministic_fallback",
            project_id=project_id,
            location=location,
            api_key_configured=api_key is not None,
            _api_key=None,
        )

    if use_vertexai:
        return LiveRuntimeResolution(
            requested_mode=mode,
            live_adk_enabled=True,
            backend="vertex_ai",
            project_id=project_id,
            location=location,
            api_key_configured=api_key is not None,
            _api_key=None,
        )

    if api_key is not None:
        return LiveRuntimeResolution(
            requested_mode=mode,
            live_adk_enabled=True,
            backend="gemini_api",
            project_id=project_id,
            location=location,
            api_key_configured=True,
            _api_key=api_key,
        )

    if mode == "live":
        raise RuntimeError(
            "EDGAR_EXECUTION_MODE='live' requires live Gemini credentials: set "
            "GOOGLE_GENAI_USE_VERTEXAI=true (with GOOGLE_CLOUD_PROJECT / GCP_PROJECT_ID) "
            "or provide GEMINI_API_KEY / GOOGLE_API_KEY."
        )

    return LiveRuntimeResolution(
        requested_mode="auto",
        live_adk_enabled=False,
        backend="deterministic_fallback",
        project_id=project_id,
        location=location,
        api_key_configured=False,
        _api_key=None,
    )


def get_live_runtime_status_metadata() -> dict[str, Any]:
    """Return secret-redacted metadata describing active ADK `Runner` / Gemini execution configuration."""
    return resolve_live_adk_runtime().to_public_dict()


def build_adk_llm_for_task(
    task_category: ModelTaskCategory,
    *,
    resolution: LiveRuntimeResolution | None = None,
    custom_llm: BaseLlm | None = None,
    model_override: str | None = None,
) -> BaseLlm | str:
    """Instantiate an ADK `Gemini` (or custom `BaseLlm`) configured with Vertex AI / API key & retries."""
    if custom_llm is not None:
        return custom_llm

    resolved = resolution or resolve_live_adk_runtime()
    model_name = (model_override or "").strip() or resolve_model_for_task(task_category)

    if resolved.backend == "vertex_ai":
        return Gemini(
            model=model_name,
            client_kwargs={
                "vertexai": True,
                "project": resolved.project_id,
                "location": resolved.location,
            },
            retry_options=_DEFAULT_GEMINI_RETRY_OPTIONS,
        )

    if resolved.backend == "gemini_api" and resolved._api_key:
        return Gemini(
            model=model_name,
            client_kwargs={"api_key": resolved._api_key},
            retry_options=_DEFAULT_GEMINI_RETRY_OPTIONS,
        )

    return model_name


def _run_coro_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Execute an async coroutine safely from either synchronous threads or active event loops."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Copy contextvars into the worker thread so the active OpenTelemetry span (and the
    # request log context) stays the parent of ADK agent / LLM / tool spans.
    ctx = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="adk-live-runner") as pool:
        future = pool.submit(ctx.run, asyncio.run, coro)
        return future.result()


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


# Retrieval budget hints for the agent; FinancialAnalystAgent runs with max_llm_calls=8.
_EVIDENCE_LEAD_EXCERPT_CHARS = 240
_MAX_EVIDENCE_LEADS = 8


def _compact_evidence_leads(
    cached_explanations: list[dict[str, Any]],
    citations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Short, citable pointers to passages the deterministic ReAct pre-pass already found.

    These are leads, not the evidence base. They give the agent chunk_ids and a short
    excerpt so it can decide what else to retrieve with its tools.
    """
    leads: list[dict[str, Any]] = []
    seen: set[str] = set()
    for c in citations:
        cid = str(c.get("chunk_id") or "").strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        leads.append(
            {
                "chunk_id": cid,
                "ticker": c.get("ticker"),
                "fiscal_year": c.get("fiscal_year"),
                "section_id": c.get("section_id"),
                "section_title": c.get("section_title"),
                "sec_filing_url": c.get("sec_filing_url"),
                "excerpt": " ".join(str(c.get("excerpt") or "").split())[
                    :_EVIDENCE_LEAD_EXCERPT_CHARS
                ],
            }
        )
        if len(leads) >= _MAX_EVIDENCE_LEADS:
            break
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
        "## 10-K Retrieval: Query What This Question Needs",
        "Full 10-K sections are NOT in this request. Retrieve only the evidence the analyst's "
        "question requires:",
        "- `query_cached_10k_context(ticker, fiscal_year, section_id, variance_query, max_passages)`: "
        "ranked passages (with `chunk_id`) from one company-year's `item_7` or `item_1a`, for "
        "explaining a specific variance driver.",
        "- `search_10k_chunks_rag(query, tickers, gics_sector, section_id, start_year, end_year, "
        "top_k)`: hybrid search across filings, for peer comparisons, thematic questions, and "
        "changes across fiscal years (query each relevant year and compare).",
        f"- Scope: tickers `{scope.get('tickers_csv', 'ALL')}`, sector "
        f"`{scope.get('gics_sector', 'ALL')}`, sections `{scope.get('section_id', 'ALL')}`, "
        f"fiscal years `FY{scope.get('start_year')}`–`FY{scope.get('end_year')}`.",
        "- Write focused queries from the question and the variance movers, keep result sets "
        "small (`max_passages` <= 4, `top_k` <= 8), skip evidence the leads already cover, and "
        "stop once every claim is supported.",
    ]
    if suggested:
        lines.append(
            "- Suggested retrieval targets from the variance evaluation:\n"
            f"```json\n{json.dumps(suggested[:8], indent=2, default=str)}\n```"
        )
    return "\n".join(lines)


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
) -> str:
    """Assemble the deterministic facts and retrieval brief for `FinancialAnalystAgent`.

    The request carries the authoritative numbers (variances, benchmarks) plus short
    evidence leads. 10-K text is retrieved by the agent itself through targeted tools,
    so only the passages relevant to the question enter the context window.
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
    if memory_summary_text.strip():
        sections.append(f"## Recalled Conversational Session Memory\n{memory_summary_text.strip()}")
    if variance_reports:
        sections.append(
            "## Authoritative Deterministic Period-over-Period Variance Calculations\n"
            "(Use these exact USD figures, percentage deltas, CAGRs, and margin bps shifts — never alter or recompute them.)\n"
            f"```json\n{json.dumps(variance_reports, indent=2, default=str)}\n```"
        )
    if variance_evaluations:
        compact_evals = [
            {k: v for k, v in ev.items() if k != "targeted_10k_queries"}
            for ev in variance_evaluations
        ]
        sections.append(
            "## Pre-Evaluated ReAct Variance Movers & Operating Leverage Diagnostics\n"
            f"```json\n{json.dumps(compact_evals, indent=2, default=str)}\n```"
        )
    if sector_benchmarks:
        sections.append(
            "## Authoritative Sector Quantitative Benchmarks\n"
            f"```json\n{json.dumps(sector_benchmarks[:5], indent=2, default=str)}\n```"
        )
    if chunk_health:
        sections.append(
            "## Vector Corpus & Chunk Health Diagnostics\n"
            f"```json\n{json.dumps(chunk_health, indent=2, default=str)}\n```"
        )
    leads = _compact_evidence_leads(cached_explanations, citations)
    if leads:
        sections.append(
            "## 10-K Evidence Leads (partial pre-retrieval — verify or extend with tools)\n"
            f"```json\n{json.dumps(leads, indent=2, default=str)}\n```"
        )
    sections.append(_retrieval_instructions(scope, variance_evaluations))
    sections.append(
        "## Synthesis Requirements\n"
        "Every percentage delta must match the deterministic calculation payload. Every 10-K claim "
        "must carry an inline `[TICKER FYxxxx | Section Title | chunk_id](sec_filing_url)` citation "
        "using a `chunk_id` returned by your tools or listed in the evidence leads."
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
    state_seed: dict[str, Any] = {
        "calculated_variances": calc_map,
        "retrieved_citations": list(citations),
        "memory_context": dict(memory_context or {}),
        "context_cache_stats": {
            "hits": cache_hits,
            "misses": cache_misses,
            "cached_tokens_saved": cached_tokens,
        },
    }

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
    )

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
) -> tuple[dict[str, Any], AdkStreamResult]:
    """Execute `NarrativeEvaluatorAgent` (`gemini-3.5-flash`) via ADK `Runner` with SSE streaming."""
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
        f"calculation outputs and retrieved citations.\n\n"
        f"### Analyst Query\n{query}\n\n"
        f"### Deterministic Variance Calculations\n"
        f"```json\n{json.dumps(variance_reports, default=str)}\n```\n\n"
        f"### Retrieved 10-K Citations (cached_explanations_count={cached_explanations_count})\n"
        f"```json\n{json.dumps(judge_citations, default=str)}\n```\n\n"
        f"### Synthesized Narrative Report Under Evaluation\n{response_markdown}\n"
    )

    stream_res = await run_adk_agent_stream_async(
        agent=evaluator_agent,
        prompt_text=eval_prompt,
        user_id="evaluator",
        session_id=f"eval-{uuid.uuid4().hex[:12]}",
        app_name="edgar_eval_app",
        state_delta={
            "calculated_variances": calc_map,
            "retrieved_citations": judge_citations,
        },
        use_adk_app=False,
        backend=resolved.backend,
        max_llm_calls=3,
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
    execution_mode: str | None = None,
    orchestrator_llm: BaseLlm | None = None,
    flash_llm: BaseLlm | None = None,
) -> AdkStreamResult:
    """Execute the full multi-agent `RootOrchestrator` ADK `App` via `Runner.run_async` (SSE)."""
    from edgar_capstone.agents.root_orchestrator import create_root_orchestrator

    resolved = resolve_live_adk_runtime(
        execution_mode=execution_mode,
        custom_llm=orchestrator_llm,
    )
    if not resolved.live_adk_enabled:
        raise RuntimeError(
            "Live ADK RootOrchestrator execution is disabled in deterministic mode or without credentials."
        )

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
