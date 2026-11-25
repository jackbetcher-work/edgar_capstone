"""Streaming BigQuery Telemetry & Audit Sinks and Firestore Feedback/Session Stores.

Implements:
1. Streaming LLM token/cost/latency telemetry (`llm_telemetry_events`), conversational session &
   tool-usage analytics (`session_telemetry_events`), async LLM-as-a-Judge evaluation updates,
   and 10-K security/access audit trails (`security_audit_logs`) to Google BigQuery via
   `google-cloud-bigquery` plus structured Cloud Logging JSON emission.
2. Persistent analyst feedback (`analyst_feedback`) and multi-instance conversational session
   storage (`analyst_sessions`) backed by `google-cloud-firestore`.
3. Sink selection (`auto` | `gcp` | `local`): Cloud Run streams to BigQuery & Firestore, while
   `local` keeps telemetry and feedback in memory so test and developer runs don't write to the
   production telemetry tables.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import threading
import time
import uuid
from collections import Counter, OrderedDict, defaultdict, deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

from google.auth.exceptions import DefaultCredentialsError, RefreshError

from edgar_capstone.common.env import env_bool
from edgar_capstone.common.formatting import percentile, utc_now_iso
from edgar_capstone.firestore_stores import (
    DEFAULT_FIRESTORE_DATABASE_ID,
    DEFAULT_FIRESTORE_FEEDBACK_COLLECTION,
    DEFAULT_FIRESTORE_SESSION_COLLECTION,
    FirestoreFeedbackStore,
    FirestoreSessionStore,
    session_firestore_doc_id,
)
from edgar_capstone.gcp.bigquery import (
    BigQueryClientWrapper,
    make_scalar_param,
    validate_bq_identifier,
)
from edgar_capstone.gcp.defaults import (
    DEFAULT_BIGQUERY_DATASET_ID,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
    resolve_bq_dataset_id,
    resolve_gcp_location,
    resolve_gcp_project_id,
)
from edgar_capstone.gcp.errors import (
    GcpClientError,
    GcpRateLimitError,
)
from edgar_capstone.observability.tracing import add_span_event, current_trace_correlation

if TYPE_CHECKING:
    from edgar_capstone.agents.core.schemas import ModelContextLatencyRecord

logger = logging.getLogger(__name__)
cloud_sink_logger = logging.getLogger("edgar_capstone.cloud_logging_sink")

TelemetryBackendMode = Literal["auto", "gcp", "local"]

DEFAULT_BQ_LLM_TELEMETRY_TABLE = "llm_telemetry_events"
DEFAULT_BQ_SESSION_TELEMETRY_TABLE = "session_telemetry_events"
DEFAULT_BQ_AUDIT_LOGS_TABLE = "security_audit_logs"

DEFAULT_MAX_LOCAL_RECORDS = 500
DEFAULT_MAX_TRACKED_SESSIONS = 2000

__all__ = [
    "BigQueryTelemetrySink",
    "DEFAULT_BIGQUERY_DATASET_ID",
    "DEFAULT_BQ_AUDIT_LOGS_TABLE",
    "DEFAULT_BQ_LLM_TELEMETRY_TABLE",
    "DEFAULT_BQ_SESSION_TELEMETRY_TABLE",
    "DEFAULT_FIRESTORE_DATABASE_ID",
    "DEFAULT_FIRESTORE_FEEDBACK_COLLECTION",
    "DEFAULT_FIRESTORE_SESSION_COLLECTION",
    "DEFAULT_LOCATION",
    "DEFAULT_PROJECT_ID",
    "FirestoreFeedbackStore",
    "FirestoreSessionStore",
    "TelemetryBackendMode",
    "TelemetryConfig",
    "TelemetryRateLimitError",
    "TelemetrySinkError",
    "cloud_sink_logger",
    "configure_telemetry_stores",
    "emit_cloud_logging_sink_record",
    "get_feedback_store",
    "get_telemetry_sink",
    "reset_telemetry_stores",
    "session_firestore_doc_id",
]


class TelemetrySinkError(GcpClientError):
    """Raised when streaming or persisting telemetry/feedback to GCP fails."""


class TelemetryRateLimitError(TelemetrySinkError, GcpRateLimitError):
    """Raised when BigQuery or Firestore returns HTTP 429 Too Many Requests."""

    def __init__(self, message: str, retry_after_seconds: float | None = None) -> None:
        GcpClientError.__init__(self, message, status_code=429)
        self.retry_after_seconds = retry_after_seconds


def _translate_gcp_error(exc: GcpClientError) -> TelemetrySinkError:
    """Translate `GcpClientError` subclasses into matching `Telemetry*Error` instances."""
    if isinstance(exc, TelemetrySinkError):
        return exc
    if isinstance(exc, GcpRateLimitError):
        return TelemetryRateLimitError(
            str(exc),
            retry_after_seconds=exc.retry_after_seconds,
        )
    return TelemetrySinkError(str(exc), status_code=exc.status_code)


def _resolve_backend_mode(raw_mode: str) -> TelemetryBackendMode:
    norm = raw_mode.strip().lower()
    if norm == "gcp":
        return "gcp"
    if norm == "local":
        return "local"
    return "auto"


@dataclass(frozen=True, slots=True)
class TelemetryConfig:
    """Runtime configuration for BigQuery telemetry/audit sinks and Firestore feedback persistence."""

    telemetry_backend: TelemetryBackendMode = "auto"
    feedback_backend: TelemetryBackendMode = "auto"
    project_id: str = DEFAULT_PROJECT_ID
    location: str = DEFAULT_LOCATION
    bq_dataset_id: str = DEFAULT_BIGQUERY_DATASET_ID
    bq_llm_telemetry_table: str = DEFAULT_BQ_LLM_TELEMETRY_TABLE
    bq_session_telemetry_table: str = DEFAULT_BQ_SESSION_TELEMETRY_TABLE
    bq_audit_logs_table: str = DEFAULT_BQ_AUDIT_LOGS_TABLE
    firestore_database_id: str = DEFAULT_FIRESTORE_DATABASE_ID
    firestore_feedback_collection: str = DEFAULT_FIRESTORE_FEEDBACK_COLLECTION
    firestore_session_collection: str = DEFAULT_FIRESTORE_SESSION_COLLECTION
    gcp_auto_triggered: bool = False

    @classmethod
    def from_env(cls) -> TelemetryConfig:
        raw_telemetry = os.environ.get("EDGAR_TELEMETRY_BACKEND") or "auto"
        raw_feedback = os.environ.get("EDGAR_FEEDBACK_BACKEND") or raw_telemetry

        telemetry_backend = _resolve_backend_mode(raw_telemetry)
        feedback_backend = _resolve_backend_mode(raw_feedback)

        auto_signals = (
            bool(os.environ.get("K_SERVICE", "").strip())
            or bool(os.environ.get("BIGQUERY_DATASET_ID", "").strip())
            or bool(os.environ.get("FIRESTORE_DATABASE_ID", "").strip())
            or bool(os.environ.get("FIRESTORE_FEEDBACK_COLLECTION", "").strip())
            or env_bool("GOOGLE_GENAI_USE_VERTEXAI", False)
        )

        return cls(
            telemetry_backend=telemetry_backend,
            feedback_backend=feedback_backend,
            project_id=resolve_gcp_project_id(),
            location=resolve_gcp_location(allow_global=False),
            bq_dataset_id=resolve_bq_dataset_id(),
            bq_llm_telemetry_table=(
                os.environ.get("BIGQUERY_LLM_TELEMETRY_TABLE") or DEFAULT_BQ_LLM_TELEMETRY_TABLE
            ).strip(),
            bq_session_telemetry_table=(
                os.environ.get("BIGQUERY_SESSION_TELEMETRY_TABLE")
                or DEFAULT_BQ_SESSION_TELEMETRY_TABLE
            ).strip(),
            bq_audit_logs_table=(
                os.environ.get("BIGQUERY_AUDIT_LOGS_TABLE") or DEFAULT_BQ_AUDIT_LOGS_TABLE
            ).strip(),
            firestore_database_id=(
                os.environ.get("FIRESTORE_DATABASE_ID") or DEFAULT_FIRESTORE_DATABASE_ID
            ).strip(),
            firestore_feedback_collection=(
                os.environ.get("FIRESTORE_FEEDBACK_COLLECTION")
                or DEFAULT_FIRESTORE_FEEDBACK_COLLECTION
            ).strip(),
            firestore_session_collection=(
                os.environ.get("FIRESTORE_SESSION_COLLECTION")
                or DEFAULT_FIRESTORE_SESSION_COLLECTION
            ).strip(),
            gcp_auto_triggered=auto_signals,
        )


def _current_trace_id() -> str | None:
    """Cloud Trace / OpenTelemetry trace ID of the active request (joins BigQuery rows to traces)."""
    corr = current_trace_correlation()
    return corr.trace_id if corr else None


def emit_cloud_logging_sink_record(sink_table: str, payload: dict[str, Any]) -> None:
    """Emit a structured telemetry log entry for Cloud Logging log-based metrics and trace correlation.

    The envelope is passed as `json_fields`, so `CloudLoggingJsonFormatter` writes
    `component` / `log_sink_target` / `event_payload` as top-level `jsonPayload` keys next to
    the active trace ID, joining telemetry rows back to their Cloud Trace trace.
    """
    envelope = {
        "component": "sec-edgar-telemetry-sink",
        "log_sink_target": sink_table,
        "event_payload": payload,
    }
    cloud_sink_logger.info(
        "telemetry sink record -> %s",
        sink_table,
        extra={"json_fields": envelope},
    )


def _should_fallback_to_local(exc: BaseException, *, strict_gcp: bool) -> bool:
    if strict_gcp:
        return False
    return isinstance(
        exc,
        (
            DefaultCredentialsError,
            RefreshError,
            GcpClientError,
            TelemetrySinkError,
            OSError,
        ),
    )


# ---------------------------------------------------------------------------
# 1. BigQuery Streaming Telemetry & Security Audit Sink
# ---------------------------------------------------------------------------


@dataclass
class _SessionCumulativeTracker:
    started_at_epoch: float
    turn_count: int = 0
    distinct_tickers: set[str] = field(default_factory=set)
    reports_generated: int = 0


class BigQueryTelemetrySink(BigQueryClientWrapper):
    """Streams LLM token/cost/latency events, session analytics, and audit logs to BigQuery."""

    def __init__(
        self,
        config: TelemetryConfig | None = None,
        *,
        bq_client: Any | None = None,
        credentials: Any | None = None,
        max_local_records: int = DEFAULT_MAX_LOCAL_RECORDS,
        max_sessions: int = DEFAULT_MAX_TRACKED_SESSIONS,
    ) -> None:
        self.config = config or TelemetryConfig.from_env()
        super().__init__(
            project_id=self.config.project_id,
            dataset_id=self.config.bq_dataset_id,
            location=self.config.location,
            bq_client=bq_client,
            credentials=credentials,
            default_timeout=20.0,
        )
        self.llm_table = validate_bq_identifier(
            self.config.bq_llm_telemetry_table, "llm_telemetry_table"
        )
        self.session_table = validate_bq_identifier(
            self.config.bq_session_telemetry_table, "session_telemetry_table"
        )
        self.audit_table = validate_bq_identifier(
            self.config.bq_audit_logs_table, "audit_logs_table"
        )

        self._max_sessions = max(1, int(max_sessions))
        self._lock = threading.Lock()
        self._llm_events: deque[dict[str, Any]] = deque(maxlen=max_local_records)
        self._session_events: deque[dict[str, Any]] = deque(maxlen=max_local_records)
        self._audit_events: deque[dict[str, Any]] = deque(maxlen=max_local_records)
        self._evaluation_events: deque[dict[str, Any]] = deque(maxlen=max_local_records)
        self._sessions: OrderedDict[str, _SessionCumulativeTracker] = OrderedDict()
        self._streamed_counts: dict[str, int] = defaultdict(int)
        self._fallback_counts: dict[str, int] = defaultdict(int)
        self._last_streamed_at: str | None = None
        self._last_error: str | None = None
        self._closed = False
        # Non-blocking background streaming pool for live BigQuery insert_rows_json when no test
        # client is injected; Cloud Logging JSON sink emission remains synchronous for trace link.
        self._async_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="edgar-bq-sink")
        self._pending_writes: list[Future[Any]] = []

    def flush_async_writes(self, timeout: float = 10.0) -> int:
        """Wait for any in-flight background BigQuery streaming writes to complete."""
        with self._lock:
            pending = list(self._pending_writes)
            self._pending_writes.clear()
        done = 0
        for fut in pending:
            try:
                fut.result(timeout=timeout)
                done += 1
            except Exception:
                logger.debug("Async BigQuery telemetry write raised during flush", exc_info=True)
        return done

    def close(self, timeout_seconds: float = 5.0) -> None:
        """Flush pending async writes, shut down the worker pool, and close pooled HTTP client."""
        with self._lock:
            already_closed = self._closed
            self._closed = True
        if not already_closed:
            try:
                self.flush_async_writes(timeout=timeout_seconds)
            finally:
                self._async_pool.shutdown(wait=False, cancel_futures=False)
        super().close()

    def _should_attempt_gcp(self) -> tuple[bool, bool]:
        """Return `(attempt_gcp, strict_gcp)`."""
        mode = self.config.telemetry_backend
        if mode == "local":
            return False, False
        if mode == "gcp":
            return True, True
        return self.config.gcp_auto_triggered, False

    def is_gcp_enabled(self) -> bool:
        """Return True when live GCP BigQuery streaming and query paths are enabled."""
        attempt_gcp, _ = self._should_attempt_gcp()
        return attempt_gcp

    def clear_local_audit_events(self) -> None:
        """Thread-safely clear the in-memory security audit event buffer."""
        with self._lock:
            self._audit_events.clear()

    def stream_rows(
        self,
        table_id: str,
        rows: list[tuple[str, dict[str, Any]]],
    ) -> dict[str, Any]:
        """Stream rows via ``google-cloud-bigquery`` ``insert_rows_json`` with deterministic IDs."""
        if not rows:
            return {"inserted": 0, "table_id": table_id}
        safe_table = validate_bq_identifier(table_id, "table_id")
        row_ids = [insert_id for insert_id, _ in rows]
        json_rows = [row_data for _, row_data in rows]
        try:
            insert_errors = self.insert_rows_json(
                safe_table,
                json_rows,
                row_ids=row_ids,
                skip_invalid_rows=False,
                ignore_unknown_values=True,
                timeout=15.0,
            )
        except GcpClientError as exc:
            raise _translate_gcp_error(exc) from exc
        if insert_errors:
            raise TelemetrySinkError(
                f"BigQuery insert_rows_json rejected rows for {self.dataset_id}.{safe_table}: "
                f"{json.dumps(insert_errors)[:400]}"
            )
        now_iso = utc_now_iso()
        with self._lock:
            self._streamed_counts[safe_table] += len(rows)
            self._last_streamed_at = now_iso
            self._last_error = None
        return {
            "inserted": len(rows),
            "table_id": safe_table,
            "dataset_id": self.dataset_id,
            "project_id": self.project_id,
        }

    def run_query(
        self,
        sql: str,
        query_parameters: Sequence[Any] | None = None,
        *,
        timeout: float | None = 20.0,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a parameterized Standard SQL query against BigQuery sink tables/views."""
        try:
            return super().run_query(
                sql,
                query_parameters=query_parameters,
                timeout=timeout,
                max_results=max_results,
            )
        except GcpClientError as exc:
            raise _translate_gcp_error(exc) from exc

    def _query_filtered_table(
        self,
        *,
        operation_name: str,
        table_id: str,
        select_columns: str,
        order_by: str,
        local_deque: deque[dict[str, Any]],
        limit: int,
        filter_column: str | None = None,
        filter_value: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query a BigQuery sink table with optional equality filter and local deque fallback."""
        bounded = max(1, min(int(limit), 500))
        cleaned_val = filter_value.strip() if filter_value and filter_value.strip() else None
        params: list[Any] = [make_scalar_param("limit", "INT64", bounded)]
        where_sql = ""
        if filter_column and cleaned_val:
            where_sql = f"WHERE {filter_column} = @{filter_column} "
            params.append(make_scalar_param(filter_column, "STRING", cleaned_val))
        sql = (
            f"SELECT {select_columns} "
            f"FROM {self.table_ref(table_id)} "
            f"{where_sql}"
            f"ORDER BY {order_by} "
            "LIMIT @limit"
        )

        def _local_read() -> list[dict[str, Any]]:
            with self._lock:
                items = list(local_deque)
            if filter_column and cleaned_val:
                items = [r for r in items if r.get(filter_column) == cleaned_val]
            return items[:bounded]

        return self._query_with_local_fallback(
            operation_name=operation_name,
            sql=sql,
            query_parameters=params,
            local_fallback=_local_read,
        )

    @staticmethod
    def _build_session_telemetry_row(
        *,
        turn_id: str,
        recorded_at: str,
        session_id: str,
        turn_index: int,
        user_id: str,
        intent: str,
        execution_mode: str,
        status: str,
        tickers_analyzed: list[str],
        evaluation_grade: str | None,
        evaluation_score: float | None,
        trace_id: str | None,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Build a schema-complete `session_telemetry_events` row with safe defaults."""
        row: dict[str, Any] = {
            "turn_id": turn_id,
            "recorded_at": recorded_at,
            "session_id": session_id,
            "turn_index": turn_index,
            "user_id": user_id,
            "intent": intent,
            "execution_mode": execution_mode,
            "status": status,
            "tickers_analyzed": tickers_analyzed,
            "companies_analyzed_count": len(tickers_analyzed),
            "gics_sector": "ALL",
            "start_year": None,
            "end_year": None,
            "report_generated": False,
            "variance_reports_count": None,
            "citations_count": None,
            "tool_invocations": [],
            "tool_invocation_count": 0,
            "llm_call_count": 0,
            "total_context_tokens": 0,
            "cached_context_tokens": 0,
            "output_tokens": 0,
            "total_cost_usd": 0.0,
            "cached_token_savings_usd": 0.0,
            "ttft_ms": 0.0,
            "total_latency_ms": 0.0,
            "session_cumulative_turns": None,
            "session_cumulative_companies_count": None,
            "session_cumulative_reports_count": None,
            "session_elapsed_seconds": None,
            "fact_check_passed": None,
            "evaluation_grade": evaluation_grade,
            "evaluation_score": (
                round(float(evaluation_score), 4) if evaluation_score is not None else None
            ),
            "trace_id": trace_id,
        }
        row.update(overrides)
        return row

    def _query_with_local_fallback(
        self,
        *,
        operation_name: str,
        sql: str,
        query_parameters: Sequence[Any],
        local_fallback: Callable[[], list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Execute a BigQuery query when GCP is enabled, falling back locally in auto mode."""
        attempt_gcp, strict_gcp = self._should_attempt_gcp()
        if attempt_gcp:
            try:
                return self.run_query(sql, query_parameters)
            except Exception as exc:
                if not _should_fallback_to_local(exc, strict_gcp=strict_gcp):
                    raise
                logger.warning(
                    "BigQuery %s failed in auto mode (%s); reading local buffer.",
                    operation_name,
                    exc,
                )
        return local_fallback()

    @staticmethod
    def serialize_llm_record(record: ModelContextLatencyRecord) -> dict[str, Any]:
        """Flatten `ModelContextLatencyRecord` to the BigQuery `llm_telemetry_events` schema."""
        cb = record.context_breakdown
        return {
            "call_id": record.call_id,
            "recorded_at": record.timestamp,
            "session_id": record.session_id,
            "user_id": record.user_id,
            "agent_name": record.agent_name,
            "model_name": record.model_name,
            "task_category": record.task_category,
            "react_phase": record.react_phase,
            "system_instruction_tokens": cb.system_instruction_tokens,
            "memory_and_history_tokens": cb.memory_and_history_tokens,
            "variance_payload_tokens": cb.variance_payload_tokens,
            "cached_10k_tokens": cb.cached_10k_tokens,
            "uncached_10k_and_tool_tokens": cb.uncached_10k_and_tool_tokens,
            "user_prompt_tokens": cb.user_prompt_tokens,
            "total_context_tokens": cb.total_context_tokens,
            "cached_context_tokens": cb.cached_context_tokens,
            "uncached_context_tokens": cb.uncached_context_tokens,
            "cache_hit": cb.cache_hit,
            "output_tokens": record.output_tokens,
            "thoughts_tokens": record.thoughts_tokens,
            "total_model_latency_ms": record.total_model_latency_ms,
            "context_prep_latency_ms": record.context_prep_latency_ms,
            "context_prefill_latency_ms": record.context_prefill_latency_ms,
            "context_latency_ms": record.context_latency_ms,
            "generation_latency_ms": record.generation_latency_ms,
            "ttft_ms": record.ttft_ms,
            "context_latency_ratio": record.context_latency_ratio,
            "latency_per_1k_context_tokens_ms": record.latency_per_1k_context_tokens_ms,
            "estimated_cache_latency_saved_ms": record.estimated_cache_latency_saved_ms,
            "within_ttft_slo": record.within_ttft_slo,
            "input_cost_usd": record.input_cost_usd,
            "output_cost_usd": record.output_cost_usd,
            "total_cost_usd": record.total_cost_usd,
            "uncached_baseline_cost_usd": record.uncached_baseline_cost_usd,
            "cached_token_savings_usd": record.cached_token_savings_usd,
        }

    def _deliver_row(
        self,
        *,
        table_id: str,
        insert_id: str,
        row: dict[str, Any],
        local_deque: deque[dict[str, Any]],
        local_entry: dict[str, Any] | None = None,
        log_payload: dict[str, Any] | None = None,
    ) -> str:
        with self._lock:
            local_deque.appendleft(local_entry if local_entry is not None else row)

        emit_cloud_logging_sink_record(table_id, log_payload if log_payload is not None else row)

        attempt_gcp, strict_gcp = self._should_attempt_gcp()
        if not attempt_gcp:
            return "local_buffer"

        # When running live (no injected test bq_client or credentials), stream BigQuery rows
        # off the request thread so interactive /api/v1/chat latency is never blocked by telemetry I/O.
        if self._injected_client is None and self._credentials is None:
            with self._lock:
                if self._closed:
                    self._fallback_counts[table_id] += 1
                    return "local_buffer_fallback"

            def _bg_stream() -> None:
                try:
                    self.stream_rows(table_id, [(insert_id, row)])
                except Exception as exc:
                    with self._lock:
                        self._fallback_counts[table_id] += 1
                        self._last_error = str(exc)
                    logger.warning(
                        "Background BigQuery telemetry sink '%s' write failed: %s",
                        table_id,
                        exc,
                    )

            ctx = contextvars.copy_context()
            try:
                fut = self._async_pool.submit(ctx.run, _bg_stream)
            except RuntimeError as exc:
                with self._lock:
                    self._fallback_counts[table_id] += 1
                    self._last_error = str(exc)
                logger.debug(
                    "Skipping background BigQuery telemetry submit after pool shutdown: %s", exc
                )
                return "local_buffer_fallback"
            with self._lock:
                self._pending_writes[:] = [f for f in self._pending_writes if not f.done()]
                self._pending_writes.append(fut)
            return "bigquery_streaming_async"

        try:
            self.stream_rows(table_id, [(insert_id, row)])
            return "bigquery_streaming"
        except Exception as exc:
            with self._lock:
                self._fallback_counts[table_id] += 1
                self._last_error = str(exc)
            if not _should_fallback_to_local(exc, strict_gcp=strict_gcp):
                raise
            logger.warning(
                "BigQuery telemetry sink '%s' unavailable in auto mode (%s); buffered locally.",
                table_id,
                exc,
            )
            return "local_buffer_fallback"

    def record_llm_telemetry(self, record: ModelContextLatencyRecord) -> dict[str, Any]:
        """Stream an LLM token/cost/latency call record to `llm_telemetry_events`."""
        row = self.serialize_llm_record(record)
        row["trace_id"] = _current_trace_id()
        # Surface the model call on the active span (ReAct step / ADK agent) for Cloud Trace.
        add_span_event(
            "llm.call",
            **{
                "gen_ai.request.model": record.model_name,
                "edgar.agent_name": record.agent_name,
                "edgar.react_phase": record.react_phase,
                "edgar.task_category": record.task_category,
                "gen_ai.usage.input_tokens": row["total_context_tokens"],
                "gen_ai.usage.output_tokens": record.output_tokens,
                "edgar.cached_context_tokens": row["cached_context_tokens"],
                "edgar.thoughts_tokens": record.thoughts_tokens,
                "edgar.ttft_ms": record.ttft_ms,
                "edgar.total_model_latency_ms": record.total_model_latency_ms,
                "edgar.cache_hit": row["cache_hit"],
                "edgar.total_cost_usd": record.total_cost_usd,
            },
        )
        delivery = self._deliver_row(
            table_id=self.llm_table,
            insert_id=record.call_id,
            row=row,
            local_deque=self._llm_events,
        )
        return {"row": row, "delivery": delivery}

    def record_session_turn_telemetry(
        self,
        *,
        session_id: str | None,
        user_id: str,
        intent: str,
        execution_mode: str,
        status_outcome: str,
        tickers_analyzed: list[str],
        gics_sector: str,
        start_year: int,
        end_year: int,
        report_generated: bool,
        variance_reports_count: int,
        citations_count: int,
        agent_trace: list[dict[str, Any]],
        turn_latency_records: list[ModelContextLatencyRecord],
        total_latency_ms: float,
        fact_check_passed: bool = True,
        evaluation_grade: str | None = None,
        evaluation_score: float | None = None,
    ) -> dict[str, Any]:
        """Record conversational session, company scope, report generation, tool usage & cost telemetry."""
        now_epoch = time.time()
        now_iso = utc_now_iso()
        eff_session_id = (session_id or "").strip() or f"ephemeral-{uuid.uuid4().hex[:10]}"
        norm_tickers = sorted({str(t).strip().upper() for t in tickers_analyzed if str(t).strip()})

        tools_used: list[str] = []
        for step in agent_trace:
            raw_tool = str(step.get("tool") or "").strip()
            if raw_tool:
                # Normalize e.g. 'query_analytics_view(chunk_health)' -> canonical base
                base_tool = raw_tool.split("(", 1)[0].strip()
                if base_tool:
                    tools_used.append(base_tool)

        with self._lock:
            tracker = self._sessions.get(eff_session_id)
            if tracker is None:
                tracker = _SessionCumulativeTracker(started_at_epoch=now_epoch)
                self._sessions[eff_session_id] = tracker
                while len(self._sessions) > self._max_sessions:
                    self._sessions.popitem(last=False)
            else:
                self._sessions.move_to_end(eff_session_id)
            tracker.turn_count += 1
            tracker.distinct_tickers.update(norm_tickers)
            if report_generated:
                tracker.reports_generated += 1
            turn_index = tracker.turn_count
            cum_turns = tracker.turn_count
            cum_companies = len(tracker.distinct_tickers)
            cum_reports = tracker.reports_generated
            elapsed_sec = round(max(0.0, now_epoch - tracker.started_at_epoch), 3)

        tot_ctx_tokens = sum(r.context_breakdown.total_context_tokens for r in turn_latency_records)
        cached_ctx_tokens = sum(
            r.context_breakdown.cached_context_tokens for r in turn_latency_records
        )
        out_tokens = sum(r.output_tokens + r.thoughts_tokens for r in turn_latency_records)
        tot_cost_usd = round(sum(r.total_cost_usd for r in turn_latency_records), 8)
        saved_cost_usd = round(sum(r.cached_token_savings_usd for r in turn_latency_records), 8)
        ttft_ms = (
            round(turn_latency_records[0].ttft_ms, 3)
            if turn_latency_records
            else round(total_latency_ms, 3)
        )

        turn_id = f"sess-turn-{uuid.uuid4().hex[:12]}"
        row = self._build_session_telemetry_row(
            turn_id=turn_id,
            recorded_at=now_iso,
            session_id=eff_session_id,
            turn_index=turn_index,
            user_id=user_id,
            intent=intent,
            execution_mode=execution_mode,
            status=status_outcome,
            tickers_analyzed=norm_tickers,
            evaluation_grade=evaluation_grade,
            evaluation_score=evaluation_score,
            trace_id=_current_trace_id(),
            gics_sector=gics_sector or "ALL",
            start_year=int(start_year),
            end_year=int(end_year),
            report_generated=bool(report_generated),
            variance_reports_count=int(variance_reports_count),
            citations_count=int(citations_count),
            tool_invocations=tools_used,
            tool_invocation_count=len(tools_used),
            llm_call_count=len(turn_latency_records),
            total_context_tokens=tot_ctx_tokens,
            cached_context_tokens=cached_ctx_tokens,
            output_tokens=out_tokens,
            total_cost_usd=tot_cost_usd,
            cached_token_savings_usd=saved_cost_usd,
            ttft_ms=ttft_ms,
            total_latency_ms=round(float(total_latency_ms), 3),
            session_cumulative_turns=cum_turns,
            session_cumulative_companies_count=cum_companies,
            session_cumulative_reports_count=cum_reports,
            session_elapsed_seconds=elapsed_sec,
            fact_check_passed=bool(fact_check_passed),
        )
        delivery = self._deliver_row(
            table_id=self.session_table,
            insert_id=turn_id,
            row=row,
            local_deque=self._session_events,
        )
        return {"row": row, "delivery": delivery}

    def record_async_evaluation_result(
        self,
        *,
        session_id: str,
        user_email: str,
        intent: str,
        evaluation_grade: str,
        evaluation_score: float,
        evaluator_model: str,
        query_text: str = "",
        tickers_analyzed: list[str] | None = None,
        rationale: str = "",
        turn_id: str | None = None,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        """Log completed background Flash LLM-as-a-Judge grade & score asynchronously.

        In BigQuery, interactive turns store ``evaluation_grade="pending"`` and
        ``evaluation_score=NULL`` (excluded from BigQuery ``AVG(evaluation_score)`` in
        ``v_session_analytics_summary``); once the
        background judge finishes, this method updates the matching local turn in place
        (without inflating local turn counts or latency averages), emits a structured Cloud
        Logging sink entry with the full judge rationale, and streams a schema-valid row to
        ``session_telemetry_events`` via ``insert_rows_json``.
        """
        now_iso = utc_now_iso()
        eff_session_id = (session_id or "").strip() or f"ephemeral-{uuid.uuid4().hex[:10]}"
        norm_tickers = sorted(
            {str(t).strip().upper() for t in (tickers_analyzed or []) if str(t).strip()}
        )
        grade_val = str(evaluation_grade).strip() or "unknown"
        score_val = round(float(evaluation_score), 4)
        eff_trace_id = trace_id or _current_trace_id()
        eval_id = f"eval-{uuid.uuid4().hex[:12]}"

        matched_turn_index = 0
        matched_turn_id = turn_id
        updated_local_turn = False
        with self._lock:
            target_idx: int | None = None
            for idx, existing in enumerate(self._session_events):
                if turn_id and existing.get("turn_id") == turn_id:
                    target_idx = idx
                    break
                if (
                    existing.get("session_id") == eff_session_id
                    and existing.get("evaluation_grade") in (None, "pending")
                    and target_idx is None
                ):
                    target_idx = idx
                    if not turn_id:
                        break
            if target_idx is not None:
                prev = self._session_events[target_idx]
                self._session_events[target_idx] = {
                    **prev,
                    "evaluation_grade": grade_val,
                    "evaluation_score": score_val,
                }
                matched_turn_index = int(prev.get("turn_index") or 0)
                matched_turn_id = str(prev.get("turn_id") or turn_id or "") or None
                updated_local_turn = True

        row = self._build_session_telemetry_row(
            turn_id=eval_id,
            recorded_at=now_iso,
            session_id=eff_session_id,
            turn_index=matched_turn_index,
            user_id=user_email,
            intent=intent,
            execution_mode="async_evaluation",
            status="evaluated",
            tickers_analyzed=norm_tickers,
            evaluation_grade=grade_val,
            evaluation_score=score_val,
            trace_id=eff_trace_id,
        )
        eval_record: dict[str, Any] = {
            **row,
            "evaluator_model": evaluator_model,
            "query_text": query_text,
            "rationale": rationale,
            "source_turn_id": matched_turn_id,
        }
        delivery = self._deliver_row(
            table_id=self.session_table,
            insert_id=eval_id,
            row=row,
            local_deque=self._evaluation_events,
            local_entry=eval_record,
            log_payload=eval_record,
        )
        return {
            "row": row,
            "evaluation": eval_record,
            "delivery": delivery,
            "updated_local_turn": updated_local_turn,
        }

    def recent_async_evaluations(
        self,
        *,
        limit: int = 50,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return recently buffered async LLM-as-a-Judge evaluation records."""
        bounded = max(1, min(int(limit), 500))
        with self._lock:
            items = list(self._evaluation_events)
        if session_id and session_id.strip():
            target = session_id.strip()
            items = [r for r in items if r.get("session_id") == target]
        return items[:bounded]

    def record_security_audit_log(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Stream a structured authentication or 10-K access audit log to `security_audit_logs`."""
        epoch = float(entry.get("recorded_at_epoch") or time.time())
        recorded_at = (
            entry.get("recorded_at")
            or entry.get("timestamp")
            or datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        event_id = str(entry.get("event_id") or f"aud-{uuid.uuid4().hex[:12]}")
        row: dict[str, Any] = {
            "event_id": event_id,
            "recorded_at": recorded_at,
            "recorded_at_epoch": round(epoch, 3),
            "event_type": str(entry.get("event_type") or "unknown"),
            "status": str(entry.get("status") or entry.get("status_outcome") or "allowed"),
            "resource": str(entry.get("resource") or ""),
            "tickers": [str(t) for t in (entry.get("tickers") or [])],
            "user_email": str(entry.get("user_email") or "unauthenticated"),
            "user_sub": entry.get("user_sub"),
            "hosted_domain": entry.get("hosted_domain"),
            "identity_provider": entry.get("identity_provider"),
            "roles": [str(r) for r in (entry.get("roles") or [])],
            "cloud_identity_groups": [str(g) for g in (entry.get("cloud_identity_groups") or [])],
            "agent_identity": entry.get("agent_identity"),
            "spiffe_id": entry.get("spiffe_id"),
            "iap_assertion_verified": bool(entry.get("iap_assertion_verified", False)),
            "detail": str(entry.get("detail") or ""),
            "trace_id": entry.get("trace_id") or _current_trace_id(),
        }
        delivery = self._deliver_row(
            table_id=self.audit_table,
            insert_id=event_id,
            row=row,
            local_deque=self._audit_events,
        )
        return {"row": row, "delivery": delivery}

    def query_session_telemetry(
        self,
        *,
        limit: int = 50,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query recent session & tool-usage telemetry from BigQuery or local buffer."""
        cols = (
            "turn_id, CAST(recorded_at AS STRING) AS recorded_at, session_id, "
            "turn_index, user_id, intent, execution_mode, status, tickers_analyzed, "
            "companies_analyzed_count, gics_sector, start_year, end_year, "
            "report_generated, variance_reports_count, citations_count, "
            "tool_invocations, tool_invocation_count, llm_call_count, "
            "total_context_tokens, cached_context_tokens, output_tokens, "
            "total_cost_usd, cached_token_savings_usd, ttft_ms, total_latency_ms, "
            "session_cumulative_turns, session_cumulative_companies_count, "
            "session_cumulative_reports_count, session_elapsed_seconds, "
            "fact_check_passed, evaluation_grade, evaluation_score, trace_id"
        )
        return self._query_filtered_table(
            operation_name="query_session_telemetry",
            table_id=self.session_table,
            select_columns=cols,
            order_by="recorded_at DESC",
            local_deque=self._session_events,
            limit=limit,
            filter_column="session_id",
            filter_value=session_id,
        )

    def summarize_session_telemetry(self) -> dict[str, Any]:
        """Compute aggregate session length, companies analyzed, report counts, and tool usage."""
        with self._lock:
            events = list(self._session_events)
            sessions_snap = {
                sid: (
                    t.turn_count,
                    len(t.distinct_tickers),
                    t.reports_generated,
                    sorted(t.distinct_tickers),
                )
                for sid, t in self._sessions.items()
            }

        tool_counter: Counter[str] = Counter()
        all_tickers: set[str] = set()
        latencies: list[float] = []
        total_reports = 0
        total_cost = 0.0
        total_savings = 0.0

        for ev in events:
            for tool_name in ev.get("tool_invocations") or []:
                tool_counter[str(tool_name)] += 1
            for ticker in ev.get("tickers_analyzed") or []:
                all_tickers.add(str(ticker))
            if ev.get("report_generated"):
                total_reports += 1
            latencies.append(float(ev.get("total_latency_ms") or 0.0))
            total_cost += float(ev.get("total_cost_usd") or 0.0)
            total_savings += float(ev.get("cached_token_savings_usd") or 0.0)

        avg_turns_per_session = (
            round(sum(v[0] for v in sessions_snap.values()) / len(sessions_snap), 2)
            if sessions_snap
            else 0.0
        )

        return {
            "total_turns_recorded": len(events),
            "distinct_sessions_count": len(sessions_snap),
            "avg_session_length_turns": avg_turns_per_session,
            "distinct_companies_analyzed": sorted(all_tickers),
            "distinct_companies_analyzed_count": len(all_tickers),
            "reports_generated_count": total_reports,
            "mean_turn_latency_ms": (
                round(sum(latencies) / len(latencies), 3) if latencies else 0.0
            ),
            "p95_turn_latency_ms": round(percentile(latencies, 95.0), 3),
            "total_turn_cost_usd": round(total_cost, 8),
            "total_cached_token_savings_usd": round(total_savings, 8),
            "tool_usage_counts": dict(tool_counter.most_common()),
        }

    def query_security_audit_logs(
        self,
        *,
        limit: int = 50,
        event_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query security & 10-K access audit events from BigQuery or local buffer."""
        cols = (
            "event_id, CAST(recorded_at AS STRING) AS recorded_at, "
            "recorded_at_epoch, event_type, status, resource, tickers, "
            "user_email, user_sub, hosted_domain, identity_provider, "
            "roles, cloud_identity_groups, agent_identity, spiffe_id, "
            "iap_assertion_verified, detail, trace_id"
        )
        return self._query_filtered_table(
            operation_name="query_security_audit_logs",
            table_id=self.audit_table,
            select_columns=cols,
            order_by="recorded_at_epoch DESC",
            local_deque=self._audit_events,
            limit=limit,
            filter_column="event_type",
            filter_value=event_type,
        )

    def status_metadata(self) -> dict[str, Any]:
        """Return secret-free diagnostic status of the BigQuery telemetry & audit sinks."""
        attempt_gcp, strict_gcp = self._should_attempt_gcp()
        with self._lock:
            streamed = dict(self._streamed_counts)
            fallbacks = dict(self._fallback_counts)
            buffered_counts = {
                self.llm_table: len(self._llm_events),
                self.session_table: len(self._session_events),
                self.audit_table: len(self._audit_events),
            }
            last_at = self._last_streamed_at
            last_err = self._last_error

        return {
            "backend_mode": self.config.telemetry_backend,
            "live_bigquery_streaming_enabled": attempt_gcp,
            "strict_gcp": strict_gcp,
            "project_id": self.project_id,
            "dataset_id": self.dataset_id,
            "sink_tables": {
                "llm_telemetry": f"{self.project_id}.{self.dataset_id}.{self.llm_table}",
                "session_telemetry": f"{self.project_id}.{self.dataset_id}.{self.session_table}",
                "security_audit_logs": f"{self.project_id}.{self.dataset_id}.{self.audit_table}",
            },
            "streamed_row_counts": streamed,
            "fallback_counts": fallbacks,
            "buffered_row_counts": buffered_counts,
            "last_streamed_at": last_at,
            "last_error": last_err,
        }

    def reset_local_buffers(self) -> None:
        with self._lock:
            self._llm_events.clear()
            self._session_events.clear()
            self._audit_events.clear()
            self._evaluation_events.clear()
            self._sessions.clear()
            self._streamed_counts.clear()
            self._fallback_counts.clear()
            self._last_streamed_at = None
            self._last_error = None


# ---------------------------------------------------------------------------
# 2. Singleton Registry & Dependency Injection Helpers
# ---------------------------------------------------------------------------

_TELEMETRY_ENV_KEYS: tuple[str, ...] = (
    "EDGAR_TELEMETRY_BACKEND",
    "EDGAR_FEEDBACK_BACKEND",
    "K_SERVICE",
    "GCP_PROJECT_ID",
    "GOOGLE_CLOUD_PROJECT",
    "EDGAR_GCP_PROJECT_ID",
    "GCP_REGION",
    "GOOGLE_CLOUD_LOCATION",
    "BIGQUERY_LOCATION",
    "EDGAR_GCP_REGION",
    "BIGQUERY_DATASET_ID",
    "BIGQUERY_LLM_TELEMETRY_TABLE",
    "BIGQUERY_SESSION_TELEMETRY_TABLE",
    "BIGQUERY_AUDIT_LOGS_TABLE",
    "FIRESTORE_DATABASE_ID",
    "FIRESTORE_FEEDBACK_COLLECTION",
    "FIRESTORE_SESSION_COLLECTION",
    "GOOGLE_GENAI_USE_VERTEXAI",
)


def _telemetry_env_snapshot() -> tuple[str | None, ...]:
    """Capture raw env values affecting `TelemetryConfig` for fast hot-path cache checks."""
    env = os.environ
    return tuple(env.get(key) for key in _TELEMETRY_ENV_KEYS)


_StoreT = TypeVar("_StoreT", BigQueryTelemetrySink, FirestoreFeedbackStore)
_REGISTRY_LOCK = threading.Lock()


class _EnvCachedStore(Generic[_StoreT]):
    """Thread-safe env-snapshot-cached singleton slot with test injection & `.close()` on eviction."""

    def __init__(self, factory: Callable[[TelemetryConfig], _StoreT]) -> None:
        self._factory = factory
        self._custom: _StoreT | None = None
        self._cached: tuple[tuple[str | None, ...], TelemetryConfig, _StoreT] | None = None

    def set_custom(self, instance: _StoreT | None) -> None:
        self._custom = instance

    def get(self) -> _StoreT:
        old: _StoreT | None = None
        with _REGISTRY_LOCK:
            if self._custom is not None:
                return self._custom
            snap = _telemetry_env_snapshot()
            if self._cached is not None and self._cached[0] == snap:
                return self._cached[2]
            cfg = TelemetryConfig.from_env()
            if self._cached is not None and self._cached[1] == cfg:
                self._cached = (snap, cfg, self._cached[2])
                return self._cached[2]
            if self._cached is not None:
                old = self._cached[2]
            instance = self._factory(cfg)
            self._cached = (snap, cfg, instance)
        if old is not None:
            old.close()
        return instance

    def drain_for_reset(self) -> list[_StoreT]:
        """Reset buffers on custom/cached stores and return them for closing (caller holds lock)."""
        items: list[_StoreT] = []
        for obj in (self._custom, self._cached[2] if self._cached is not None else None):
            if obj is not None:
                obj.reset_local_buffers()
                items.append(obj)
        self._custom = None
        self._cached = None
        return items


_TELEMETRY_STORE_SLOT: _EnvCachedStore[BigQueryTelemetrySink] = _EnvCachedStore(
    BigQueryTelemetrySink
)
_FEEDBACK_STORE_SLOT: _EnvCachedStore[FirestoreFeedbackStore] = _EnvCachedStore(
    FirestoreFeedbackStore
)


def configure_telemetry_stores(
    *,
    telemetry_sink: BigQueryTelemetrySink | None = None,
    feedback_store: FirestoreFeedbackStore | None = None,
) -> None:
    """Explicitly inject `BigQueryTelemetrySink` and/or `FirestoreFeedbackStore`."""
    with _REGISTRY_LOCK:
        _TELEMETRY_STORE_SLOT.set_custom(telemetry_sink)
        _FEEDBACK_STORE_SLOT.set_custom(feedback_store)


def reset_telemetry_stores() -> None:
    """Reset and cleanly close injected and environment-cached telemetry/feedback singletons."""
    to_close: list[Any] = []
    seen_ids: set[int] = set()
    with _REGISTRY_LOCK:
        for slot in (_TELEMETRY_STORE_SLOT, _FEEDBACK_STORE_SLOT):
            for obj in slot.drain_for_reset():
                if id(obj) not in seen_ids:
                    seen_ids.add(id(obj))
                    to_close.append(obj)
    for obj in to_close:
        try:
            obj.close()
        except Exception:
            logger.debug("Ignoring close error during telemetry store reset", exc_info=True)


def get_telemetry_sink() -> BigQueryTelemetrySink:
    """Return the active `BigQueryTelemetrySink` instance."""
    return _TELEMETRY_STORE_SLOT.get()


def get_feedback_store() -> FirestoreFeedbackStore:
    """Return the active `FirestoreFeedbackStore` instance."""
    return _FEEDBACK_STORE_SLOT.get()
