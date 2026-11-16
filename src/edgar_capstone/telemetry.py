"""Streaming BigQuery Telemetry & Audit Sinks, Firestore Feedback Persistence, and LookML Catalog.

Implements:
1. Streaming LLM token/cost/latency telemetry (`llm_telemetry_events`), conversational session &
   tool-usage analytics (`session_telemetry_events`), and 10-K security/access audit trails
   (`security_audit_logs`) to Google BigQuery sink tables via REST v2 `tabledata.insertAll`
   plus structured Cloud Logging JSON emission for `google_logging_project_sink` durability.
2. Persistent user feedback storage (`analyst_feedback`) backed by Google Cloud Firestore Native
   REST v1 (`documents` & `:runQuery`) with ADC IAM authentication (`roles/datastore.user`).
3. Dual-mode (`auto` | `gcp` | `local`) dispatch so live Cloud Run deployments stream directly
   to BigQuery & Firestore while offline CI/pytest runs execute deterministically with zero
   external credentials.
4. LookML semantic layer catalog introspection over `looker/` models, views, explores, and dashboards.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import TYPE_CHECKING, Any, Literal
import urllib.parse
import uuid

import google.auth
from google.auth.exceptions import DefaultCredentialsError, RefreshError
from google.auth.transport.requests import Request as GoogleAuthRequest
import httpx

from edgar_capstone.dataset.gcs_storage import DEFAULT_LOCATION, DEFAULT_PROJECT_ID

if TYPE_CHECKING:
    from edgar_capstone.agents.schemas import ModelContextLatencyRecord

logger = logging.getLogger(__name__)
cloud_sink_logger = logging.getLogger("edgar_capstone.cloud_logging_sink")

TelemetryBackendMode = Literal["auto", "gcp", "local"]

DEFAULT_BIGQUERY_DATASET_ID = "sec_edgar"
DEFAULT_BQ_LLM_TELEMETRY_TABLE = "llm_telemetry_events"
DEFAULT_BQ_SESSION_TELEMETRY_TABLE = "session_telemetry_events"
DEFAULT_BQ_AUDIT_LOGS_TABLE = "security_audit_logs"
DEFAULT_BQ_TELEMETRY_COST_VIEW = "v_telemetry_cost_and_latency"

DEFAULT_FIRESTORE_DATABASE_ID = "(default)"
DEFAULT_FIRESTORE_FEEDBACK_COLLECTION = "analyst_feedback"

_SAFE_BQ_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


class TelemetrySinkError(RuntimeError):
    """Raised when streaming or persisting telemetry/feedback to GCP fails."""


class TelemetryRateLimitError(TelemetrySinkError):
    """Raised when BigQuery or Firestore returns HTTP 429 Too Many Requests."""

    def __init__(self, message: str, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class TelemetryTimeoutError(TelemetrySinkError, TimeoutError):
    """Raised when a BigQuery or Firestore REST request times out."""


def _validate_bq_id(value: str, label: str) -> str:
    cleaned = value.strip()
    if not cleaned or not _SAFE_BQ_IDENTIFIER_RE.match(cleaned):
        raise ValueError(f"Unsafe BigQuery {label} identifier: {value!r}")
    return cleaned


def _resolve_backend_mode(raw_mode: str, exec_mode: str) -> TelemetryBackendMode:
    norm = raw_mode.strip().lower()
    if norm in ("gcp", "live", "firestore", "bigquery"):
        return "gcp"
    if norm in ("local", "deterministic", "memory"):
        return "local"
    if exec_mode == "deterministic":
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
    bq_telemetry_cost_view: str = DEFAULT_BQ_TELEMETRY_COST_VIEW
    firestore_database_id: str = DEFAULT_FIRESTORE_DATABASE_ID
    firestore_feedback_collection: str = DEFAULT_FIRESTORE_FEEDBACK_COLLECTION
    gcp_auto_triggered: bool = False

    @classmethod
    def from_env(cls) -> TelemetryConfig:
        exec_mode = os.environ.get("EDGAR_EXECUTION_MODE", "auto").strip().lower()
        raw_telemetry = (
            os.environ.get("EDGAR_TELEMETRY_BACKEND")
            or os.environ.get("EDGAR_DATA_BACKEND")
            or "auto"
        )
        raw_feedback = os.environ.get("EDGAR_FEEDBACK_BACKEND") or raw_telemetry

        telemetry_backend = _resolve_backend_mode(raw_telemetry, exec_mode)
        feedback_backend = _resolve_backend_mode(raw_feedback, exec_mode)

        auto_signals = (
            os.environ.get("EDGAR_USE_GCP_TELEMETRY", "").strip().lower() in _TRUTHY_ENV_VALUES
            or os.environ.get("EDGAR_USE_GCP_TOOLS", "").strip().lower() in _TRUTHY_ENV_VALUES
            or bool(os.environ.get("K_SERVICE", "").strip())
            or bool(os.environ.get("BIGQUERY_DATASET_ID", "").strip())
            or bool(os.environ.get("FIRESTORE_DATABASE_ID", "").strip())
            or bool(os.environ.get("FIRESTORE_FEEDBACK_COLLECTION", "").strip())
            or exec_mode == "live"
            or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower()
            in _TRUTHY_ENV_VALUES
        )

        project_id = (
            os.environ.get("GCP_PROJECT_ID")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or DEFAULT_PROJECT_ID
        ).strip()
        location = (
            os.environ.get("GCP_REGION")
            or os.environ.get("GOOGLE_CLOUD_LOCATION")
            or DEFAULT_LOCATION
        ).strip()

        return cls(
            telemetry_backend=telemetry_backend,
            feedback_backend=feedback_backend,
            project_id=project_id,
            location=location,
            bq_dataset_id=(
                os.environ.get("BIGQUERY_DATASET_ID") or DEFAULT_BIGQUERY_DATASET_ID
            ).strip(),
            bq_llm_telemetry_table=(
                os.environ.get("BIGQUERY_LLM_TELEMETRY_TABLE")
                or DEFAULT_BQ_LLM_TELEMETRY_TABLE
            ).strip(),
            bq_session_telemetry_table=(
                os.environ.get("BIGQUERY_SESSION_TELEMETRY_TABLE")
                or DEFAULT_BQ_SESSION_TELEMETRY_TABLE
            ).strip(),
            bq_audit_logs_table=(
                os.environ.get("BIGQUERY_AUDIT_LOGS_TABLE") or DEFAULT_BQ_AUDIT_LOGS_TABLE
            ).strip(),
            bq_telemetry_cost_view=(
                os.environ.get("BIGQUERY_TELEMETRY_COST_VIEW")
                or DEFAULT_BQ_TELEMETRY_COST_VIEW
            ).strip(),
            firestore_database_id=(
                os.environ.get("FIRESTORE_DATABASE_ID") or DEFAULT_FIRESTORE_DATABASE_ID
            ).strip(),
            firestore_feedback_collection=(
                os.environ.get("FIRESTORE_FEEDBACK_COLLECTION")
                or DEFAULT_FIRESTORE_FEEDBACK_COLLECTION
            ).strip(),
            gcp_auto_triggered=auto_signals,
        )


def emit_cloud_logging_sink_record(sink_table: str, payload: dict[str, Any]) -> None:
    """Emit structured JSON log line targeted at Cloud Logging -> BigQuery project sinks."""
    envelope = {
        "severity": "INFO",
        "component": "sec-edgar-telemetry-sink",
        "log_sink_target": sink_table,
        "event_payload": payload,
    }
    cloud_sink_logger.info(json.dumps(envelope, default=str, separators=(",", ":")))


class _AuthenticatedRestTransport:
    """ADC OAuth2 bearer authentication and HTTP helper for BigQuery & Firestore REST APIs."""

    def __init__(
        self,
        *,
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
        default_timeout: float = 15.0,
    ) -> None:
        self._http_client = http_client
        self._credentials = credentials
        self._default_timeout = default_timeout

    def _get_bearer_token(self) -> str:
        if self._credentials is None:
            self._credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        if not getattr(self._credentials, "valid", False) or not getattr(
            self._credentials, "token", None
        ):
            self._credentials.refresh(GoogleAuthRequest())
        token = getattr(self._credentials, "token", None)
        if not token:
            raise DefaultCredentialsError(
                "Application Default Credentials did not return an OAuth2 bearer token."
            )
        return str(token)

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        token = self._get_bearer_token()
        headers: dict[str, str] = {"Authorization": f"Bearer {token}"}
        if json_body is not None:
            headers["Content-Type"] = "application/json"

        client = self._http_client or httpx.Client(timeout=self._default_timeout)
        close_client = self._http_client is None
        try:
            resp = client.request(
                method,
                url,
                json=json_body,
                params=params,
                headers=headers,
            )
        except httpx.TimeoutException as exc:
            raise TelemetryTimeoutError(f"Timed out calling GCP endpoint {url}: {exc}") from exc
        finally:
            if close_client:
                client.close()

        if resp.status_code == 429:
            raise TelemetryRateLimitError(
                f"GCP API HTTP 429 Too Many Requests for {url}: {resp.text[:300]}"
            )
        if resp.status_code >= 400:
            raise TelemetrySinkError(
                f"GCP REST request failed (HTTP {resp.status_code}) for {url}: {resp.text[:500]}"
            )
        if not resp.content:
            return {}
        return resp.json()


def _should_fallback_to_local(exc: BaseException, *, strict_gcp: bool) -> bool:
    if strict_gcp:
        return False
    if isinstance(
        exc,
        (
            DefaultCredentialsError,
            RefreshError,
            TelemetrySinkError,
            TelemetryRateLimitError,
            TelemetryTimeoutError,
            httpx.HTTPError,
            OSError,
        ),
    ):
        return True
    return False


def _bq_scalar_param(name: str, bq_type: str, value: Any) -> dict[str, Any]:
    return {
        "name": name,
        "parameterType": {"type": bq_type},
        "parameterValue": {"value": None if value is None else str(value)},
    }


def _decode_bq_cell(raw_val: Any, field_schema: dict[str, Any]) -> Any:
    mode = str(field_schema.get("mode") or "NULLABLE").upper()
    field_type = str(field_schema.get("type") or "STRING").upper()
    if mode == "REPEATED":
        if raw_val is None:
            return []
        elem_schema = {**field_schema, "mode": "NULLABLE"}
        items: list[Any] = []
        if isinstance(raw_val, list):
            for item in raw_val:
                inner = item.get("v") if isinstance(item, dict) and "v" in item else item
                items.append(_decode_bq_cell(inner, elem_schema))
        return items
    if raw_val is None:
        return None
    if field_type in ("INTEGER", "INT64"):
        return int(raw_val)
    if field_type in ("FLOAT", "FLOAT64", "NUMERIC", "BIGNUMERIC"):
        return float(raw_val)
    if field_type in ("BOOLEAN", "BOOL"):
        if isinstance(raw_val, bool):
            return raw_val
        return str(raw_val).strip().lower() == "true"
    return str(raw_val)


def _decode_bq_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    schema_fields = (data.get("schema") or {}).get("fields") or []
    raw_rows = data.get("rows") or []
    decoded_rows: list[dict[str, Any]] = []
    for row_obj in raw_rows:
        cells = row_obj.get("f")
        if not isinstance(cells, list):
            decoded_rows.append(dict(row_obj))
            continue
        row_dict: dict[str, Any] = {}
        for idx, f_schema in enumerate(schema_fields):
            col_name = str(f_schema.get("name") or f"col_{idx}")
            cell = cells[idx] if idx < len(cells) else {"v": None}
            raw_val = cell.get("v") if isinstance(cell, dict) and "v" in cell else cell
            row_dict[col_name] = _decode_bq_cell(raw_val, f_schema)
        decoded_rows.append(row_dict)
    return decoded_rows


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


# ---------------------------------------------------------------------------
# 1. BigQuery Streaming Telemetry & Security Audit Sink
# ---------------------------------------------------------------------------


@dataclass
class _SessionCumulativeTracker:
    started_at_epoch: float
    turn_count: int = 0
    distinct_tickers: set[str] = field(default_factory=set)
    reports_generated: int = 0


class BigQueryTelemetrySink(_AuthenticatedRestTransport):
    """Streams LLM token/cost/latency events, session analytics, and audit logs to BigQuery."""

    def __init__(
        self,
        config: TelemetryConfig | None = None,
        *,
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
        max_local_records: int = 500,
    ) -> None:
        super().__init__(http_client=http_client, credentials=credentials, default_timeout=15.0)
        self.config = config or TelemetryConfig.from_env()
        self.project_id = _validate_bq_id(self.config.project_id, "project_id")
        self.dataset_id = _validate_bq_id(self.config.bq_dataset_id, "dataset_id")
        self.llm_table = _validate_bq_id(
            self.config.bq_llm_telemetry_table, "llm_telemetry_table"
        )
        self.session_table = _validate_bq_id(
            self.config.bq_session_telemetry_table, "session_telemetry_table"
        )
        self.audit_table = _validate_bq_id(self.config.bq_audit_logs_table, "audit_logs_table")
        self.telemetry_cost_view = _validate_bq_id(
            self.config.bq_telemetry_cost_view, "telemetry_cost_view"
        )

        self._lock = threading.Lock()
        self._llm_events: deque[dict[str, Any]] = deque(maxlen=max_local_records)
        self._session_events: deque[dict[str, Any]] = deque(maxlen=max_local_records)
        self._audit_events: deque[dict[str, Any]] = deque(maxlen=max_local_records)
        self._sessions: dict[str, _SessionCumulativeTracker] = {}
        self._streamed_counts: dict[str, int] = defaultdict(int)
        self._fallback_counts: dict[str, int] = defaultdict(int)
        self._last_streamed_at: str | None = None
        self._last_error: str | None = None

    def _should_attempt_gcp(self) -> tuple[bool, bool]:
        """Return `(attempt_gcp, strict_gcp)`."""
        mode = self.config.telemetry_backend
        if mode == "local":
            return False, False
        if mode == "gcp":
            return True, True
        return self.config.gcp_auto_triggered, False

    def _table_ref(self, table_or_view: str) -> str:
        safe = _validate_bq_id(table_or_view, "table_or_view")
        return f"`{self.project_id}.{self.dataset_id}.{safe}`"

    def stream_rows(
        self,
        table_id: str,
        rows: list[tuple[str, dict[str, Any]]],
    ) -> dict[str, Any]:
        """Stream rows to BigQuery REST v2 `tabledata.insertAll` with deterministic `insertId`s."""
        if not rows:
            return {"inserted": 0, "table_id": table_id}
        safe_table = _validate_bq_id(table_id, "table_id")
        url = (
            f"https://bigquery.googleapis.com/bigquery/v2/projects/{self.project_id}"
            f"/datasets/{self.dataset_id}/tables/{safe_table}/insertAll"
        )
        payload = {
            "kind": "bigquery#tableDataInsertAllRequest",
            "skipInvalidRows": False,
            "ignoreUnknownValues": True,
            "rows": [
                {"insertId": insert_id, "json": row_data}
                for insert_id, row_data in rows
            ],
        }
        resp_data = self._request_json("POST", url, json_body=payload)
        insert_errors = resp_data.get("insertErrors") if isinstance(resp_data, dict) else None
        if insert_errors:
            raise TelemetrySinkError(
                f"BigQuery insertAll rejected rows for {self.dataset_id}.{safe_table}: "
                f"{json.dumps(insert_errors)[:400]}"
            )
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
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
        query_parameters: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a parameterized Standard SQL query against BigQuery sink tables/views."""
        url = f"https://bigquery.googleapis.com/bigquery/v2/projects/{self.project_id}/queries"
        payload: dict[str, Any] = {
            "query": sql,
            "useLegacySql": False,
            "location": self.config.location,
            "timeoutMs": 20000,
        }
        if query_parameters:
            payload["parameterMode"] = "NAMED"
            payload["queryParameters"] = query_parameters
        data = self._request_json("POST", url, json_body=payload)
        return _decode_bq_rows(data if isinstance(data, dict) else {})

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
    ) -> str:
        with self._lock:
            local_deque.appendleft(row)

        emit_cloud_logging_sink_record(table_id, row)

        attempt_gcp, strict_gcp = self._should_attempt_gcp()
        if not attempt_gcp:
            return "local_buffer"

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
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        eff_session_id = (session_id or "").strip() or f"ephemeral-{uuid.uuid4().hex[:10]}"
        norm_tickers = sorted({str(t).strip().upper() for t in tickers_analyzed if str(t).strip()})

        tools_used: list[str] = []
        for step in agent_trace:
            raw_tool = str(step.get("tool") or "").strip()
            if raw_tool:
                # Normalize e.g. 'query_looker_financial_explore(chunk_health)' -> canonical base or keep descriptive
                base_tool = raw_tool.split("(", 1)[0].strip()
                if base_tool:
                    tools_used.append(base_tool)

        with self._lock:
            tracker = self._sessions.get(eff_session_id)
            if tracker is None:
                tracker = _SessionCumulativeTracker(started_at_epoch=now_epoch)
                self._sessions[eff_session_id] = tracker
            tracker.turn_count += 1
            tracker.distinct_tickers.update(norm_tickers)
            if report_generated:
                tracker.reports_generated += 1
            turn_index = tracker.turn_count
            cum_turns = tracker.turn_count
            cum_companies = len(tracker.distinct_tickers)
            cum_reports = tracker.reports_generated
            elapsed_sec = round(max(0.0, now_epoch - tracker.started_at_epoch), 3)

        tot_ctx_tokens = sum(
            r.context_breakdown.total_context_tokens for r in turn_latency_records
        )
        cached_ctx_tokens = sum(
            r.context_breakdown.cached_context_tokens for r in turn_latency_records
        )
        out_tokens = sum(r.output_tokens + r.thoughts_tokens for r in turn_latency_records)
        tot_cost_usd = round(sum(r.total_cost_usd for r in turn_latency_records), 8)
        saved_cost_usd = round(
            sum(r.cached_token_savings_usd for r in turn_latency_records), 8
        )
        ttft_ms = (
            round(turn_latency_records[0].ttft_ms, 3)
            if turn_latency_records
            else round(total_latency_ms, 3)
        )

        turn_id = f"sess-turn-{uuid.uuid4().hex[:12]}"
        row: dict[str, Any] = {
            "turn_id": turn_id,
            "recorded_at": now_iso,
            "session_id": eff_session_id,
            "turn_index": turn_index,
            "user_id": user_id,
            "intent": intent,
            "execution_mode": execution_mode,
            "status": status_outcome,
            "tickers_analyzed": norm_tickers,
            "companies_analyzed_count": len(norm_tickers),
            "gics_sector": gics_sector or "ALL",
            "start_year": int(start_year),
            "end_year": int(end_year),
            "report_generated": bool(report_generated),
            "variance_reports_count": int(variance_reports_count),
            "citations_count": int(citations_count),
            "tool_invocations": tools_used,
            "tool_invocation_count": len(tools_used),
            "llm_call_count": len(turn_latency_records),
            "total_context_tokens": tot_ctx_tokens,
            "cached_context_tokens": cached_ctx_tokens,
            "output_tokens": out_tokens,
            "total_cost_usd": tot_cost_usd,
            "cached_token_savings_usd": saved_cost_usd,
            "ttft_ms": ttft_ms,
            "total_latency_ms": round(float(total_latency_ms), 3),
            "session_cumulative_turns": cum_turns,
            "session_cumulative_companies_count": cum_companies,
            "session_cumulative_reports_count": cum_reports,
            "session_elapsed_seconds": elapsed_sec,
            "fact_check_passed": bool(fact_check_passed),
            "evaluation_grade": evaluation_grade,
            "evaluation_score": (
                round(float(evaluation_score), 4) if evaluation_score is not None else None
            ),
        }
        delivery = self._deliver_row(
            table_id=self.session_table,
            insert_id=turn_id,
            row=row,
            local_deque=self._session_events,
        )
        return {"row": row, "delivery": delivery}

    def record_security_audit_log(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Stream a structured authentication or 10-K access audit log to `security_audit_logs`."""
        epoch = float(entry.get("recorded_at_epoch") or time.time())
        recorded_at = entry.get("recorded_at") or datetime.fromtimestamp(
            epoch, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        event_id = str(entry.get("event_id") or f"aud-{uuid.uuid4().hex[:12]}")
        row: dict[str, Any] = {
            "event_id": event_id,
            "recorded_at": recorded_at,
            "recorded_at_epoch": round(epoch, 3),
            "event_type": str(entry.get("event_type") or "unknown"),
            "status": str(entry.get("status") or "allowed"),
            "resource": str(entry.get("resource") or ""),
            "tickers": [str(t) for t in (entry.get("tickers") or [])],
            "user_email": str(entry.get("user_email") or "unauthenticated"),
            "user_sub": entry.get("user_sub"),
            "hosted_domain": entry.get("hosted_domain"),
            "identity_provider": entry.get("identity_provider"),
            "roles": [str(r) for r in (entry.get("roles") or [])],
            "cloud_identity_groups": [
                str(g) for g in (entry.get("cloud_identity_groups") or [])
            ],
            "agent_identity": entry.get("agent_identity"),
            "spiffe_id": entry.get("spiffe_id"),
            "iap_assertion_verified": bool(entry.get("iap_assertion_verified", False)),
            "detail": str(entry.get("detail") or ""),
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
        bounded = max(1, min(int(limit), 500))
        attempt_gcp, strict_gcp = self._should_attempt_gcp()
        if attempt_gcp:
            try:
                clauses: list[str] = []
                params = [_bq_scalar_param("limit", "INT64", bounded)]
                if session_id and session_id.strip():
                    clauses.append("session_id = @session_id")
                    params.append(
                        _bq_scalar_param("session_id", "STRING", session_id.strip())
                    )
                where_sql = f"WHERE {' AND '.join(clauses)} " if clauses else ""
                sql = (
                    "SELECT turn_id, CAST(recorded_at AS STRING) AS recorded_at, session_id, "
                    "turn_index, user_id, intent, execution_mode, status, tickers_analyzed, "
                    "companies_analyzed_count, gics_sector, start_year, end_year, "
                    "report_generated, variance_reports_count, citations_count, "
                    "tool_invocations, tool_invocation_count, llm_call_count, "
                    "total_context_tokens, cached_context_tokens, output_tokens, "
                    "total_cost_usd, cached_token_savings_usd, ttft_ms, total_latency_ms, "
                    "session_cumulative_turns, session_cumulative_companies_count, "
                    "session_cumulative_reports_count, session_elapsed_seconds, "
                    "fact_check_passed, evaluation_grade, evaluation_score "
                    f"FROM {self._table_ref(self.session_table)} "
                    f"{where_sql}"
                    "ORDER BY recorded_at DESC "
                    "LIMIT @limit"
                )
                return self.run_query(sql, params)
            except Exception as exc:
                if not _should_fallback_to_local(exc, strict_gcp=strict_gcp):
                    raise
                logger.warning(
                    "BigQuery query_session_telemetry failed in auto mode (%s); reading local buffer.",
                    exc,
                )

        with self._lock:
            items = list(self._session_events)
        if session_id and session_id.strip():
            items = [r for r in items if r.get("session_id") == session_id.strip()]
        return items[:bounded]

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

        latencies.sort()
        avg_turns_per_session = (
            round(
                sum(v[0] for v in sessions_snap.values()) / len(sessions_snap), 2
            )
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
            "p95_turn_latency_ms": round(_percentile(latencies, 95.0), 3),
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
        bounded = max(1, min(int(limit), 500))
        attempt_gcp, strict_gcp = self._should_attempt_gcp()
        if attempt_gcp:
            try:
                clauses: list[str] = []
                params = [_bq_scalar_param("limit", "INT64", bounded)]
                if event_type and event_type.strip():
                    clauses.append("event_type = @event_type")
                    params.append(
                        _bq_scalar_param("event_type", "STRING", event_type.strip())
                    )
                where_sql = f"WHERE {' AND '.join(clauses)} " if clauses else ""
                sql = (
                    "SELECT event_id, CAST(recorded_at AS STRING) AS recorded_at, "
                    "recorded_at_epoch, event_type, status, resource, tickers, "
                    "user_email, user_sub, hosted_domain, identity_provider, "
                    "roles, cloud_identity_groups, agent_identity, spiffe_id, "
                    "iap_assertion_verified, detail "
                    f"FROM {self._table_ref(self.audit_table)} "
                    f"{where_sql}"
                    "ORDER BY recorded_at_epoch DESC "
                    "LIMIT @limit"
                )
                return self.run_query(sql, params)
            except Exception as exc:
                if not _should_fallback_to_local(exc, strict_gcp=strict_gcp):
                    raise
                logger.warning(
                    "BigQuery query_security_audit_logs failed in auto mode (%s); reading local buffer.",
                    exc,
                )

        with self._lock:
            items = list(self._audit_events)
        if event_type and event_type.strip():
            items = [r for r in items if r.get("event_type") == event_type.strip()]
        return items[:bounded]

    def query_llm_telemetry_explore(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Query aggregated model/agent cost, cached-token savings, and latency rows for Looker."""
        bounded = max(1, min(int(limit), 200))
        attempt_gcp, strict_gcp = self._should_attempt_gcp()
        if attempt_gcp:
            try:
                sql = (
                    "SELECT CAST(telemetry_date AS STRING) AS telemetry_date, "
                    "model_name, agent_name, task_category, call_count, "
                    "total_context_tokens, cached_context_tokens, uncached_context_tokens, "
                    "output_tokens, cache_hit_rate, avg_total_latency_ms, "
                    "p95_total_latency_ms, avg_ttft_ms, p95_ttft_ms, "
                    "ttft_slo_compliance_rate, total_cost_usd, "
                    "uncached_baseline_cost_usd, cached_token_savings_usd "
                    f"FROM {self._table_ref(self.telemetry_cost_view)} "
                    "ORDER BY telemetry_date DESC, total_cost_usd DESC "
                    "LIMIT @limit"
                )
                return self.run_query(sql, [_bq_scalar_param("limit", "INT64", bounded)])
            except Exception as exc:
                if not _should_fallback_to_local(exc, strict_gcp=strict_gcp):
                    raise
                logger.warning(
                    "BigQuery query_llm_telemetry_explore failed in auto mode (%s); aggregating locally.",
                    exc,
                )

        with self._lock:
            events = list(self._llm_events)

        grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for ev in events:
            day = str(ev.get("recorded_at") or "")[:10] or datetime.now(timezone.utc).strftime(
                "%Y-%m-%d"
            )
            key = (
                day,
                str(ev.get("model_name") or "unknown"),
                str(ev.get("agent_name") or "unknown"),
                str(ev.get("task_category") or "financial_reasoning"),
            )
            grouped[key].append(ev)

        rows: list[dict[str, Any]] = []
        for (day, model_name, agent_name, task_cat), recs in sorted(
            grouped.items(), reverse=True
        ):
            tot_lats = sorted(float(r.get("total_model_latency_ms") or 0.0) for r in recs)
            ttfts = sorted(float(r.get("ttft_ms") or 0.0) for r in recs)
            hits = sum(1 for r in recs if r.get("cache_hit"))
            slo_ok = sum(1 for r in recs if r.get("within_ttft_slo", True))
            rows.append(
                {
                    "telemetry_date": day,
                    "model_name": model_name,
                    "agent_name": agent_name,
                    "task_category": task_cat,
                    "call_count": len(recs),
                    "total_context_tokens": sum(
                        int(r.get("total_context_tokens") or 0) for r in recs
                    ),
                    "cached_context_tokens": sum(
                        int(r.get("cached_context_tokens") or 0) for r in recs
                    ),
                    "uncached_context_tokens": sum(
                        int(r.get("uncached_context_tokens") or 0) for r in recs
                    ),
                    "output_tokens": sum(
                        int(r.get("output_tokens") or 0) + int(r.get("thoughts_tokens") or 0)
                        for r in recs
                    ),
                    "cache_hit_rate": round(hits / len(recs), 4),
                    "avg_total_latency_ms": round(sum(tot_lats) / len(tot_lats), 3),
                    "p95_total_latency_ms": round(_percentile(tot_lats, 95.0), 3),
                    "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 3),
                    "p95_ttft_ms": round(_percentile(ttfts, 95.0), 3),
                    "ttft_slo_compliance_rate": round(slo_ok / len(recs), 4),
                    "total_cost_usd": round(
                        sum(float(r.get("total_cost_usd") or 0.0) for r in recs), 8
                    ),
                    "uncached_baseline_cost_usd": round(
                        sum(float(r.get("uncached_baseline_cost_usd") or 0.0) for r in recs),
                        8,
                    ),
                    "cached_token_savings_usd": round(
                        sum(float(r.get("cached_token_savings_usd") or 0.0) for r in recs), 8
                    ),
                }
            )
        return rows[:bounded]

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
                "telemetry_cost_view": f"{self.project_id}.{self.dataset_id}.{self.telemetry_cost_view}",
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
            self._sessions.clear()
            self._streamed_counts.clear()
            self._fallback_counts.clear()
            self._last_streamed_at = None
            self._last_error = None


# ---------------------------------------------------------------------------
# 2. Google Cloud Firestore Persistent User Feedback Store
# ---------------------------------------------------------------------------


def encode_firestore_value(value: Any, *, field_name: str = "") -> dict[str, Any]:
    """Encode a Python value into Firestore REST v1 `Value` JSON wire representation."""
    if value is None:
        return {"nullValue": None}
    if isinstance(value, bool):
        return {"booleanValue": value}
    if isinstance(value, int):
        return {"integerValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        if field_name.endswith("_at") and _ISO_UTC_RE.match(value):
            return {"timestampValue": value}
        return {"stringValue": value}
    if isinstance(value, (list, tuple)):
        return {
            "arrayValue": {
                "values": [encode_firestore_value(v) for v in value]
            }
        }
    if isinstance(value, dict):
        return {
            "mapValue": {
                "fields": {
                    str(k): encode_firestore_value(v, field_name=str(k))
                    for k, v in value.items()
                }
            }
        }
    return {"stringValue": str(value)}


def encode_firestore_fields(record: dict[str, Any]) -> dict[str, Any]:
    return {
        str(k): encode_firestore_value(v, field_name=str(k))
        for k, v in record.items()
    }


def decode_firestore_value(val_obj: dict[str, Any]) -> Any:
    """Decode a Firestore REST v1 `Value` JSON object back into native Python types."""
    if "nullValue" in val_obj:
        return None
    if "booleanValue" in val_obj:
        return bool(val_obj["booleanValue"])
    if "integerValue" in val_obj:
        return int(val_obj["integerValue"])
    if "doubleValue" in val_obj:
        return float(val_obj["doubleValue"])
    if "timestampValue" in val_obj:
        return str(val_obj["timestampValue"])
    if "stringValue" in val_obj:
        return str(val_obj["stringValue"])
    if "arrayValue" in val_obj:
        arr = val_obj["arrayValue"] or {}
        return [decode_firestore_value(item) for item in (arr.get("values") or [])]
    if "mapValue" in val_obj:
        mp = val_obj["mapValue"] or {}
        return {
            str(k): decode_firestore_value(v)
            for k, v in (mp.get("fields") or {}).items()
        }
    return None


def decode_firestore_document(doc: dict[str, Any]) -> dict[str, Any]:
    """Decode a Firestore REST v1 `Document` payload into a flat Python dictionary."""
    fields = doc.get("fields") or {}
    decoded = {str(k): decode_firestore_value(v) for k, v in fields.items()}
    if doc.get("name") and "document_name" not in decoded:
        decoded["document_name"] = str(doc["name"])
    return decoded


class FirestoreFeedbackStore(_AuthenticatedRestTransport):
    """Persists analyst thumbs-up/down ratings and audit comments to Google Cloud Firestore."""

    def __init__(
        self,
        config: TelemetryConfig | None = None,
        *,
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
    ) -> None:
        super().__init__(http_client=http_client, credentials=credentials, default_timeout=15.0)
        self.config = config or TelemetryConfig.from_env()
        self.project_id = self.config.project_id
        self.database_id = self.config.firestore_database_id
        self.collection_id = self.config.firestore_feedback_collection
        self._lock = threading.Lock()
        self._local_feedback: list[dict[str, Any]] = []
        self._persisted_firestore_count = 0
        self._last_error: str | None = None

    def _should_attempt_gcp(self) -> tuple[bool, bool]:
        mode = self.config.feedback_backend
        if mode == "local":
            return False, False
        if mode == "gcp":
            return True, True
        return self.config.gcp_auto_triggered, False

    def _documents_base_url(self) -> str:
        encoded_db = urllib.parse.quote(self.database_id, safe="()")
        return (
            f"https://firestore.googleapis.com/v1/projects/{urllib.parse.quote(self.project_id, safe='')}"
            f"/databases/{encoded_db}/documents"
        )

    def record_feedback(
        self,
        *,
        message_id: str,
        rating: Literal["up", "down"],
        comment: str = "",
        query: str = "",
        user_email: str,
        identity_provider: str | None = None,
        session_id: str | None = None,
        tickers: list[str] | None = None,
        intent: str = "",
    ) -> dict[str, Any]:
        """Persist an analyst feedback entry to Firestore (with local fallback in offline/auto mode)."""
        now_epoch = round(time.time(), 3)
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        feedback_id = f"fbk-{uuid.uuid4().hex[:12]}"
        norm_tickers = [
            str(t).strip().upper() for t in (tickers or []) if str(t).strip()
        ]

        entry: dict[str, Any] = {
            "feedback_id": feedback_id,
            "message_id": message_id.strip(),
            "session_id": session_id.strip() if session_id and session_id.strip() else None,
            "rating": rating,
            "comment": comment.strip(),
            "query": query.strip(),
            "tickers": norm_tickers,
            "intent": intent.strip(),
            "user_email": user_email,
            "identity_provider": identity_provider or "unknown",
            "recorded_at": now_iso,
            "recorded_at_epoch": now_epoch,
        }

        emit_cloud_logging_sink_record(self.collection_id, entry)

        document_name = (
            f"projects/{self.project_id}/databases/{self.database_id}"
            f"/documents/{self.collection_id}/{feedback_id}"
        )
        backend_used = "local_memory"
        attempt_gcp, strict_gcp = self._should_attempt_gcp()

        if attempt_gcp:
            url = (
                f"{self._documents_base_url()}/"
                f"{urllib.parse.quote(self.collection_id, safe='')}"
            )
            doc_body = {"fields": encode_firestore_fields(entry)}
            try:
                created_doc = self._request_json(
                    "POST",
                    url,
                    json_body=doc_body,
                    params={"documentId": feedback_id},
                )
                if isinstance(created_doc, dict) and created_doc.get("name"):
                    document_name = str(created_doc["name"])
                backend_used = "firestore"
                with self._lock:
                    self._persisted_firestore_count += 1
                    self._last_error = None
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                if not _should_fallback_to_local(exc, strict_gcp=strict_gcp):
                    raise
                logger.warning(
                    "Firestore feedback write failed in auto mode (%s); persisting in local fallback store.",
                    exc,
                )
                backend_used = "local_memory_fallback"

        with self._lock:
            self._local_feedback.append(entry)
            total_count = len(self._local_feedback)

        return {
            "entry": entry,
            "backend_used": backend_used,
            "collection": self.collection_id,
            "database_id": self.database_id,
            "document_name": document_name,
            "total_feedback_count": total_count,
        }

    def list_feedback(
        self,
        *,
        limit: int = 50,
        rating: Literal["up", "down"] | None = None,
    ) -> dict[str, Any]:
        """List persisted analyst feedback entries from Firestore `:runQuery` or local fallback."""
        bounded = max(1, min(int(limit), 500))
        attempt_gcp, strict_gcp = self._should_attempt_gcp()
        items: list[dict[str, Any]] | None = None
        source = "local_memory"

        if attempt_gcp:
            structured_query: dict[str, Any] = {
                "from": [{"collectionId": self.collection_id}],
                "orderBy": [
                    {
                        "field": {"fieldPath": "recorded_at_epoch"},
                        "direction": "DESCENDING",
                    }
                ],
                "limit": bounded,
            }
            if rating in ("up", "down"):
                structured_query["where"] = {
                    "fieldFilter": {
                        "field": {"fieldPath": "rating"},
                        "op": "EQUAL",
                        "value": {"stringValue": rating},
                    }
                }
            url = f"{self._documents_base_url()}:runQuery"
            try:
                raw_results = self._request_json(
                    "POST", url, json_body={"structuredQuery": structured_query}
                )
                parsed: list[dict[str, Any]] = []
                if isinstance(raw_results, list):
                    for row in raw_results:
                        if isinstance(row, dict) and isinstance(row.get("document"), dict):
                            parsed.append(decode_firestore_document(row["document"]))
                items = parsed
                source = "firestore"
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                if not _should_fallback_to_local(exc, strict_gcp=strict_gcp):
                    raise
                logger.warning(
                    "Firestore feedback query failed in auto mode (%s); reading local fallback store.",
                    exc,
                )
                source = "local_memory_fallback"

        if items is None:
            with self._lock:
                local_copy = list(reversed(self._local_feedback))
            if rating in ("up", "down"):
                local_copy = [x for x in local_copy if x.get("rating") == rating]
            items = local_copy[:bounded]

        up_count = sum(1 for x in items if x.get("rating") == "up")
        down_count = sum(1 for x in items if x.get("rating") == "down")
        total = len(items)
        return {
            "source": source,
            "database_id": self.database_id,
            "collection": self.collection_id,
            "feedback_count": total,
            "thumbs_up_count": up_count,
            "thumbs_down_count": down_count,
            "positive_rating_ratio": round(up_count / total, 4) if total > 0 else 1.0,
            "feedback": items,
        }

    def status_metadata(self) -> dict[str, Any]:
        attempt_gcp, strict_gcp = self._should_attempt_gcp()
        with self._lock:
            local_cnt = len(self._local_feedback)
            fs_cnt = self._persisted_firestore_count
            last_err = self._last_error
        return {
            "backend_mode": self.config.feedback_backend,
            "live_firestore_enabled": attempt_gcp,
            "strict_gcp": strict_gcp,
            "project_id": self.project_id,
            "database_id": self.database_id,
            "collection": self.collection_id,
            "persisted_firestore_count": fs_cnt,
            "buffered_feedback_count": local_cnt,
            "last_error": last_err,
        }

    def reset_local_buffers(self) -> None:
        with self._lock:
            self._local_feedback.clear()
            self._persisted_firestore_count = 0
            self._last_error = None


# ---------------------------------------------------------------------------
# 3. Singleton Registry & Dependency Injection Helpers
# ---------------------------------------------------------------------------

_CUSTOM_TELEMETRY_SINK: BigQueryTelemetrySink | None = None
_CUSTOM_FEEDBACK_STORE: FirestoreFeedbackStore | None = None
_ENV_TELEMETRY_CACHE: tuple[TelemetryConfig, BigQueryTelemetrySink] | None = None
_ENV_FEEDBACK_CACHE: tuple[TelemetryConfig, FirestoreFeedbackStore] | None = None
_REGISTRY_LOCK = threading.Lock()


def configure_telemetry_stores(
    *,
    telemetry_sink: BigQueryTelemetrySink | None = None,
    feedback_store: FirestoreFeedbackStore | None = None,
) -> None:
    """Explicitly inject `BigQueryTelemetrySink` and/or `FirestoreFeedbackStore` (for testing)."""
    global _CUSTOM_TELEMETRY_SINK, _CUSTOM_FEEDBACK_STORE
    with _REGISTRY_LOCK:
        _CUSTOM_TELEMETRY_SINK = telemetry_sink
        _CUSTOM_FEEDBACK_STORE = feedback_store


def reset_telemetry_stores() -> None:
    """Reset injected and environment-cached telemetry sink & feedback store singletons."""
    global _CUSTOM_TELEMETRY_SINK, _CUSTOM_FEEDBACK_STORE, _ENV_TELEMETRY_CACHE, _ENV_FEEDBACK_CACHE
    with _REGISTRY_LOCK:
        if _CUSTOM_TELEMETRY_SINK is not None:
            _CUSTOM_TELEMETRY_SINK.reset_local_buffers()
        if _CUSTOM_FEEDBACK_STORE is not None:
            _CUSTOM_FEEDBACK_STORE.reset_local_buffers()
        if _ENV_TELEMETRY_CACHE is not None:
            _ENV_TELEMETRY_CACHE[1].reset_local_buffers()
        if _ENV_FEEDBACK_CACHE is not None:
            _ENV_FEEDBACK_CACHE[1].reset_local_buffers()
        _CUSTOM_TELEMETRY_SINK = None
        _CUSTOM_FEEDBACK_STORE = None
        _ENV_TELEMETRY_CACHE = None
        _ENV_FEEDBACK_CACHE = None


def get_telemetry_sink() -> BigQueryTelemetrySink:
    """Return the active `BigQueryTelemetrySink` instance."""
    global _ENV_TELEMETRY_CACHE
    with _REGISTRY_LOCK:
        if _CUSTOM_TELEMETRY_SINK is not None:
            return _CUSTOM_TELEMETRY_SINK
        cfg = TelemetryConfig.from_env()
        if _ENV_TELEMETRY_CACHE is None or _ENV_TELEMETRY_CACHE[0] != cfg:
            _ENV_TELEMETRY_CACHE = (cfg, BigQueryTelemetrySink(cfg))
        return _ENV_TELEMETRY_CACHE[1]


def get_feedback_store() -> FirestoreFeedbackStore:
    """Return the active `FirestoreFeedbackStore` instance."""
    global _ENV_FEEDBACK_CACHE
    with _REGISTRY_LOCK:
        if _CUSTOM_FEEDBACK_STORE is not None:
            return _CUSTOM_FEEDBACK_STORE
        cfg = TelemetryConfig.from_env()
        if _ENV_FEEDBACK_CACHE is None or _ENV_FEEDBACK_CACHE[0] != cfg:
            _ENV_FEEDBACK_CACHE = (cfg, FirestoreFeedbackStore(cfg))
        return _ENV_FEEDBACK_CACHE[1]


# ---------------------------------------------------------------------------
# 4. LookML Project Catalog Introspection (`looker/`)
# ---------------------------------------------------------------------------

_LOOKML_EXPLORE_RE = re.compile(r"^\s*explore:\s*([A-Za-z0-9_]+)\s*\{", re.MULTILINE)
_LOOKML_VIEW_RE = re.compile(r"^\s*view:\s*([A-Za-z0-9_]+)\s*\{", re.MULTILINE)
_LOOKML_DIMENSION_RE = re.compile(
    r"^\s*(?:dimension|dimension_group):\s*([A-Za-z0-9_]+)\s*\{", re.MULTILINE
)
_LOOKML_MEASURE_RE = re.compile(r"^\s*measure:\s*([A-Za-z0-9_]+)\s*\{", re.MULTILINE)
_LOOKML_SQL_TABLE_RE = re.compile(r"sql_table_name:\s*([^;]+);;", re.MULTILINE)
_LOOKML_CONNECTION_RE = re.compile(r'connection:\s*"([^"]+)"')
_LOOKML_DASHBOARD_RE = re.compile(r"^\s*-\s*dashboard:\s*([A-Za-z0-9_]+)", re.MULTILINE)
_LOOKML_TITLE_RE = re.compile(r"^\s*title:\s*(.+)$", re.MULTILINE)
_LOOKML_ELEMENT_NAME_RE = re.compile(r"^\s*-\s*name:\s*([A-Za-z0-9_]+)", re.MULTILINE)


def resolve_looker_project_dir() -> Path:
    """Locate the repository `looker/` semantic layer directory."""
    env_dir = os.environ.get("EDGAR_LOOKML_DIR", "").strip()
    if env_dir:
        return Path(env_dir).resolve()
    candidates = [
        Path.cwd() / "looker",
        Path(__file__).resolve().parents[2] / "looker",
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand.resolve()
    return candidates[-1].resolve()


def load_lookml_project_catalog(looker_dir: Path | None = None) -> dict[str, Any]:
    """Parse LookML models, views, explores, and dashboards in `looker/` into a catalog."""
    root = (looker_dir or resolve_looker_project_dir()).resolve()
    if not root.is_dir():
        return {
            "status": "missing",
            "looker_dir": str(root),
            "models": [],
            "views": [],
            "dashboards": [],
        }

    manifest_path = root / "manifest.lkml"
    models: list[dict[str, Any]] = []
    for model_file in sorted((root / "models").glob("*.model.lkml")) if (root / "models").is_dir() else []:
        content = model_file.read_text(encoding="utf-8")
        conn_match = _LOOKML_CONNECTION_RE.search(content)
        explores = _LOOKML_EXPLORE_RE.findall(content)
        models.append(
            {
                "model_name": model_file.name.replace(".model.lkml", ""),
                "file": f"looker/models/{model_file.name}",
                "connection": conn_match.group(1) if conn_match else None,
                "explores": explores,
                "explore_count": len(explores),
            }
        )

    views: list[dict[str, Any]] = []
    for view_file in sorted((root / "views").glob("*.view.lkml")) if (root / "views").is_dir() else []:
        content = view_file.read_text(encoding="utf-8")
        view_names = _LOOKML_VIEW_RE.findall(content)
        sql_table_match = _LOOKML_SQL_TABLE_RE.search(content)
        dims = _LOOKML_DIMENSION_RE.findall(content)
        measures = _LOOKML_MEASURE_RE.findall(content)
        views.append(
            {
                "view_name": view_names[0] if view_names else view_file.stem.replace(".view", ""),
                "file": f"looker/views/{view_file.name}",
                "sql_table_name": (
                    sql_table_match.group(1).strip() if sql_table_match else None
                ),
                "dimensions": dims,
                "measures": measures,
                "dimension_count": len(dims),
                "measure_count": len(measures),
            }
        )

    dashboards: list[dict[str, Any]] = []
    for dash_file in (
        sorted((root / "dashboards").glob("*.dashboard.lookml"))
        if (root / "dashboards").is_dir()
        else []
    ):
        content = dash_file.read_text(encoding="utf-8")
        dash_ids = _LOOKML_DASHBOARD_RE.findall(content)
        titles = _LOOKML_TITLE_RE.findall(content)
        tiles = _LOOKML_ELEMENT_NAME_RE.findall(content)
        dashboards.append(
            {
                "dashboard_id": (
                    dash_ids[0] if dash_ids else dash_file.name.replace(".dashboard.lookml", "")
                ),
                "title": titles[0].strip().strip("\"'") if titles else "",
                "file": f"looker/dashboards/{dash_file.name}",
                "elements": tiles,
                "element_count": len(tiles),
            }
        )

    return {
        "status": "ok",
        "looker_dir": str(root),
        "manifest_present": manifest_path.is_file(),
        "model_count": len(models),
        "view_count": len(views),
        "dashboard_count": len(dashboards),
        "models": models,
        "views": views,
        "dashboards": dashboards,
    }
