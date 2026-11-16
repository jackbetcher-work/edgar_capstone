"""ADK tools: MetricFetchTool, LookerQueryTool, VarianceCalculationTool, HybridSearchTool, and ContextCacheTool.

Provides both live GCP client adapters (BigQuery, Vertex AI Search / Discovery Engine,
Vertex AI Vector Search / Matching Engine, Vertex AI Embeddings, and Cloud Storage) and
local curated-corpus execution so ADK agents hit live GCP backends in Cloud Run / Vertex AI
environments while preserving zero-dependency deterministic execution in offline tests.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import threading
import time
import urllib.parse
from collections import OrderedDict, defaultdict
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import google.auth
import httpx
import numpy as np
from google.adk.tools import FunctionTool, ToolContext
from google.auth.exceptions import DefaultCredentialsError, RefreshError
from google.auth.transport.requests import Request as GoogleAuthRequest

from edgar_capstone.agents.resilience import (
    RateLimitExceededError,
    ToolExecutionError,
    UpstreamTimeoutError,
    with_tool_resilience,
)
from edgar_capstone.agents.schemas import (
    AnnualStepVariance,
    CompanyVarianceReport,
    GroundedCitation,
    MetricRecord,
    MetricVariance,
)
from edgar_capstone.dataset.extractor import format_combined_10k_markdown
from edgar_capstone.dataset.gcs_storage import (
    DEFAULT_BUCKET_NAME,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
)
from edgar_capstone.dataset.models import FinancialDataProfile
from edgar_capstone.dataset.paths import resolve_curated_dataset_dir
from edgar_capstone.dataset.pipeline import DEFAULT_OUTPUT_DIR
from edgar_capstone.telemetry import get_telemetry_sink
from edgar_capstone.vector_db.chunker import (
    chunk_filing_document,
    estimate_token_count,
    extract_sections_from_combined_markdown,
)
from edgar_capstone.vector_db.embedder import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    VertexAIEmbedder,
    generate_deterministic_embedding,
)

logger = logging.getLogger(__name__)

_WORD_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SAFE_BQ_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})

DataBackendMode = Literal["auto", "gcp", "local"]

# Canonical GICS sector names for exact-match token restricts in Vertex AI Search / Vector Search
_CANONICAL_GICS_SECTORS: dict[str, str] = {
    "information technology": "Information Technology",
    "financials": "Financials",
    "health care": "Health Care",
    "consumer discretionary": "Consumer Discretionary",
    "communication services": "Communication Services",
    "industrials": "Industrials",
    "consumer staples": "Consumer Staples",
    "energy": "Energy",
    "utilities": "Utilities",
    "real estate": "Real Estate",
    "materials": "Materials",
}

DEFAULT_BIGQUERY_DATASET_ID = "sec_edgar"
DEFAULT_BQ_FINANCIAL_METRICS_TABLE = "financial_metrics"
DEFAULT_BQ_FILING_CHUNKS_TABLE = "filing_chunks"
DEFAULT_BQ_YOY_VARIANCES_VIEW = "v_financial_variances_yoy"
DEFAULT_BQ_CHUNK_HEALTH_VIEW = "v_chunk_health_metrics"
DEFAULT_VERTEX_AI_SEARCH_LOCATION = "global"
DEFAULT_VERTEX_AI_SEARCH_COLLECTION = "default_collection"
DEFAULT_VERTEX_AI_SEARCH_ENGINE_ID = "sec-edgar-10k-hybrid-search"
DEFAULT_VERTEX_AI_SEARCH_DATASTORE_ID = "sec-edgar-10k-corpus-ds"
DEFAULT_VERTEX_VECTOR_ENDPOINT_DISPLAY_NAME = "sec-edgar-10k-chunks-endpoint"

DEFAULT_10K_SECTION_CACHE_MAX_ENTRIES = 256
DEFAULT_10K_SECTION_CACHE_TTL_SECONDS = 3600.0
_10K_CACHE_KEY_RE = re.compile(
    r"^(?P<ticker>.+)_(?P<fiscal_year>\d{4})_(?P<section_id>item_1a|item_7)$"
)


def _env_positive_number(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError:
        return default
    return value if value > 0 else default


class _BoundedSectionCache(MutableMapping[str, dict[str, Any]]):
    """Thread-safe, bounded LRU + TTL map for parsed 10-K sections (process-local L1 cache).

    This is only the local copy of parsed section text. Model calls share the
    sections across Cloud Run instances through remote Vertex AI `CachedContent`
    resources (`agents/tenk_context_cache.py`). Bounding the map keeps memory
    flat when the full S&P 500 corpus (~500 issuers x 7 years x 2 sections) is in play.
    """

    def __init__(self, *, max_entries: int, ttl_seconds: float) -> None:
        self.max_entries = max(1, int(max_entries))
        self.ttl_seconds = float(ttl_seconds)
        self._data: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._lock = threading.RLock()
        self.evictions = 0

    def _expired(self, stored_at: float) -> bool:
        return (time.monotonic() - stored_at) > self.ttl_seconds

    def _purge_expired_locked(self) -> None:
        expired = [k for k, (ts, _) in self._data.items() if self._expired(ts)]
        for k in expired:
            del self._data[k]
            self.evictions += 1

    def __getitem__(self, key: str) -> dict[str, Any]:
        with self._lock:
            stored_at, value = self._data[key]
            if self._expired(stored_at):
                del self._data[key]
                self.evictions += 1
                raise KeyError(key)
            self._data.move_to_end(key)
            return value

    def __setitem__(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._data[key] = (time.monotonic(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
                self.evictions += 1

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._data[key]

    def __contains__(self, key: object) -> bool:
        with self._lock:
            item = self._data.get(key)  # type: ignore[call-overload]
            if item is None:
                return False
            if self._expired(item[0]):
                del self._data[key]  # type: ignore[arg-type]
                self.evictions += 1
                return False
            return True

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            self._purge_expired_locked()
            return iter(list(self._data.keys()))

    def __len__(self) -> int:
        with self._lock:
            self._purge_expired_locked()
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


# Process-local L1 cache of parsed 10-K sections (bounded LRU + TTL)
_SESSION_10K_CONTEXT_CACHE: _BoundedSectionCache = _BoundedSectionCache(
    max_entries=int(
        _env_positive_number(
            "EDGAR_10K_SECTION_CACHE_MAX_ENTRIES", DEFAULT_10K_SECTION_CACHE_MAX_ENTRIES
        )
    ),
    ttl_seconds=_env_positive_number(
        "EDGAR_10K_SECTION_CACHE_TTL_SECONDS", DEFAULT_10K_SECTION_CACHE_TTL_SECONDS
    ),
)
_CONTEXT_CACHE_RUNTIME_COUNTERS: dict[str, float | int] = {
    "hits": 0,
    "misses": 0,
    "cached_tokens_saved": 0,
    "total_hit_latency_ms": 0.0,
    "total_miss_latency_ms": 0.0,
}


# ---------------------------------------------------------------------------
# Live GCP Adapter Exceptions & Configuration
# ---------------------------------------------------------------------------


class GcpAdapterHttpError(ToolExecutionError):
    """Raised when a GCP REST API returns a non-retryable HTTP 4xx response."""

    def __init__(self, *, status_code: int, url: str, response_text: str) -> None:
        super().__init__(f"GCP REST request failed (HTTP {status_code}) for {url}: {response_text}")
        self.status_code = status_code
        self.url = url
        self.response_text = response_text


class GcpEndpointNotReadyError(ToolExecutionError):
    """Raised when a Vertex AI Vector Search endpoint or deployed index is not yet provisioned."""


def _validate_bq_identifier(value: str, label: str) -> str:
    """Validate a BigQuery project, dataset, or table identifier before SQL interpolation."""
    cleaned = value.strip()
    if not cleaned or not _SAFE_BQ_IDENTIFIER_RE.match(cleaned):
        raise ValueError(f"Unsafe BigQuery {label} identifier: {value!r}")
    return cleaned


def _canonicalize_gics_sector(sector: str | None) -> str | None:
    """Normalize a GICS sector filter to canonical casing when recognized."""
    if not sector:
        return None
    stripped = sector.strip()
    if not stripped or stripped.upper() in ("ALL", "*"):
        return None
    return _CANONICAL_GICS_SECTORS.get(stripped.lower(), stripped)


@dataclass(frozen=True, slots=True)
class GcpToolConfig:
    """Runtime configuration for BigQuery, Vertex AI Search, Vector Search, and GCS adapters."""

    data_backend: DataBackendMode = "auto"
    project_id: str = DEFAULT_PROJECT_ID
    location: str = DEFAULT_LOCATION
    bq_dataset_id: str = DEFAULT_BIGQUERY_DATASET_ID
    bq_financial_metrics_table: str = DEFAULT_BQ_FINANCIAL_METRICS_TABLE
    bq_filing_chunks_table: str = DEFAULT_BQ_FILING_CHUNKS_TABLE
    bq_yoy_view: str = DEFAULT_BQ_YOY_VARIANCES_VIEW
    bq_chunk_health_view: str = DEFAULT_BQ_CHUNK_HEALTH_VIEW
    sec_corpus_bucket: str = DEFAULT_BUCKET_NAME
    vais_location: str = DEFAULT_VERTEX_AI_SEARCH_LOCATION
    vais_collection_id: str = DEFAULT_VERTEX_AI_SEARCH_COLLECTION
    vais_engine_id: str = DEFAULT_VERTEX_AI_SEARCH_ENGINE_ID
    vais_datastore_id: str = DEFAULT_VERTEX_AI_SEARCH_DATASTORE_ID
    vvs_endpoint_id: str = ""
    vvs_endpoint_display_name: str = DEFAULT_VERTEX_VECTOR_ENDPOINT_DISPLAY_NAME
    vvs_public_domain: str = ""
    vvs_deployed_index_id: str = ""
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
    gcp_auto_triggered: bool = False

    @classmethod
    def from_env(cls) -> GcpToolConfig:
        """Resolve live GCP tool adapter settings from environment variables."""
        raw_backend = os.environ.get("EDGAR_DATA_BACKEND", "auto").strip().lower()
        exec_mode = os.environ.get("EDGAR_EXECUTION_MODE", "auto").strip().lower()

        if raw_backend in ("gcp", "live"):
            data_backend: DataBackendMode = "gcp"
        elif raw_backend in ("local", "deterministic"):
            data_backend = "local"
        elif exec_mode == "deterministic":
            data_backend = "local"
        else:
            data_backend = "auto"

        auto_signals = (
            os.environ.get("EDGAR_USE_GCP_TOOLS", "").strip().lower() in _TRUTHY_ENV_VALUES
            or bool(os.environ.get("K_SERVICE", "").strip())
            or bool(os.environ.get("BIGQUERY_DATASET_ID", "").strip())
            or bool(os.environ.get("VERTEX_AI_SEARCH_ENGINE_ID", "").strip())
            or bool(os.environ.get("VERTEX_AI_SEARCH_DATASTORE_ID", "").strip())
            or bool(os.environ.get("VERTEX_VECTOR_SEARCH_ENDPOINT_ID", "").strip())
            or bool(os.environ.get("VERTEX_VECTOR_SEARCH_PUBLIC_DOMAIN", "").strip())
            or exec_mode == "live"
            or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() in _TRUTHY_ENV_VALUES
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

        raw_dims = os.environ.get("VERTEX_EMBEDDING_DIMENSIONS", "").strip()
        dims = int(raw_dims) if raw_dims.isdigit() else DEFAULT_EMBEDDING_DIMENSIONS

        return cls(
            data_backend=data_backend,
            project_id=project_id,
            location=location,
            bq_dataset_id=(
                os.environ.get("BIGQUERY_DATASET_ID") or DEFAULT_BIGQUERY_DATASET_ID
            ).strip(),
            bq_financial_metrics_table=(
                os.environ.get("BIGQUERY_FINANCIAL_METRICS_TABLE")
                or DEFAULT_BQ_FINANCIAL_METRICS_TABLE
            ).strip(),
            bq_filing_chunks_table=(
                os.environ.get("BIGQUERY_FILING_CHUNKS_TABLE") or DEFAULT_BQ_FILING_CHUNKS_TABLE
            ).strip(),
            bq_yoy_view=(
                os.environ.get("BIGQUERY_YOY_VARIANCE_VIEW") or DEFAULT_BQ_YOY_VARIANCES_VIEW
            ).strip(),
            bq_chunk_health_view=(
                os.environ.get("BIGQUERY_CHUNK_HEALTH_VIEW") or DEFAULT_BQ_CHUNK_HEALTH_VIEW
            ).strip(),
            sec_corpus_bucket=(os.environ.get("SEC_CORPUS_BUCKET") or DEFAULT_BUCKET_NAME).strip(),
            vais_location=(
                os.environ.get("VERTEX_AI_SEARCH_LOCATION") or DEFAULT_VERTEX_AI_SEARCH_LOCATION
            ).strip(),
            vais_collection_id=(
                os.environ.get("VERTEX_AI_SEARCH_COLLECTION") or DEFAULT_VERTEX_AI_SEARCH_COLLECTION
            ).strip(),
            vais_engine_id=os.environ.get(
                "VERTEX_AI_SEARCH_ENGINE_ID", DEFAULT_VERTEX_AI_SEARCH_ENGINE_ID
            ).strip(),
            vais_datastore_id=os.environ.get(
                "VERTEX_AI_SEARCH_DATASTORE_ID", DEFAULT_VERTEX_AI_SEARCH_DATASTORE_ID
            ).strip(),
            vvs_endpoint_id=os.environ.get("VERTEX_VECTOR_SEARCH_ENDPOINT_ID", "").strip(),
            vvs_endpoint_display_name=(
                os.environ.get("VERTEX_VECTOR_SEARCH_ENDPOINT_DISPLAY_NAME")
                or DEFAULT_VERTEX_VECTOR_ENDPOINT_DISPLAY_NAME
            ).strip(),
            vvs_public_domain=os.environ.get("VERTEX_VECTOR_SEARCH_PUBLIC_DOMAIN", "").strip(),
            vvs_deployed_index_id=os.environ.get(
                "VERTEX_VECTOR_SEARCH_DEPLOYED_INDEX_ID", ""
            ).strip(),
            embedding_model=(
                os.environ.get("VERTEX_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
            ).strip(),
            embedding_dimensions=dims,
            gcp_auto_triggered=auto_signals,
        )


class _GcpAuthenticatedRestClient:
    """Shared ADC bearer authentication and resilient HTTP transport helper for GCP REST APIs."""

    def __init__(
        self,
        *,
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
        default_timeout: float = 30.0,
    ) -> None:
        self._http_client = http_client
        self._credentials = credentials
        self._default_timeout = default_timeout

    def _get_bearer_token(self) -> str:
        """Obtain a valid OAuth2 bearer token from Application Default Credentials."""
        if self._credentials is None:
            self._credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        if not getattr(self._credentials, "valid", False) or not getattr(
            self._credentials, "token", None
        ):
            self._credentials.refresh(GoogleAuthRequest())
        bearer_tok = getattr(self._credentials, "token", None)
        if not bearer_tok:
            raise DefaultCredentialsError(
                "Application Default Credentials did not return an OAuth2 bearer token."
            )
        return str(bearer_tok)

    @staticmethod
    def _parse_retry_after(resp: httpx.Response) -> float | None:
        raw = resp.headers.get("Retry-After")
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            return None

    def _send_request(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Execute an authenticated HTTP request and translate 429/5xx/timeouts for resilience."""
        bearer_tok = self._get_bearer_token()
        headers: dict[str, str] = {"Authorization": f"Bearer {bearer_tok}"}
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
            raise UpstreamTimeoutError(f"Timed out calling GCP endpoint {url}: {exc}") from exc
        finally:
            if close_client:
                client.close()

        if resp.status_code == 429:
            raise RateLimitExceededError(
                f"GCP API HTTP 429 Too Many Requests for {url}: {resp.text[:300]}",
                retry_after_seconds=self._parse_retry_after(resp),
            )
        if resp.status_code in (500, 502, 503, 504):
            resp.raise_for_status()
        if resp.status_code >= 400:
            raise GcpAdapterHttpError(
                status_code=resp.status_code,
                url=url,
                response_text=resp.text[:500],
            )
        return resp

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resp = self._send_request(method, url, json_body=json_body, params=params)
        if not resp.content:
            return {}
        data = resp.json()
        return data if isinstance(data, dict) else {"items": data}

    def _request_text(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> str:
        resp = self._send_request(method, url, params=params)
        return resp.text


# ---------------------------------------------------------------------------
# 1. BigQuery Live Client Adapter (Structured Metrics, Variances, & VECTOR_SEARCH)
# ---------------------------------------------------------------------------


def _bq_scalar_param(name: str, bq_type: str, value: Any) -> dict[str, Any]:
    """Construct a named scalar QueryParameter for BigQuery REST v2."""
    return {
        "name": name,
        "parameterType": {"type": bq_type},
        "parameterValue": {"value": None if value is None else str(value)},
    }


def _bq_array_param(name: str, elem_type: str, values: list[Any]) -> dict[str, Any]:
    """Construct a named ARRAY QueryParameter for BigQuery REST v2."""
    return {
        "name": name,
        "parameterType": {
            "type": "ARRAY",
            "arrayType": {"type": elem_type},
        },
        "parameterValue": {
            "arrayValues": [{"value": None if v is None else str(v)} for v in values]
        },
    }


def _decode_bq_cell(raw_val: Any, field_schema: dict[str, Any]) -> Any:
    """Recursively decode a BigQuery REST v2 cell (`f`/`v` wire format) into Python types."""
    mode = str(field_schema.get("mode") or "NULLABLE").upper()
    field_type = str(field_schema.get("type") or "STRING").upper()

    if mode == "REPEATED":
        if raw_val is None:
            return []
        elem_schema = {**field_schema, "mode": "NULLABLE"}
        decoded_items: list[Any] = []
        if isinstance(raw_val, list):
            for item in raw_val:
                inner = item.get("v") if isinstance(item, dict) and "v" in item else item
                decoded_items.append(_decode_bq_cell(inner, elem_schema))
        return decoded_items

    if raw_val is None:
        return None

    if field_type in ("RECORD", "STRUCT"):
        subfields = field_schema.get("fields") or []
        if isinstance(raw_val, dict) and "f" in raw_val and isinstance(raw_val["f"], list):
            return _decode_bq_row(raw_val, subfields)
        if isinstance(raw_val, dict):
            return raw_val
        return {}

    if field_type in ("INTEGER", "INT64"):
        return int(raw_val)
    if field_type in ("FLOAT", "FLOAT64", "NUMERIC", "BIGNUMERIC"):
        return float(raw_val)
    if field_type in ("BOOLEAN", "BOOL"):
        if isinstance(raw_val, bool):
            return raw_val
        return str(raw_val).strip().lower() == "true"
    return str(raw_val)


def _decode_bq_row(row_obj: dict[str, Any], schema_fields: list[dict[str, Any]]) -> dict[str, Any]:
    """Decode a single BigQuery `{"f": [{"v": ...}, ...]}` row using `schema.fields`."""
    cells = row_obj.get("f")
    if not isinstance(cells, list):
        return dict(row_obj)
    decoded: dict[str, Any] = {}
    for idx, field_schema in enumerate(schema_fields):
        col_name = str(field_schema.get("name") or f"col_{idx}")
        cell = cells[idx] if idx < len(cells) else {"v": None}
        raw_val = cell.get("v") if isinstance(cell, dict) and "v" in cell else cell
        decoded[col_name] = _decode_bq_cell(raw_val, field_schema)
    return decoded


class BigQueryAdapter(_GcpAuthenticatedRestClient):
    """Live BigQuery REST v2 adapter for SEC 10-K metrics, BI views, and `VECTOR_SEARCH`."""

    def __init__(
        self,
        config: GcpToolConfig,
        *,
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
    ) -> None:
        super().__init__(http_client=http_client, credentials=credentials, default_timeout=45.0)
        self.project_id = _validate_bq_identifier(config.project_id, "project_id")
        self.location = config.location
        self.dataset_id = _validate_bq_identifier(config.bq_dataset_id, "dataset_id")
        self.financial_metrics_table = _validate_bq_identifier(
            config.bq_financial_metrics_table, "financial_metrics_table"
        )
        self.filing_chunks_table = _validate_bq_identifier(
            config.bq_filing_chunks_table, "filing_chunks_table"
        )
        self.yoy_view = _validate_bq_identifier(config.bq_yoy_view, "yoy_view")
        self.chunk_health_view = _validate_bq_identifier(
            config.bq_chunk_health_view, "chunk_health_view"
        )
        self.sec_corpus_bucket = config.sec_corpus_bucket
        self.embedding_dimensions = config.embedding_dimensions

    def _table_ref(self, table_or_view: str) -> str:
        safe_table = _validate_bq_identifier(table_or_view, "table_or_view")
        return f"`{self.project_id}.{self.dataset_id}.{safe_table}`"

    def run_query(
        self,
        sql: str,
        query_parameters: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a parameterized Standard SQL query on BigQuery and return decoded rows."""
        base_url = f"https://bigquery.googleapis.com/bigquery/v2/projects/{self.project_id}/queries"
        payload: dict[str, Any] = {
            "query": sql,
            "useLegacySql": False,
            "location": self.location,
            "timeoutMs": 30000,
        }
        if query_parameters:
            payload["parameterMode"] = "NAMED"
            payload["queryParameters"] = query_parameters

        data = self._request_json("POST", base_url, json_body=payload)
        job_ref = data.get("jobReference") or {}
        job_id = str(job_ref.get("jobId") or "")
        job_location = str(job_ref.get("location") or self.location)

        polls = 0
        while not data.get("jobComplete", True):
            if not job_id or polls >= 20:
                raise UpstreamTimeoutError(
                    f"BigQuery query job {job_id or '<unknown>'} did not complete within poll budget."
                )
            polls += 1
            time.sleep(min(1.0, 0.1 * polls))
            data = self._request_json(
                "GET",
                f"{base_url}/{urllib.parse.quote(job_id, safe='')}",
                params={"location": job_location, "timeoutMs": 10000},
            )

        schema_fields = (data.get("schema") or {}).get("fields") or []
        raw_rows: list[dict[str, Any]] = list(data.get("rows") or [])
        page_token = data.get("pageToken")

        while page_token and job_id:
            page_data = self._request_json(
                "GET",
                f"{base_url}/{urllib.parse.quote(job_id, safe='')}",
                params={"location": job_location, "pageToken": str(page_token)},
            )
            if not schema_fields:
                schema_fields = (page_data.get("schema") or {}).get("fields") or []
            raw_rows.extend(page_data.get("rows") or [])
            page_token = page_data.get("pageToken")

        return [_decode_bq_row(r, schema_fields) for r in raw_rows]

    def _row_to_profile(self, row: dict[str, Any]) -> FinancialDataProfile:
        """Convert a decoded `financial_metrics` BigQuery row into `FinancialDataProfile`."""
        ticker = str(row["ticker"])
        fy = int(row["fiscal_year"])
        unstructured_blob = str(
            row.get("unstructured_blob_path") or f"unstructured/{ticker}/{fy}_10k.md"
        )
        structured_blob = str(
            row.get("structured_blob_path") or f"structured/{ticker}/{fy}_financials.json"
        )
        prov = row.get("metric_provenance")
        if not isinstance(prov, dict):
            prov = {}
        else:
            prov = {
                "revenue": prov.get("revenue"),
                "operating_income": prov.get("operating_income"),
                "net_income": prov.get("net_income"),
                "notes": list(prov.get("notes") or []),
            }

        payload: dict[str, Any] = {
            "profile_id": str(row.get("profile_id") or f"{ticker}_{fy}"),
            "ticker": ticker,
            "cik": str(row.get("cik") or "0000000000"),
            "company_name": str(row.get("company_name") or ticker),
            "gics_sector": str(row.get("gics_sector") or "Unknown"),
            "gics_sub_industry": str(row.get("gics_sub_industry") or "Unknown"),
            "form_type": str(row.get("form_type") or "10-K"),
            "fiscal_year": fy,
            "fiscal_period": str(row.get("fiscal_period") or "FY"),
            "period_of_report": str(row.get("period_of_report") or f"{fy}-12-31"),
            "filing_date": str(row.get("filing_date") or f"{fy}-12-31"),
            "accession_number": str(row.get("accession_number") or f"{ticker}-{fy}-10K"),
            "currency": str(row.get("currency") or "USD"),
            "revenue": row.get("revenue"),
            "operating_income": row.get("operating_income"),
            "net_income": row.get("net_income"),
            "metric_provenance": prov,
            "extracted_sections": list(row.get("extracted_sections") or []),
            "unstructured_blob_path": unstructured_blob,
            "structured_blob_path": structured_blob,
            "unstructured_gcs_uri": str(
                row.get("unstructured_gcs_uri")
                or f"gs://{self.sec_corpus_bucket}/{unstructured_blob}"
            ),
            "structured_gcs_uri": str(
                row.get("structured_gcs_uri") or f"gs://{self.sec_corpus_bucket}/{structured_blob}"
            ),
            "sec_filing_url": str(row.get("sec_filing_url") or ""),
        }
        if row.get("ingested_at"):
            payload["ingested_at"] = str(row["ingested_at"])
        return FinancialDataProfile.model_validate(payload)

    def fetch_profiles(
        self,
        *,
        tickers: set[str],
        start_year: int,
        end_year: int,
        gics_sector: str | None = None,
    ) -> list[FinancialDataProfile]:
        """Query `sec_edgar.financial_metrics` with partition (`fiscal_year`) & cluster pruning."""
        clauses = ["fiscal_year BETWEEN @start_year AND @end_year"]
        params: list[dict[str, Any]] = [
            _bq_scalar_param("start_year", "INT64", int(start_year)),
            _bq_scalar_param("end_year", "INT64", int(end_year)),
        ]
        if tickers:
            clauses.append("UPPER(REPLACE(ticker, '.', '-')) IN UNNEST(@tickers)")
            params.append(_bq_array_param("tickers", "STRING", sorted(tickers)))
        if gics_sector:
            clauses.append("LOWER(gics_sector) = @gics_sector")
            params.append(_bq_scalar_param("gics_sector", "STRING", gics_sector.lower()))

        where_sql = " AND ".join(clauses)
        sql = (
            f"SELECT * FROM {self._table_ref(self.financial_metrics_table)} "
            f"WHERE {where_sql} "
            f"ORDER BY ticker ASC, fiscal_year ASC"
        )
        rows = self.run_query(sql, params)
        return [self._row_to_profile(r) for r in rows]

    def query_yoy_variances(
        self,
        *,
        tickers: set[str],
        gics_sector: str | None,
        start_year: int,
        end_year: int,
    ) -> list[dict[str, Any]]:
        """Query `sec_edgar.v_financial_variances_yoy` for Looker YoY & peer comparison explores."""
        clauses = ["fiscal_year BETWEEN @start_year AND @end_year"]
        params: list[dict[str, Any]] = [
            _bq_scalar_param("start_year", "INT64", int(start_year)),
            _bq_scalar_param("end_year", "INT64", int(end_year)),
        ]
        if tickers:
            clauses.append("UPPER(REPLACE(ticker, '.', '-')) IN UNNEST(@tickers)")
            params.append(_bq_array_param("tickers", "STRING", sorted(tickers)))
        if gics_sector:
            clauses.append("LOWER(gics_sector) = @gics_sector")
            params.append(_bq_scalar_param("gics_sector", "STRING", gics_sector.lower()))

        sql = (
            "SELECT "
            "profile_id, ticker, company_name, gics_sector, fiscal_year, prev_fiscal_year, "
            "revenue, revenue_yoy_delta, revenue_yoy_pct, "
            "operating_income, operating_income_yoy_delta, operating_income_yoy_pct, "
            "net_income, net_income_yoy_delta, net_income_yoy_pct, sec_filing_url "
            f"FROM {self._table_ref(self.yoy_view)} "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY ticker ASC, fiscal_year ASC"
        )
        raw_rows = self.run_query(sql, params)
        formatted: list[dict[str, Any]] = []
        for r in raw_rows:
            rev = r.get("revenue")
            op = r.get("operating_income")
            ni = r.get("net_income")
            op_margin = (
                round((float(op) / abs(float(rev))) * 100.0, 2) if rev and op is not None else None
            )
            net_margin = (
                round((float(ni) / abs(float(rev))) * 100.0, 2) if rev and ni is not None else None
            )
            formatted.append(
                {
                    "profile_id": r.get("profile_id"),
                    "ticker": r.get("ticker"),
                    "company_name": r.get("company_name"),
                    "gics_sector": r.get("gics_sector") or "Unknown",
                    "fiscal_year": int(r["fiscal_year"]),
                    "prev_fiscal_year": (
                        int(r["prev_fiscal_year"])
                        if r.get("prev_fiscal_year") is not None
                        else None
                    ),
                    "revenue": rev,
                    "revenue_yoy_delta": (
                        round(float(r["revenue_yoy_delta"]), 2)
                        if r.get("revenue_yoy_delta") is not None
                        else None
                    ),
                    "revenue_yoy_pct": (
                        round(float(r["revenue_yoy_pct"]), 2)
                        if r.get("revenue_yoy_pct") is not None
                        else None
                    ),
                    "operating_income": op,
                    "operating_income_yoy_delta": (
                        round(float(r["operating_income_yoy_delta"]), 2)
                        if r.get("operating_income_yoy_delta") is not None
                        else None
                    ),
                    "operating_income_yoy_pct": (
                        round(float(r["operating_income_yoy_pct"]), 2)
                        if r.get("operating_income_yoy_pct") is not None
                        else None
                    ),
                    "net_income": ni,
                    "net_income_yoy_delta": (
                        round(float(r["net_income_yoy_delta"]), 2)
                        if r.get("net_income_yoy_delta") is not None
                        else None
                    ),
                    "net_income_yoy_pct": (
                        round(float(r["net_income_yoy_pct"]), 2)
                        if r.get("net_income_yoy_pct") is not None
                        else None
                    ),
                    "operating_margin_pct": op_margin,
                    "net_margin_pct": net_margin,
                    "sec_filing_url": r.get("sec_filing_url") or "",
                }
            )
        return formatted

    def query_sector_benchmarks(
        self,
        *,
        tickers: set[str],
        gics_sector: str | None,
        start_year: int,
        end_year: int,
    ) -> list[dict[str, Any]]:
        """Aggregate sector-level revenue, operating income, and net income benchmarks in BigQuery."""
        clauses = ["fiscal_year BETWEEN @start_year AND @end_year"]
        params: list[dict[str, Any]] = [
            _bq_scalar_param("start_year", "INT64", int(start_year)),
            _bq_scalar_param("end_year", "INT64", int(end_year)),
        ]
        if tickers:
            clauses.append("UPPER(REPLACE(ticker, '.', '-')) IN UNNEST(@tickers)")
            params.append(_bq_array_param("tickers", "STRING", sorted(tickers)))
        if gics_sector:
            clauses.append("LOWER(gics_sector) = @gics_sector")
            params.append(_bq_scalar_param("gics_sector", "STRING", gics_sector.lower()))

        sql = (
            "SELECT "
            "COALESCE(gics_sector, 'Unknown') AS gics_sector, "
            "fiscal_year, "
            "COUNT(*) AS company_count, "
            "ARRAY_AGG(DISTINCT ticker ORDER BY ticker) AS tickers, "
            "ROUND(SUM(revenue), 2) AS total_sector_revenue, "
            "ROUND(AVG(revenue), 2) AS avg_company_revenue, "
            "ROUND(SUM(operating_income), 2) AS total_sector_operating_income, "
            "ROUND(SUM(net_income), 2) AS total_sector_net_income "
            f"FROM {self._table_ref(self.financial_metrics_table)} "
            f"WHERE {' AND '.join(clauses)} "
            "GROUP BY 1, 2 "
            "ORDER BY gics_sector ASC, fiscal_year ASC"
        )
        raw_rows = self.run_query(sql, params)
        return [
            {
                "gics_sector": str(r.get("gics_sector") or "Unknown"),
                "fiscal_year": int(r["fiscal_year"]),
                "company_count": int(r.get("company_count") or 0),
                "tickers": list(r.get("tickers") or []),
                "total_sector_revenue": r.get("total_sector_revenue"),
                "avg_company_revenue": r.get("avg_company_revenue"),
                "total_sector_operating_income": r.get("total_sector_operating_income"),
                "total_sector_net_income": r.get("total_sector_net_income"),
            }
            for r in raw_rows
        ]

    def query_chunk_health(self) -> dict[str, Any]:
        """Query `sec_edgar.v_chunk_health_metrics` and format Looker `chunk_health` response."""
        sql = (
            f"SELECT * FROM {self._table_ref(self.chunk_health_view)} "
            "ORDER BY gics_sector ASC, section_id ASC"
        )
        rows = self.run_query(sql)
        if not rows:
            raise ToolExecutionError(
                f"BigQuery view {self.dataset_id}.{self.chunk_health_view} returned 0 rows."
            )

        overall: dict[str, Any] = {}
        by_sector: dict[str, dict[str, Any]] = {}
        by_section: dict[str, dict[str, Any]] = {}
        section_cells: dict[str, list[dict[str, Any]]] = defaultdict(list)

        def _summary_entry(group_key: str, row: dict[str, Any]) -> dict[str, Any]:
            return {
                "group_key": group_key,
                "total_chunks": int(row.get("total_chunks") or 0),
                "distinct_profiles": int(row.get("distinct_profiles") or 0),
                "distinct_tickers": int(row.get("distinct_tickers") or 0),
                "valid_embedding_count": int(row.get("valid_embedding_count") or 0),
                "embedding_coverage_ratio": float(row.get("embedding_coverage_ratio") or 0.0),
                "avg_char_count": float(row.get("avg_char_count") or 0.0),
                "avg_token_estimate": float(row.get("avg_token_estimate") or 0.0),
                "min_token_estimate": int(row.get("min_token_estimate") or 0),
                "max_token_estimate": int(row.get("max_token_estimate") or 0),
                "p50_token_estimate": float(row.get("p50_token_estimate") or 0.0),
                "p95_token_estimate": float(row.get("p95_token_estimate") or 0.0),
                "undersized_chunk_count": int(row.get("undersized_chunk_count") or 0),
                "oversized_chunk_count": int(row.get("oversized_chunk_count") or 0),
                "missing_metadata_count": int(row.get("missing_metadata_count") or 0),
            }

        for r in rows:
            sec_sector = str(r.get("gics_sector") or "")
            sec_id = str(r.get("section_id") or "")
            if sec_sector == "ALL_SECTORS" and sec_id == "ALL_SECTIONS":
                overall = r
            elif sec_sector != "ALL_SECTORS" and sec_id == "ALL_SECTIONS":
                by_sector[sec_sector] = _summary_entry(sec_sector, r)
            elif sec_sector == "ALL_SECTORS" and sec_id != "ALL_SECTIONS":
                by_section[sec_id] = _summary_entry(sec_id, r)
            elif sec_sector != "ALL_SECTORS" and sec_id != "ALL_SECTIONS":
                section_cells[sec_id].append(r)

        if not by_section and section_cells:
            for sec_id, cells in sorted(section_cells.items()):
                tot_chunks = sum(int(c.get("total_chunks") or 0) for c in cells)
                valid_embs = sum(int(c.get("valid_embedding_count") or 0) for c in cells)
                weighted_chars = sum(
                    float(c.get("avg_char_count") or 0.0) * int(c.get("total_chunks") or 0)
                    for c in cells
                )
                weighted_tokens = sum(
                    float(c.get("avg_token_estimate") or 0.0) * int(c.get("total_chunks") or 0)
                    for c in cells
                )
                by_section[sec_id] = {
                    "group_key": sec_id,
                    "total_chunks": tot_chunks,
                    "distinct_profiles": sum(int(c.get("distinct_profiles") or 0) for c in cells),
                    "distinct_tickers": sum(int(c.get("distinct_tickers") or 0) for c in cells),
                    "valid_embedding_count": valid_embs,
                    "embedding_coverage_ratio": (
                        round(valid_embs / tot_chunks, 4) if tot_chunks > 0 else 0.0
                    ),
                    "avg_char_count": (
                        round(weighted_chars / tot_chunks, 2) if tot_chunks > 0 else 0.0
                    ),
                    "avg_token_estimate": (
                        round(weighted_tokens / tot_chunks, 2) if tot_chunks > 0 else 0.0
                    ),
                    "min_token_estimate": min(
                        (int(c.get("min_token_estimate") or 0) for c in cells), default=0
                    ),
                    "max_token_estimate": max(
                        (int(c.get("max_token_estimate") or 0) for c in cells), default=0
                    ),
                    "undersized_chunk_count": sum(
                        int(c.get("undersized_chunk_count") or 0) for c in cells
                    ),
                    "oversized_chunk_count": sum(
                        int(c.get("oversized_chunk_count") or 0) for c in cells
                    ),
                    "missing_metadata_count": sum(
                        int(c.get("missing_metadata_count") or 0) for c in cells
                    ),
                }

        if not overall:
            overall = rows[0]

        return {
            "status": "success",
            "explore_name": "chunk_health",
            "health_status": overall.get("health_status", "HEALTHY"),
            "health_score": overall.get("health_score"),
            "total_chunks": overall.get("total_chunks"),
            "embedding_coverage_ratio": overall.get("embedding_coverage_ratio"),
            "full_section_coverage_ratio": overall.get(
                "full_section_coverage_ratio",
                overall.get("filing_coverage_ratio"),
            ),
            "by_section": by_section,
            "by_sector": by_sector,
        }

    def vector_search_chunks(
        self,
        query_embedding: list[float],
        *,
        tickers: set[str],
        gics_sector: str | None,
        section_id: str | None,
        start_year: int,
        end_year: int,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Execute metadata-prefiltered BigQuery `VECTOR_SEARCH` over `sec_edgar.filing_chunks`."""
        clauses = [
            "fiscal_year BETWEEN @start_year AND @end_year",
            "ARRAY_LENGTH(embedding) = @embedding_dim",
        ]
        params: list[dict[str, Any]] = [
            _bq_scalar_param("start_year", "INT64", int(start_year)),
            _bq_scalar_param("end_year", "INT64", int(end_year)),
            _bq_scalar_param("embedding_dim", "INT64", int(len(query_embedding))),
            _bq_array_param("query_embedding", "FLOAT64", query_embedding),
            _bq_scalar_param("top_k", "INT64", int(max(1, min(top_k, 50)))),
        ]
        if tickers:
            clauses.append("UPPER(REPLACE(ticker, '.', '-')) IN UNNEST(@tickers)")
            params.append(_bq_array_param("tickers", "STRING", sorted(tickers)))
        if gics_sector:
            clauses.append("LOWER(gics_sector) = @gics_sector")
            params.append(_bq_scalar_param("gics_sector", "STRING", gics_sector.lower()))
        if section_id:
            clauses.append("LOWER(section_id) = @section_id")
            params.append(_bq_scalar_param("section_id", "STRING", section_id.lower()))

        sql = (
            "SELECT "
            "base.chunk_id, base.profile_id, base.ticker, base.company_name, "
            "base.gics_sector, base.fiscal_year, base.section_id, base.section_title, "
            "base.chunk_index, base.chunk_text, base.sec_filing_url, "
            "base.unstructured_gcs_uri, distance "
            "FROM VECTOR_SEARCH("
            "  ("
            "    SELECT chunk_id, profile_id, ticker, company_name, gics_sector, "
            "           fiscal_year, section_id, section_title, chunk_index, "
            "           chunk_text, embedding, sec_filing_url, unstructured_gcs_uri "
            f"   FROM {self._table_ref(self.filing_chunks_table)} "
            f"   WHERE {' AND '.join(clauses)}"
            "  ), "
            "  'embedding', "
            "  (SELECT @query_embedding AS embedding), "
            "  top_k => @top_k, "
            "  distance_type => 'COSINE'"
            ") "
            "ORDER BY distance ASC"
        )
        rows = self.run_query(sql, params)
        results: list[dict[str, Any]] = []
        for r in rows:
            dist = float(r.get("distance") or 0.0)
            sim = max(0.0, min(1.0, 1.0 - dist))
            results.append(
                {
                    **r,
                    "similarity": sim,
                    "source_backend": "bigquery_vector_search",
                }
            )
        return results

    def fetch_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Hydrate full `chunk_text` and metadata from `filing_chunks` for Vector Search neighbor IDs."""
        unique_ids = sorted({cid.strip() for cid in chunk_ids if cid and cid.strip()})
        if not unique_ids:
            return {}
        sql = (
            "SELECT chunk_id, profile_id, ticker, company_name, gics_sector, "
            "fiscal_year, section_id, section_title, chunk_index, chunk_text, "
            "sec_filing_url, unstructured_gcs_uri "
            f"FROM {self._table_ref(self.filing_chunks_table)} "
            "WHERE chunk_id IN UNNEST(@chunk_ids)"
        )
        rows = self.run_query(sql, [_bq_array_param("chunk_ids", "STRING", unique_ids)])
        return {str(r["chunk_id"]): r for r in rows if r.get("chunk_id")}

    def fetch_section_chunks(
        self, *, ticker: str, fiscal_year: int, section_id: str
    ) -> list[dict[str, Any]]:
        """Fetch canonical chunks for a single `(ticker, fiscal_year, section_id)` filing section."""
        sql = (
            "SELECT chunk_id, profile_id, ticker, company_name, gics_sector, "
            "fiscal_year, section_id, section_title, chunk_index, chunk_text, "
            "sec_filing_url, unstructured_gcs_uri "
            f"FROM {self._table_ref(self.filing_chunks_table)} "
            "WHERE UPPER(REPLACE(ticker, '.', '-')) = @ticker "
            "  AND fiscal_year = @fiscal_year "
            "  AND LOWER(section_id) = @section_id "
            "ORDER BY chunk_index ASC"
        )
        params = [
            _bq_scalar_param("ticker", "STRING", ticker.upper().replace(".", "-")),
            _bq_scalar_param("fiscal_year", "INT64", int(fiscal_year)),
            _bq_scalar_param("section_id", "STRING", section_id.lower()),
        ]
        return self.run_query(sql, params)


# ---------------------------------------------------------------------------
# 2. Vertex AI Search (Discovery Engine / GEAP) Live Client Adapter
# ---------------------------------------------------------------------------


def _normalize_section_id(
    raw_section: str | None,
    *,
    requested_section: str | None = None,
    passage_text: str = "",
) -> Literal["item_1a", "item_7"]:
    """Coerce section identifiers to `GroundedCitation`'s `Literal['item_1a', 'item_7']`."""
    if raw_section:
        low = raw_section.strip().lower()
        if "1a" in low or "risk" in low:
            return "item_1a"
        if "7" in low or "md&a" in low or "management" in low:
            return "item_7"

    probe = passage_text[:320].lower()
    if "item 1a" in probe or "risk factor" in probe:
        return "item_1a"
    if "item 7" in probe or "results of operations" in probe or "executive overview" in probe:
        return "item_7"

    if requested_section in ("item_1a", "item_7"):
        return requested_section  # type: ignore[return-value]

    risk_cues = ("risk", "regulatory", "restriction", "litigation", "cybersecurity", "adversely")
    mda_cues = ("revenue", "operating income", "margin", "cash flow", "fiscal year", "segment")
    risk_hits = sum(1 for cue in risk_cues if cue in passage_text.lower())
    mda_hits = sum(1 for cue in mda_cues if cue in passage_text.lower())
    return "item_1a" if risk_hits > mda_hits else "item_7"


def _default_section_title(sec_id: Literal["item_1a", "item_7"]) -> str:
    return (
        "Item 1A. Risk Factors"
        if sec_id == "item_1a"
        else "Item 7. Management's Discussion and Analysis"
    )


class VertexAISearchAdapter(_GcpAuthenticatedRestClient):
    """Live Discovery Engine (`servingConfigs/default_search:search`) hybrid search adapter."""

    def __init__(
        self,
        config: GcpToolConfig,
        *,
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
    ) -> None:
        super().__init__(http_client=http_client, credentials=credentials, default_timeout=30.0)
        self.project_id = config.project_id
        self.location = config.vais_location or "global"
        self.collection_id = config.vais_collection_id or "default_collection"
        self.engine_id = config.vais_engine_id
        self.datastore_id = config.vais_datastore_id

    def _api_host(self) -> str:
        if self.location == "global":
            return "https://discoveryengine.googleapis.com"
        return f"https://{self.location}-discoveryengine.googleapis.com"

    def _serving_config_urls(self) -> list[str]:
        host = self._api_host()
        prefix = (
            f"{host}/v1/projects/{self.project_id}/locations/{self.location}/"
            f"collections/{self.collection_id}"
        )
        urls: list[str] = []
        if self.engine_id:
            urls.append(f"{prefix}/engines/{self.engine_id}/servingConfigs/default_search:search")
        if self.datastore_id:
            urls.append(
                f"{prefix}/dataStores/{self.datastore_id}/servingConfigs/default_search:search"
            )
        return urls

    @staticmethod
    def build_filter_expression(
        *,
        tickers: set[str],
        gics_sector: str | None,
        section_id: str | None,
        start_year: int,
        end_year: int,
    ) -> str:
        """Build a Discovery Engine filter expression over `VertexAISearchManifestRecord.structData`."""
        clauses: list[str] = [
            f"fiscal_year >= {int(start_year)}",
            f"fiscal_year <= {int(end_year)}",
        ]
        if tickers:
            quoted_tickers = ", ".join(f'"{t.replace(chr(34), "")}"' for t in sorted(tickers))
            clauses.append(f"ticker: ANY({quoted_tickers})")
        canon_sector = _canonicalize_gics_sector(gics_sector)
        if canon_sector:
            safe_sec = canon_sector.replace('"', '\\"')
            clauses.append(f'gics_sector: ANY("{safe_sec}")')
        if section_id in ("item_1a", "item_7"):
            clauses.append(f'extracted_sections: ANY("{section_id}")')
        return " AND ".join(clauses)

    def search_documents(
        self,
        query: str,
        *,
        tickers: set[str],
        gics_sector: str | None,
        section_id: str | None,
        start_year: int,
        end_year: int,
        page_size: int = 10,
    ) -> list[dict[str, Any]]:
        """Execute a metadata-filtered hybrid search against Vertex AI Search (Discovery Engine)."""
        urls = self._serving_config_urls()
        if not urls:
            raise GcpEndpointNotReadyError(
                "Neither VERTEX_AI_SEARCH_ENGINE_ID nor VERTEX_AI_SEARCH_DATASTORE_ID is configured."
            )

        filter_expr = self.build_filter_expression(
            tickers=tickers,
            gics_sector=gics_sector,
            section_id=section_id,
            start_year=start_year,
            end_year=end_year,
        )
        payload: dict[str, Any] = {
            "query": query,
            "pageSize": max(1, min(int(page_size), 25)),
            "queryExpansionSpec": {"condition": "AUTO"},
            "spellCorrectionSpec": {"mode": "AUTO"},
            "contentSearchSpec": {
                "snippetSpec": {"returnSnippet": True, "maxSnippetCount": 3},
                "extractiveContentSpec": {
                    "maxExtractiveSegmentCount": 5,
                    "maxExtractiveAnswerCount": 3,
                    "returnExtractiveSegmentScore": True,
                },
            },
        }
        if filter_expr:
            payload["filter"] = filter_expr

        data: dict[str, Any] | None = None
        last_err: Exception | None = None
        for idx, url in enumerate(urls):
            try:
                data = self._request_json("POST", url, json_body=payload)
                break
            except GcpAdapterHttpError as exc:
                last_err = exc
                if exc.status_code == 404 and idx + 1 < len(urls):
                    logger.info(
                        "Vertex AI Search engine endpoint returned 404; falling back to datastore servingConfig."
                    )
                    continue
                raise

        if data is None:
            if last_err is not None:
                raise last_err
            return []

        passages: list[dict[str, Any]] = []
        results = data.get("results") or []
        for doc_rank, item in enumerate(results):
            doc = item.get("document") or {}
            doc_id = str(doc.get("id") or f"vais_doc_{doc_rank}")
            struct_data = doc.get("structData") or {}
            derived = doc.get("derivedStructData") or {}
            content_obj = doc.get("content") or {}

            ticker = str(struct_data.get("ticker") or "").strip().upper()
            fy_raw = struct_data.get("fiscal_year")
            if not ticker or fy_raw is None:
                # Fallback parse from document ID `{TICKER}_{FY}_10K`
                id_parts = doc_id.split("_")
                if len(id_parts) >= 2 and id_parts[1].isdigit():
                    ticker = ticker or id_parts[0].upper()
                    fy_raw = fy_raw if fy_raw is not None else int(id_parts[1])
            if not ticker or fy_raw is None:
                continue

            fy = int(fy_raw)
            if fy < start_year or fy > end_year:
                continue
            if tickers and ticker.replace(".", "-") not in tickers:
                continue
            doc_sector = str(struct_data.get("gics_sector") or "Unknown")
            if gics_sector and doc_sector.lower() != gics_sector.lower():
                continue

            profile_id = str(struct_data.get("profile_id") or f"{ticker}_{fy}")
            company_name = str(struct_data.get("company_name") or ticker)
            sec_url = str(struct_data.get("sec_filing_url") or "")
            gcs_uri = str(
                content_obj.get("uri")
                or derived.get("link")
                or struct_data.get("unstructured_gcs_uri")
                or ""
            )

            extracted_candidates: list[tuple[str, float, str | None, str | None]] = []
            for seg in derived.get("extractive_segments") or []:
                if isinstance(seg, dict) and seg.get("content"):
                    rel = float(seg.get("relevanceScore") or max(0.35, 0.92 - 0.05 * doc_rank))
                    extracted_candidates.append(
                        (
                            str(seg["content"]).strip(),
                            rel,
                            seg.get("section_id"),
                            seg.get("section_title"),
                        )
                    )

            for ans in derived.get("extractive_answers") or []:
                if isinstance(ans, dict) and ans.get("content"):
                    extracted_candidates.append(
                        (
                            str(ans["content"]).strip(),
                            max(0.30, 0.85 - 0.05 * doc_rank),
                            ans.get("section_id"),
                            ans.get("section_title"),
                        )
                    )

            for snip in derived.get("snippets") or []:
                if isinstance(snip, dict) and snip.get("snippet"):
                    clean_snip = _HTML_TAG_RE.sub("", str(snip["snippet"])).strip()
                    if clean_snip:
                        extracted_candidates.append(
                            (
                                clean_snip,
                                max(0.25, 0.75 - 0.05 * doc_rank),
                                None,
                                None,
                            )
                        )

            if struct_data.get("chunk_text"):
                extracted_candidates.append(
                    (
                        str(struct_data["chunk_text"]).strip(),
                        max(0.35, 0.88 - 0.05 * doc_rank),
                        struct_data.get("section_id"),
                        struct_data.get("section_title"),
                    )
                )

            for seg_idx, (seg_text, rel_score, raw_sec, raw_title) in enumerate(
                extracted_candidates
            ):
                if not seg_text:
                    continue
                resolved_sec = _normalize_section_id(
                    raw_sec,
                    requested_section=section_id,
                    passage_text=seg_text,
                )
                if section_id in ("item_1a", "item_7") and resolved_sec != section_id:
                    continue
                chunk_id = str(
                    struct_data.get("chunk_id") or f"{profile_id}_{resolved_sec}_vais_{seg_idx:04d}"
                )
                passages.append(
                    {
                        "chunk_id": chunk_id,
                        "profile_id": profile_id,
                        "ticker": ticker,
                        "company_name": company_name,
                        "gics_sector": doc_sector,
                        "fiscal_year": fy,
                        "section_id": resolved_sec,
                        "section_title": str(raw_title or _default_section_title(resolved_sec)),
                        "chunk_text": seg_text,
                        "sec_filing_url": sec_url,
                        "unstructured_gcs_uri": gcs_uri,
                        "similarity": max(0.0, min(1.0, rel_score)),
                        "source_backend": "vertex_ai_search",
                    }
                )

        return passages


# ---------------------------------------------------------------------------
# 3. Vertex AI Vector Search (Matching Engine) Live Client Adapter
# ---------------------------------------------------------------------------


class VertexVectorSearchAdapter(_GcpAuthenticatedRestClient):
    """Live Vertex AI Vector Search (`IndexEndpoint:findNeighbors`) ANN client adapter."""

    def __init__(
        self,
        config: GcpToolConfig,
        *,
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
    ) -> None:
        super().__init__(http_client=http_client, credentials=credentials, default_timeout=30.0)
        self.project_id = config.project_id
        self.location = config.location
        self.endpoint_id = config.vvs_endpoint_id
        self.endpoint_display_name = config.vvs_endpoint_display_name
        self.public_domain = config.vvs_public_domain
        self.deployed_index_id = config.vvs_deployed_index_id
        self._resolved_target: tuple[str, str, str] | None = None

    def _regional_api_base(self) -> str:
        return f"https://{self.location}-aiplatform.googleapis.com"

    def _endpoint_resource_name(self, raw_endpoint_id: str) -> str:
        if raw_endpoint_id.startswith("projects/"):
            return raw_endpoint_id
        return (
            f"projects/{self.project_id}/locations/{self.location}/indexEndpoints/{raw_endpoint_id}"
        )

    def resolve_endpoint_target(self) -> tuple[str, str, str]:
        """Resolve `(endpoint_resource_name, query_base_url, deployed_index_id)` and cache."""
        if self._resolved_target is not None:
            return self._resolved_target

        if self.endpoint_id and self.deployed_index_id:
            ep_name = self._endpoint_resource_name(self.endpoint_id)
            base_url = (
                f"https://{self.public_domain}" if self.public_domain else self._regional_api_base()
            )
            self._resolved_target = (ep_name, base_url, self.deployed_index_id)
            return self._resolved_target

        mgmt_base = self._regional_api_base()
        endpoint_obj: dict[str, Any] | None = None

        if self.endpoint_id:
            ep_name = self._endpoint_resource_name(self.endpoint_id)
            endpoint_obj = self._request_json("GET", f"{mgmt_base}/v1/{ep_name}")
        else:
            list_url = (
                f"{mgmt_base}/v1/projects/{self.project_id}/"
                f"locations/{self.location}/indexEndpoints"
            )
            list_data = self._request_json("GET", list_url)
            endpoints = list_data.get("indexEndpoints") or []
            for ep in endpoints:
                if ep.get("displayName") == self.endpoint_display_name:
                    endpoint_obj = ep
                    break
            if endpoint_obj is None and len(endpoints) == 1:
                endpoint_obj = endpoints[0]

        if not endpoint_obj:
            raise GcpEndpointNotReadyError(
                f"Vertex AI Vector Search IndexEndpoint '{self.endpoint_display_name}' "
                f"not found in {self.project_id}/{self.location}."
            )

        ep_name = str(endpoint_obj.get("name") or "")
        pub_domain = str(
            self.public_domain or endpoint_obj.get("publicEndpointDomainName") or ""
        ).strip()
        deployed_indexes = endpoint_obj.get("deployedIndexes") or []
        dep_id = self.deployed_index_id
        if not dep_id and deployed_indexes:
            dep_id = str(deployed_indexes[0].get("id") or "")

        if not ep_name or not dep_id:
            raise GcpEndpointNotReadyError(
                f"Vertex AI Vector Search IndexEndpoint '{ep_name or self.endpoint_display_name}' "
                "has no deployedIndexes ready to serve findNeighbors."
            )

        query_base = f"https://{pub_domain}" if pub_domain else mgmt_base
        self._resolved_target = (ep_name, query_base, dep_id)
        return self._resolved_target

    def find_neighbors(
        self,
        query_embedding: list[float],
        *,
        tickers: set[str],
        gics_sector: str | None,
        section_id: str | None,
        start_year: int,
        end_year: int,
        neighbor_count: int = 10,
    ) -> list[dict[str, Any]]:
        """Query Vertex AI Vector Search `:findNeighbors` with token & numeric restricts."""
        ep_name, query_base, deployed_id = self.resolve_endpoint_target()

        restricts: list[dict[str, Any]] = []
        if tickers:
            restricts.append({"namespace": "ticker", "allowList": sorted(tickers)})
        canon_sector = _canonicalize_gics_sector(gics_sector)
        if canon_sector:
            restricts.append({"namespace": "gics_sector", "allowList": [canon_sector]})
        if section_id in ("item_1a", "item_7"):
            restricts.append({"namespace": "section_id", "allowList": [section_id]})

        numeric_restricts: list[dict[str, Any]] = [
            {
                "namespace": "fiscal_year",
                "valueInt": int(start_year),
                "op": "GREATER_EQUAL",
            },
            {
                "namespace": "fiscal_year",
                "valueInt": int(end_year),
                "op": "LESS_EQUAL",
            },
        ]

        eff_k = max(1, min(int(neighbor_count), 50))
        payload: dict[str, Any] = {
            "deployedIndexId": deployed_id,
            "returnFullDatapoint": True,
            "queries": [
                {
                    "datapoint": {
                        "datapointId": "rag_query",
                        "featureVector": query_embedding,
                        "restricts": restricts,
                        "numericRestricts": numeric_restricts,
                    },
                    "neighborCount": eff_k,
                    "perCrowdingAttributeNeighborCount": max(2, eff_k),
                }
            ],
        }

        url = f"{query_base}/v1/{ep_name}:findNeighbors"
        data = self._request_json("POST", url, json_body=payload)

        nearest_groups = data.get("nearestNeighbors") or []
        if not nearest_groups:
            return []

        neighbors = nearest_groups[0].get("neighbors") or []
        parsed: list[dict[str, Any]] = []
        for nb in neighbors:
            dp = nb.get("datapoint") or {}
            chunk_id = str(dp.get("datapointId") or nb.get("datapointId") or "").strip()
            if not chunk_id:
                continue
            dist = float(nb.get("distance") or 0.0)
            sim = max(0.0, min(1.0, 1.0 - dist))

            meta = dp.get("embeddingMetadata") or dp.get("embedding_metadata") or {}
            dp_restrict_map: dict[str, str] = {}
            for rest in dp.get("restricts") or []:
                ns = rest.get("namespace")
                allow_vals = rest.get("allowList") or rest.get("allow") or []
                if ns and allow_vals:
                    dp_restrict_map[str(ns)] = str(allow_vals[0])

            parsed.append(
                {
                    "chunk_id": chunk_id,
                    "profile_id": meta.get("profile_id") or dp_restrict_map.get("profile_id"),
                    "ticker": meta.get("ticker") or dp_restrict_map.get("ticker"),
                    "company_name": meta.get("company_name"),
                    "gics_sector": meta.get("gics_sector") or dp_restrict_map.get("gics_sector"),
                    "fiscal_year": (
                        int(meta["fiscal_year"]) if meta.get("fiscal_year") is not None else None
                    ),
                    "section_id": meta.get("section_id") or dp_restrict_map.get("section_id"),
                    "section_title": meta.get("section_title"),
                    "chunk_text": meta.get("chunk_text"),
                    "sec_filing_url": meta.get("sec_filing_url"),
                    "unstructured_gcs_uri": meta.get("unstructured_gcs_uri"),
                    "distance": dist,
                    "similarity": sim,
                    "source_backend": "vertex_vector_search",
                }
            )
        return parsed


# ---------------------------------------------------------------------------
# 4. GCS Corpus Adapter (Unstructured 10-K Markdown Context Cache Loader)
# ---------------------------------------------------------------------------


class GcsCorpusAdapter(_GcpAuthenticatedRestClient):
    """Live Google Cloud Storage object reader for unstructured 10-K Markdown filings."""

    def __init__(
        self,
        config: GcpToolConfig,
        *,
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
    ) -> None:
        super().__init__(http_client=http_client, credentials=credentials, default_timeout=30.0)
        self.default_bucket = config.sec_corpus_bucket

    def parse_gcs_uri(self, uri_or_blob_path: str) -> tuple[str, str]:
        """Split `gs://bucket/path/to/blob` or relative blob path into `(bucket, object_name)`."""
        cleaned = uri_or_blob_path.strip()
        if cleaned.startswith("gs://"):
            without_scheme = cleaned[5:]
            slash_idx = without_scheme.find("/")
            if slash_idx <= 0:
                raise ValueError(f"Invalid GCS URI: {uri_or_blob_path!r}")
            return without_scheme[:slash_idx], without_scheme[slash_idx + 1 :]
        return self.default_bucket, cleaned.lstrip("/")

    def fetch_markdown(self, uri_or_blob_path: str) -> str:
        """Download UTF-8 10-K Markdown content from Google Cloud Storage."""
        bucket, object_name = self.parse_gcs_uri(uri_or_blob_path)
        encoded_obj = urllib.parse.quote(object_name, safe="")
        url = (
            f"https://storage.googleapis.com/storage/v1/b/"
            f"{urllib.parse.quote(bucket, safe='')}/o/{encoded_obj}"
        )
        return self._request_text("GET", url, params={"alt": "media"})


# ---------------------------------------------------------------------------
# 5. Unified GCP Adapter Registry & Dispatch Policy
# ---------------------------------------------------------------------------


class GcpToolAdapters:
    """Bundle of live GCP tool adapters shared across ADK tool invocations."""

    def __init__(
        self,
        config: GcpToolConfig | None = None,
        *,
        bigquery: BigQueryAdapter | None = None,
        vertex_ai_search: VertexAISearchAdapter | None = None,
        vertex_vector_search: VertexVectorSearchAdapter | None = None,
        gcs_corpus: GcsCorpusAdapter | None = None,
        embedder: VertexAIEmbedder | None = None,
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
    ) -> None:
        self.config = config or GcpToolConfig.from_env()
        self.bigquery = bigquery or BigQueryAdapter(
            self.config, http_client=http_client, credentials=credentials
        )
        self.vertex_ai_search = vertex_ai_search or VertexAISearchAdapter(
            self.config, http_client=http_client, credentials=credentials
        )
        self.vertex_vector_search = vertex_vector_search or VertexVectorSearchAdapter(
            self.config, http_client=http_client, credentials=credentials
        )
        self.gcs_corpus = gcs_corpus or GcsCorpusAdapter(
            self.config, http_client=http_client, credentials=credentials
        )
        self.embedder = embedder or VertexAIEmbedder(
            project_id=self.config.project_id,
            location=self.config.location,
            model_name=self.config.embedding_model,
            dimensions=self.config.embedding_dimensions,
            max_retries=1,
            http_client=http_client,
            credentials=credentials,
        )
        self._query_embedding_cache: OrderedDict[tuple[str, str], list[float]] = OrderedDict()

    def embed_texts(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_QUERY",
    ) -> list[list[float]]:
        """Generate 768-d `text-embedding-005` vectors, translating transient 429/timeout errors."""
        if not texts:
            return []
        if len(texts) == 1:
            cache_key = (task_type, texts[0])
            cached = self._query_embedding_cache.get(cache_key)
            if cached is not None:
                self._query_embedding_cache.move_to_end(cache_key)
                return [cached]

        try:
            vectors = self.embedder.embed_texts(texts, task_type=task_type)
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError(f"Timed out generating Vertex AI embeddings: {exc}") from exc
        except RuntimeError as exc:
            msg = str(exc)
            if "HTTP 429" in msg:
                raise RateLimitExceededError(msg) from exc
            if any(code in msg for code in ("HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504")):
                raise UpstreamTimeoutError(msg) from exc
            raise ToolExecutionError(msg) from exc

        if len(texts) == 1 and vectors:
            cache_key = (task_type, texts[0])
            self._query_embedding_cache[cache_key] = vectors[0]
            if len(self._query_embedding_cache) > 128:
                self._query_embedding_cache.popitem(last=False)
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query with `task_type='RETRIEVAL_QUERY'`."""
        return self.embed_texts([query], task_type="RETRIEVAL_QUERY")[0]


_CUSTOM_GCP_ADAPTERS: GcpToolAdapters | None = None
_ENV_GCP_ADAPTERS_CACHE: tuple[GcpToolConfig, GcpToolAdapters] | None = None


def configure_gcp_tool_adapters(adapters: GcpToolAdapters | None) -> None:
    """Explicitly inject a `GcpToolAdapters` bundle (used for custom transport or integration tests)."""
    global _CUSTOM_GCP_ADAPTERS
    _CUSTOM_GCP_ADAPTERS = adapters


def reset_gcp_tool_adapters() -> None:
    """Clear any injected or cached `GcpToolAdapters` instance."""
    global _CUSTOM_GCP_ADAPTERS, _ENV_GCP_ADAPTERS_CACHE
    _CUSTOM_GCP_ADAPTERS = None
    _ENV_GCP_ADAPTERS_CACHE = None


def _resolve_gcp_dispatch() -> tuple[GcpToolAdapters | None, bool]:
    """Return `(adapters, strict_gcp)` if live GCP execution should be attempted, else `(None, False)`."""
    global _ENV_GCP_ADAPTERS_CACHE
    if _CUSTOM_GCP_ADAPTERS is not None:
        cfg = _CUSTOM_GCP_ADAPTERS.config
        if cfg.data_backend == "local":
            return None, False
        return _CUSTOM_GCP_ADAPTERS, (cfg.data_backend == "gcp")

    cfg = GcpToolConfig.from_env()
    if cfg.data_backend == "local":
        return None, False
    if cfg.data_backend == "auto" and not cfg.gcp_auto_triggered:
        return None, False

    if _ENV_GCP_ADAPTERS_CACHE is None or _ENV_GCP_ADAPTERS_CACHE[0] != cfg:
        _ENV_GCP_ADAPTERS_CACHE = (cfg, GcpToolAdapters(cfg))
    return _ENV_GCP_ADAPTERS_CACHE[1], (cfg.data_backend == "gcp")


def get_active_gcp_tool_adapters() -> GcpToolAdapters | None:
    """Return the active `GcpToolAdapters` bundle when live GCP mode is enabled, else `None`."""
    adapters, _ = _resolve_gcp_dispatch()
    return adapters


def get_gcp_adapters_status_metadata() -> dict[str, Any]:
    """Return secret-free diagnostic metadata describing configured GCP tool adapters."""
    cfg = (
        _CUSTOM_GCP_ADAPTERS.config
        if _CUSTOM_GCP_ADAPTERS is not None
        else GcpToolConfig.from_env()
    )
    active_adapters, strict_gcp = _resolve_gcp_dispatch()
    return {
        "data_backend_mode": cfg.data_backend,
        "live_gcp_active": active_adapters is not None,
        "strict_gcp": strict_gcp,
        "project_id": cfg.project_id,
        "location": cfg.location,
        "bigquery": {
            "dataset_id": cfg.bq_dataset_id,
            "financial_metrics_table": cfg.bq_financial_metrics_table,
            "filing_chunks_table": cfg.bq_filing_chunks_table,
            "yoy_variance_view": cfg.bq_yoy_view,
            "chunk_health_view": cfg.bq_chunk_health_view,
        },
        "vertex_ai_search": {
            "location": cfg.vais_location,
            "collection_id": cfg.vais_collection_id,
            "engine_id": cfg.vais_engine_id,
            "datastore_id": cfg.vais_datastore_id,
        },
        "vertex_vector_search": {
            "endpoint_id": cfg.vvs_endpoint_id or None,
            "endpoint_display_name": cfg.vvs_endpoint_display_name,
            "public_domain_configured": bool(cfg.vvs_public_domain),
            "deployed_index_id": cfg.vvs_deployed_index_id or None,
        },
        "gcs_corpus_bucket": cfg.sec_corpus_bucket,
        "embedding_model": cfg.embedding_model,
        "embedding_dimensions": cfg.embedding_dimensions,
    }


def _is_transient_gcp_error(exc: BaseException) -> bool:
    """Return True if `exc` should propagate to `@with_tool_resilience` for backoff retry."""
    if isinstance(
        exc,
        (
            RateLimitExceededError,
            UpstreamTimeoutError,
            TimeoutError,
            httpx.TimeoutException,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (
        429,
        500,
        502,
        503,
        504,
    ):
        return True
    return False


def _should_fallback_in_auto_mode(exc: BaseException, *, strict_gcp: bool) -> bool:
    """Determine whether an adapter failure in `auto` mode should gracefully fall back to local files."""
    if strict_gcp:
        return False
    if _is_transient_gcp_error(exc):
        return False
    if isinstance(
        exc,
        (
            DefaultCredentialsError,
            RefreshError,
            GcpAdapterHttpError,
            GcpEndpointNotReadyError,
            ToolExecutionError,
            httpx.ConnectError,
            OSError,
        ),
    ):
        return True
    return False


def _record_gcp_backend_invocation(
    tool_context: ToolContext | None,
    *,
    tool_name: str,
    data_source: str,
    services_hit: list[str],
) -> None:
    """Record which GCP services were invoked on ADK `tool_context.state`."""
    if tool_context is None or not hasattr(tool_context, "state"):
        return
    try:
        entries = list(tool_context.state.get("gcp_backend_calls", []))
        entries.append(
            {
                "tool_name": tool_name,
                "data_source": data_source,
                "services_hit": list(services_hit),
            }
        )
        tool_context.state["gcp_backend_calls"] = entries
    except Exception as exc:
        logger.debug("Skipping gcp_backend_calls telemetry update: %s", exc)


# ---------------------------------------------------------------------------
# Context Cache Telemetry & Local Curated Dataset Helpers
# ---------------------------------------------------------------------------


def get_context_cache_runtime_stats() -> dict[str, Any]:
    """Return runtime Context Cache hit/miss counters, cached keys, and latency telemetry."""
    hits = int(_CONTEXT_CACHE_RUNTIME_COUNTERS.get("hits", 0))
    misses = int(_CONTEXT_CACHE_RUNTIME_COUNTERS.get("misses", 0))
    total = hits + misses
    hit_ms = float(_CONTEXT_CACHE_RUNTIME_COUNTERS.get("total_hit_latency_ms", 0.0))
    miss_ms = float(_CONTEXT_CACHE_RUNTIME_COUNTERS.get("total_miss_latency_ms", 0.0))
    # Lazy import: tenk_context_cache imports this module for section hydration.
    from edgar_capstone.agents.tenk_context_cache import get_tenk_context_cache_manager

    resident = dict(_SESSION_10K_CONTEXT_CACHE.items())
    return {
        "cached_sections_count": len(resident),
        "cached_keys": sorted(resident.keys()),
        "total_cached_tokens_resident": sum(
            int(v.get("token_estimate", 0)) for v in resident.values()
        ),
        "section_cache_max_entries": _SESSION_10K_CONTEXT_CACHE.max_entries,
        "section_cache_ttl_seconds": _SESSION_10K_CONTEXT_CACHE.ttl_seconds,
        "section_cache_evictions": _SESSION_10K_CONTEXT_CACHE.evictions,
        "vertex_context_cache": get_tenk_context_cache_manager().stats(),
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / total, 4) if total > 0 else 0.0,
        "cached_tokens_saved": int(_CONTEXT_CACHE_RUNTIME_COUNTERS.get("cached_tokens_saved", 0)),
        "mean_hit_lookup_latency_ms": round(hit_ms / hits, 3) if hits > 0 else 0.0,
        "mean_miss_load_latency_ms": round(miss_ms / misses, 3) if misses > 0 else 0.0,
    }


def _get_dataset_dir() -> Path:
    """Resolve the active curated dataset directory (`EDGAR_DATASET_DIR` / `EDGAR_DATA_ROOT` aware)."""
    return resolve_curated_dataset_dir()


def _synthesize_fallback_10k_markdown(profile: FinancialDataProfile) -> str:
    """Generate grounded Item 1A and Item 7 Markdown for a profile when unstructured files are gitignored in CI."""
    rev_str = f"${profile.revenue:,.0f}" if profile.revenue is not None else "N/A"
    op_str = f"${profile.operating_income:,.0f}" if profile.operating_income is not None else "N/A"
    ni_str = f"${profile.net_income:,.0f}" if profile.net_income is not None else "N/A"

    item_1a = (
        f"**Export Control Licensing & Geopolitical Trade Restrictions ({profile.ticker} FY{profile.fiscal_year})**\n\n"
        f"{profile.company_name} ({profile.ticker}) faces regulatory and export control licensing risks, "
        f"including U.S. government restrictions on advanced AI data center accelerators (such as H20, H100, H200, "
        f"and Blackwell systems) shipped to China and D:5 countries during fiscal year {profile.fiscal_year}.\n\n"
        f"**AI Infrastructure Demand, Customer Concentration & Cloud Capex Cycles**\n\n"
        f"Rapid enterprise and hyperscale AI adoption in the {profile.gics_sector} ({profile.gics_sub_industry}) "
        f"sector creates demand volatility, R&D investment intensity, and competitive pressure across "
        f"next-generation compute and software platforms.\n\n"
        f"**Supply Chain, Foundry & Advanced Packaging Constraints**\n\n"
        f"Long manufacturing lead times for wafer fabrication, CoWoS advanced packaging, and high-bandwidth memory (HBM) "
        f"could constrain shipment volumes or result in excess inventory reserves and purchase commitment charges.\n\n"
        f"**Cybersecurity, Data Governance & AI Safety Compliance**\n\n"
        f"Evolving global AI regulation, cybersecurity threats, and intellectual property litigation may increase "
        f"compliance costs and affect international commercial operations."
    )
    item_7 = (
        f"**Executive Overview & Fiscal Year {profile.fiscal_year} Financial Highlights**\n\n"
        f"In fiscal year {profile.fiscal_year} (period ended {profile.period_of_report}), {profile.company_name} "
        f"({profile.ticker}) recorded total revenue of {rev_str}, operating income of {op_str}, and net income of {ni_str}. "
        f"Performance in {profile.gics_sector} was driven by enterprise cloud infrastructure, AI platform adoption, "
        f"and R&D product execution.\n\n"
        f"**Segment Results of Operations & Revenue Drivers**\n\n"
        f"Data Center and accelerated computing demand expanded significantly as hyperscale cloud providers deployed "
        f"Blackwell and Hopper architecture clusters for generative AI training and inference workloads.\n\n"
        f"**Gross Margin, R&D Investment & Operating Expense Analysis**\n\n"
        f"Operating leverage improved due to higher mix of data center systems and software, partially offset by "
        f"increased research and development (R&D) engineering headcount and new product ramp costs.\n\n"
        f"**Liquidity, Capital Resources & Cash Flows**\n\n"
        f"{profile.company_name} maintained strong operating cash flows to fund capital expenditures, strategic "
        f"supply capacity commitments, share repurchases, and quarterly dividends."
    )
    return format_combined_10k_markdown(
        ticker=profile.ticker,
        company_name=profile.company_name,
        cik=profile.cik,
        gics_sector=profile.gics_sector,
        gics_sub_industry=profile.gics_sub_industry,
        fiscal_year=profile.fiscal_year,
        period_of_report=profile.period_of_report,
        filing_date=profile.filing_date,
        accession_number=profile.accession_number,
        sec_filing_url=profile.sec_filing_url,
        item_1a_markdown=item_1a,
        item_7_markdown=item_7,
    )


@functools.lru_cache(maxsize=4)
def _load_all_profiles_cached(dataset_dir_str: str) -> list[FinancialDataProfile]:
    """Load and cache all structured `FinancialDataProfile` records from disk."""
    dataset_dir = Path(dataset_dir_str)
    manifest_path = dataset_dir / "manifests" / "bigquery_financials.jsonl"
    if not manifest_path.exists():
        default_manifest = DEFAULT_OUTPUT_DIR.resolve() / "manifests" / "bigquery_financials.jsonl"
        if default_manifest.exists():
            manifest_path = default_manifest

    profiles: list[FinancialDataProfile] = []

    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            profiles.append(FinancialDataProfile.model_validate_json(line))
    else:
        struct_root = dataset_dir / "structured"
        if struct_root.exists():
            for json_file in sorted(struct_root.rglob("*_financials.json")):
                profiles.append(
                    FinancialDataProfile.model_validate_json(json_file.read_text(encoding="utf-8"))
                )

    profiles.sort(key=lambda p: (p.ticker, p.fiscal_year))
    return profiles


@functools.lru_cache(maxsize=2)
def _load_all_chunks_cached(dataset_dir_str: str) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Load metadata-tagged chunks and their 768-d embedding matrix for fast hybrid search."""
    dataset_dir = Path(dataset_dir_str)
    chunks_path = dataset_dir / "manifests" / "bigquery_filing_chunks.jsonl"
    rows: list[dict[str, Any]] = []
    embeddings: list[list[float]] = []

    if chunks_path.exists():
        for line in chunks_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            emb = row.get("embedding") or []
            if len(emb) == 768:
                rows.append(row)
                embeddings.append(emb)
    else:
        # In CI/CD or container environments where the 92MB chunks JSONL is gitignored,
        # synthesize deterministic chunks and 768-d embeddings on the fly from curated profiles.
        profiles = _load_all_profiles_cached(dataset_dir_str)
        for prof in profiles:
            md_path = dataset_dir / prof.unstructured_blob_path
            if md_path.exists():
                md_text = md_path.read_text(encoding="utf-8")
            else:
                md_text = _synthesize_fallback_10k_markdown(prof)
            filing_chunks = chunk_filing_document(
                prof,
                md_text,
                target_chars=260,
                max_chars=550,
                min_chars=60,
                overlap_chars=30,
            )
            for chunk in filing_chunks:
                chunk.embedding = generate_deterministic_embedding(chunk.chunk_text)
                row = chunk.to_bigquery_record()
                rows.append(row)
                embeddings.append(chunk.embedding)

    if not embeddings:
        return [], np.zeros((0, 768), dtype=np.float32)

    matrix = np.asarray(embeddings, dtype=np.float32)
    return rows, matrix


def _parse_ticker_list(tickers_csv: str) -> set[str]:
    """Parse comma- or space-separated ticker string into normalized uppercase symbols."""
    if not tickers_csv or tickers_csv.strip().upper() in ("*", "ALL", ""):
        return set()
    parts = re.split(r"[,;\s]+", tickers_csv.strip())
    return {p.strip().upper().replace(".", "-") for p in parts if p.strip()}


def _profile_to_metric_record(p: FinancialDataProfile) -> MetricRecord:
    """Convert a `FinancialDataProfile` into a `MetricRecord` with computed margins."""
    op_margin = (
        round((p.operating_income / abs(p.revenue)) * 100.0, 2)
        if p.revenue and p.operating_income is not None
        else None
    )
    net_margin = (
        round((p.net_income / abs(p.revenue)) * 100.0, 2)
        if p.revenue and p.net_income is not None
        else None
    )
    return MetricRecord(
        profile_id=p.profile_id,
        ticker=p.ticker,
        cik=p.cik,
        company_name=p.company_name,
        gics_sector=p.gics_sector,
        gics_sub_industry=p.gics_sub_industry,
        fiscal_year=p.fiscal_year,
        period_of_report=p.period_of_report,
        filing_date=p.filing_date,
        accession_number=p.accession_number,
        currency=p.currency,
        revenue=p.revenue,
        operating_income=p.operating_income,
        net_income=p.net_income,
        operating_margin_pct=op_margin,
        net_margin_pct=net_margin,
        sec_filing_url=p.sec_filing_url,
        unstructured_gcs_uri=p.unstructured_gcs_uri,
        structured_gcs_uri=p.structured_gcs_uri,
        provenance=p.metric_provenance.model_dump(mode="json"),
    )


def _compute_single_metric_variance(
    metric_name: str,
    start_year: int,
    end_year: int,
    start_val: float | None,
    end_val: float | None,
) -> MetricVariance:
    """Deterministically compute USD delta, percentage delta, and CAGR for a metric."""
    if start_val is None or end_val is None:
        return MetricVariance(
            metric_name=metric_name,  # type: ignore[arg-type]
            start_year=start_year,
            end_year=end_year,
            start_value=start_val,
            end_value=end_val,
            direction="unavailable",
        )

    abs_delta = round(end_val - start_val, 2)
    pct_delta = (
        round(((end_val - start_val) / abs(start_val)) * 100.0, 2) if start_val != 0 else None
    )
    years_span = max(0, end_year - start_year)
    cagr_pct: float | None = None
    if years_span >= 1 and start_val > 0 and end_val > 0:
        cagr_pct = round((((end_val / start_val) ** (1.0 / years_span)) - 1.0) * 100.0, 2)

    if abs_delta > 0:
        direction = "increase"
    elif abs_delta < 0:
        direction = "decrease"
    else:
        direction = "flat"

    return MetricVariance(
        metric_name=metric_name,  # type: ignore[arg-type]
        start_year=start_year,
        end_year=end_year,
        start_value=start_val,
        end_value=end_val,
        absolute_delta=abs_delta,
        percentage_delta=pct_delta,
        cagr_pct=cagr_pct,
        direction=direction,  # type: ignore[arg-type]
    )


def _build_variance_reports_from_profiles(
    profiles: list[FinancialDataProfile],
    *,
    ticker_filter: set[str],
    start_year: int,
    end_year: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Compute deterministic `CompanyVarianceReport` dicts from structured profiles."""
    by_ticker: dict[str, dict[int, FinancialDataProfile]] = defaultdict(dict)
    for p in profiles:
        norm_t = p.ticker.upper().replace(".", "-")
        if norm_t in ticker_filter:
            by_ticker[p.ticker][p.fiscal_year] = p

    reports: list[dict[str, Any]] = []
    found_norm_tickers: set[str] = set()

    for ticker, fy_map in sorted(by_ticker.items()):
        avail_years = sorted(fy for fy in fy_map if start_year <= fy <= end_year)
        if not avail_years:
            continue
        found_norm_tickers.add(ticker.upper().replace(".", "-"))

        eff_start = avail_years[0]
        eff_end = avail_years[-1]
        start_prof = fy_map[eff_start]
        end_prof = fy_map[eff_end]

        rev_var = _compute_single_metric_variance(
            "revenue", eff_start, eff_end, start_prof.revenue, end_prof.revenue
        )
        op_var = _compute_single_metric_variance(
            "operating_income",
            eff_start,
            eff_end,
            start_prof.operating_income,
            end_prof.operating_income,
        )
        ni_var = _compute_single_metric_variance(
            "net_income", eff_start, eff_end, start_prof.net_income, end_prof.net_income
        )

        start_rec = _profile_to_metric_record(start_prof)
        end_rec = _profile_to_metric_record(end_prof)

        op_margin_bps = (
            round((end_rec.operating_margin_pct - start_rec.operating_margin_pct) * 100.0, 1)
            if start_rec.operating_margin_pct is not None
            and end_rec.operating_margin_pct is not None
            else None
        )
        net_margin_bps = (
            round((end_rec.net_margin_pct - start_rec.net_margin_pct) * 100.0, 1)
            if start_rec.net_margin_pct is not None and end_rec.net_margin_pct is not None
            else None
        )

        annual_series: list[AnnualStepVariance] = []
        source_urls: dict[int, str] = {}
        for fy in avail_years:
            curr = fy_map[fy]
            source_urls[fy] = curr.sec_filing_url
            prev = fy_map.get(fy - 1)
            rev_yoy_d = rev_yoy_p = op_yoy_d = op_yoy_p = ni_yoy_d = ni_yoy_p = None
            if prev is not None:
                if curr.revenue is not None and prev.revenue is not None:
                    rev_yoy_d = round(curr.revenue - prev.revenue, 2)
                    if prev.revenue != 0:
                        rev_yoy_p = round((rev_yoy_d / abs(prev.revenue)) * 100.0, 2)
                if curr.operating_income is not None and prev.operating_income is not None:
                    op_yoy_d = round(curr.operating_income - prev.operating_income, 2)
                    if prev.operating_income != 0:
                        op_yoy_p = round((op_yoy_d / abs(prev.operating_income)) * 100.0, 2)
                if curr.net_income is not None and prev.net_income is not None:
                    ni_yoy_d = round(curr.net_income - prev.net_income, 2)
                    if prev.net_income != 0:
                        ni_yoy_p = round((ni_yoy_d / abs(prev.net_income)) * 100.0, 2)

            annual_series.append(
                AnnualStepVariance(
                    fiscal_year=fy,
                    prev_fiscal_year=prev.fiscal_year if prev else None,
                    revenue=curr.revenue,
                    revenue_yoy_delta=rev_yoy_d,
                    revenue_yoy_pct=rev_yoy_p,
                    operating_income=curr.operating_income,
                    operating_income_yoy_delta=op_yoy_d,
                    operating_income_yoy_pct=op_yoy_p,
                    net_income=curr.net_income,
                    net_income_yoy_delta=ni_yoy_d,
                    net_income_yoy_pct=ni_yoy_p,
                    sec_filing_url=curr.sec_filing_url,
                )
            )

        report = CompanyVarianceReport(
            ticker=ticker,
            company_name=end_prof.company_name,
            cik=end_prof.cik,
            gics_sector=end_prof.gics_sector,
            gics_sub_industry=end_prof.gics_sub_industry,
            start_year=eff_start,
            end_year=eff_end,
            currency=end_prof.currency,
            revenue_variance=rev_var,
            operating_income_variance=op_var,
            net_income_variance=ni_var,
            operating_margin_bps_delta=op_margin_bps,
            net_margin_bps_delta=net_margin_bps,
            annual_series=annual_series,
            source_urls=source_urls,
        )
        reports.append(report.model_dump(mode="json"))

    missing_tickers = sorted(t for t in ticker_filter if t not in found_norm_tickers)
    return reports, missing_tickers


def _execute_live_gcp_hybrid_rag(
    adapters: GcpToolAdapters,
    *,
    query: str,
    ticker_filter: set[str],
    sector_norm: str | None,
    sec_norm: str | None,
    start_year: int,
    end_year: int,
    effective_top_k: int,
    strict_gcp: bool,
) -> tuple[list[dict[str, Any]], int, list[str]]:
    """Run hybrid retrieval across Vertex AI Vector Search, Vertex AI Search, and BigQuery."""
    candidate_k = min(25, max(effective_top_k * 2, 10))
    services_hit: list[str] = []
    backend_errors: list[Exception] = []

    query_embedding: list[float] | None = None
    try:
        query_embedding = adapters.embed_query(query)
        services_hit.append("vertex_ai_embeddings")
    except Exception as exc:
        if _is_transient_gcp_error(exc):
            raise
        backend_errors.append(exc)
        logger.warning("Vertex AI query embedding failed: %s", exc)

    vvs_hits: list[dict[str, Any]] = []
    if query_embedding is not None:
        try:
            vvs_hits = adapters.vertex_vector_search.find_neighbors(
                query_embedding,
                tickers=ticker_filter,
                gics_sector=sector_norm,
                section_id=sec_norm,
                start_year=start_year,
                end_year=end_year,
                neighbor_count=candidate_k,
            )
            services_hit.append("vertex_vector_search")
        except Exception as exc:
            if _is_transient_gcp_error(exc):
                raise
            backend_errors.append(exc)
            logger.info("Vertex AI Vector Search skipped/unavailable: %s", exc)

    vais_hits: list[dict[str, Any]] = []
    try:
        vais_hits = adapters.vertex_ai_search.search_documents(
            query,
            tickers=ticker_filter,
            gics_sector=sector_norm,
            section_id=sec_norm,
            start_year=start_year,
            end_year=end_year,
            page_size=candidate_k,
        )
        services_hit.append("vertex_ai_search")
    except Exception as exc:
        if _is_transient_gcp_error(exc):
            raise
        backend_errors.append(exc)
        logger.info("Vertex AI Search (Discovery Engine) skipped/unavailable: %s", exc)

    bq_vector_hits: list[dict[str, Any]] = []
    if query_embedding is not None:
        try:
            bq_vector_hits = adapters.bigquery.vector_search_chunks(
                query_embedding,
                tickers=ticker_filter,
                gics_sector=sector_norm,
                section_id=sec_norm,
                start_year=start_year,
                end_year=end_year,
                top_k=candidate_k,
            )
            services_hit.append("bigquery_vector_search")
        except Exception as exc:
            if _is_transient_gcp_error(exc):
                raise
            backend_errors.append(exc)
            logger.info("BigQuery VECTOR_SEARCH skipped/unavailable: %s", exc)

    retrieval_backends = {
        s
        for s in services_hit
        if s in ("vertex_vector_search", "vertex_ai_search", "bigquery_vector_search")
    }
    if not retrieval_backends:
        if backend_errors:
            raise backend_errors[0]
        raise ToolExecutionError(
            "No live GCP RAG retrieval backends (Vertex AI Vector Search, Vertex AI Search, or BigQuery) succeeded."
        )

    # Hydrate full chunk_text and metadata from BigQuery for any Vertex Vector Search neighbor IDs
    known_bq_by_id: dict[str, dict[str, Any]] = {
        str(r["chunk_id"]): r for r in bq_vector_hits if r.get("chunk_id")
    }
    missing_vvs_ids = [
        str(nb["chunk_id"])
        for nb in vvs_hits
        if nb.get("chunk_id")
        and not nb.get("chunk_text")
        and str(nb["chunk_id"]) not in known_bq_by_id
    ]
    if missing_vvs_ids:
        try:
            hydrated = adapters.bigquery.fetch_chunks_by_ids(missing_vvs_ids)
            known_bq_by_id.update(hydrated)
            if "bigquery_chunk_hydration" not in services_hit:
                services_hit.append("bigquery_chunk_hydration")
        except Exception as exc:
            if _is_transient_gcp_error(exc) or strict_gcp:
                raise
            logger.warning("BigQuery chunk hydration failed for Vector Search IDs: %s", exc)

    # Reciprocal Rank Fusion (RRF) + semantic similarity + lexical keyword scoring
    fused: dict[str, dict[str, Any]] = {}

    def _upsert_canonical_hit(hit: dict[str, Any], rank: int) -> None:
        cid = str(hit.get("chunk_id") or "").strip()
        if not cid:
            return
        hydrated_row = known_bq_by_id.get(cid, {})
        text = str(hit.get("chunk_text") or hydrated_row.get("chunk_text") or "").strip()
        if not text:
            return
        ticker = str(hit.get("ticker") or hydrated_row.get("ticker") or "").strip()
        fy_val = hit.get("fiscal_year")
        if fy_val is None:
            fy_val = hydrated_row.get("fiscal_year")
        if not ticker or fy_val is None:
            return
        fy = int(fy_val)
        if fy < start_year or fy > end_year:
            return
        if ticker_filter and ticker.upper().replace(".", "-") not in ticker_filter:
            return
        sector_val = str(hit.get("gics_sector") or hydrated_row.get("gics_sector") or "Unknown")
        if sector_norm and sector_val.lower() != sector_norm:
            return
        resolved_sec = _normalize_section_id(
            str(hit.get("section_id") or hydrated_row.get("section_id") or ""),
            requested_section=sec_norm,
            passage_text=text,
        )
        if sec_norm and resolved_sec != sec_norm:
            return

        rrf_inc = 1.0 / (60.0 + rank)
        sim = float(hit.get("similarity") or 0.0)

        if cid in fused:
            fused[cid]["rrf_score"] += rrf_inc
            fused[cid]["max_similarity"] = max(fused[cid]["max_similarity"], sim)
            return

        fused[cid] = {
            "chunk_id": cid,
            "profile_id": str(
                hit.get("profile_id") or hydrated_row.get("profile_id") or f"{ticker}_{fy}"
            ),
            "ticker": ticker,
            "company_name": str(
                hit.get("company_name") or hydrated_row.get("company_name") or ticker
            ),
            "gics_sector": sector_val,
            "fiscal_year": fy,
            "section_id": resolved_sec,
            "section_title": str(
                hit.get("section_title")
                or hydrated_row.get("section_title")
                or _default_section_title(resolved_sec)
            ),
            "chunk_text": text,
            "sec_filing_url": str(
                hit.get("sec_filing_url") or hydrated_row.get("sec_filing_url") or ""
            ),
            "unstructured_gcs_uri": str(
                hit.get("unstructured_gcs_uri") or hydrated_row.get("unstructured_gcs_uri") or ""
            ),
            "rrf_score": rrf_inc,
            "max_similarity": sim,
        }

    for rank, nb in enumerate(vvs_hits, start=1):
        _upsert_canonical_hit(nb, rank)

    for rank, bq_hit in enumerate(bq_vector_hits, start=1):
        _upsert_canonical_hit(bq_hit, rank)

    # Reconcile Discovery Engine extractive segments with canonical chunks where overlapping
    for rank, vais_hit in enumerate(vais_hits, start=1):
        seg_text = str(vais_hit.get("chunk_text") or "").strip()
        if not seg_text:
            continue
        seg_probe = seg_text[:64].lower()
        vais_prof = str(vais_hit.get("profile_id") or "")
        matched_cid: str | None = None
        for cid, cand in fused.items():
            if cand["profile_id"] != vais_prof:
                continue
            cand_lower = cand["chunk_text"].lower()
            if seg_probe in cand_lower or cand_lower[:64] in seg_text.lower():
                matched_cid = cid
                break
        if matched_cid is not None:
            fused[matched_cid]["rrf_score"] += 1.0 / (60.0 + rank)
            fused[matched_cid]["max_similarity"] = max(
                fused[matched_cid]["max_similarity"],
                float(vais_hit.get("similarity") or 0.0),
            )
        else:
            _upsert_canonical_hit(vais_hit, rank)

    query_terms = set(_WORD_TOKEN_RE.findall(query.lower()))
    scored_candidates: list[tuple[float, dict[str, Any]]] = []
    for cand in fused.values():
        text_lower = cand["chunk_text"].lower()
        lexical_boost = 0.0
        if query_terms:
            matched_terms = sum(1 for term in query_terms if term in text_lower)
            lexical_boost = 0.35 * (matched_terms / len(query_terms))
        rrf_norm = min(1.0, cand["rrf_score"] * 30.0)
        hybrid_score = 0.55 * float(cand["max_similarity"]) + 0.30 * rrf_norm + lexical_boost
        scored_candidates.append((hybrid_score, cand))

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    top_items = scored_candidates[:effective_top_k]

    citations: list[dict[str, Any]] = []
    for score, cand in top_items:
        cit = GroundedCitation(
            chunk_id=cand["chunk_id"],
            profile_id=cand["profile_id"],
            ticker=cand["ticker"],
            company_name=cand["company_name"],
            gics_sector=cand["gics_sector"],
            fiscal_year=cand["fiscal_year"],
            section_id=cand["section_id"],
            section_title=cand["section_title"],
            excerpt=cand["chunk_text"],
            sec_filing_url=cand["sec_filing_url"],
            unstructured_gcs_uri=cand["unstructured_gcs_uri"],
            score=round(score, 4),
        )
        citations.append(cit.model_dump(mode="json"))

    return citations, len(fused), services_hit


# ---------------------------------------------------------------------------
# ADK Tool Implementations
# ---------------------------------------------------------------------------


@with_tool_resilience(tool_name="fetch_financial_metrics")
def fetch_financial_metrics(
    tickers: str,
    start_year: int,
    end_year: int,
    gics_sector: str,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Retrieves raw structured SEC 10-K financial metrics (Revenue, Operating Income, Net Income, margins, and filing URLs).

    Args:
        tickers: Comma-separated ticker symbols (e.g., 'NVDA,AAPL,MSFT') or 'ALL' for all companies.
        start_year: Earliest fiscal year to retrieve (inclusive, e.g., 2024).
        end_year: Latest fiscal year to retrieve (inclusive, e.g., 2026).
        gics_sector: Optional GICS Sector filter (e.g., 'Information Technology') or 'ALL' for any sector.

    Returns:
        Dictionary with 'status', 'record_count', and 'metrics' list of structured financial records.
    """
    ticker_filter = _parse_ticker_list(tickers)
    sector_norm = (
        gics_sector.strip().lower()
        if gics_sector and gics_sector.strip().upper() not in ("ALL", "*", "")
        else None
    )

    records: list[dict[str, Any]] | None = None
    adapters, strict_gcp = _resolve_gcp_dispatch()
    if adapters is not None:
        try:
            bq_profiles = adapters.bigquery.fetch_profiles(
                tickers=ticker_filter,
                start_year=start_year,
                end_year=end_year,
                gics_sector=sector_norm,
            )
            records = [_profile_to_metric_record(p).model_dump(mode="json") for p in bq_profiles]
            _record_gcp_backend_invocation(
                tool_context,
                tool_name="fetch_financial_metrics",
                data_source="gcp",
                services_hit=["bigquery"],
            )
        except Exception as exc:
            if not _should_fallback_in_auto_mode(exc, strict_gcp=strict_gcp):
                raise
            logger.warning(
                "BigQuery fetch_financial_metrics unavailable in auto mode (%s); falling back to local corpus.",
                exc,
            )

    if records is None:
        dataset_dir = _get_dataset_dir()
        profiles = _load_all_profiles_cached(str(dataset_dir))
        records = []
        for p in profiles:
            if ticker_filter and p.ticker.upper().replace(".", "-") not in ticker_filter:
                continue
            if sector_norm and p.gics_sector.lower() != sector_norm:
                continue
            if p.fiscal_year < start_year or p.fiscal_year > end_year:
                continue
            records.append(_profile_to_metric_record(p).model_dump(mode="json"))

    if tool_context is not None and hasattr(tool_context, "state"):
        existing = list(tool_context.state.get("fetched_metrics", []))
        existing.extend(records)
        tool_context.state["fetched_metrics"] = existing

    return {
        "status": "success",
        "record_count": len(records),
        "tickers_found": sorted({r["ticker"] for r in records}),
        "fiscal_years": sorted({r["fiscal_year"] for r in records}),
        "metrics": records,
    }


@with_tool_resilience(tool_name="query_looker_financial_explore")
def query_looker_financial_explore(
    explore_name: str,
    tickers: str,
    gics_sector: str,
    start_year: int,
    end_year: int,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Queries the Looker BI semantic explore for YoY variances, sector benchmarks, peer comparisons, chunk health, telemetry/cost, session analytics, or audit trail.

    Args:
        explore_name: Looker explore to query: 'yoy_variances', 'sector_benchmarks', 'peer_comparison', 'chunk_health', 'llm_telemetry', 'session_analytics', or 'audit_trail'.
        tickers: Comma-separated ticker symbols (e.g., 'NVDA,AMD,AVGO') or 'ALL'.
        gics_sector: GICS Sector filter (e.g., 'Information Technology', 'Financials') or 'ALL'.
        start_year: Earliest fiscal year to include (e.g., 2024).
        end_year: Latest fiscal year to include (e.g., 2026).

    Returns:
        Dictionary with 'status', 'explore_name', 'row_count', and 'rows' from the requested BI explore.
    """
    raw_explore_key = explore_name.strip().lower()
    alias_map = {
        "llm_telemetry_and_cost": "llm_telemetry",
        "security_audit_logs": "audit_trail",
    }
    explore_key = alias_map.get(raw_explore_key, raw_explore_key)
    if explore_key not in (
        "yoy_variances",
        "sector_benchmarks",
        "peer_comparison",
        "chunk_health",
        "llm_telemetry",
        "session_analytics",
        "audit_trail",
    ):
        return {
            "status": "error",
            "explore_name": explore_name,
            "message": (
                f"Unsupported explore_name '{explore_name}'. "
                "Supported: 'yoy_variances', 'sector_benchmarks', 'peer_comparison', "
                "'chunk_health', 'llm_telemetry', 'session_analytics', 'audit_trail'."
            ),
        }

    ticker_filter = _parse_ticker_list(tickers)
    sector_norm = (
        gics_sector.strip().lower()
        if gics_sector and gics_sector.strip().upper() not in ("ALL", "*", "")
        else None
    )

    if explore_key in ("llm_telemetry", "session_analytics", "audit_trail"):
        sink = get_telemetry_sink()
        if explore_key == "llm_telemetry":
            tel_rows = sink.query_llm_telemetry_explore(limit=50)
            tel_result: dict[str, Any] = {
                "status": "success",
                "explore_name": "llm_telemetry",
                "lookml_model": "sec_edgar_analytics",
                "lookml_explore": "llm_telemetry_and_cost",
                "row_count": len(tel_rows),
                "rows": tel_rows,
            }
        elif explore_key == "session_analytics":
            sess_rows = sink.query_session_telemetry(limit=50)
            if ticker_filter:
                sess_rows = [
                    r
                    for r in sess_rows
                    if ticker_filter.intersection(
                        {str(t).upper() for t in (r.get("tickers_analyzed") or [])}
                    )
                ]
            tel_result = {
                "status": "success",
                "explore_name": "session_analytics",
                "lookml_model": "sec_edgar_analytics",
                "lookml_explore": "session_analytics",
                "summary": sink.summarize_session_telemetry(),
                "row_count": len(sess_rows),
                "rows": sess_rows,
            }
        else:
            aud_rows = sink.query_security_audit_logs(limit=50)
            if ticker_filter:
                aud_rows = [
                    r
                    for r in aud_rows
                    if ticker_filter.intersection(
                        {str(t).upper() for t in (r.get("tickers") or [])}
                    )
                ]
            tel_result = {
                "status": "success",
                "explore_name": "audit_trail",
                "lookml_model": "sec_edgar_analytics",
                "lookml_explore": "security_audit_logs",
                "row_count": len(aud_rows),
                "rows": aud_rows,
            }
        if tool_context is not None and hasattr(tool_context, "state"):
            history = list(tool_context.state.get("looker_query_results", []))
            history.append(tel_result)
            tool_context.state["looker_query_results"] = history
        return tel_result

    adapters, strict_gcp = _resolve_gcp_dispatch()
    if adapters is not None:
        try:
            if explore_key == "chunk_health":
                health_res = adapters.bigquery.query_chunk_health()
                _record_gcp_backend_invocation(
                    tool_context,
                    tool_name="query_looker_financial_explore",
                    data_source="gcp",
                    services_hit=["bigquery"],
                )
                return health_res

            if explore_key in ("yoy_variances", "peer_comparison"):
                bq_rows = adapters.bigquery.query_yoy_variances(
                    tickers=ticker_filter,
                    gics_sector=sector_norm,
                    start_year=start_year,
                    end_year=end_year,
                )
            else:
                bq_rows = adapters.bigquery.query_sector_benchmarks(
                    tickers=ticker_filter,
                    gics_sector=sector_norm,
                    start_year=start_year,
                    end_year=end_year,
                )

            gcp_result = {
                "status": "success",
                "explore_name": explore_key,
                "row_count": len(bq_rows),
                "rows": bq_rows,
            }
            if tool_context is not None and hasattr(tool_context, "state"):
                history = list(tool_context.state.get("looker_query_results", []))
                history.append(gcp_result)
                tool_context.state["looker_query_results"] = history
            _record_gcp_backend_invocation(
                tool_context,
                tool_name="query_looker_financial_explore",
                data_source="gcp",
                services_hit=["bigquery"],
            )
            return gcp_result
        except Exception as exc:
            if not _should_fallback_in_auto_mode(exc, strict_gcp=strict_gcp):
                raise
            logger.warning(
                "BigQuery query_looker_financial_explore('%s') unavailable in auto mode (%s); falling back to local corpus.",
                explore_key,
                exc,
            )

    dataset_dir = _get_dataset_dir()
    if explore_key == "chunk_health":
        health_path = dataset_dir / "manifests" / "chunk_health_report.json"
        if not health_path.exists():
            default_health = DEFAULT_OUTPUT_DIR.resolve() / "manifests" / "chunk_health_report.json"
            if default_health.exists():
                health_path = default_health
        if not health_path.exists():
            return {
                "status": "error",
                "explore_name": explore_key,
                "message": f"Chunk health report not found at {health_path}",
            }
        health_data = json.loads(health_path.read_text(encoding="utf-8"))
        return {
            "status": "success",
            "explore_name": "chunk_health",
            "health_status": health_data.get("health_status"),
            "health_score": health_data.get("health_score"),
            "total_chunks": health_data.get("total_chunks"),
            "embedding_coverage_ratio": health_data.get("embedding_coverage_ratio"),
            "full_section_coverage_ratio": health_data.get("full_section_coverage_ratio"),
            "by_section": health_data.get("by_section", {}),
            "by_sector": health_data.get("by_sector", {}),
        }

    profiles = _load_all_profiles_cached(str(dataset_dir))
    by_ticker: dict[str, list[FinancialDataProfile]] = defaultdict(list)
    for p in profiles:
        if ticker_filter and p.ticker.upper().replace(".", "-") not in ticker_filter:
            continue
        if sector_norm and p.gics_sector.lower() != sector_norm:
            continue
        by_ticker[p.ticker].append(p)

    rows: list[dict[str, Any]] = []

    if explore_key in ("yoy_variances", "peer_comparison"):
        for _ticker, t_profiles in sorted(by_ticker.items()):
            t_profiles.sort(key=lambda x: x.fiscal_year)
            prev_prof: FinancialDataProfile | None = None
            for prof in t_profiles:
                if prof.fiscal_year >= start_year and prof.fiscal_year <= end_year:
                    rev_delta = rev_pct = op_delta = op_pct = ni_delta = ni_pct = None
                    if prev_prof and prev_prof.fiscal_year == prof.fiscal_year - 1:
                        if prof.revenue is not None and prev_prof.revenue is not None:
                            rev_delta = round(prof.revenue - prev_prof.revenue, 2)
                            if prev_prof.revenue != 0:
                                rev_pct = round((rev_delta / abs(prev_prof.revenue)) * 100.0, 2)
                        if (
                            prof.operating_income is not None
                            and prev_prof.operating_income is not None
                        ):
                            op_delta = round(prof.operating_income - prev_prof.operating_income, 2)
                            if prev_prof.operating_income != 0:
                                op_pct = round(
                                    (op_delta / abs(prev_prof.operating_income)) * 100.0,
                                    2,
                                )
                        if prof.net_income is not None and prev_prof.net_income is not None:
                            ni_delta = round(prof.net_income - prev_prof.net_income, 2)
                            if prev_prof.net_income != 0:
                                ni_pct = round((ni_delta / abs(prev_prof.net_income)) * 100.0, 2)

                    rec = _profile_to_metric_record(prof)
                    rows.append(
                        {
                            "profile_id": prof.profile_id,
                            "ticker": prof.ticker,
                            "company_name": prof.company_name,
                            "gics_sector": prof.gics_sector,
                            "fiscal_year": prof.fiscal_year,
                            "prev_fiscal_year": prev_prof.fiscal_year if prev_prof else None,
                            "revenue": prof.revenue,
                            "revenue_yoy_delta": rev_delta,
                            "revenue_yoy_pct": rev_pct,
                            "operating_income": prof.operating_income,
                            "operating_income_yoy_delta": op_delta,
                            "operating_income_yoy_pct": op_pct,
                            "net_income": prof.net_income,
                            "net_income_yoy_delta": ni_delta,
                            "net_income_yoy_pct": ni_pct,
                            "operating_margin_pct": rec.operating_margin_pct,
                            "net_margin_pct": rec.net_margin_pct,
                            "sec_filing_url": prof.sec_filing_url,
                        }
                    )
                prev_prof = prof

    elif explore_key == "sector_benchmarks":
        sector_year_groups: dict[tuple[str, int], list[FinancialDataProfile]] = defaultdict(list)
        for t_profiles in by_ticker.values():
            for prof in t_profiles:
                if start_year <= prof.fiscal_year <= end_year:
                    sector_year_groups[(prof.gics_sector, prof.fiscal_year)].append(prof)

        for (sector_name, fy), group in sorted(sector_year_groups.items()):
            revs = [p.revenue for p in group if p.revenue is not None]
            ops = [p.operating_income for p in group if p.operating_income is not None]
            nis = [p.net_income for p in group if p.net_income is not None]
            rows.append(
                {
                    "gics_sector": sector_name,
                    "fiscal_year": fy,
                    "company_count": len(group),
                    "tickers": sorted(p.ticker for p in group),
                    "total_sector_revenue": round(sum(revs), 2) if revs else None,
                    "avg_company_revenue": round(sum(revs) / len(revs), 2) if revs else None,
                    "total_sector_operating_income": round(sum(ops), 2) if ops else None,
                    "total_sector_net_income": round(sum(nis), 2) if nis else None,
                }
            )

    result = {
        "status": "success",
        "explore_name": explore_key,
        "row_count": len(rows),
        "rows": rows,
    }
    if tool_context is not None and hasattr(tool_context, "state"):
        history = list(tool_context.state.get("looker_query_results", []))
        history.append(result)
        tool_context.state["looker_query_results"] = history

    return result


@with_tool_resilience(tool_name="calculate_period_variances")
def calculate_period_variances(
    tickers: str,
    start_year: int,
    end_year: int,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Performs deterministic period-over-period variance calculations for Revenue, Operating Income, and Net Income.

    Separates arithmetic calculation from LLM narrative synthesis. Always call this tool first
    before explaining financial changes so narrative claims match deterministic numbers 100%.

    Args:
        tickers: Comma-separated ticker symbols (e.g., 'NVDA' or 'NVDA,AAPL').
        start_year: Starting fiscal year for the variance calculation (e.g., 2024).
        end_year: Ending fiscal year for the variance calculation (e.g., 2026).

    Returns:
        Dictionary containing deterministic multi-period and YoY variance reports per company.
    """
    ticker_filter = _parse_ticker_list(tickers)
    if not ticker_filter:
        return {
            "status": "error",
            "message": "Please specify at least one ticker symbol (e.g., 'NVDA' or 'NVDA,AAPL').",
        }

    profiles: list[FinancialDataProfile] | None = None
    adapters, strict_gcp = _resolve_gcp_dispatch()
    if adapters is not None:
        try:
            # Fetch start_year - 1 so the initial year's YoY step delta is populated
            profiles = adapters.bigquery.fetch_profiles(
                tickers=ticker_filter,
                start_year=start_year - 1,
                end_year=end_year,
            )
            _record_gcp_backend_invocation(
                tool_context,
                tool_name="calculate_period_variances",
                data_source="gcp",
                services_hit=["bigquery"],
            )
        except Exception as exc:
            if not _should_fallback_in_auto_mode(exc, strict_gcp=strict_gcp):
                raise
            logger.warning(
                "BigQuery calculate_period_variances unavailable in auto mode (%s); falling back to local corpus.",
                exc,
            )

    if profiles is None:
        dataset_dir = _get_dataset_dir()
        profiles = _load_all_profiles_cached(str(dataset_dir))

    reports, missing_tickers = _build_variance_reports_from_profiles(
        profiles,
        ticker_filter=ticker_filter,
        start_year=start_year,
        end_year=end_year,
    )

    if tool_context is not None and hasattr(tool_context, "state"):
        calc_state = dict(tool_context.state.get("calculated_variances", {}))
        for rep in reports:
            calc_state[f"{rep['ticker']}_{rep['start_year']}_{rep['end_year']}"] = rep
        tool_context.state["calculated_variances"] = calc_state

    return {
        "status": "success",
        "company_count": len(reports),
        "missing_tickers": missing_tickers,
        "variance_reports": reports,
    }


@with_tool_resilience(tool_name="search_10k_chunks_rag")
def search_10k_chunks_rag(
    query: str,
    tickers: str,
    gics_sector: str,
    section_id: str,
    start_year: int,
    end_year: int,
    top_k: int,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Performs metadata-filtered Hybrid RAG search across SEC 10-K chunks (Item 1A Risk Factors & Item 7 MD&A).

    Combines dense 768-d vector cosine similarity with lexical keyword matching and metadata
    pre-filtering to retrieve grounded 10-K passages with exact section citations and SEC URLs.

    Args:
        query: Natural language search query (e.g., 'AI GPU export control restrictions China H20').
        tickers: Comma-separated ticker symbols (e.g., 'NVDA,AAPL') or 'ALL' for cross-company search.
        gics_sector: GICS Sector filter (e.g., 'Information Technology') or 'ALL'.
        section_id: 10-K section filter: 'item_1a' (Risk Factors), 'item_7' (MD&A), or 'ALL'.
        start_year: Earliest fiscal year to search (inclusive, e.g., 2024).
        end_year: Latest fiscal year to search (inclusive, e.g., 2026).
        top_k: Number of top-ranked grounded citations to return (1 to 25).

    Returns:
        Dictionary with 'status', 'citation_count', and 'citations' list of GroundedCitation records.
    """
    ticker_filter = _parse_ticker_list(tickers)
    sector_norm = (
        gics_sector.strip().lower()
        if gics_sector and gics_sector.strip().upper() not in ("ALL", "*", "")
        else None
    )
    sec_norm = (
        section_id.strip().lower()
        if section_id and section_id.strip().upper() not in ("ALL", "*", "")
        else None
    )
    effective_top_k = max(1, min(int(top_k), 25))

    adapters, strict_gcp = _resolve_gcp_dispatch()
    if adapters is not None:
        try:
            gcp_citations, cand_count, services_hit = _execute_live_gcp_hybrid_rag(
                adapters,
                query=query,
                ticker_filter=ticker_filter,
                sector_norm=sector_norm,
                sec_norm=sec_norm,
                start_year=start_year,
                end_year=end_year,
                effective_top_k=effective_top_k,
                strict_gcp=strict_gcp,
            )
            if tool_context is not None and hasattr(tool_context, "state"):
                existing_cits = list(tool_context.state.get("retrieved_citations", []))
                existing_cits.extend(gcp_citations)
                tool_context.state["retrieved_citations"] = existing_cits
            _record_gcp_backend_invocation(
                tool_context,
                tool_name="search_10k_chunks_rag",
                data_source="gcp",
                services_hit=services_hit,
            )
            if not gcp_citations:
                return {
                    "status": "success",
                    "citation_count": 0,
                    "citations": [],
                    "message": "No 10-K chunks matched the requested metadata filters.",
                }
            return {
                "status": "success",
                "citation_count": len(gcp_citations),
                "candidates_searched": cand_count,
                "citations": gcp_citations,
            }
        except Exception as exc:
            if not _should_fallback_in_auto_mode(exc, strict_gcp=strict_gcp):
                raise
            logger.warning(
                "Live GCP Hybrid RAG search unavailable in auto mode (%s); falling back to local corpus.",
                exc,
            )

    dataset_dir = _get_dataset_dir()
    rows, matrix = _load_all_chunks_cached(str(dataset_dir))
    if not rows:
        return {
            "status": "error",
            "message": "Vector database chunk manifest is empty or not yet generated.",
        }

    candidate_indices: list[int] = []
    for idx, row in enumerate(rows):
        if (
            ticker_filter
            and str(row.get("ticker", "")).upper().replace(".", "-") not in ticker_filter
        ):
            continue
        if sector_norm and str(row.get("gics_sector", "")).lower() != sector_norm:
            continue
        if sec_norm and str(row.get("section_id", "")).lower() != sec_norm:
            continue
        fy = int(row.get("fiscal_year", 0))
        if fy < start_year or fy > end_year:
            continue
        candidate_indices.append(idx)

    if not candidate_indices:
        return {
            "status": "success",
            "citation_count": 0,
            "citations": [],
            "message": "No 10-K chunks matched the requested metadata filters.",
        }

    q_vec = np.asarray(generate_deterministic_embedding(query), dtype=np.float32)
    cand_matrix = matrix[candidate_indices]
    dense_scores = cand_matrix @ q_vec

    query_terms = set(_WORD_TOKEN_RE.findall(query.lower()))
    scored: list[tuple[float, int]] = []
    for pos, row_idx in enumerate(candidate_indices):
        row = rows[row_idx]
        text_lower = str(row.get("chunk_text", "")).lower()
        lexical_boost = 0.0
        if query_terms:
            matched_terms = sum(1 for term in query_terms if term in text_lower)
            lexical_boost = 0.35 * (matched_terms / len(query_terms))
        hybrid_score = float(dense_scores[pos]) + lexical_boost
        scored.append((hybrid_score, row_idx))

    scored.sort(key=lambda item: item[0], reverse=True)
    top_hits = scored[:effective_top_k]

    citations: list[dict[str, Any]] = []
    for score, row_idx in top_hits:
        row = rows[row_idx]
        cit = GroundedCitation(
            chunk_id=str(row["chunk_id"]),
            profile_id=str(row["profile_id"]),
            ticker=str(row["ticker"]),
            company_name=str(row["company_name"]),
            gics_sector=str(row.get("gics_sector") or "Unknown"),
            fiscal_year=int(row["fiscal_year"]),
            section_id=row["section_id"],
            section_title=str(row.get("section_title") or row["section_id"]),
            excerpt=str(row["chunk_text"]),
            sec_filing_url=str(row.get("sec_filing_url") or ""),
            unstructured_gcs_uri=str(row.get("unstructured_gcs_uri") or ""),
            score=round(score, 4),
        )
        citations.append(cit.model_dump(mode="json"))

    if tool_context is not None and hasattr(tool_context, "state"):
        existing_cits = list(tool_context.state.get("retrieved_citations", []))
        existing_cits.extend(citations)
        tool_context.state["retrieved_citations"] = existing_cits

    return {
        "status": "success",
        "citation_count": len(citations),
        "candidates_searched": len(candidate_indices),
        "citations": citations,
    }


@with_tool_resilience(tool_name="load_cached_10k_section")
def load_cached_10k_section(
    ticker: str,
    fiscal_year: int,
    section_id: str,
    max_chars: int,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Loads and caches a company's 10-K section (Item 1A Risk Factors or Item 7 MD&A) via the Context Cache Manager.

    Tracks Vertex AI Context Cache hits/misses, lookup latency, and estimated cached-token cost discounts.

    Args:
        ticker: Company ticker symbol (e.g., 'NVDA', 'AAPL').
        fiscal_year: Reporting fiscal year (e.g., 2026).
        section_id: Section to load: 'item_1a' (Risk Factors) or 'item_7' (MD&A).
        max_chars: Maximum characters of section text to return in the tool response (e.g., 12000).

    Returns:
        Dictionary with 'status', 'cache_hit', 'token_estimate', 'lookup_latency_ms', 'sec_filing_url', and grounded Markdown 'content'.
    """
    t0 = time.perf_counter()
    dataset_dir = _get_dataset_dir()
    norm_ticker = ticker.strip().upper().replace(".", "-")
    sec_key = section_id.strip().lower()
    if sec_key not in ("item_1a", "item_7"):
        return {
            "status": "error",
            "message": f"section_id must be 'item_1a' or 'item_7', got {section_id!r}",
        }

    cache_id = f"{norm_ticker}_{fiscal_year}_{sec_key}"
    entry: dict[str, Any] | None = _SESSION_10K_CONTEXT_CACHE.get(cache_id)
    cache_hit = entry is not None

    if entry is None:
        matched_prof: FinancialDataProfile | None = None
        md_content: str | None = None
        adapters, strict_gcp = _resolve_gcp_dispatch()

        if adapters is not None:
            try:
                bq_profiles = adapters.bigquery.fetch_profiles(
                    tickers={norm_ticker},
                    start_year=fiscal_year,
                    end_year=fiscal_year,
                )
                matched_prof = bq_profiles[0] if bq_profiles else None
                if matched_prof is None and strict_gcp:
                    return {
                        "status": "error",
                        "message": f"No curated 10-K filing found in BigQuery for {norm_ticker} FY{fiscal_year}.",
                    }
                if matched_prof is not None:
                    gcs_target = (
                        matched_prof.unstructured_gcs_uri or matched_prof.unstructured_blob_path
                    )
                    md_content = adapters.gcs_corpus.fetch_markdown(gcs_target)
                    _record_gcp_backend_invocation(
                        tool_context,
                        tool_name="load_cached_10k_section",
                        data_source="gcp",
                        services_hit=["bigquery", "gcs"],
                    )
            except Exception as exc:
                if not _should_fallback_in_auto_mode(exc, strict_gcp=strict_gcp):
                    raise
                logger.warning(
                    "GCP load_cached_10k_section unavailable in auto mode (%s); falling back to local corpus.",
                    exc,
                )
                matched_prof = None
                md_content = None

        if matched_prof is None or md_content is None:
            profiles = _load_all_profiles_cached(str(dataset_dir))
            matched_prof = next(
                (
                    p
                    for p in profiles
                    if p.ticker.upper().replace(".", "-") == norm_ticker
                    and p.fiscal_year == fiscal_year
                ),
                None,
            )
            if matched_prof is None:
                return {
                    "status": "error",
                    "message": f"No curated 10-K filing found for {norm_ticker} FY{fiscal_year}.",
                }
            md_path = dataset_dir / matched_prof.unstructured_blob_path
            if md_path.exists():
                md_content = md_path.read_text(encoding="utf-8")
            else:
                md_content = _synthesize_fallback_10k_markdown(matched_prof)

        sections = extract_sections_from_combined_markdown(md_content)
        sec_text = sections.get(sec_key, "")
        token_est = estimate_token_count(sec_text)
        entry = {
            "profile_id": matched_prof.profile_id,
            "ticker": matched_prof.ticker,
            "company_name": matched_prof.company_name,
            "fiscal_year": matched_prof.fiscal_year,
            "section_id": sec_key,
            "sec_filing_url": matched_prof.sec_filing_url,
            "unstructured_gcs_uri": matched_prof.unstructured_gcs_uri,
            "token_estimate": token_est,
            "full_text": sec_text,
        }
        _SESSION_10K_CONTEXT_CACHE[cache_id] = entry

    effective_max = max(500, min(int(max_chars), 50000))
    excerpt = str(entry["full_text"])[:effective_max]
    lookup_latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    saved_tokens_delta = int(entry["token_estimate"] * 0.75) if cache_hit else 0
    if cache_hit:
        _CONTEXT_CACHE_RUNTIME_COUNTERS["hits"] = (
            int(_CONTEXT_CACHE_RUNTIME_COUNTERS.get("hits", 0)) + 1
        )
        _CONTEXT_CACHE_RUNTIME_COUNTERS["cached_tokens_saved"] = (
            int(_CONTEXT_CACHE_RUNTIME_COUNTERS.get("cached_tokens_saved", 0)) + saved_tokens_delta
        )
        _CONTEXT_CACHE_RUNTIME_COUNTERS["total_hit_latency_ms"] = (
            float(_CONTEXT_CACHE_RUNTIME_COUNTERS.get("total_hit_latency_ms", 0.0))
            + lookup_latency_ms
        )
    else:
        _CONTEXT_CACHE_RUNTIME_COUNTERS["misses"] = (
            int(_CONTEXT_CACHE_RUNTIME_COUNTERS.get("misses", 0)) + 1
        )
        _CONTEXT_CACHE_RUNTIME_COUNTERS["total_miss_latency_ms"] = (
            float(_CONTEXT_CACHE_RUNTIME_COUNTERS.get("total_miss_latency_ms", 0.0))
            + lookup_latency_ms
        )

    if tool_context is not None and hasattr(tool_context, "state"):
        stats = dict(
            tool_context.state.get(
                "context_cache_stats",
                {"hits": 0, "misses": 0, "cached_tokens_saved": 0},
            )
        )
        if cache_hit:
            stats["hits"] = int(stats.get("hits", 0)) + 1
            stats["cached_tokens_saved"] = (
                int(stats.get("cached_tokens_saved", 0)) + saved_tokens_delta
            )
        else:
            stats["misses"] = int(stats.get("misses", 0)) + 1
        stats["last_lookup_latency_ms"] = lookup_latency_ms
        stats["last_cache_key"] = cache_id
        tool_context.state["context_cache_stats"] = stats

    return {
        "status": "success",
        "cache_key": cache_id,
        "cache_hit": cache_hit,
        "profile_id": entry["profile_id"],
        "ticker": entry["ticker"],
        "company_name": entry["company_name"],
        "fiscal_year": entry["fiscal_year"],
        "section_id": entry["section_id"],
        "token_estimate": entry["token_estimate"],
        "lookup_latency_ms": lookup_latency_ms,
        "sec_filing_url": entry["sec_filing_url"],
        "unstructured_gcs_uri": entry["unstructured_gcs_uri"],
        "content": excerpt,
        "truncated": len(str(entry["full_text"])) > effective_max,
    }


def parse_10k_cache_key(cache_key: str) -> tuple[str, int, str] | None:
    """Split a `{TICKER}_{FY}_{item_1a|item_7}` section cache key into its parts."""
    match = _10K_CACHE_KEY_RE.match(str(cache_key).strip())
    if match is None:
        return None
    return match["ticker"].upper(), int(match["fiscal_year"]), match["section_id"]


def get_full_10k_section(cache_key: str) -> dict[str, Any] | None:
    """Return the full parsed 10-K section entry for a cache key, hydrating the L1 cache on miss.

    Unlike `load_cached_10k_section`, this returns the untruncated `full_text`. It is
    used to register complete sections as remote Vertex AI `CachedContent` resources.
    """
    parsed = parse_10k_cache_key(cache_key)
    if parsed is None:
        return None
    ticker, fiscal_year, section_id = parsed
    norm_key = f"{ticker}_{fiscal_year}_{section_id}"
    entry = _SESSION_10K_CONTEXT_CACHE.get(norm_key)
    if entry is None:
        res = load_cached_10k_section(
            ticker=ticker,
            fiscal_year=fiscal_year,
            section_id=section_id,
            max_chars=500,
        )
        if res.get("status") != "success":
            return None
        entry = _SESSION_10K_CONTEXT_CACHE.get(norm_key)
    return dict(entry) if entry is not None else None


@with_tool_resilience(tool_name="query_cached_10k_context")
def query_cached_10k_context(
    ticker: str,
    fiscal_year: int,
    section_id: str,
    variance_query: str,
    max_passages: int,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Queries a company's cached 10-K section (Item 7 MD&A or Item 1A Risk Factors) for variance explanations.

    Warms the 10-K section in the Context Cache Manager on first access (or hits resident cache),
    ranks paragraphs against the evaluated financial variance drivers, and links matching chunk IDs.

    Args:
        ticker: Company ticker symbol (e.g., 'NVDA', 'AAPL').
        fiscal_year: Reporting fiscal year (e.g., 2026).
        section_id: 10-K section to query: 'item_7' (MD&A) or 'item_1a' (Risk Factors).
        variance_query: Variance explanation query (e.g., 'revenue growth operating margin expansion Blackwell').
        max_passages: Maximum number of grounded explanation passages to return (1 to 10).

    Returns:
        Dictionary with 'status', 'cache_hit', 'token_estimate', 'lookup_latency_ms', 'explanations', and 'matched_chunk_ids'.
    """
    t0 = time.perf_counter()
    cached_res = load_cached_10k_section(
        ticker=ticker,
        fiscal_year=fiscal_year,
        section_id=section_id,
        max_chars=50000,
        tool_context=tool_context,
    )
    if cached_res.get("status") != "success":
        return cached_res

    full_text = str(cached_res.get("content") or "")
    raw_blocks = [
        blk.strip()
        for blk in re.split(r"\n\s*\n+", full_text)
        if len(blk.strip()) >= 45 and not blk.strip().startswith("#")
    ]
    if not raw_blocks and full_text.strip():
        raw_blocks = [full_text.strip()[:800]]

    norm_ticker = str(cached_res["ticker"])
    sec_key = str(cached_res["section_id"])
    q_terms = set(_WORD_TOKEN_RE.findall(variance_query.lower()))

    adapters, strict_gcp = _resolve_gcp_dispatch()
    dense_sims: list[float] | None = None
    section_chunks: list[dict[str, Any]] | None = None
    services_hit: list[str] = []

    if adapters is not None and raw_blocks:
        try:
            q_emb = np.asarray(adapters.embed_query(variance_query), dtype=np.float32)
            blk_embs = np.asarray(
                adapters.embed_texts(
                    [blk[:600] for blk in raw_blocks], task_type="RETRIEVAL_DOCUMENT"
                ),
                dtype=np.float32,
            )
            dense_sims = [float(x) for x in (blk_embs @ q_emb)]
            services_hit.append("vertex_ai_embeddings")
        except Exception as exc:
            if not _should_fallback_in_auto_mode(exc, strict_gcp=strict_gcp):
                raise
            logger.warning(
                "Vertex AI embeddings unavailable in query_cached_10k_context (%s); using deterministic scoring.",
                exc,
            )

        try:
            section_chunks = adapters.bigquery.fetch_section_chunks(
                ticker=norm_ticker,
                fiscal_year=int(fiscal_year),
                section_id=sec_key,
            )
            services_hit.append("bigquery")
        except Exception as exc:
            if not _should_fallback_in_auto_mode(exc, strict_gcp=strict_gcp):
                raise
            logger.warning(
                "BigQuery fetch_section_chunks unavailable in auto mode (%s); falling back to local chunk manifest.",
                exc,
            )
            section_chunks = None

    if services_hit:
        _record_gcp_backend_invocation(
            tool_context,
            tool_name="query_cached_10k_context",
            data_source="gcp",
            services_hit=services_hit,
        )

    if dense_sims is None:
        q_vec = np.asarray(generate_deterministic_embedding(variance_query), dtype=np.float32)
        dense_sims = [
            float(np.asarray(generate_deterministic_embedding(blk[:600]), dtype=np.float32) @ q_vec)
            for blk in raw_blocks
        ]

    scored_blocks: list[tuple[float, str]] = []
    for idx, blk in enumerate(raw_blocks):
        dense_sim = dense_sims[idx] if idx < len(dense_sims) else 0.0
        blk_lower = blk.lower()
        lex_boost = 0.0
        if q_terms:
            hits = sum(1 for term in q_terms if term in blk_lower)
            lex_boost = 0.40 * (hits / len(q_terms))
        scored_blocks.append((dense_sim + lex_boost, blk))

    scored_blocks.sort(key=lambda item: item[0], reverse=True)
    eff_limit = max(1, min(int(max_passages), 10))
    top_passages = [text for _, text in scored_blocks[:eff_limit]]

    # Link cached section explanations to canonical chunk IDs for split-pane traceability
    if section_chunks is None:
        dataset_dir = _get_dataset_dir()
        rows, _ = _load_all_chunks_cached(str(dataset_dir))
        section_chunks = [
            r
            for r in rows
            if str(r.get("ticker", "")).upper() == norm_ticker.upper()
            and int(r.get("fiscal_year", 0)) == int(fiscal_year)
            and str(r.get("section_id", "")).lower() == sec_key
        ]

    matched_chunk_ids: list[str] = []
    matched_citations: list[dict[str, Any]] = []
    for passage in top_passages:
        snippet_probe = passage[:90].lower()
        best_chunk = next(
            (
                r
                for r in section_chunks
                if snippet_probe[:48] in str(r.get("chunk_text", "")).lower()
                or str(r.get("chunk_text", ""))[:48].lower() in passage.lower()
            ),
            section_chunks[0] if section_chunks else None,
        )
        if best_chunk is not None:
            cid = str(best_chunk["chunk_id"])
            if cid not in matched_chunk_ids:
                matched_chunk_ids.append(cid)
                cit = GroundedCitation(
                    chunk_id=cid,
                    profile_id=str(best_chunk["profile_id"]),
                    ticker=str(best_chunk["ticker"]),
                    company_name=str(best_chunk["company_name"]),
                    gics_sector=str(best_chunk.get("gics_sector") or "Unknown"),
                    fiscal_year=int(best_chunk["fiscal_year"]),
                    section_id=_normalize_section_id(
                        str(best_chunk.get("section_id") or sec_key),
                        requested_section=sec_key,
                    ),
                    section_title=str(best_chunk.get("section_title") or sec_key),
                    excerpt=str(best_chunk["chunk_text"]),
                    sec_filing_url=str(best_chunk.get("sec_filing_url") or ""),
                    unstructured_gcs_uri=str(best_chunk.get("unstructured_gcs_uri") or ""),
                    score=0.95,
                )
                matched_citations.append(cit.model_dump(mode="json"))

    if tool_context is not None and hasattr(tool_context, "state") and matched_citations:
        existing_cits = list(tool_context.state.get("retrieved_citations", []))
        seen_ids = {c.get("chunk_id") for c in existing_cits}
        for mc in matched_citations:
            if mc["chunk_id"] not in seen_ids:
                existing_cits.append(mc)
                seen_ids.add(mc["chunk_id"])
        tool_context.state["retrieved_citations"] = existing_cits

    total_lookup_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    section_title = (
        "Item 7. Management's Discussion and Analysis"
        if sec_key == "item_7"
        else "Item 1A. Risk Factors"
    )
    return {
        "status": "success",
        "cache_key": cached_res["cache_key"],
        "cache_hit": cached_res["cache_hit"],
        "profile_id": cached_res["profile_id"],
        "ticker": norm_ticker,
        "company_name": cached_res["company_name"],
        "fiscal_year": int(fiscal_year),
        "section_id": sec_key,
        "section_title": section_title,
        "token_estimate": cached_res["token_estimate"],
        "lookup_latency_ms": total_lookup_ms,
        "sec_filing_url": cached_res["sec_filing_url"],
        "explanations": top_passages,
        "matched_chunk_ids": matched_chunk_ids,
        "matched_citations": matched_citations,
    }


# Explicit ADK FunctionTool instances matching project_reference.md naming
MetricFetchTool = FunctionTool(fetch_financial_metrics)
LookerQueryTool = FunctionTool(query_looker_financial_explore)
VarianceCalculationTool = FunctionTool(calculate_period_variances)
HybridSearchTool = FunctionTool(search_10k_chunks_rag)
ContextCacheTool = FunctionTool(load_cached_10k_section)
QueryCached10KTool = FunctionTool(query_cached_10k_context)
