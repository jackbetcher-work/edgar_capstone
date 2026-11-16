"""Publish the curated SEC 10-K corpus to its serving stores on Google Cloud.

Two publishing steps, both driven by manifests already staged in the GCS corpus bucket:

1. ``reload_financial_metrics_table``: BigQuery load job from
   ``gs://<bucket>/manifests/bigquery_financials.jsonl`` into the native, partitioned
   ``sec_edgar.financial_metrics`` table, using the explicit schema from
   ``terraform/schemas/financial_metrics.json``.
2. ``import_vertex_ai_search_documents``: Discovery Engine ``documents:import`` from
   ``gs://<bucket>/manifests/vertex_ai_search_metadata.jsonl`` into the Vertex AI Search
   data store, followed by long-running-operation polling. ``search_vertex_ai_datastore``
   runs a post-import smoke query against the ``default_search`` serving config.

All calls are plain REST over ``httpx`` with Application Default Credentials bearer tokens,
mirroring the rest of the code base (no extra client libraries required).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import google.auth
import httpx
from google.auth.transport.requests import Request as GoogleAuthRequest

from edgar_capstone.dataset.gcs_storage import (
    DEFAULT_BUCKET_NAME,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
    normalize_bucket_name,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]

_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
BIGQUERY_API_ROOT = "https://bigquery.googleapis.com/bigquery/v2"
DISCOVERY_ENGINE_API_ROOT = "https://discoveryengine.googleapis.com/v1"

# --- BigQuery financial_metrics defaults -------------------------------------------------
DEFAULT_BIGQUERY_DATASET_ID = "sec_edgar"
DEFAULT_FINANCIAL_METRICS_TABLE_ID = "financial_metrics"
DEFAULT_BIGQUERY_LOCATION = DEFAULT_LOCATION
FINANCIALS_MANIFEST_BLOB = "manifests/bigquery_financials.jsonl"
DEFAULT_FINANCIAL_METRICS_SCHEMA_PATH = (
    REPO_ROOT / "terraform" / "schemas" / "financial_metrics.json"
)
# Mirrors `google_bigquery_table.financial_metrics` in terraform/bigquery.tf. Restating the
# existing spec on a WRITE_TRUNCATE load is a no-op that guarantees the table keeps it.
FINANCIAL_METRICS_RANGE_PARTITIONING: dict[str, Any] = {
    "field": "fiscal_year",
    "range": {"start": "2015", "end": "2036", "interval": "1"},
}
FINANCIAL_METRICS_CLUSTERING_FIELDS: tuple[str, ...] = ("gics_sector", "ticker")

# --- Vertex AI Search (Discovery Engine) defaults ----------------------------------------
DEFAULT_VERTEX_SEARCH_DATASTORE_ID = "sec-edgar-10k-corpus-ds"
DEFAULT_VERTEX_SEARCH_LOCATION = "global"
DEFAULT_VERTEX_SEARCH_COLLECTION = "default_collection"
DEFAULT_VERTEX_SEARCH_BRANCH = "default_branch"
DEFAULT_VERTEX_SEARCH_SERVING_CONFIG = "default_search"
VERTEX_SEARCH_MANIFEST_BLOB = "manifests/vertex_ai_search_metadata.jsonl"
VERTEX_SEARCH_IMPORT_ERRORS_PREFIX = "vertex_ai_search/import_errors/"
DEFAULT_SMOKE_QUERY = "Agilent risk factors"

ReconciliationMode = Literal["INCREMENTAL", "FULL"]
RECONCILIATION_MODES: tuple[str, ...] = ("INCREMENTAL", "FULL")


class CloudPublishError(RuntimeError):
    """Raised when a BigQuery load job or Discovery Engine operation fails."""


class CloudPublishTimeoutError(CloudPublishError):
    """Raised when a job/operation is still running after the polling deadline."""


# ------------------------------------------------------------------------------------------
# Shared REST helpers
# ------------------------------------------------------------------------------------------


def _bearer_token(credentials: Any | None) -> str:
    """Return a valid OAuth2 bearer token from `credentials` (or ADC when omitted)."""
    if credentials is None:
        credentials, _ = google.auth.default(scopes=[_CLOUD_PLATFORM_SCOPE])
    if not getattr(credentials, "valid", False) or not getattr(credentials, "token", None):
        credentials.refresh(GoogleAuthRequest())
    token = getattr(credentials, "token", None)
    if not token:
        raise CloudPublishError("Application Default Credentials did not return a bearer token.")
    return str(token)


def _auth_headers(token: str, *, quota_project_id: str | None, json_body: bool) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {token}"}
    if json_body:
        headers["Content-Type"] = "application/json"
    if quota_project_id:
        headers["x-goog-user-project"] = quota_project_id
    return headers


def _raise_for_http(resp: httpx.Response, action: str) -> None:
    if resp.status_code not in (200, 201):
        raise CloudPublishError(f"{action} failed (HTTP {resp.status_code}): {resp.text[:400]}")


def _to_int(value: Any) -> int | None:
    """Coerce int64 JSON values (which the Google APIs serialize as strings) to `int`."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def gcs_uri(bucket_name: str, blob_path: str) -> str:
    """Build a `gs://` URI from a (possibly `gs://`-prefixed) bucket name and blob path."""
    return f"gs://{normalize_bucket_name(bucket_name)}/{blob_path.lstrip('/')}"


# ------------------------------------------------------------------------------------------
# BigQuery load jobs
# ------------------------------------------------------------------------------------------


@dataclass(slots=True)
class BigQueryLoadJobResult:
    """Outcome of a BigQuery load job submission (and optional completion polling)."""

    job_id: str
    project_id: str
    location: str
    destination_table: str
    source_uris: list[str]
    state: str = "PENDING"
    output_rows: int | None = None
    input_files: int | None = None
    input_file_bytes: int | None = None
    bad_records: int | None = None
    schema_field_count: int | None = None
    schema_update_options: list[str] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.state == "DONE"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["done"] = self.done
        return payload


def load_bigquery_schema_fields(path: Path | str) -> list[dict[str, Any]]:
    """Load a BigQuery `TableFieldSchema` list (the terraform `schemas/*.json` format)."""
    schema_path = Path(path)
    data = json.loads(schema_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("fields"), list):
        data = data["fields"]
    if not isinstance(data, list) or not all(isinstance(f, dict) and "name" in f for f in data):
        raise ValueError(f"{schema_path} is not a BigQuery schema field list")
    return data


def resolve_schema_path(schema_path: Path | str | None) -> Path | None:
    """Resolve an explicit (or the default terraform) schema path; `None` if it is absent.

    Relative paths are resolved against the repository root, not the current directory.
    """
    if schema_path is None:
        candidate = DEFAULT_FINANCIAL_METRICS_SCHEMA_PATH
        return candidate if candidate.exists() else None
    candidate = Path(schema_path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    if not candidate.exists():
        raise FileNotFoundError(f"BigQuery schema file not found: {candidate}")
    return candidate


def run_bigquery_load_job(
    *,
    project_id: str,
    dataset_id: str,
    table_id: str,
    source_uris: Sequence[str],
    location: str = DEFAULT_BIGQUERY_LOCATION,
    schema_fields: list[dict[str, Any]] | None = None,
    schema_update_options: Sequence[str] | None = None,
    write_disposition: str = "WRITE_TRUNCATE",
    source_format: str = "NEWLINE_DELIMITED_JSON",
    range_partitioning: dict[str, Any] | None = None,
    clustering_fields: Sequence[str] | None = None,
    ignore_unknown_values: bool = False,
    max_bad_records: int = 0,
    job_id_prefix: str = "load",
    wait: bool = True,
    poll_timeout_seconds: float = 120.0,
    poll_interval_seconds: float | None = None,
    raise_on_timeout: bool = True,
    quota_project_id: str | None = None,
    http_client: httpx.Client | None = None,
    credentials: Any | None = None,
) -> BigQueryLoadJobResult:
    """Submit a BigQuery load job via `jobs.insert` and (optionally) poll it until `DONE`.

    Raises `CloudPublishError` on submission failure or a terminal `errorResult`, and
    `CloudPublishTimeoutError` if the job is still running at the deadline (unless
    `raise_on_timeout=False`, in which case a warning is logged and the pending result returned).

    BigQuery rejects `schemaUpdateOptions` combined with `WRITE_TRUNCATE` unless the destination
    is a single partition (`table$partition`); a plain `WRITE_TRUNCATE` with an explicit schema
    already replaces the table schema, so that combination is refused here up front.
    """
    update_options = list(schema_update_options or [])
    if update_options and write_disposition == "WRITE_TRUNCATE" and "$" not in table_id:
        raise ValueError(
            "schemaUpdateOptions require WRITE_APPEND (or WRITE_TRUNCATE on a `table$partition`); "
            "WRITE_TRUNCATE with an explicit schema already replaces the table schema."
        )

    token = _bearer_token(credentials)
    job_id = f"{job_id_prefix}_{uuid.uuid4().hex[:12]}"
    load_config: dict[str, Any] = {
        "sourceUris": list(source_uris),
        "destinationTable": {
            "projectId": project_id,
            "datasetId": dataset_id,
            "tableId": table_id,
        },
        "sourceFormat": source_format,
        "writeDisposition": write_disposition,
        "autodetect": False,
    }
    if schema_fields is not None:
        load_config["schema"] = {"fields": schema_fields}
    if update_options:
        load_config["schemaUpdateOptions"] = update_options
    if range_partitioning is not None:
        load_config["rangePartitioning"] = range_partitioning
    if clustering_fields:
        load_config["clustering"] = {"fields": list(clustering_fields)}
    if ignore_unknown_values:
        load_config["ignoreUnknownValues"] = True
    if max_bad_records:
        load_config["maxBadRecords"] = max_bad_records

    payload = {
        "jobReference": {"projectId": project_id, "jobId": job_id, "location": location},
        "configuration": {"load": load_config},
    }
    result = BigQueryLoadJobResult(
        job_id=job_id,
        project_id=project_id,
        location=location,
        destination_table=f"{project_id}.{dataset_id}.{table_id}",
        source_uris=list(source_uris),
        schema_field_count=len(schema_fields) if schema_fields is not None else None,
        schema_update_options=update_options,
    )

    client = http_client or httpx.Client(timeout=60.0)
    close_client = http_client is None
    try:
        resp = client.post(
            f"{BIGQUERY_API_ROOT}/projects/{project_id}/jobs",
            json=payload,
            headers=_auth_headers(token, quota_project_id=quota_project_id, json_body=True),
        )
        _raise_for_http(resp, f"BigQuery load job submission ({job_id})")
        logger.info("Submitted BigQuery load job %s -> %s", job_id, result.destination_table)
        if not wait:
            result.state = str(resp.json().get("status", {}).get("state") or "PENDING")
            return result

        status_url = f"{BIGQUERY_API_ROOT}/projects/{project_id}/jobs/{job_id}"
        interval = poll_interval_seconds
        if interval is None:
            interval = 1.5 if poll_timeout_seconds <= 120 else 5.0
        deadline = time.monotonic() + poll_timeout_seconds
        while time.monotonic() < deadline:
            poll_resp = client.get(
                status_url,
                params={"location": location},
                headers=_auth_headers(token, quota_project_id=quota_project_id, json_body=False),
            )
            if poll_resp.status_code == 200:
                job_data = poll_resp.json()
                status = job_data.get("status", {})
                result.state = str(status.get("state") or result.state)
                if result.done:
                    err = status.get("errorResult")
                    if err:
                        details = status.get("errors") or []
                        raise CloudPublishError(
                            f"BigQuery load job {job_id} failed: {err}"
                            + (f" (first errors: {details[:3]})" if details else "")
                        )
                    load_stats = job_data.get("statistics", {}).get("load", {})
                    result.output_rows = _to_int(load_stats.get("outputRows"))
                    result.input_files = _to_int(load_stats.get("inputFiles"))
                    result.input_file_bytes = _to_int(load_stats.get("inputFileBytes"))
                    result.bad_records = _to_int(load_stats.get("badRecords"))
                    return result
            else:
                logger.debug(
                    "BigQuery job poll for %s returned HTTP %s", job_id, poll_resp.status_code
                )
            time.sleep(interval)

        message = (
            f"BigQuery load job {job_id} still running after {poll_timeout_seconds:.0f}s; "
            f"check `bq show -j --location={location} {job_id}`."
        )
        if raise_on_timeout:
            raise CloudPublishTimeoutError(message)
        logger.warning(message)
        return result
    finally:
        if close_client:
            client.close()


def reload_financial_metrics_table(
    *,
    project_id: str = DEFAULT_PROJECT_ID,
    bucket_name: str = DEFAULT_BUCKET_NAME,
    dataset_id: str = DEFAULT_BIGQUERY_DATASET_ID,
    table_id: str = DEFAULT_FINANCIAL_METRICS_TABLE_ID,
    location: str = DEFAULT_BIGQUERY_LOCATION,
    source_uri: str | None = None,
    schema_path: Path | str | None = None,
    use_schema_file: bool = True,
    write_disposition: Literal["WRITE_TRUNCATE", "WRITE_APPEND"] = "WRITE_TRUNCATE",
    allow_field_addition: bool = True,
    keep_partitioning: bool = True,
    ignore_unknown_values: bool = False,
    wait: bool = True,
    timeout_seconds: float = 600.0,
    poll_interval_seconds: float | None = None,
    http_client: httpx.Client | None = None,
    credentials: Any | None = None,
) -> BigQueryLoadJobResult:
    """Reload native `sec_edgar.financial_metrics` from the GCS financials manifest.

    The explicit schema from `terraform/schemas/financial_metrics.json` (or `schema_path`) is
    sent with the job, so newly added NULLABLE columns land without a separate terraform apply:

    * `WRITE_TRUNCATE` (default) replaces both rows and schema with the supplied schema.
      BigQuery forbids `schemaUpdateOptions` here, so none are sent.
    * `WRITE_APPEND` sends `schemaUpdateOptions: ["ALLOW_FIELD_ADDITION"]` (when
      `allow_field_addition`) so the new column is added to the existing table schema.
    """
    uri = source_uri or gcs_uri(bucket_name, FINANCIALS_MANIFEST_BLOB)
    resolved_schema = resolve_schema_path(schema_path) if use_schema_file else None
    schema_fields = load_bigquery_schema_fields(resolved_schema) if resolved_schema else None
    if resolved_schema is None:
        logger.warning("No financial_metrics schema file found; relying on the table schema.")

    update_options: list[str] = []
    if allow_field_addition and schema_fields is not None:
        if write_disposition == "WRITE_APPEND" or "$" in table_id:
            update_options.append("ALLOW_FIELD_ADDITION")

    return run_bigquery_load_job(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
        source_uris=[uri],
        location=location,
        schema_fields=schema_fields,
        schema_update_options=update_options,
        write_disposition=write_disposition,
        range_partitioning=FINANCIAL_METRICS_RANGE_PARTITIONING if keep_partitioning else None,
        clustering_fields=FINANCIAL_METRICS_CLUSTERING_FIELDS if keep_partitioning else None,
        ignore_unknown_values=ignore_unknown_values,
        job_id_prefix=f"load_{table_id.split('$', 1)[0]}",
        wait=wait,
        poll_timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        raise_on_timeout=True,
        quota_project_id=project_id,
        http_client=http_client,
        credentials=credentials,
    )


# ------------------------------------------------------------------------------------------
# Vertex AI Search (Discovery Engine)
# ------------------------------------------------------------------------------------------


def vertex_search_datastore_path(
    *,
    project_id: str,
    data_store_id: str,
    location: str = DEFAULT_VERTEX_SEARCH_LOCATION,
    collection: str = DEFAULT_VERTEX_SEARCH_COLLECTION,
) -> str:
    return (
        f"projects/{project_id}/locations/{location}/collections/{collection}"
        f"/dataStores/{data_store_id}"
    )


def _discovery_engine_root(location: str) -> str:
    if location == "global":
        return DISCOVERY_ENGINE_API_ROOT
    return f"https://{location}-discoveryengine.googleapis.com/v1"


@dataclass(slots=True)
class VertexSearchImportResult:
    """Outcome of a Discovery Engine `documents:import` long-running operation."""

    operation_name: str
    data_store: str
    input_uris: list[str]
    reconciliation_mode: str
    error_gcs_prefix: str
    done: bool = False
    success_count: int | None = None
    failure_count: int | None = None
    total_count: int | None = None
    error_samples: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _apply_operation(result: VertexSearchImportResult, operation: dict[str, Any]) -> None:
    metadata = operation.get("metadata") or {}
    result.metadata = metadata
    result.done = bool(operation.get("done"))
    result.success_count = _to_int(metadata.get("successCount"))
    result.failure_count = _to_int(metadata.get("failureCount"))
    result.total_count = _to_int(metadata.get("totalCount"))
    response = operation.get("response") or {}
    samples = response.get("errorSamples") or []
    result.error_samples = [
        str(s.get("message") or s) if isinstance(s, dict) else str(s) for s in samples[:10]
    ]
    gcs_prefix = (response.get("errorConfig") or {}).get("gcsPrefix")
    if gcs_prefix:
        result.error_gcs_prefix = gcs_prefix


def poll_discovery_engine_operation(
    operation_name: str,
    *,
    location: str = DEFAULT_VERTEX_SEARCH_LOCATION,
    quota_project_id: str | None = None,
    timeout_seconds: float = 3600.0,
    initial_interval_seconds: float = 5.0,
    max_interval_seconds: float = 60.0,
    backoff_multiplier: float = 1.5,
    http_client: httpx.Client | None = None,
    credentials: Any | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Poll `GET /v1/{operation_name}` with exponential backoff until `done`.

    Returns the final operation JSON. Raises `CloudPublishError` if the operation finished
    with an `error`, and `CloudPublishTimeoutError` if it is still running at the deadline.
    """
    url = f"{_discovery_engine_root(location)}/{operation_name}"
    client = http_client or httpx.Client(timeout=60.0)
    close_client = http_client is None
    interval = initial_interval_seconds
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            token = _bearer_token(credentials)
            resp = client.get(
                url,
                headers=_auth_headers(token, quota_project_id=quota_project_id, json_body=False),
            )
            if resp.status_code == 200:
                operation = resp.json()
                if on_progress is not None:
                    on_progress(operation)
                if operation.get("done"):
                    if operation.get("error"):
                        raise CloudPublishError(
                            f"Discovery Engine operation {operation_name} failed: "
                            f"{operation['error']}"
                        )
                    return operation
            elif resp.status_code in (429,) or resp.status_code >= 500:
                logger.warning("Operation poll HTTP %s; retrying", resp.status_code)
            else:
                _raise_for_http(resp, f"Polling Discovery Engine operation {operation_name}")

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CloudPublishTimeoutError(
                    f"Discovery Engine operation {operation_name} not done after "
                    f"{timeout_seconds:.0f}s; re-poll with GET {url}"
                )
            time.sleep(min(interval, remaining))
            interval = min(interval * backoff_multiplier, max_interval_seconds)
    finally:
        if close_client:
            client.close()


def import_vertex_ai_search_documents(
    *,
    project_id: str = DEFAULT_PROJECT_ID,
    bucket_name: str = DEFAULT_BUCKET_NAME,
    data_store_id: str = DEFAULT_VERTEX_SEARCH_DATASTORE_ID,
    location: str = DEFAULT_VERTEX_SEARCH_LOCATION,
    collection: str = DEFAULT_VERTEX_SEARCH_COLLECTION,
    branch: str = DEFAULT_VERTEX_SEARCH_BRANCH,
    input_uris: Sequence[str] | None = None,
    error_gcs_prefix: str | None = None,
    reconciliation_mode: ReconciliationMode = "INCREMENTAL",
    wait: bool = True,
    timeout_seconds: float = 3600.0,
    initial_poll_interval_seconds: float = 5.0,
    max_poll_interval_seconds: float = 60.0,
    http_client: httpx.Client | None = None,
    credentials: Any | None = None,
) -> VertexSearchImportResult:
    """Import `vertex_ai_search_metadata.jsonl` documents into the Vertex AI Search data store.

    Each manifest line is a Discovery Engine `Document`
    (`{"id", "structData", "content": {"mimeType", "uri"}}`), hence `dataSchema: "document"`.
    """
    mode = reconciliation_mode.upper()
    if mode not in RECONCILIATION_MODES:
        raise ValueError(f"reconciliation_mode must be one of {RECONCILIATION_MODES}, got {mode}")

    uris = list(input_uris or [gcs_uri(bucket_name, VERTEX_SEARCH_MANIFEST_BLOB)])
    errors_prefix = error_gcs_prefix or gcs_uri(bucket_name, VERTEX_SEARCH_IMPORT_ERRORS_PREFIX)
    data_store = vertex_search_datastore_path(
        project_id=project_id, data_store_id=data_store_id, location=location, collection=collection
    )
    url = f"{_discovery_engine_root(location)}/{data_store}/branches/{branch}/documents:import"
    payload = {
        "gcsSource": {"inputUris": uris, "dataSchema": "document"},
        "reconciliationMode": mode,
        "errorConfig": {"gcsPrefix": errors_prefix},
    }

    client = http_client or httpx.Client(timeout=60.0)
    close_client = http_client is None
    try:
        token = _bearer_token(credentials)
        resp = client.post(
            url,
            json=payload,
            headers=_auth_headers(token, quota_project_id=project_id, json_body=True),
        )
        _raise_for_http(resp, f"Vertex AI Search documents:import into {data_store_id}")
        operation = resp.json()
        operation_name = str(operation.get("name") or "")
        if not operation_name:
            raise CloudPublishError(f"documents:import returned no operation name: {operation}")
        logger.info("Started Vertex AI Search import operation %s (%s)", operation_name, mode)

        result = VertexSearchImportResult(
            operation_name=operation_name,
            data_store=data_store,
            input_uris=uris,
            reconciliation_mode=mode,
            error_gcs_prefix=errors_prefix,
        )
        if operation.get("done") and operation.get("error"):
            raise CloudPublishError(
                f"Discovery Engine operation {operation_name} failed: {operation['error']}"
            )
        if wait and not operation.get("done"):

            def _log_progress(op: dict[str, Any]) -> None:
                meta = op.get("metadata") or {}
                logger.info(
                    "Import %s: success=%s failure=%s",
                    operation_name,
                    meta.get("successCount", 0),
                    meta.get("failureCount", 0),
                )

            operation = poll_discovery_engine_operation(
                operation_name,
                location=location,
                quota_project_id=project_id,
                timeout_seconds=timeout_seconds,
                initial_interval_seconds=initial_poll_interval_seconds,
                max_interval_seconds=max_poll_interval_seconds,
                http_client=client,
                credentials=credentials,
                on_progress=_log_progress,
            )
        _apply_operation(result, operation)
        return result
    finally:
        if close_client:
            client.close()


def search_vertex_ai_datastore(
    query: str = DEFAULT_SMOKE_QUERY,
    *,
    project_id: str = DEFAULT_PROJECT_ID,
    data_store_id: str = DEFAULT_VERTEX_SEARCH_DATASTORE_ID,
    location: str = DEFAULT_VERTEX_SEARCH_LOCATION,
    collection: str = DEFAULT_VERTEX_SEARCH_COLLECTION,
    serving_config: str = DEFAULT_VERTEX_SEARCH_SERVING_CONFIG,
    page_size: int = 5,
    http_client: httpx.Client | None = None,
    credentials: Any | None = None,
) -> list[str]:
    """Run a `servingConfigs/default_search:search` query and return the hit document ids."""
    data_store = vertex_search_datastore_path(
        project_id=project_id, data_store_id=data_store_id, location=location, collection=collection
    )
    url = f"{_discovery_engine_root(location)}/{data_store}/servingConfigs/{serving_config}:search"
    client = http_client or httpx.Client(timeout=60.0)
    close_client = http_client is None
    try:
        token = _bearer_token(credentials)
        resp = client.post(
            url,
            json={"query": query, "pageSize": page_size},
            headers=_auth_headers(token, quota_project_id=project_id, json_body=True),
        )
        _raise_for_http(resp, f"Vertex AI Search query against {data_store_id}")
        hits: list[str] = []
        for item in resp.json().get("results") or []:
            doc_id = item.get("id") or (item.get("document") or {}).get("id")
            if doc_id:
                hits.append(str(doc_id))
        return hits
    finally:
        if close_client:
            client.close()


__all__ = [
    "BigQueryLoadJobResult",
    "CloudPublishError",
    "CloudPublishTimeoutError",
    "DEFAULT_BIGQUERY_DATASET_ID",
    "DEFAULT_BIGQUERY_LOCATION",
    "DEFAULT_FINANCIAL_METRICS_SCHEMA_PATH",
    "DEFAULT_FINANCIAL_METRICS_TABLE_ID",
    "DEFAULT_SMOKE_QUERY",
    "DEFAULT_VERTEX_SEARCH_DATASTORE_ID",
    "FINANCIALS_MANIFEST_BLOB",
    "RECONCILIATION_MODES",
    "VERTEX_SEARCH_MANIFEST_BLOB",
    "VertexSearchImportResult",
    "gcs_uri",
    "import_vertex_ai_search_documents",
    "load_bigquery_schema_fields",
    "poll_discovery_engine_operation",
    "reload_financial_metrics_table",
    "resolve_schema_path",
    "run_bigquery_load_job",
    "search_vertex_ai_datastore",
]
