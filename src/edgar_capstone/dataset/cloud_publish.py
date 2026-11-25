"""Publish the curated SEC 10-K corpus to its serving stores on Google Cloud.

Two publishing steps, both driven by manifests already staged in the GCS corpus bucket:

1. ``reload_financial_metrics_table``: BigQuery load job via ``google-cloud-bigquery`` from
   ``gs://<bucket>/manifests/bigquery_financials.jsonl`` into the native, partitioned
   ``sec_edgar.financial_metrics`` table, using the explicit schema from
   ``terraform/schemas/financial_metrics.json``.
2. ``import_vertex_ai_search_documents``: Discovery Engine ``import_documents`` via
   ``google-cloud-discoveryengine`` from ``gs://<bucket>/manifests/vertex_ai_search_metadata.jsonl``
   into the Vertex AI Search data store, using native SDK LRO waiting. ``search_vertex_ai_datastore``
   runs a post-import smoke query via ``SearchServiceClient``.
"""

from __future__ import annotations

import concurrent.futures
import logging
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from google.auth.exceptions import DefaultCredentialsError

from edgar_capstone.dataset.gcs_storage import normalize_bucket_name
from edgar_capstone.gcp.bigquery import (
    load_bigquery_schema_fields as _gcp_load_bigquery_schema_fields,
)
from edgar_capstone.gcp.bigquery import (
    schema_fields_to_bigquery_schema,
)
from edgar_capstone.gcp.defaults import (
    DEFAULT_BIGQUERY_DATASET_ID,
    DEFAULT_BUCKET_NAME,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
)
from edgar_capstone.gcp.discovery_engine import (
    DEFAULT_VERTEX_AI_SEARCH_COLLECTION,
    DEFAULT_VERTEX_AI_SEARCH_DATASTORE_ID,
    DEFAULT_VERTEX_AI_SEARCH_LOCATION,
    discovery_engine_host,
)
from edgar_capstone.gcp.discovery_engine import (
    vertex_search_datastore_path as _gcp_vertex_search_datastore_path,
)
from edgar_capstone.gcp.errors import build_client_options
from edgar_capstone.gcp.paths import resolve_repo_root

logger = logging.getLogger(__name__)

REPO_ROOT = resolve_repo_root()

# --- BigQuery financial_metrics defaults -------------------------------------------------
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
    "range": {"start": 2015, "end": 2036, "interval": 1},
}
FINANCIAL_METRICS_CLUSTERING_FIELDS: tuple[str, ...] = ("gics_sector", "ticker")

# --- Vertex AI Search (Discovery Engine) defaults ----------------------------------------
DEFAULT_VERTEX_SEARCH_DATASTORE_ID = DEFAULT_VERTEX_AI_SEARCH_DATASTORE_ID
DEFAULT_VERTEX_SEARCH_LOCATION = DEFAULT_VERTEX_AI_SEARCH_LOCATION
DEFAULT_VERTEX_SEARCH_COLLECTION = DEFAULT_VERTEX_AI_SEARCH_COLLECTION
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


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def gcs_uri(bucket_name: str, blob_path: str) -> str:
    """Build a `gs://` URI from a (possibly `gs://`-prefixed) bucket name and blob path."""
    return f"gs://{normalize_bucket_name(bucket_name)}/{blob_path.lstrip('/')}"


def _build_bq_client(
    *,
    project_id: str,
    location: str,
    quota_project_id: str | None = None,
    credentials: Any | None = None,
) -> Any:
    from google.cloud import bigquery

    opts = build_client_options(quota_project_id=quota_project_id)
    kwargs: dict[str, Any] = {"project": project_id, "location": location}
    if credentials is not None:
        kwargs["credentials"] = credentials
    if opts is not None:
        kwargs["client_options"] = opts
    return bigquery.Client(**kwargs)


def _close_quietly(client: Any) -> None:
    close_fn = getattr(client, "close", None)
    if callable(close_fn):
        close_fn()
        return
    transport = getattr(client, "transport", None)
    t_close = getattr(transport, "close", None)
    if callable(t_close):
        t_close()


# ------------------------------------------------------------------------------------------
# BigQuery load jobs
# ------------------------------------------------------------------------------------------


@dataclass(slots=True)
class BigQueryLoadJobResult:
    """Outcome of a BigQuery load job submission (and optional completion wait)."""

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
    fields = _gcp_load_bigquery_schema_fields(path)
    if not all("name" in f for f in fields):
        raise ValueError(f"{path} is not a BigQuery schema field list")
    return fields


def resolve_schema_path(schema_path: Path | str | None) -> Path | None:
    """Resolve an explicit (or the default terraform) schema path; `None` if it is absent."""
    if schema_path is None:
        candidate = DEFAULT_FINANCIAL_METRICS_SCHEMA_PATH
        return candidate if candidate.exists() else None
    candidate = Path(schema_path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    if not candidate.exists():
        raise FileNotFoundError(f"BigQuery schema file not found: {candidate}")
    return candidate


def _build_range_partitioning(spec: dict[str, Any]) -> Any:
    from google.cloud import bigquery

    rng = spec.get("range") or {}
    return bigquery.RangePartitioning(
        field=str(spec["field"]),
        range_=bigquery.PartitionRange(
            start=int(rng["start"]),
            end=int(rng["end"]),
            interval=int(rng.get("interval", 1)),
        ),
    )


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
    raise_on_timeout: bool = True,
    quota_project_id: str | None = None,
    bq_client: Any | None = None,
    credentials: Any | None = None,
) -> BigQueryLoadJobResult:
    """Submit a BigQuery load job via ``google-cloud-bigquery`` and optionally wait until DONE."""
    from google.cloud import bigquery

    update_options = list(schema_update_options or [])
    if update_options and write_disposition == "WRITE_TRUNCATE" and "$" not in table_id:
        raise ValueError(
            "schemaUpdateOptions require WRITE_APPEND (or WRITE_TRUNCATE on a `table$partition`); "
            "WRITE_TRUNCATE with an explicit schema already replaces the table schema."
        )

    job_id = f"{job_id_prefix}_{uuid.uuid4().hex[:12]}"
    destination_table = f"{project_id}.{dataset_id}.{table_id}"

    cfg_kwargs: dict[str, Any] = {
        "source_format": source_format,
        "write_disposition": write_disposition,
        "autodetect": False,
        "ignore_unknown_values": ignore_unknown_values,
        "max_bad_records": max_bad_records,
    }
    if schema_fields is not None:
        cfg_kwargs["schema"] = schema_fields_to_bigquery_schema(schema_fields)
    if update_options:
        cfg_kwargs["schema_update_options"] = update_options
    if range_partitioning is not None:
        cfg_kwargs["range_partitioning"] = _build_range_partitioning(range_partitioning)
    if clustering_fields:
        cfg_kwargs["clustering_fields"] = list(clustering_fields)

    job_config = bigquery.LoadJobConfig(**cfg_kwargs)
    result = BigQueryLoadJobResult(
        job_id=job_id,
        project_id=project_id,
        location=location,
        destination_table=destination_table,
        source_uris=list(source_uris),
        schema_field_count=len(schema_fields) if schema_fields is not None else None,
        schema_update_options=update_options,
    )

    owns_client = bq_client is None
    try:
        client = bq_client or _build_bq_client(
            project_id=project_id,
            location=location,
            quota_project_id=quota_project_id,
            credentials=credentials,
        )
    except DefaultCredentialsError as exc:
        raise CloudPublishError(str(exc)) from exc

    try:
        try:
            job = client.load_table_from_uri(
                list(source_uris),
                destination_table,
                job_id=job_id,
                location=location,
                job_config=job_config,
            )
        except Exception as exc:
            raise CloudPublishError(
                f"BigQuery load job submission ({job_id}) failed: {exc}"
            ) from exc

        actual_job_id = str(getattr(job, "job_id", "") or job_id)
        result.job_id = actual_job_id
        logger.info("Submitted BigQuery load job %s -> %s", actual_job_id, result.destination_table)

        if not wait:
            result.state = str(getattr(job, "state", None) or "PENDING")
            return result

        try:
            job.result(timeout=poll_timeout_seconds)
        except (TimeoutError, concurrent.futures.TimeoutError) as exc:
            result.state = str(getattr(job, "state", None) or "RUNNING")
            message = (
                f"BigQuery load job {actual_job_id} still running after {poll_timeout_seconds:.0f}s; "
                f"check `bq show -j --location={location} {actual_job_id}`."
            )
            if raise_on_timeout:
                raise CloudPublishTimeoutError(message) from exc
            logger.warning(message)
            return result
        except Exception as exc:
            err_result = getattr(job, "error_result", None)
            details = getattr(job, "errors", None) or []
            if err_result:
                raise CloudPublishError(
                    f"BigQuery load job {actual_job_id} failed: {err_result}"
                    + (f" (first errors: {list(details)[:3]})" if details else "")
                ) from exc
            raise CloudPublishError(f"BigQuery load job {actual_job_id} failed: {exc}") from exc

        err_result = getattr(job, "error_result", None)
        if err_result:
            details = getattr(job, "errors", None) or []
            raise CloudPublishError(
                f"BigQuery load job {actual_job_id} failed: {err_result}"
                + (f" (first errors: {list(details)[:3]})" if details else "")
            )

        result.state = str(getattr(job, "state", None) or "DONE")
        result.output_rows = _to_int(getattr(job, "output_rows", None))
        result.input_files = _to_int(getattr(job, "input_files", None))
        result.input_file_bytes = _to_int(getattr(job, "input_file_bytes", None))
        result.bad_records = _to_int(getattr(job, "bad_records", None))
        return result
    finally:
        if owns_client:
            _close_quietly(client)


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
    bq_client: Any | None = None,
    credentials: Any | None = None,
) -> BigQueryLoadJobResult:
    """Reload native `sec_edgar.financial_metrics` from the GCS financials manifest."""
    uri = source_uri or gcs_uri(bucket_name, FINANCIALS_MANIFEST_BLOB)
    resolved_schema = resolve_schema_path(schema_path) if use_schema_file else None
    schema_fields = load_bigquery_schema_fields(resolved_schema) if resolved_schema else None
    if resolved_schema is None:
        logger.warning("No financial_metrics schema file found; relying on the table schema.")

    update_options: list[str] = []
    if (
        allow_field_addition
        and schema_fields is not None
        and (write_disposition == "WRITE_APPEND" or "$" in table_id)
    ):
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
        raise_on_timeout=True,
        quota_project_id=project_id,
        bq_client=bq_client,
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
    return _gcp_vertex_search_datastore_path(
        project_id,
        data_store_id=data_store_id,
        location=location,
        collection=collection,
    )


@dataclass(slots=True)
class VertexSearchImportResult:
    """Outcome of a Discovery Engine `import_documents` long-running operation."""

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


def _apply_sdk_operation(
    result: VertexSearchImportResult,
    operation: Any,
    response: Any | None = None,
) -> None:
    meta = getattr(operation, "metadata", None)
    if meta is not None:
        result.success_count = _to_int(getattr(meta, "success_count", None))
        result.failure_count = _to_int(getattr(meta, "failure_count", None))
        result.total_count = _to_int(getattr(meta, "total_count", None))
        to_dict = getattr(type(meta), "to_dict", None)
        result.metadata = to_dict(meta) if callable(to_dict) else {}
    done_fn = getattr(operation, "done", None)
    result.done = bool(done_fn() if callable(done_fn) else done_fn)

    if response is not None:
        samples = getattr(response, "error_samples", None) or ()
        result.error_samples = [str(getattr(s, "message", None) or s) for s in list(samples)[:10]]
        err_cfg = getattr(response, "error_config", None)
        gcs_prefix = getattr(err_cfg, "gcs_prefix", None) if err_cfg is not None else None
        if gcs_prefix:
            result.error_gcs_prefix = str(gcs_prefix)


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
    document_client: Any | None = None,
    credentials: Any | None = None,
) -> VertexSearchImportResult:
    """Import `vertex_ai_search_metadata.jsonl` documents via ``DocumentServiceClient``."""
    from google.cloud import discoveryengine_v1 as de

    mode = reconciliation_mode.upper()
    if mode not in RECONCILIATION_MODES:
        raise ValueError(f"reconciliation_mode must be one of {RECONCILIATION_MODES}, got {mode}")

    uris = list(input_uris or [gcs_uri(bucket_name, VERTEX_SEARCH_MANIFEST_BLOB)])
    errors_prefix = error_gcs_prefix or gcs_uri(bucket_name, VERTEX_SEARCH_IMPORT_ERRORS_PREFIX)
    data_store = vertex_search_datastore_path(
        project_id=project_id, data_store_id=data_store_id, location=location, collection=collection
    )
    parent = f"{data_store}/branches/{branch}"

    reconcile_enum = (
        de.ImportDocumentsRequest.ReconciliationMode.FULL
        if mode == "FULL"
        else de.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL
    )
    request = de.ImportDocumentsRequest(
        parent=parent,
        gcs_source=de.GcsSource(input_uris=uris, data_schema="document"),
        reconciliation_mode=reconcile_enum,
        error_config=de.ImportErrorConfig(gcs_prefix=errors_prefix),
    )

    owns_client = document_client is None
    try:
        if document_client is not None:
            client = document_client
        else:
            opts = build_client_options(
                api_endpoint=discovery_engine_host(location),
                quota_project_id=project_id,
            )
            kwargs: dict[str, Any] = {"client_options": opts}
            if credentials is not None:
                kwargs["credentials"] = credentials
            client = de.DocumentServiceClient(**kwargs)
    except DefaultCredentialsError as exc:
        raise CloudPublishError(str(exc)) from exc

    try:
        try:
            operation = client.import_documents(request=request)
        except Exception as exc:
            raise CloudPublishError(
                f"Vertex AI Search import_documents into {data_store_id} failed: {exc}"
            ) from exc

        raw_op = getattr(operation, "operation", None)
        operation_name = str(getattr(raw_op, "name", None) or getattr(operation, "name", "") or "")
        if not operation_name:
            raise CloudPublishError("import_documents returned an operation without a name.")
        logger.info("Started Vertex AI Search import operation %s (%s)", operation_name, mode)

        result = VertexSearchImportResult(
            operation_name=operation_name,
            data_store=data_store,
            input_uris=uris,
            reconciliation_mode=mode,
            error_gcs_prefix=errors_prefix,
        )

        response: Any | None = None
        if wait:
            try:
                response = operation.result(timeout=timeout_seconds)
            except (TimeoutError, concurrent.futures.TimeoutError) as exc:
                raise CloudPublishTimeoutError(
                    f"Discovery Engine operation {operation_name} not done after {timeout_seconds:.0f}s."
                ) from exc
            except Exception as exc:
                raise CloudPublishError(
                    f"Discovery Engine operation {operation_name} failed: {exc}"
                ) from exc
            result.done = True

        _apply_sdk_operation(result, operation, response)
        if wait:
            result.done = True
        return result
    finally:
        if owns_client:
            _close_quietly(client)


def search_vertex_ai_datastore(
    query: str = DEFAULT_SMOKE_QUERY,
    *,
    project_id: str = DEFAULT_PROJECT_ID,
    data_store_id: str = DEFAULT_VERTEX_SEARCH_DATASTORE_ID,
    location: str = DEFAULT_VERTEX_SEARCH_LOCATION,
    collection: str = DEFAULT_VERTEX_SEARCH_COLLECTION,
    serving_config: str = DEFAULT_VERTEX_SEARCH_SERVING_CONFIG,
    page_size: int = 5,
    search_client: Any | None = None,
    credentials: Any | None = None,
) -> list[str]:
    """Run a ``SearchServiceClient.search`` smoke query and return the hit document IDs."""
    from google.cloud import discoveryengine_v1 as de

    data_store = vertex_search_datastore_path(
        project_id=project_id, data_store_id=data_store_id, location=location, collection=collection
    )
    serving_config_path = f"{data_store}/servingConfigs/{serving_config}"
    request = de.SearchRequest(
        serving_config=serving_config_path,
        query=query,
        page_size=page_size,
    )

    owns_client = search_client is None
    try:
        if search_client is not None:
            client = search_client
        else:
            opts = build_client_options(
                api_endpoint=discovery_engine_host(location),
                quota_project_id=project_id,
            )
            kwargs: dict[str, Any] = {"client_options": opts}
            if credentials is not None:
                kwargs["credentials"] = credentials
            client = de.SearchServiceClient(**kwargs)
    except DefaultCredentialsError as exc:
        raise CloudPublishError(str(exc)) from exc

    try:
        try:
            response = client.search(request=request, timeout=30.0)
        except Exception as exc:
            raise CloudPublishError(
                f"Vertex AI Search query against {data_store_id} failed: {exc}"
            ) from exc

        if hasattr(response, "pages"):
            first_page = next(iter(response.pages), None)
            results = list(getattr(first_page, "results", None) or ()) if first_page else []
        else:
            results = list(getattr(response, "results", None) or ())

        hits: list[str] = []
        for item in results:
            doc = getattr(item, "document", None)
            doc_id = getattr(item, "id", None) or (getattr(doc, "id", None) if doc else None)
            if doc_id:
                hits.append(str(doc_id))
        return hits
    finally:
        if owns_client:
            _close_quietly(client)


__all__ = [
    "DEFAULT_BIGQUERY_DATASET_ID",
    "DEFAULT_BIGQUERY_LOCATION",
    "DEFAULT_FINANCIAL_METRICS_SCHEMA_PATH",
    "DEFAULT_FINANCIAL_METRICS_TABLE_ID",
    "DEFAULT_SMOKE_QUERY",
    "DEFAULT_VERTEX_SEARCH_DATASTORE_ID",
    "FINANCIALS_MANIFEST_BLOB",
    "RECONCILIATION_MODES",
    "VERTEX_SEARCH_MANIFEST_BLOB",
    "BigQueryLoadJobResult",
    "CloudPublishError",
    "CloudPublishTimeoutError",
    "VertexSearchImportResult",
    "gcs_uri",
    "import_vertex_ai_search_documents",
    "load_bigquery_schema_fields",
    "reload_financial_metrics_table",
    "resolve_schema_path",
    "run_bigquery_load_job",
    "search_vertex_ai_datastore",
]
