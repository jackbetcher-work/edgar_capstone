"""BigQuery I/O for the golden eval set and eval results (plain REST + ADC).

Provided on `EvalBigQueryClient` (inheriting thread-safe ADC refresh, pooled HTTP
transport, and BigQuery REST v2 wire decoding from `BigQueryRestClient`):
- `load_golden_dataset`: multipart-upload load job (`WRITE_TRUNCATE`) into
  `sec_edgar.golden_eval_set`, so no GCS staging object is needed.
- `read_golden_dataset`: `tabledata.list` (needs only `bigquery.tables.getData`).
- `stream_eval_result`: `tabledata.insertAll` of one run summary row into
  `sec_edgar.eval_results`.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import httpx
from google.auth.exceptions import DefaultCredentialsError

from edgar_capstone.common.formatting import utc_now_iso
from edgar_capstone.evals.golden import (
    REPO_ROOT,
    GoldenEvalItem,
    dataset_fingerprint,
)
from edgar_capstone.gcp.bigquery import (
    BIGQUERY_API_ROOT,
    BIGQUERY_UPLOAD_ROOT,
    BigQueryRestClient,
    decode_bq_row,
    load_bigquery_schema_fields,
)
from edgar_capstone.gcp.defaults import (
    DEFAULT_BIGQUERY_DATASET_ID,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
)
from edgar_capstone.gcp.lro import OperationError, OperationTimeoutError, poll_operation
from edgar_capstone.gcp.rest_client import GcpRestError

DEFAULT_DATASET_ID = DEFAULT_BIGQUERY_DATASET_ID
GOLDEN_EVAL_SET_TABLE = "golden_eval_set"
EVAL_RESULTS_TABLE = "eval_results"
SCHEMA_DIR = REPO_ROOT / "terraform" / "schemas"


class EvalBigQueryError(RuntimeError):
    """Raised when a BigQuery REST call for the eval tables fails."""


def load_schema_fields(table_id: str, schema_dir: Path = SCHEMA_DIR) -> list[dict[str, Any]]:
    """Load and validate a BigQuery table schema JSON file from `schema_dir`."""
    return load_bigquery_schema_fields(table_id, schema_dir=schema_dir)


class EvalBigQueryClient(BigQueryRestClient):
    """BigQuery REST v2 client for `golden_eval_set` and `eval_results`."""

    def __init__(
        self,
        *,
        project_id: str = DEFAULT_PROJECT_ID,
        dataset_id: str = DEFAULT_DATASET_ID,
        location: str = DEFAULT_LOCATION,
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
    ) -> None:
        super().__init__(
            project_id=project_id,
            dataset_id=dataset_id,
            location=location,
            http_client=http_client,
            credentials=credentials,
            default_timeout=60.0,
            quota_project_id=project_id,
        )

    def _request(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            payload = self._request_json(
                method,
                url,
                json_body=json_body,
                content=content,
                headers=headers,
                params=params,
            )
        except (GcpRestError, DefaultCredentialsError) as exc:
            raise EvalBigQueryError(f"BigQuery {method} {url} failed: {exc}") from exc
        return payload if isinstance(payload, dict) else {}

    def _table_url(self, table_id: str) -> str:
        return (
            f"{BIGQUERY_API_ROOT}/projects/{self.project_id}/datasets/{self.dataset_id}"
            f"/tables/{table_id}"
        )

    # -- golden_eval_set ---------------------------------------------------------------------
    def load_golden_dataset(
        self,
        items: list[GoldenEvalItem],
        *,
        table_id: str = GOLDEN_EVAL_SET_TABLE,
        poll_timeout_seconds: float = 120.0,
    ) -> dict[str, Any]:
        """Replace `golden_eval_set` with `items` via a multipart-upload load job."""
        version = dataset_fingerprint(items)
        loaded_at = utc_now_iso()
        ndjson = "\n".join(
            json.dumps(i.to_bigquery_row(dataset_version=version, loaded_at=loaded_at))
            for i in items
        )
        job_id = f"load_golden_eval_set_{uuid.uuid4().hex[:12]}"
        config = {
            "jobReference": {
                "projectId": self.project_id,
                "jobId": job_id,
                "location": self.location,
            },
            "configuration": {
                "load": {
                    "destinationTable": {
                        "projectId": self.project_id,
                        "datasetId": self.dataset_id,
                        "tableId": table_id,
                    },
                    "schema": {"fields": load_schema_fields(GOLDEN_EVAL_SET_TABLE)},
                    "sourceFormat": "NEWLINE_DELIMITED_JSON",
                    "writeDisposition": "WRITE_TRUNCATE",
                    "createDisposition": "CREATE_NEVER",
                    "autodetect": False,
                }
            },
        }
        boundary = f"golden-{uuid.uuid4().hex}"
        body = (
            f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n"
            f"{json.dumps(config)}\r\n"
            f"--{boundary}\r\nContent-Type: application/octet-stream\r\n\r\n"
            f"{ndjson}\r\n--{boundary}--\r\n"
        ).encode()
        self._request(
            "POST",
            f"{BIGQUERY_UPLOAD_ROOT}/projects/{self.project_id}/jobs",
            content=body,
            headers={"Content-Type": f"multipart/related; boundary={boundary}"},
            params={"uploadType": "multipart"},
        )
        job = self._wait_for_job(job_id, poll_timeout_seconds)
        stats = (job.get("statistics") or {}).get("load") or {}
        return {
            "job_id": job_id,
            "table": f"{self.project_id}.{self.dataset_id}.{table_id}",
            "dataset_version": version,
            "rows_loaded": int(stats.get("outputRows") or 0),
        }

    def _wait_for_job(self, job_id: str, timeout_seconds: float) -> dict[str, Any]:
        url = f"{BIGQUERY_API_ROOT}/projects/{self.project_id}/jobs/{job_id}"
        try:
            job = poll_operation(
                lambda: self._raw_request("GET", url, params={"location": self.location}),
                lambda payload: (payload.get("status") or {}).get("state") == "DONE",
                label=f"BigQuery job {job_id}",
                timeout_seconds=timeout_seconds,
                initial_interval_seconds=1.5,
                max_interval_seconds=1.5,
                backoff_multiplier=1.0,
            )
        except OperationTimeoutError as exc:
            raise EvalBigQueryError(f"BigQuery job {job_id} still running after timeout") from exc
        except (OperationError, DefaultCredentialsError) as exc:
            raise EvalBigQueryError(str(exc)) from exc

        status = job.get("status") or {}
        if status.get("errorResult"):
            raise EvalBigQueryError(
                f"BigQuery job {job_id} failed: {status['errorResult']} "
                f"{(status.get('errors') or [])[:3]}"
            )
        return job

    def read_golden_dataset(self, *, table_id: str = GOLDEN_EVAL_SET_TABLE) -> list[GoldenEvalItem]:
        table = self._request("GET", self._table_url(table_id))
        fields = (table.get("schema") or {}).get("fields") or []
        rows: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            params: dict[str, Any] = {"maxResults": 1000}
            if page_token:
                params["pageToken"] = page_token
            data = self._request("GET", f"{self._table_url(table_id)}/data", params=params)
            rows.extend(decode_bq_row(r, fields) for r in data.get("rows") or [])
            page_token = data.get("pageToken")
            if not page_token:
                break
        items = [GoldenEvalItem.from_bigquery_row(_drop_nulls(r)) for r in rows]
        return sorted(items, key=lambda i: i.id)

    # -- eval_results ------------------------------------------------------------------------
    def stream_eval_result(
        self, row: dict[str, Any], *, table_id: str = EVAL_RESULTS_TABLE
    ) -> dict[str, Any]:
        try:
            resp = self.insert_all(
                table_id,
                [{"insertId": row["run_id"], "json": row}],
                skip_invalid_rows=False,
                ignore_unknown_values=False,
            )
        except (GcpRestError, DefaultCredentialsError) as exc:
            raise EvalBigQueryError(f"BigQuery insertAll failed: {exc}") from exc
        if resp.get("insertErrors"):
            raise EvalBigQueryError(
                f"insertAll rejected eval_results row: {json.dumps(resp['insertErrors'])[:500]}"
            )
        return {"table": f"{self.project_id}.{self.dataset_id}.{table_id}", "run_id": row["run_id"]}


def _drop_nulls(value: Any) -> Any:
    """BigQuery returns NULL for absent optional fields; let pydantic apply model defaults."""
    if isinstance(value, dict):
        return {k: _drop_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_nulls(v) for v in value]
    return value
