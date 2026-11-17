"""BigQuery I/O for the golden eval set and eval results (plain REST + ADC, like the rest of
the code base).

- `load_golden_dataset_to_bigquery`: multipart-upload load job (WRITE_TRUNCATE) into
  `sec_edgar.golden_eval_set`, so no GCS staging object is needed.
- `read_golden_dataset_from_bigquery`: `tabledata.list` (needs only `bigquery.tables.getData`).
- `stream_eval_result`: `tabledata.insertAll` of one run summary row into `sec_edgar.eval_results`.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import google.auth
import httpx
from google.auth.transport.requests import Request as GoogleAuthRequest

from edgar_capstone.dataset.gcs_storage import DEFAULT_LOCATION, DEFAULT_PROJECT_ID
from edgar_capstone.evals.golden import (
    REPO_ROOT,
    GoldenEvalItem,
    dataset_fingerprint,
)

BIGQUERY_API_ROOT = "https://bigquery.googleapis.com/bigquery/v2"
BIGQUERY_UPLOAD_ROOT = "https://bigquery.googleapis.com/upload/bigquery/v2"
DEFAULT_DATASET_ID = "sec_edgar"
GOLDEN_EVAL_SET_TABLE = "golden_eval_set"
EVAL_RESULTS_TABLE = "eval_results"
SCHEMA_DIR = REPO_ROOT / "terraform" / "schemas"


class EvalBigQueryError(RuntimeError):
    """Raised when a BigQuery REST call for the eval tables fails."""


def load_schema_fields(table_id: str, schema_dir: Path = SCHEMA_DIR) -> list[dict[str, Any]]:
    return json.loads((schema_dir / f"{table_id}.json").read_text(encoding="utf-8"))


class EvalBigQueryClient:
    def __init__(
        self,
        *,
        project_id: str = DEFAULT_PROJECT_ID,
        dataset_id: str = DEFAULT_DATASET_ID,
        location: str = DEFAULT_LOCATION,
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
    ) -> None:
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.location = location
        self._http = http_client
        self._credentials = credentials

    # -- transport ---------------------------------------------------------------------------
    def _token(self) -> str:
        if self._credentials is None:
            self._credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        if not getattr(self._credentials, "valid", False) or not getattr(
            self._credentials, "token", None
        ):
            self._credentials.refresh(GoogleAuthRequest())
        return str(self._credentials.token)

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
        hdrs = {"Authorization": f"Bearer {self._token()}", "x-goog-user-project": self.project_id}
        hdrs.update(headers or {})
        client = self._http or httpx.Client(timeout=60.0)
        try:
            resp = client.request(
                method, url, json=json_body, content=content, headers=hdrs, params=params
            )
        finally:
            if self._http is None:
                client.close()
        if resp.status_code >= 400:
            raise EvalBigQueryError(
                f"BigQuery {method} {url} failed (HTTP {resp.status_code}): {resp.text[:500]}"
            )
        return resp.json() if resp.content else {}

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
        loaded_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
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
        deadline = time.monotonic() + timeout_seconds
        url = f"{BIGQUERY_API_ROOT}/projects/{self.project_id}/jobs/{job_id}"
        while True:
            job = self._request("GET", url, params={"location": self.location})
            status = job.get("status") or {}
            if status.get("state") == "DONE":
                if status.get("errorResult"):
                    raise EvalBigQueryError(
                        f"BigQuery job {job_id} failed: {status['errorResult']} "
                        f"{(status.get('errors') or [])[:3]}"
                    )
                return job
            if time.monotonic() > deadline:
                raise EvalBigQueryError(f"BigQuery job {job_id} still running after timeout")
            time.sleep(1.5)

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
            rows.extend(_decode_record(r, fields) for r in data.get("rows") or [])
            page_token = data.get("pageToken")
            if not page_token:
                break
        items = [GoldenEvalItem.from_bigquery_row(_drop_nulls(r)) for r in rows]
        return sorted(items, key=lambda i: i.id)

    # -- eval_results ------------------------------------------------------------------------
    def stream_eval_result(
        self, row: dict[str, Any], *, table_id: str = EVAL_RESULTS_TABLE
    ) -> dict[str, Any]:
        payload = {
            "kind": "bigquery#tableDataInsertAllRequest",
            "skipInvalidRows": False,
            "ignoreUnknownValues": False,
            "rows": [{"insertId": row["run_id"], "json": row}],
        }
        resp = self._request("POST", f"{self._table_url(table_id)}/insertAll", json_body=payload)
        if resp.get("insertErrors"):
            raise EvalBigQueryError(
                f"insertAll rejected eval_results row: {json.dumps(resp['insertErrors'])[:500]}"
            )
        return {"table": f"{self.project_id}.{self.dataset_id}.{table_id}", "run_id": row["run_id"]}


def _decode_value(raw: Any, field: dict[str, Any]) -> Any:
    mode = str(field.get("mode") or "NULLABLE").upper()
    if mode == "REPEATED":
        inner = {**field, "mode": "NULLABLE"}
        return [_decode_value(v.get("v") if isinstance(v, dict) else v, inner) for v in raw or []]
    if raw is None:
        return None
    ftype = str(field.get("type") or "STRING").upper()
    if ftype in ("RECORD", "STRUCT"):
        return _decode_record(raw, field.get("fields") or [])
    if ftype in ("INTEGER", "INT64"):
        return int(raw)
    if ftype in ("FLOAT", "FLOAT64", "NUMERIC", "BIGNUMERIC"):
        return float(raw)
    if ftype in ("BOOLEAN", "BOOL"):
        return raw if isinstance(raw, bool) else str(raw).lower() == "true"
    return raw


def _decode_record(row: dict[str, Any], fields: list[dict[str, Any]]) -> dict[str, Any]:
    cells = row.get("f") or []
    out: dict[str, Any] = {}
    for idx, field in enumerate(fields):
        cell = cells[idx] if idx < len(cells) else {"v": None}
        out[field["name"]] = _decode_value(cell.get("v") if isinstance(cell, dict) else cell, field)
    return out


def _drop_nulls(value: Any) -> Any:
    """BigQuery returns NULL for absent optional fields; let pydantic apply model defaults."""
    if isinstance(value, dict):
        return {k: _drop_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_nulls(v) for v in value]
    return value
