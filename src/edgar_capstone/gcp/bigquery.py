"""Shared BigQuery REST v2 wire encoding/decoding, schema loading, and client utilities."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from edgar_capstone.gcp.defaults import (
    DEFAULT_BIGQUERY_DATASET_ID,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
)
from edgar_capstone.gcp.paths import resolve_asset_dir
from edgar_capstone.gcp.rest_client import GcpRestClient, GcpRestError

BIGQUERY_API_ROOT = "https://bigquery.googleapis.com/bigquery/v2"
BIGQUERY_UPLOAD_ROOT = "https://bigquery.googleapis.com/upload/bigquery/v2"
SAFE_BQ_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_bq_identifier(value: str, label: str) -> str:
    """Ensure a BigQuery project, dataset, or table identifier contains only safe characters."""
    cleaned = value.strip()
    if not cleaned or not SAFE_BQ_IDENTIFIER_RE.match(cleaned):
        raise ValueError(f"Unsafe BigQuery {label} identifier: {value!r}")
    return cleaned


def bq_scalar_param(name: str, bq_type: str, value: Any) -> dict[str, Any]:
    """Construct a named scalar QueryParameter for BigQuery REST v2."""
    return {
        "name": name,
        "parameterType": {"type": bq_type},
        "parameterValue": {"value": None if value is None else str(value)},
    }


def bq_array_param(name: str, elem_type: str, values: list[Any]) -> dict[str, Any]:
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


def decode_bq_cell(raw_val: Any, field_schema: dict[str, Any]) -> Any:
    """Recursively decode a BigQuery REST v2 wire cell (``f``/``v``) into native Python types.

    Supports ``REPEATED`` arrays, nested ``RECORD``/``STRUCT`` rows, ``INTEGER``/``INT64``,
    ``FLOAT``/``FLOAT64``/``NUMERIC``/``BIGNUMERIC``, ``BOOLEAN``/``BOOL``, and ``STRING``.
    """
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
                decoded_items.append(decode_bq_cell(inner, elem_schema))
        return decoded_items

    if raw_val is None:
        return None

    if field_type in ("RECORD", "STRUCT"):
        subfields = field_schema.get("fields") or []
        if isinstance(raw_val, dict) and "f" in raw_val and isinstance(raw_val["f"], list):
            return decode_bq_row(raw_val, subfields)
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


def decode_bq_row(row_obj: dict[str, Any], schema_fields: list[dict[str, Any]]) -> dict[str, Any]:
    """Decode a single BigQuery ``{"f": [{"v": ...}, ...]}`` row using ``schema.fields``."""
    cells = row_obj.get("f")
    if not isinstance(cells, list):
        return dict(row_obj)
    decoded: dict[str, Any] = {}
    for idx, field_schema in enumerate(schema_fields):
        col_name = str(field_schema.get("name") or f"col_{idx}")
        cell = cells[idx] if idx < len(cells) else {"v": None}
        raw_val = cell.get("v") if isinstance(cell, dict) and "v" in cell else cell
        decoded[col_name] = decode_bq_cell(raw_val, field_schema)
    return decoded


def decode_bq_rows(
    payload: dict[str, Any],
    *,
    schema_fields: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Decode all rows from a BigQuery ``queries`` / ``tabledata.list`` response payload."""
    fields = (
        schema_fields
        if schema_fields is not None
        else ((payload.get("schema") or {}).get("fields") or [])
    )
    raw_rows = payload.get("rows") or []
    return [decode_bq_row(row, fields) for row in raw_rows if isinstance(row, dict)]


def load_bigquery_schema_fields(
    table_or_path: str | Path,
    *,
    schema_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Load and validate a BigQuery table schema JSON file (list or ``{"fields": [...]}``)."""
    path = Path(table_or_path)
    if not path.suffix and not path.is_file():
        base = schema_dir if schema_dir is not None else resolve_asset_dir("terraform/schemas")
        path = base / f"{table_or_path}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    fields = raw.get("fields") if isinstance(raw, dict) else raw
    if not isinstance(fields, list) or not all(isinstance(f, dict) for f in fields):
        raise ValueError(f"Invalid BigQuery schema JSON in {path}: expected list of field objects.")
    return fields


class BigQueryRestClient(GcpRestClient):
    """Reusable BigQuery REST v2 client with query polling, pagination, and ``insertAll``."""

    def __init__(
        self,
        *,
        project_id: str = DEFAULT_PROJECT_ID,
        dataset_id: str = DEFAULT_BIGQUERY_DATASET_ID,
        location: str = DEFAULT_LOCATION,
        http_client: Any | None = None,
        credentials: Any | None = None,
        default_timeout: float = 30.0,
        quota_project_id: str | None = None,
    ) -> None:
        super().__init__(
            http_client=http_client,
            credentials=credentials,
            default_timeout=default_timeout,
            quota_project_id=quota_project_id,
        )
        self.project_id = validate_bq_identifier(project_id, "project_id")
        self.dataset_id = validate_bq_identifier(dataset_id, "dataset_id")
        self.location = location.strip() or DEFAULT_LOCATION

    def table_ref(self, table_id: str) -> str:
        safe_table = validate_bq_identifier(table_id, "table_id")
        return f"`{self.project_id}.{self.dataset_id}.{safe_table}`"

    def run_query(
        self,
        sql: str,
        *,
        query_parameters: list[dict[str, Any]] | None = None,
        timeout_ms: int = 30000,
        max_results: int = 10000,
        max_polls: int = 20,
        poll_interval_seconds: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Execute a parameterized Standard SQL query, polling if incomplete and paging all rows."""
        url = f"{BIGQUERY_API_ROOT}/projects/{self.project_id}/queries"
        body: dict[str, Any] = {
            "query": sql,
            "useLegacySql": False,
            "location": self.location,
            "timeoutMs": timeout_ms,
            "maxResults": max_results,
        }
        if query_parameters:
            body["parameterMode"] = "NAMED"
            body["queryParameters"] = query_parameters

        payload = self._request_json("POST", url, json_body=body)
        if not isinstance(payload, dict):
            return []

        job_ref = payload.get("jobReference") or {}
        job_id = str(job_ref.get("jobId") or "")
        job_location = str(job_ref.get("location") or self.location)

        polls = 0
        while not payload.get("jobComplete", True) and job_id and polls < max_polls:
            time.sleep(poll_interval_seconds)
            polls += 1
            results_url = f"{BIGQUERY_API_ROOT}/projects/{self.project_id}/queries/{job_id}"
            payload = self._request_json(
                "GET",
                results_url,
                params={"location": job_location, "maxResults": max_results},
            )

        if not payload.get("jobComplete", True):
            raise GcpRestError(
                f"BigQuery query job {job_id!r} did not complete after {polls} polls."
            )

        schema_fields = (payload.get("schema") or {}).get("fields") or []
        all_rows = decode_bq_rows(payload, schema_fields=schema_fields)
        page_token = payload.get("pageToken")

        while page_token and job_id:
            results_url = f"{BIGQUERY_API_ROOT}/projects/{self.project_id}/queries/{job_id}"
            page_payload = self._request_json(
                "GET",
                results_url,
                params={
                    "location": job_location,
                    "pageToken": page_token,
                    "maxResults": max_results,
                },
            )
            all_rows.extend(decode_bq_rows(page_payload, schema_fields=schema_fields))
            page_token = page_payload.get("pageToken")

        return all_rows

    def insert_all(
        self,
        table_id: str,
        rows: list[dict[str, Any]],
        *,
        skip_invalid_rows: bool = False,
        ignore_unknown_values: bool = False,
    ) -> dict[str, Any]:
        """Stream rows via BigQuery REST v2 ``tabledata.insertAll``."""
        safe_table = validate_bq_identifier(table_id, "table_id")
        url = (
            f"{BIGQUERY_API_ROOT}/projects/{self.project_id}"
            f"/datasets/{self.dataset_id}/tables/{safe_table}/insertAll"
        )
        payload = {
            "skipInvalidRows": skip_invalid_rows,
            "ignoreUnknownValues": ignore_unknown_values,
            "rows": rows,
        }
        return self._request_json("POST", url, json_body=payload)
