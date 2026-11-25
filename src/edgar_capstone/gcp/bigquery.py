"""Shared BigQuery SDK wrapper, identifier validation, and schema loading utilities."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path
from types import TracebackType
from typing import Any

from edgar_capstone.gcp.defaults import (
    DEFAULT_BIGQUERY_DATASET_ID,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
)
from edgar_capstone.gcp.errors import build_client_options, translate_gcp_sdk_error
from edgar_capstone.gcp.paths import resolve_asset_dir

SAFE_BQ_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_bq_identifier(value: str, label: str) -> str:
    """Ensure a BigQuery project, dataset, or table identifier contains only safe characters."""
    cleaned = value.strip()
    if not cleaned or not SAFE_BQ_IDENTIFIER_RE.match(cleaned):
        raise ValueError(f"Unsafe BigQuery {label} identifier: {value!r}")
    return cleaned


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


def schema_fields_to_bigquery_schema(fields: Sequence[dict[str, Any]]) -> list[Any]:
    """Convert JSON ``TableFieldSchema`` dicts into ``google.cloud.bigquery.SchemaField`` objects."""
    from google.cloud import bigquery

    return [bigquery.SchemaField.from_api_repr(f) for f in fields]


def normalize_bq_value(value: Any) -> Any:
    """Normalize native ``google-cloud-bigquery`` cell types to JSON-friendly Python values.

    - ``datetime`` -> ISO-8601 UTC ``YYYY-MM-DDTHH:MM:SSZ`` string
    - ``date`` / ``time`` -> ISO-8601 string
    - ``Decimal`` -> ``float``
    - Nested ``Mapping`` / ``list`` -> recursively normalized
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Mapping):
        return {str(k): normalize_bq_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_bq_value(item) for item in value]
    return value


def normalize_bq_row(row: Any) -> dict[str, Any]:
    """Convert a ``bigquery.Row`` (or dict) into a normalized plain Python ``dict``."""
    if isinstance(row, Mapping):
        return {str(k): normalize_bq_value(v) for k, v in row.items()}
    items = getattr(row, "items", None)
    if callable(items):
        return {str(k): normalize_bq_value(v) for k, v in items()}
    return dict(row)


def make_scalar_param(name: str, bq_type: str, value: Any) -> Any:
    """Construct a ``google.cloud.bigquery.ScalarQueryParameter``."""
    from google.cloud import bigquery

    return bigquery.ScalarQueryParameter(name, bq_type, value)


def make_array_param(name: str, elem_type: str, values: Sequence[Any]) -> Any:
    """Construct a ``google.cloud.bigquery.ArrayQueryParameter``."""
    from google.cloud import bigquery

    return bigquery.ArrayQueryParameter(name, elem_type, list(values))


class BigQueryClientWrapper:
    """Thread-safe lazy wrapper around the official ``google.cloud.bigquery.Client``."""

    def __init__(
        self,
        *,
        project_id: str = DEFAULT_PROJECT_ID,
        dataset_id: str = DEFAULT_BIGQUERY_DATASET_ID,
        location: str = DEFAULT_LOCATION,
        bq_client: Any | None = None,
        credentials: Any | None = None,
        default_timeout: float = 30.0,
        quota_project_id: str | None = None,
    ) -> None:
        self.project_id = validate_bq_identifier(project_id, "project_id")
        self.dataset_id = validate_bq_identifier(dataset_id, "dataset_id")
        self.location = location.strip() or DEFAULT_LOCATION
        self._injected_client = bq_client
        self._owned_client: Any | None = None
        self._credentials = credentials
        self._default_timeout = float(default_timeout)
        self._quota_project_id = (quota_project_id or "").strip() or None
        self._client_lock = threading.Lock()

    def _get_client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        with self._client_lock:
            if self._owned_client is None:
                from google.cloud import bigquery

                opts = build_client_options(quota_project_id=self._quota_project_id)
                kwargs: dict[str, Any] = {
                    "project": self.project_id,
                    "location": self.location,
                }
                if self._credentials is not None:
                    kwargs["credentials"] = self._credentials
                if opts is not None:
                    kwargs["client_options"] = opts
                self._owned_client = bigquery.Client(**kwargs)
            return self._owned_client

    def close(self) -> None:
        """Close the internally managed ``bigquery.Client`` (leaves injected clients open)."""
        with self._client_lock:
            if self._owned_client is not None:
                close_fn = getattr(self._owned_client, "close", None)
                if callable(close_fn):
                    close_fn()
                self._owned_client = None

    def __enter__(self) -> BigQueryClientWrapper:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def table_ref(self, table_id: str) -> str:
        safe_table = validate_bq_identifier(table_id, "table_id")
        return f"`{self.project_id}.{self.dataset_id}.{safe_table}`"

    def full_table_id(self, table_id: str) -> str:
        safe_table = validate_bq_identifier(table_id, "table_id")
        return f"{self.project_id}.{self.dataset_id}.{safe_table}"

    def run_query(
        self,
        sql: str,
        *,
        query_parameters: Sequence[Any] | None = None,
        timeout: float | None = None,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a parameterized Standard SQL query and return normalized Python row dicts."""
        from google.cloud import bigquery

        eff_timeout = self._default_timeout if timeout is None else float(timeout)
        client = self._get_client()
        job_config = bigquery.QueryJobConfig(
            use_legacy_sql=False,
            query_parameters=list(query_parameters or ()),
        )
        try:
            job = client.query(
                sql,
                job_config=job_config,
                location=self.location,
                timeout=eff_timeout,
            )
            result_kwargs: dict[str, Any] = {"timeout": eff_timeout}
            if max_results is not None:
                result_kwargs["max_results"] = max_results
            rows_iter = job.result(**result_kwargs)
            return [normalize_bq_row(row) for row in rows_iter]
        except Exception as exc:
            from google.auth.exceptions import DefaultCredentialsError, RefreshError

            if isinstance(exc, (DefaultCredentialsError, RefreshError)):
                raise
            raise translate_gcp_sdk_error(
                exc, context=f"BigQuery query ({self.project_id})"
            ) from exc

    def insert_rows_json(
        self,
        table_id: str,
        rows: Sequence[dict[str, Any]],
        *,
        row_ids: Sequence[str] | None = None,
        skip_invalid_rows: bool = False,
        ignore_unknown_values: bool = False,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """Stream rows via ``bigquery.Client.insert_rows_json``; returns any per-row error dicts."""
        if not rows:
            return []
        table_fqn = self.full_table_id(table_id)
        eff_timeout = self._default_timeout if timeout is None else float(timeout)
        client = self._get_client()
        try:
            errors = client.insert_rows_json(
                table_fqn,
                list(rows),
                row_ids=list(row_ids) if row_ids is not None else None,
                skip_invalid_rows=skip_invalid_rows,
                ignore_unknown_values=ignore_unknown_values,
                timeout=eff_timeout,
            )
            return list(errors or ())
        except Exception as exc:
            from google.auth.exceptions import DefaultCredentialsError, RefreshError

            if isinstance(exc, (DefaultCredentialsError, RefreshError)):
                raise
            raise translate_gcp_sdk_error(
                exc, context=f"BigQuery insert_rows_json ({table_fqn})"
            ) from exc


__all__ = [
    "BigQueryClientWrapper",
    "SAFE_BQ_IDENTIFIER_RE",
    "load_bigquery_schema_fields",
    "make_array_param",
    "make_scalar_param",
    "normalize_bq_row",
    "normalize_bq_value",
    "schema_fields_to_bigquery_schema",
    "validate_bq_identifier",
]
