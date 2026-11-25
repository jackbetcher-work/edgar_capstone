"""BigQuery I/O for the golden eval set and eval results via ``google-cloud-bigquery``.

Provided on ``EvalBigQueryClient``:
- ``load_golden_dataset``: ``load_table_from_file`` (``WRITE_TRUNCATE``) into
  ``sec_edgar.golden_eval_set`` without GCS staging or manual multipart framing.
- ``read_golden_dataset``: ``list_rows`` (automatically paginates and decodes all column types;
  needs only ``bigquery.tables.getData``).
- ``stream_eval_result``: ``insert_rows_json`` of one run summary row into
  ``sec_edgar.eval_results``.
"""

from __future__ import annotations

import io
import json
import uuid
from pathlib import Path
from typing import Any

from google.auth.exceptions import DefaultCredentialsError

from edgar_capstone.common.formatting import utc_now_iso
from edgar_capstone.evals.golden import (
    REPO_ROOT,
    GoldenEvalItem,
    dataset_fingerprint,
)
from edgar_capstone.gcp.bigquery import (
    BigQueryClientWrapper,
    load_bigquery_schema_fields,
    normalize_bq_row,
    schema_fields_to_bigquery_schema,
)
from edgar_capstone.gcp.defaults import (
    DEFAULT_BIGQUERY_DATASET_ID,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
)
from edgar_capstone.gcp.errors import GcpClientError

DEFAULT_DATASET_ID = DEFAULT_BIGQUERY_DATASET_ID
GOLDEN_EVAL_SET_TABLE = "golden_eval_set"
EVAL_RESULTS_TABLE = "eval_results"
SCHEMA_DIR = REPO_ROOT / "terraform" / "schemas"


class EvalBigQueryError(RuntimeError):
    """Raised when a BigQuery SDK call for the eval tables fails."""


def load_schema_fields(table_id: str, schema_dir: Path = SCHEMA_DIR) -> list[dict[str, Any]]:
    """Load and validate a BigQuery table schema JSON file from `schema_dir`."""
    return load_bigquery_schema_fields(table_id, schema_dir=schema_dir)


class EvalBigQueryClient(BigQueryClientWrapper):
    """BigQuery SDK client for ``golden_eval_set`` and ``eval_results``."""

    def __init__(
        self,
        *,
        project_id: str = DEFAULT_PROJECT_ID,
        dataset_id: str = DEFAULT_DATASET_ID,
        location: str = DEFAULT_LOCATION,
        bq_client: Any | None = None,
        credentials: Any | None = None,
    ) -> None:
        super().__init__(
            project_id=project_id,
            dataset_id=dataset_id,
            location=location,
            bq_client=bq_client,
            credentials=credentials,
            default_timeout=60.0,
            quota_project_id=project_id,
        )

    # -- golden_eval_set ---------------------------------------------------------------------
    def load_golden_dataset(
        self,
        items: list[GoldenEvalItem],
        *,
        table_id: str = GOLDEN_EVAL_SET_TABLE,
        poll_timeout_seconds: float = 120.0,
    ) -> dict[str, Any]:
        """Replace ``golden_eval_set`` with ``items`` via ``load_table_from_file``."""
        from google.cloud import bigquery

        version = dataset_fingerprint(items)
        loaded_at = utc_now_iso()
        ndjson = "\n".join(
            json.dumps(i.to_bigquery_row(dataset_version=version, loaded_at=loaded_at))
            for i in items
        )
        job_id = f"load_golden_eval_set_{uuid.uuid4().hex[:12]}"
        table_fqn = self.full_table_id(table_id)
        job_config = bigquery.LoadJobConfig(
            schema=schema_fields_to_bigquery_schema(load_schema_fields(GOLDEN_EVAL_SET_TABLE)),
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            create_disposition=bigquery.CreateDisposition.CREATE_NEVER,
            autodetect=False,
        )
        try:
            client = self._get_client()
            job = client.load_table_from_file(
                io.BytesIO(ndjson.encode("utf-8")),
                table_fqn,
                job_id=job_id,
                location=self.location,
                job_config=job_config,
            )
            job.result(timeout=poll_timeout_seconds)
        except Exception as exc:
            raise EvalBigQueryError(
                f"BigQuery load_golden_dataset ({job_id}) failed: {exc}"
            ) from exc

        err_result = getattr(job, "error_result", None)
        if err_result:
            details = list(getattr(job, "errors", None) or ())[:3]
            raise EvalBigQueryError(f"BigQuery job {job_id} failed: {err_result} {details}")

        return {
            "job_id": str(getattr(job, "job_id", "") or job_id),
            "table": table_fqn,
            "dataset_version": version,
            "rows_loaded": int(getattr(job, "output_rows", None) or 0),
        }

    def read_golden_dataset(self, *, table_id: str = GOLDEN_EVAL_SET_TABLE) -> list[GoldenEvalItem]:
        """Read all rows from ``golden_eval_set`` via ``bigquery.Client.list_rows``."""
        table_fqn = self.full_table_id(table_id)
        try:
            client = self._get_client()
            rows_iter = client.list_rows(table_fqn, page_size=1000, timeout=self._default_timeout)
            rows = [normalize_bq_row(r) for r in rows_iter]
        except Exception as exc:
            raise EvalBigQueryError(f"BigQuery list_rows ({table_fqn}) failed: {exc}") from exc

        items = [GoldenEvalItem.from_bigquery_row(_drop_nulls(r)) for r in rows]
        return sorted(items, key=lambda i: i.id)

    # -- eval_results ------------------------------------------------------------------------
    def stream_eval_result(
        self, row: dict[str, Any], *, table_id: str = EVAL_RESULTS_TABLE
    ) -> dict[str, Any]:
        try:
            errors = self.insert_rows_json(
                table_id,
                [row],
                row_ids=[row["run_id"]],
                skip_invalid_rows=False,
                ignore_unknown_values=False,
            )
        except (GcpClientError, DefaultCredentialsError) as exc:
            raise EvalBigQueryError(f"BigQuery insert_rows_json failed: {exc}") from exc
        if errors:
            raise EvalBigQueryError(
                f"insert_rows_json rejected eval_results row: {json.dumps(errors)[:500]}"
            )
        return {"table": self.full_table_id(table_id), "run_id": row["run_id"]}


def _drop_nulls(value: Any) -> Any:
    """BigQuery returns NULL for absent optional fields; let pydantic apply model defaults."""
    if isinstance(value, dict):
        return {k: _drop_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_nulls(v) for v in value]
    return value
