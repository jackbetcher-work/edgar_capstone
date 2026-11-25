"""Offline tests for BigQuery / Vertex AI Search corpus publishing (`dataset.cloud_publish`)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from google.api_core import exceptions as gcp_exceptions
from google.cloud import discoveryengine_v1 as de

from edgar_capstone.dataset.cloud_publish import (
    DEFAULT_FINANCIAL_METRICS_SCHEMA_PATH,
    CloudPublishError,
    CloudPublishTimeoutError,
    import_vertex_ai_search_documents,
    load_bigquery_schema_fields,
    reload_financial_metrics_table,
    resolve_schema_path,
    run_bigquery_load_job,
    search_vertex_ai_datastore,
)
from edgar_capstone.vector_db.pipeline import trigger_bigquery_chunks_load_job
from scripts import publish_corpus

PROJECT = "iron-radio-514217-q7"
BUCKET = "fde-sec-edgar-sandbox-dev-sec-corpus"
DATA_STORE = "sec-edgar-10k-corpus-ds"
DS_PATH = (
    f"projects/{PROJECT}/locations/global/collections/default_collection/dataStores/{DATA_STORE}"
)


@pytest.fixture
def schema_file(tmp_path: Path) -> Path:
    path = tmp_path / "financial_metrics.json"
    path.write_text(
        json.dumps(
            [
                {"name": "profile_id", "type": "STRING", "mode": "REQUIRED"},
                {"name": "fiscal_year", "type": "INTEGER", "mode": "REQUIRED"},
                {"name": "research_and_development", "type": "FLOAT", "mode": "NULLABLE"},
            ]
        ),
        encoding="utf-8",
    )
    return path


class _FakeLoadJob:
    def __init__(
        self,
        job_id: str,
        *,
        state: str = "RUNNING",
        output_rows: int | None = None,
        input_files: int | None = None,
        bad_records: int | None = None,
        error_result: dict[str, Any] | None = None,
        errors: list[dict[str, Any]] | None = None,
        result_error: Exception | None = None,
    ) -> None:
        self.job_id = job_id
        self.state = state
        self.output_rows = output_rows
        self.input_files = input_files
        self.input_file_bytes = None
        self.bad_records = bad_records
        self.error_result = error_result
        self.errors = errors
        self._result_error = result_error
        self.result_timeouts: list[float | None] = []

    def result(self, *, timeout: float | None = None) -> None:
        self.result_timeouts.append(timeout)
        if self._result_error is not None:
            raise self._result_error
        self.state = "DONE"


class _FakeBigQueryClient:
    def __init__(
        self,
        *,
        job_kwargs: dict[str, Any] | None = None,
        submit_error: Exception | None = None,
    ) -> None:
        self._job_kwargs = dict(job_kwargs or {})
        self._submit_error = submit_error
        self.calls: list[dict[str, Any]] = []
        self.jobs: list[_FakeLoadJob] = []

    def load_table_from_uri(
        self,
        source_uris: list[str],
        destination: str,
        *,
        job_id: str,
        location: str,
        job_config: Any,
    ) -> _FakeLoadJob:
        self.calls.append(
            {
                "source_uris": source_uris,
                "destination": destination,
                "job_id": job_id,
                "location": location,
                "job_config": job_config,
            }
        )
        if self._submit_error is not None:
            raise self._submit_error
        job = _FakeLoadJob(job_id, **self._job_kwargs)
        self.jobs.append(job)
        return job


# --- BigQuery ------------------------------------------------------------------------------


def test_default_schema_file_is_resolved_from_repo_root() -> None:
    assert DEFAULT_FINANCIAL_METRICS_SCHEMA_PATH.name == "financial_metrics.json"
    assert resolve_schema_path(None) == DEFAULT_FINANCIAL_METRICS_SCHEMA_PATH
    assert resolve_schema_path("terraform/schemas/financial_metrics.json") == (
        DEFAULT_FINANCIAL_METRICS_SCHEMA_PATH
    )
    fields = load_bigquery_schema_fields(DEFAULT_FINANCIAL_METRICS_SCHEMA_PATH)
    assert {"profile_id", "ticker", "fiscal_year"} <= {f["name"] for f in fields}
    with pytest.raises(FileNotFoundError):
        resolve_schema_path("does/not/exist.json")


def test_reload_financial_metrics_payload_waits_to_done(schema_file: Path) -> None:
    fake = _FakeBigQueryClient(job_kwargs={"output_rows": 3022, "input_files": 1, "bad_records": 0})
    result = reload_financial_metrics_table(
        project_id=PROJECT,
        bucket_name=BUCKET,
        schema_path=schema_file,
        bq_client=fake,
        timeout_seconds=30,
    )

    assert result.done and result.output_rows == 3022 and result.input_files == 1
    assert result.job_id.startswith("load_financial_metrics_")
    assert result.destination_table == f"{PROJECT}.sec_edgar.financial_metrics"

    call = fake.calls[0]
    assert call["source_uris"] == [f"gs://{BUCKET}/manifests/bigquery_financials.jsonl"]
    assert call["destination"] == f"{PROJECT}.sec_edgar.financial_metrics"
    assert call["location"] == "us-central1"
    cfg = call["job_config"]
    assert cfg.source_format == "NEWLINE_DELIMITED_JSON"
    assert cfg.write_disposition == "WRITE_TRUNCATE"
    assert cfg.autodetect is False
    assert [f.name for f in cfg.schema][-1] == "research_and_development"
    assert not cfg.schema_update_options
    assert cfg.range_partitioning.field == "fiscal_year"
    assert cfg.clustering_fields == ["gics_sector", "ticker"]
    assert fake.jobs[0].result_timeouts == [30]


def test_reload_financial_metrics_write_append_allows_field_addition(schema_file: Path) -> None:
    fake = _FakeBigQueryClient()
    result = reload_financial_metrics_table(
        schema_path=schema_file,
        write_disposition="WRITE_APPEND",
        source_uri="gs://other/x.jsonl",
        bq_client=fake,
    )
    cfg = fake.calls[0]["job_config"]
    assert cfg.schema_update_options == ["ALLOW_FIELD_ADDITION"]
    assert cfg.write_disposition == "WRITE_APPEND"
    assert fake.calls[0]["source_uris"] == ["gs://other/x.jsonl"]
    assert result.schema_update_options == ["ALLOW_FIELD_ADDITION"]
    assert result.output_rows is None


def test_reload_financial_metrics_without_schema_file() -> None:
    fake = _FakeBigQueryClient()
    reload_financial_metrics_table(use_schema_file=False, bq_client=fake)
    cfg = fake.calls[0]["job_config"]
    assert cfg.schema is None and not cfg.schema_update_options


def test_load_job_error_result_raises(schema_file: Path) -> None:
    fake = _FakeBigQueryClient(
        job_kwargs={
            "error_result": {"reason": "invalid", "message": "No such field: foo"},
            "errors": [{"message": "No such field: foo"}],
        }
    )
    with pytest.raises(CloudPublishError, match="No such field: foo"):
        reload_financial_metrics_table(schema_path=schema_file, bq_client=fake)


def test_load_job_submission_error_raises() -> None:
    fake = _FakeBigQueryClient(submit_error=gcp_exceptions.Forbidden("denied"))
    with pytest.raises(CloudPublishError, match="denied"):
        run_bigquery_load_job(
            project_id=PROJECT,
            dataset_id="sec_edgar",
            table_id="t",
            source_uris=["gs://b/x"],
            bq_client=fake,
        )


def test_load_job_timeout_and_invalid_option_combo() -> None:
    fake = _FakeBigQueryClient(job_kwargs={"result_error": TimeoutError("still running")})
    with pytest.raises(CloudPublishTimeoutError):
        run_bigquery_load_job(
            project_id=PROJECT,
            dataset_id="sec_edgar",
            table_id="t",
            source_uris=["gs://b/x"],
            poll_timeout_seconds=120,
            bq_client=fake,
        )
    with pytest.raises(ValueError, match="WRITE_APPEND"):
        run_bigquery_load_job(
            project_id=PROJECT,
            dataset_id="sec_edgar",
            table_id="t",
            source_uris=["gs://b/x"],
            schema_update_options=["ALLOW_FIELD_ADDITION"],
            write_disposition="WRITE_TRUNCATE",
        )


def test_load_job_no_wait_returns_after_submission() -> None:
    fake = _FakeBigQueryClient(job_kwargs={"state": "RUNNING"})
    result = reload_financial_metrics_table(wait=False, use_schema_file=False, bq_client=fake)
    assert result.state == "RUNNING" and not result.done
    assert fake.jobs[0].result_timeouts == []


def test_chunks_load_job_wrapper_keeps_behavior() -> None:
    fake = _FakeBigQueryClient()
    job_id = trigger_bigquery_chunks_load_job(
        project_id=PROJECT,
        dataset_id="sec_edgar",
        table_id="filing_chunks",
        source_gcs_uri=f"gs://{BUCKET}/manifests/bigquery_filing_chunks.jsonl",
        bq_client=fake,
    )
    assert job_id.startswith("load_filing_chunks_")
    cfg = fake.calls[0]["job_config"]
    assert cfg.write_disposition == "WRITE_TRUNCATE"
    assert cfg.autodetect is False
    assert cfg.schema is None and not cfg.schema_update_options

    failing = _FakeBigQueryClient(job_kwargs={"error_result": {"reason": "x"}})
    with pytest.raises(RuntimeError, match="failed"):
        trigger_bigquery_chunks_load_job(
            project_id=PROJECT,
            dataset_id="sec_edgar",
            table_id="filing_chunks",
            source_gcs_uri="gs://b/x",
            bq_client=failing,
        )


# --- Vertex AI Search ----------------------------------------------------------------------


class _FakeImportOperation:
    def __init__(
        self,
        name: str,
        *,
        metadata: Any | None = None,
        response: Any | None = None,
        result_error: Exception | None = None,
    ) -> None:
        self.operation = SimpleNamespace(name=name)
        self.metadata = metadata
        self._response = response
        self._result_error = result_error
        self._done = False
        self.timeouts: list[float | None] = []

    def done(self) -> bool:
        return self._done

    def result(self, *, timeout: float | None = None) -> Any:
        self.timeouts.append(timeout)
        if self._result_error is not None:
            raise self._result_error
        self._done = True
        return self._response


class _FakeDocumentServiceClient:
    def __init__(self, operation: _FakeImportOperation) -> None:
        self._operation = operation
        self.requests: list[de.ImportDocumentsRequest] = []

    def import_documents(self, *, request: de.ImportDocumentsRequest) -> _FakeImportOperation:
        self.requests.append(request)
        return self._operation


def test_import_documents_payload_and_lro_result() -> None:
    op = _FakeImportOperation(
        f"{DS_PATH}/branches/0/operations/import-documents-123",
        metadata=de.ImportDocumentsMetadata(success_count=113, failure_count=1, total_count=114),
        response=de.ImportDocumentsResponse(
            error_samples=[{"code": 3, "message": "gs://bad/doc.md not found"}],
            error_config=de.ImportErrorConfig(
                gcs_prefix=f"gs://{BUCKET}/vertex_ai_search/import_errors/123/"
            ),
        ),
    )
    fake = _FakeDocumentServiceClient(op)
    result = import_vertex_ai_search_documents(
        project_id=PROJECT,
        bucket_name=BUCKET,
        document_client=fake,
        timeout_seconds=1800,
    )

    req = fake.requests[0]
    assert req.parent == f"{DS_PATH}/branches/default_branch"
    assert list(req.gcs_source.input_uris) == [
        f"gs://{BUCKET}/manifests/vertex_ai_search_metadata.jsonl"
    ]
    assert req.gcs_source.data_schema == "document"
    assert req.reconciliation_mode == de.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL
    assert op.timeouts == [1800]

    assert result.done
    assert (result.success_count, result.failure_count, result.total_count) == (113, 1, 114)
    assert result.error_samples == ["gs://bad/doc.md not found"]
    assert result.error_gcs_prefix.endswith("/import_errors/123/")
    assert result.operation_name.endswith("import-documents-123")


def test_import_documents_full_mode_no_wait() -> None:
    op = _FakeImportOperation(f"{DS_PATH}/branches/0/operations/import-full")
    fake = _FakeDocumentServiceClient(op)
    result = import_vertex_ai_search_documents(
        reconciliation_mode="FULL",
        wait=False,
        document_client=fake,
    )
    assert fake.requests[0].reconciliation_mode == de.ImportDocumentsRequest.ReconciliationMode.FULL
    assert op.timeouts == []
    assert not result.done and result.success_count is None
    with pytest.raises(ValueError):
        import_vertex_ai_search_documents(reconciliation_mode="BOGUS")  # type: ignore[arg-type]


def test_import_documents_operation_error_raises() -> None:
    op = _FakeImportOperation(
        f"{DS_PATH}/branches/0/operations/op",
        result_error=gcp_exceptions.PermissionDenied("permission denied"),
    )
    with pytest.raises(CloudPublishError, match="permission denied"):
        import_vertex_ai_search_documents(document_client=_FakeDocumentServiceClient(op))


def test_search_helper_parses_hit_ids() -> None:
    seen: list[de.SearchRequest] = []

    class _FakeSearchClient:
        def search(
            self, *, request: de.SearchRequest, timeout: float | None = None
        ) -> de.SearchResponse:
            _ = timeout
            seen.append(request)
            return de.SearchResponse(
                results=[
                    de.SearchResponse.SearchResult(
                        id="A_2024_10K", document=de.Document(id="A_2024_10K")
                    ),
                    de.SearchResponse.SearchResult(document=de.Document(id="A_2023_10K")),
                    de.SearchResponse.SearchResult(document=de.Document()),
                ],
                total_size=2,
            )

    hits = search_vertex_ai_datastore(
        "Agilent risk factors",
        page_size=3,
        search_client=_FakeSearchClient(),
    )
    assert hits == ["A_2024_10K", "A_2023_10K"]
    assert seen[0].serving_config == f"{DS_PATH}/servingConfigs/default_search"
    assert seen[0].query == "Agilent risk factors" and seen[0].page_size == 3


# --- CLI -----------------------------------------------------------------------------------


def test_cli_parsing_defaults_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("GOOGLE_CLOUD_PROJECT", "SEC_CORPUS_BUCKET"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GCP_PROJECT_ID", "stale-sandbox-project")  # intentionally ignored
    monkeypatch.setenv("VERTEX_AI_SEARCH_DATASTORE_ID", "custom-ds")

    args = publish_corpus.parse_args(["vertex-search"])
    assert args.command == "vertex-search"
    assert (args.project_id, args.bucket, args.data_store_id) == (PROJECT, BUCKET, "custom-ds")
    assert args.reconciliation_mode == "INCREMENTAL" and not args.no_wait
    assert args.smoke_query == "Agilent risk factors" and args.timeout_seconds is None

    args = publish_corpus.parse_args(
        [
            "vertex-search",
            "--reconciliation-mode",
            "full",
            "--timeout-seconds",
            "120",
            "--no-wait",
            "--smoke-query",
            "NVDA supply chain",
        ]
    )
    assert args.reconciliation_mode == "FULL" and args.no_wait and args.timeout_seconds == 120
    assert args.smoke_query == "NVDA supply chain"

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "other-proj")
    monkeypatch.setenv("SEC_CORPUS_BUCKET", "other-bucket")
    args = publish_corpus.parse_args(["financial-metrics", "--schema-path", "x.json"])
    assert (args.project_id, args.bucket) == ("other-proj", "other-bucket")
    assert args.schema_path == Path("x.json") and args.write_disposition == "WRITE_TRUNCATE"
    assert args.table_id == "financial_metrics" and args.bq_location == "us-central1"

    args = publish_corpus.parse_args(["all", "--skip-smoke"])
    assert args.command == "all" and args.skip_smoke and args.dataset_id == "sec_edgar"

    with pytest.raises(SystemExit):
        publish_corpus.parse_args([])
    with pytest.raises(SystemExit):
        publish_corpus.parse_args(["vertex-search", "--reconciliation-mode", "PARTIAL"])


def test_cli_all_runs_both_steps(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: dict[str, dict[str, Any]] = {}

    class _Result:
        def __init__(self, payload: dict[str, Any]) -> None:
            self.payload = payload
            self.done = payload.get("done", True)

        def to_dict(self) -> dict[str, Any]:
            return dict(self.payload)

    def fake_reload(**kwargs: Any) -> _Result:
        calls["reload"] = kwargs
        return _Result({"job_id": "load_financial_metrics_x", "output_rows": 3022})

    def fake_import(**kwargs: Any) -> _Result:
        calls["import"] = kwargs
        return _Result({"operation_name": "op", "done": True, "success_count": 114})

    def fake_search(query: str, **kwargs: Any) -> list[str]:
        calls["search"] = {"query": query, **kwargs}
        return ["A_2024_10K"]

    monkeypatch.setattr(publish_corpus, "reload_financial_metrics_table", fake_reload)
    monkeypatch.setattr(publish_corpus, "import_vertex_ai_search_documents", fake_import)
    monkeypatch.setattr(publish_corpus, "search_vertex_ai_datastore", fake_search)

    rc = publish_corpus.main(["all", "--json", "--timeout-seconds", "900"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["financial_metrics"]["output_rows"] == 3022
    assert out["vertex_search"]["smoke_hits"] == ["A_2024_10K"]
    assert calls["reload"]["timeout_seconds"] == 900 and calls["reload"]["wait"] is True
    assert calls["import"]["reconciliation_mode"] == "INCREMENTAL"
    assert calls["search"]["query"] == "Agilent risk factors"


def test_cloud_publish_timeout_non_raising_and_discovery_engine_timeout() -> None:
    """Covers `run_bigquery_load_job(raise_on_timeout=False)` and Discovery Engine LRO timeout."""
    fake_bq = _FakeBigQueryClient(job_kwargs={"result_error": TimeoutError("still running")})
    res = run_bigquery_load_job(
        project_id=PROJECT,
        dataset_id="sec_edgar",
        table_id="filing_chunks",
        source_uris=["gs://b/x"],
        ignore_unknown_values=True,
        max_bad_records=5,
        poll_timeout_seconds=100,
        raise_on_timeout=False,
        bq_client=fake_bq,
    )
    assert not res.done and res.state == "RUNNING"
    cfg = fake_bq.calls[0]["job_config"]
    assert cfg.ignore_unknown_values is True and cfg.max_bad_records == 5

    job_id = trigger_bigquery_chunks_load_job(
        project_id=PROJECT,
        dataset_id="sec_edgar",
        table_id="filing_chunks",
        source_gcs_uri=f"gs://{BUCKET}/manifests/bigquery_filing_chunks.jsonl",
        poll_timeout_seconds=100,
        bq_client=_FakeBigQueryClient(job_kwargs={"result_error": TimeoutError("slow")}),
    )
    assert job_id.startswith("load_filing_chunks_")

    timeout_op = _FakeImportOperation(
        f"{DS_PATH}/branches/0/operations/op",
        result_error=TimeoutError("still importing"),
    )
    with pytest.raises(CloudPublishTimeoutError):
        import_vertex_ai_search_documents(
            project_id=PROJECT,
            bucket_name=BUCKET,
            timeout_seconds=100,
            document_client=_FakeDocumentServiceClient(timeout_op),
        )
