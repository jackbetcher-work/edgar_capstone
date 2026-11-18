"""Offline tests for BigQuery / Vertex AI Search corpus publishing (`dataset.cloud_publish`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from edgar_capstone.dataset import cloud_publish
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


class _FakeCredentials:
    def __init__(self) -> None:
        self.valid = True
        self.token = "fake-token"
        self.refresh_calls = 0

    def refresh(self, request: Any) -> None:
        self.refresh_calls += 1
        self.valid = True


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setattr(cloud_publish.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


def _client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


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


class _BigQueryFake:
    """Records `jobs.insert` payloads and serves a scripted sequence of job states."""

    def __init__(self, poll_states: list[dict[str, Any]]) -> None:
        self.poll_states = list(poll_states)
        self.inserted: list[dict[str, Any]] = []
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "POST":
            body = json.loads(request.content)
            self.inserted.append(body)
            return httpx.Response(200, json={**body, "status": {"state": "RUNNING"}})
        state = self.poll_states.pop(0) if len(self.poll_states) > 1 else self.poll_states[0]
        return httpx.Response(200, json=state)


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


def test_reload_financial_metrics_payload_polls_to_done(schema_file: Path) -> None:
    fake = _BigQueryFake(
        [
            {"status": {"state": "PENDING"}},
            {"status": {"state": "RUNNING"}},
            {
                "status": {"state": "DONE"},
                "statistics": {
                    "load": {"outputRows": "3022", "inputFiles": "1", "badRecords": "0"}
                },
            },
        ]
    )
    result = reload_financial_metrics_table(
        project_id=PROJECT,
        bucket_name=BUCKET,
        schema_path=schema_file,
        http_client=_client(fake),
        credentials=_FakeCredentials(),
        timeout_seconds=30,
    )

    assert result.done and result.output_rows == 3022 and result.input_files == 1
    assert result.job_id.startswith("load_financial_metrics_")
    assert result.destination_table == f"{PROJECT}.sec_edgar.financial_metrics"

    body = fake.inserted[0]
    assert body["jobReference"] == {
        "projectId": PROJECT,
        "jobId": result.job_id,
        "location": "us-central1",
    }
    load = body["configuration"]["load"]
    assert load["sourceUris"] == [f"gs://{BUCKET}/manifests/bigquery_financials.jsonl"]
    assert load["destinationTable"] == {
        "projectId": PROJECT,
        "datasetId": "sec_edgar",
        "tableId": "financial_metrics",
    }
    assert load["sourceFormat"] == "NEWLINE_DELIMITED_JSON"
    assert load["writeDisposition"] == "WRITE_TRUNCATE"
    assert load["autodetect"] is False
    assert [f["name"] for f in load["schema"]["fields"]][-1] == "research_and_development"
    # BigQuery rejects schemaUpdateOptions with WRITE_TRUNCATE on a whole table; the explicit
    # schema alone replaces the table schema, so the option must not be sent here.
    assert "schemaUpdateOptions" not in load
    assert load["rangePartitioning"]["field"] == "fiscal_year"
    assert load["clustering"] == {"fields": ["gics_sector", "ticker"]}

    insert_req, *polls = fake.requests
    assert (
        str(insert_req.url)
        == f"https://bigquery.googleapis.com/bigquery/v2/projects/{PROJECT}/jobs"
    )
    assert insert_req.headers["Authorization"] == "Bearer fake-token"
    assert insert_req.headers["x-goog-user-project"] == PROJECT
    assert len(polls) == 3
    assert polls[0].url.path.endswith(f"/jobs/{result.job_id}")
    assert polls[0].url.params["location"] == "us-central1"


def test_reload_financial_metrics_write_append_allows_field_addition(schema_file: Path) -> None:
    fake = _BigQueryFake([{"status": {"state": "DONE"}, "statistics": {"load": {}}}])
    result = reload_financial_metrics_table(
        schema_path=schema_file,
        write_disposition="WRITE_APPEND",
        source_uri="gs://other/x.jsonl",
        http_client=_client(fake),
        credentials=_FakeCredentials(),
    )
    load = fake.inserted[0]["configuration"]["load"]
    assert load["schemaUpdateOptions"] == ["ALLOW_FIELD_ADDITION"]
    assert load["writeDisposition"] == "WRITE_APPEND"
    assert load["sourceUris"] == ["gs://other/x.jsonl"]
    assert result.schema_update_options == ["ALLOW_FIELD_ADDITION"]
    assert result.output_rows is None


def test_reload_financial_metrics_without_schema_file() -> None:
    fake = _BigQueryFake([{"status": {"state": "DONE"}}])
    reload_financial_metrics_table(
        use_schema_file=False, http_client=_client(fake), credentials=_FakeCredentials()
    )
    load = fake.inserted[0]["configuration"]["load"]
    assert "schema" not in load and "schemaUpdateOptions" not in load


def test_load_job_error_result_raises(schema_file: Path) -> None:
    fake = _BigQueryFake(
        [
            {
                "status": {
                    "state": "DONE",
                    "errorResult": {"reason": "invalid", "message": "No such field: foo"},
                    "errors": [{"message": "No such field: foo"}],
                }
            }
        ]
    )
    with pytest.raises(CloudPublishError, match="No such field: foo"):
        reload_financial_metrics_table(
            schema_path=schema_file, http_client=_client(fake), credentials=_FakeCredentials()
        )


def test_load_job_submission_http_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": {"message": "denied"}})

    with pytest.raises(CloudPublishError, match="HTTP 403"):
        run_bigquery_load_job(
            project_id=PROJECT,
            dataset_id="sec_edgar",
            table_id="t",
            source_uris=["gs://b/x"],
            http_client=_client(handler),
            credentials=_FakeCredentials(),
        )


def test_load_job_timeout_and_invalid_option_combo(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _BigQueryFake([{"status": {"state": "RUNNING"}}])
    clock = iter(range(0, 10_000, 50))
    monkeypatch.setattr(cloud_publish.time, "monotonic", lambda: float(next(clock)))
    with pytest.raises(CloudPublishTimeoutError):
        run_bigquery_load_job(
            project_id=PROJECT,
            dataset_id="sec_edgar",
            table_id="t",
            source_uris=["gs://b/x"],
            poll_timeout_seconds=120,
            http_client=_client(fake),
            credentials=_FakeCredentials(),
        )
    with pytest.raises(ValueError, match="WRITE_APPEND"):
        run_bigquery_load_job(
            project_id=PROJECT,
            dataset_id="sec_edgar",
            table_id="t",
            source_uris=["gs://b/x"],
            schema_update_options=["ALLOW_FIELD_ADDITION"],
            write_disposition="WRITE_TRUNCATE",
            credentials=_FakeCredentials(),
        )


def test_load_job_no_wait_returns_after_submission() -> None:
    fake = _BigQueryFake([{"status": {"state": "DONE"}}])
    result = reload_financial_metrics_table(
        wait=False, use_schema_file=False, http_client=_client(fake), credentials=_FakeCredentials()
    )
    assert result.state == "RUNNING" and not result.done
    assert len(fake.requests) == 1


def test_chunks_load_job_wrapper_keeps_behavior() -> None:
    fake = _BigQueryFake([{"status": {"state": "RUNNING"}}, {"status": {"state": "DONE"}}])
    job_id = trigger_bigquery_chunks_load_job(
        project_id=PROJECT,
        dataset_id="sec_edgar",
        table_id="filing_chunks",
        source_gcs_uri=f"gs://{BUCKET}/manifests/bigquery_filing_chunks.jsonl",
        http_client=_client(fake),
        credentials=_FakeCredentials(),
    )
    assert job_id.startswith("load_filing_chunks_")
    load = fake.inserted[0]["configuration"]["load"]
    assert load["writeDisposition"] == "WRITE_TRUNCATE"
    assert load["autodetect"] is False
    assert "schema" not in load and "schemaUpdateOptions" not in load
    assert "x-goog-user-project" not in fake.requests[0].headers

    failing = _BigQueryFake([{"status": {"state": "DONE", "errorResult": {"reason": "x"}}}])
    with pytest.raises(RuntimeError, match="failed"):
        trigger_bigquery_chunks_load_job(
            project_id=PROJECT,
            dataset_id="sec_edgar",
            table_id="filing_chunks",
            source_gcs_uri="gs://b/x",
            http_client=_client(failing),
            credentials=_FakeCredentials(),
        )


# --- Vertex AI Search ----------------------------------------------------------------------


class _DiscoveryEngineFake:
    def __init__(self, operations: list[dict[str, Any]]) -> None:
        self.operations = list(operations)
        self.requests: list[httpx.Request] = []
        self.import_body: dict[str, Any] | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "POST" and request.url.path.endswith("documents:import"):
            self.import_body = json.loads(request.content)
            return httpx.Response(
                200,
                json={"name": f"{DS_PATH}/branches/0/operations/import-documents-123"},
            )
        op = self.operations.pop(0) if len(self.operations) > 1 else self.operations[0]
        return httpx.Response(200, json=op)


def test_import_documents_payload_and_polling(_no_sleep: list[float]) -> None:
    fake = _DiscoveryEngineFake(
        [
            {"name": "op", "metadata": {"successCount": "10"}},
            {"name": "op", "metadata": {"successCount": "60", "failureCount": "1"}},
            {
                "name": "op",
                "done": True,
                "metadata": {"successCount": "113", "failureCount": "1", "totalCount": "114"},
                "response": {
                    "errorSamples": [{"code": 3, "message": "gs://bad/doc.md not found"}],
                    "errorConfig": {
                        "gcsPrefix": f"gs://{BUCKET}/vertex_ai_search/import_errors/123/"
                    },
                },
            },
        ]
    )
    result = import_vertex_ai_search_documents(
        project_id=PROJECT,
        bucket_name=BUCKET,
        http_client=_client(fake),
        credentials=_FakeCredentials(),
        initial_poll_interval_seconds=2.0,
        max_poll_interval_seconds=3.0,
    )

    assert fake.import_body == {
        "gcsSource": {
            "inputUris": [f"gs://{BUCKET}/manifests/vertex_ai_search_metadata.jsonl"],
            "dataSchema": "document",
        },
        "reconciliationMode": "INCREMENTAL",
        "errorConfig": {"gcsPrefix": f"gs://{BUCKET}/vertex_ai_search/import_errors/"},
    }
    import_req, *polls = fake.requests
    assert str(import_req.url) == (
        f"https://discoveryengine.googleapis.com/v1/{DS_PATH}/branches/default_branch/documents:import"
    )
    assert import_req.headers["x-goog-user-project"] == PROJECT
    assert len(polls) == 3
    assert str(polls[0].url) == (
        f"https://discoveryengine.googleapis.com/v1/{DS_PATH}/branches/0/operations/import-documents-123"
    )
    assert _no_sleep == [2.0, 3.0]  # exponential backoff capped at max interval

    assert result.done
    assert (result.success_count, result.failure_count, result.total_count) == (113, 1, 114)
    assert result.error_samples == ["gs://bad/doc.md not found"]
    assert result.error_gcs_prefix.endswith("/import_errors/123/")
    assert result.operation_name.endswith("import-documents-123")


def test_import_documents_full_mode_no_wait() -> None:
    fake = _DiscoveryEngineFake([{"done": True}])
    result = import_vertex_ai_search_documents(
        reconciliation_mode="FULL",
        wait=False,
        http_client=_client(fake),
        credentials=_FakeCredentials(),
    )
    assert fake.import_body is not None and fake.import_body["reconciliationMode"] == "FULL"
    assert len(fake.requests) == 1
    assert not result.done and result.success_count is None
    with pytest.raises(ValueError):
        import_vertex_ai_search_documents(reconciliation_mode="BOGUS")  # type: ignore[arg-type]


def test_import_documents_operation_error_raises() -> None:
    fake = _DiscoveryEngineFake(
        [{"name": "op", "done": True, "error": {"code": 7, "message": "permission denied"}}]
    )
    with pytest.raises(CloudPublishError, match="permission denied"):
        import_vertex_ai_search_documents(http_client=_client(fake), credentials=_FakeCredentials())


def test_search_helper_parses_hit_ids() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"id": "A_2024_10K", "document": {"id": "A_2024_10K"}},
                    {"document": {"id": "A_2023_10K"}},
                    {"document": {}},
                ],
                "totalSize": 2,
            },
        )

    hits = search_vertex_ai_datastore(
        "Agilent risk factors",
        page_size=3,
        http_client=_client(handler),
        credentials=_FakeCredentials(),
    )
    assert hits == ["A_2024_10K", "A_2023_10K"]
    assert seen["url"] == (
        f"https://discoveryengine.googleapis.com/v1/{DS_PATH}/servingConfigs/default_search:search"
    )
    assert seen["body"] == {"query": "Agilent risk factors", "pageSize": 3}


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


def test_cloud_publish_timeout_non_raising_and_discovery_engine_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers `run_bigquery_load_job(raise_on_timeout=False)` and Discovery Engine LRO timeout + retry."""
    fake_bq = _BigQueryFake([{"status": {"state": "RUNNING"}}])
    clock_bq = iter(range(0, 10_000, 60))
    monkeypatch.setattr(cloud_publish.time, "monotonic", lambda: float(next(clock_bq)))
    res = run_bigquery_load_job(
        project_id=PROJECT,
        dataset_id="sec_edgar",
        table_id="filing_chunks",
        source_uris=["gs://b/x"],
        ignore_unknown_values=True,
        max_bad_records=5,
        poll_timeout_seconds=100,
        raise_on_timeout=False,
        http_client=_client(fake_bq),
        credentials=_FakeCredentials(),
    )
    assert not res.done and res.state == "RUNNING"
    load_cfg = fake_bq.inserted[0]["configuration"]["load"]
    assert load_cfg["ignoreUnknownValues"] is True and load_cfg["maxBadRecords"] == 5

    # `trigger_bigquery_chunks_load_job` returns the job_id without raising on timeout (`raise_on_timeout=False`)
    clock_chunks = iter(range(0, 10_000, 60))
    monkeypatch.setattr(cloud_publish.time, "monotonic", lambda: float(next(clock_chunks)))
    job_id = trigger_bigquery_chunks_load_job(
        project_id=PROJECT,
        dataset_id="sec_edgar",
        table_id="filing_chunks",
        source_gcs_uri=f"gs://{BUCKET}/manifests/bigquery_filing_chunks.jsonl",
        poll_timeout_seconds=100,
        http_client=_client(_BigQueryFake([{"status": {"state": "RUNNING"}}])),
        credentials=_FakeCredentials(),
    )
    assert job_id.startswith("load_filing_chunks_")

    # Discovery Engine LRO poll timeout raises CloudPublishTimeoutError
    clock_de = iter(range(0, 10_000, 50))
    monkeypatch.setattr(cloud_publish.time, "monotonic", lambda: float(next(clock_de)))
    with pytest.raises(CloudPublishTimeoutError):
        import_vertex_ai_search_documents(
            project_id=PROJECT,
            bucket_name=BUCKET,
            timeout_seconds=100,
            http_client=_client(_DiscoveryEngineFake([{"name": "op", "done": False}])),
            credentials=_FakeCredentials(),
        )
