"""Unit tests for `edgar_capstone.gcp` REST clients, codecs, defaults, LRO polling, and paths."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import pytest
from google.auth.exceptions import DefaultCredentialsError


class _DummyCreds:
    """Fake ADC credential object supporting thread-safe refresh testing."""

    def __init__(self, token: str | None = "fake-token", *, valid: bool = True) -> None:
        self.token = token
        self.valid = valid
        self.refresh_count = 0

    def refresh(self, _request: Any) -> None:
        self.refresh_count += 1
        if self.token is None:
            self.valid = False
            return
        self.valid = True


def test_gcp_defaults_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    from edgar_capstone.gcp.defaults import (
        DEFAULT_BIGQUERY_DATASET_ID,
        DEFAULT_BUCKET_NAME,
        DEFAULT_LOCATION,
        DEFAULT_PROJECT_ID,
        DEFAULT_SEC_USER_AGENT,
        resolve_bq_dataset_id,
        resolve_corpus_bucket,
        resolve_gcp_location,
        resolve_gcp_project_id,
        resolve_sec_user_agent,
    )

    for key in (
        "GCP_PROJECT_ID",
        "GOOGLE_CLOUD_PROJECT",
        "GCLOUD_PROJECT",
        "EDGAR_GCP_PROJECT_ID",
        "GCP_REGION",
        "GOOGLE_CLOUD_LOCATION",
        "BIGQUERY_LOCATION",
        "EDGAR_GCP_REGION",
        "SEC_CORPUS_BUCKET",
        "EDGAR_GCS_BUCKET",
        "BIGQUERY_DATASET_ID",
        "EDGAR_BQ_DATASET",
        "SEC_EDGAR_USER_AGENT",
        "SEC_USER_AGENT",
    ):
        monkeypatch.delenv(key, raising=False)

    assert resolve_gcp_project_id() == DEFAULT_PROJECT_ID
    monkeypatch.setenv("EDGAR_GCP_PROJECT_ID", "p-legacy")
    assert resolve_gcp_project_id() == "p-legacy"
    monkeypatch.setenv("GCLOUD_PROJECT", "p-gcloud")
    assert resolve_gcp_project_id() == "p-gcloud"
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p-google")
    assert resolve_gcp_project_id() == "p-google"
    monkeypatch.setenv("GCP_PROJECT_ID", "p-canonical")
    assert resolve_gcp_project_id() == "p-canonical"
    assert resolve_gcp_project_id("p-explicit") == "p-explicit"

    monkeypatch.setenv("GCP_REGION", "global")
    assert resolve_gcp_location(allow_global=True) == "global"
    assert resolve_gcp_location(allow_global=False) == DEFAULT_LOCATION

    assert resolve_corpus_bucket("gs://my-bucket/") == "my-bucket"
    monkeypatch.setenv("EDGAR_GCS_BUCKET", "gs://legacy-bucket/")
    assert resolve_corpus_bucket() == "legacy-bucket"
    monkeypatch.setenv("SEC_CORPUS_BUCKET", "canonical-bucket")
    assert resolve_corpus_bucket() == "canonical-bucket"
    monkeypatch.delenv("SEC_CORPUS_BUCKET")
    monkeypatch.delenv("EDGAR_GCS_BUCKET")
    assert resolve_corpus_bucket() == DEFAULT_BUCKET_NAME

    assert resolve_bq_dataset_id() == DEFAULT_BIGQUERY_DATASET_ID
    monkeypatch.setenv("EDGAR_BQ_DATASET", "legacy_ds")
    assert resolve_bq_dataset_id() == "legacy_ds"
    monkeypatch.setenv("BIGQUERY_DATASET_ID", "canon_ds")
    assert resolve_bq_dataset_id() == "canon_ds"

    assert resolve_sec_user_agent() == DEFAULT_SEC_USER_AGENT
    monkeypatch.setenv("SEC_USER_AGENT", "LegacyUA test@example.com")
    assert resolve_sec_user_agent() == "LegacyUA test@example.com"
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "CanonicalUA test@example.com")
    assert resolve_sec_user_agent() == "CanonicalUA test@example.com"


def test_gcp_rest_client_auth_headers_errors_and_close() -> None:
    from edgar_capstone.gcp.rest_client import (
        GcpHttpError,
        GcpNotFoundError,
        GcpRateLimitError,
        GcpRestClient,
        GcpTimeoutError,
        parse_retry_after,
    )

    refresh_barrier = threading.Barrier(4)
    creds = _DummyCreds(token="tok-1", valid=False)
    orig_refresh = creds.refresh

    def _slow_refresh(req: Any) -> None:
        time.sleep(0.005)
        orig_refresh(req)

    creds.refresh = _slow_refresh  # type: ignore[method-assign]
    c_auth = GcpRestClient(credentials=creds)

    def _get_tok(_: int) -> str:
        refresh_barrier.wait()
        return c_auth._get_bearer_token()

    with ThreadPoolExecutor(max_workers=4) as pool:
        tokens = list(pool.map(_get_tok, range(4)))
    assert tokens == ["tok-1"] * 4
    assert creds.refresh_count == 1

    empty_creds = _DummyCreds(token=None, valid=False)
    with pytest.raises(DefaultCredentialsError):
        GcpRestClient(credentials=empty_creds)._get_bearer_token()

    seen_headers: list[dict[str, str]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(dict(request.headers))
        path = request.url.path
        if path.endswith("/ok"):
            return httpx.Response(200, json={"ok": True})
        if path.endswith("/429"):
            return httpx.Response(429, headers={"Retry-After": "12.5"}, text="slow down")
        if path.endswith("/404"):
            return httpx.Response(404, text="not found")
        if path.endswith("/timeout"):
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(500, text="boom")

    injected = httpx.Client(transport=httpx.MockTransport(_handler))
    with GcpRestClient(
        http_client=injected,
        credentials=_DummyCreds("bearer-xyz"),
        quota_project_id="quota-proj-1",
    ) as client:
        assert client._request_json("GET", "https://example.googleapis.com/ok") == {"ok": True}
        assert seen_headers[0]["x-goog-user-project"] == "quota-proj-1"
        assert seen_headers[0]["authorization"] == "Bearer bearer-xyz"

        with pytest.raises(GcpRateLimitError) as exc_429:
            client._send_request("GET", "https://example.googleapis.com/429")
        assert exc_429.value.retry_after_seconds == 12.5

        with pytest.raises(GcpNotFoundError):
            client._send_request(
                "GET",
                "https://example.googleapis.com/404",
                not_found_as_dedicated_error=True,
            )

        with pytest.raises(GcpTimeoutError):
            client._send_request("GET", "https://example.googleapis.com/timeout")

        with pytest.raises(GcpHttpError) as exc_500:
            client._send_request("GET", "https://example.googleapis.com/boom")
        assert exc_500.value.status_code == 500

        with pytest.raises(httpx.HTTPStatusError):
            client._send_request(
                "GET",
                "https://example.googleapis.com/boom",
                raise_5xx_as_httpx=True,
            )

        assert client._request_text("GET", "https://example.googleapis.com/ok") == '{"ok":true}'

    assert not injected.is_closed
    injected.close()

    owned = GcpRestClient(credentials=_DummyCreds("tok"))
    pooled = owned._get_http_client()
    assert not pooled.is_closed
    owned.close()
    assert pooled.is_closed

    assert parse_retry_after(httpx.Response(429, headers={"Retry-After": "3"})) == 3.0
    assert parse_retry_after(httpx.Response(429, headers={"Retry-After": "bad"})) is None


def test_gcp_bigquery_identifiers_decode_schema_and_client(tmp_path: Path) -> None:
    from edgar_capstone.gcp.bigquery import (
        BigQueryRestClient,
        bq_array_param,
        bq_scalar_param,
        decode_bq_cell,
        decode_bq_row,
        decode_bq_rows,
        load_bigquery_schema_fields,
        validate_bq_identifier,
    )
    from edgar_capstone.gcp.rest_client import GcpRestError

    assert validate_bq_identifier("sec_edgar-1", "dataset") == "sec_edgar-1"
    for bad in ("", "foo; DROP TABLE x", "a.b", "`table`", "has space"):
        with pytest.raises(ValueError):
            validate_bq_identifier(bad, "dataset")

    assert bq_scalar_param("fy", "INT64", 2026) == {
        "name": "fy",
        "parameterType": {"type": "INT64"},
        "parameterValue": {"value": "2026"},
    }
    assert bq_array_param("tickers", "STRING", ["NVDA", "AAPL"]) == {
        "name": "tickers",
        "parameterType": {"type": "ARRAY", "arrayType": {"type": "STRING"}},
        "parameterValue": {"arrayValues": [{"value": "NVDA"}, {"value": "AAPL"}]},
    }

    schema_fields = [
        {"name": "ticker", "type": "STRING", "mode": "REQUIRED"},
        {"name": "fiscal_year", "type": "INT64", "mode": "REQUIRED"},
        {"name": "revenue", "type": "FLOAT64", "mode": "NULLABLE"},
        {"name": "verified", "type": "BOOL", "mode": "NULLABLE"},
        {"name": "tags", "type": "STRING", "mode": "REPEATED"},
        {
            "name": "prov",
            "type": "RECORD",
            "mode": "NULLABLE",
            "fields": [
                {"name": "concept", "type": "STRING", "mode": "NULLABLE"},
                {"name": "notes", "type": "STRING", "mode": "REPEATED"},
            ],
        },
    ]
    wire_row = {
        "f": [
            {"v": "NVDA"},
            {"v": "2026"},
            {"v": "130500000000.5"},
            {"v": "true"},
            {"v": [{"v": "ai"}, {"v": "gpu"}]},
            {"v": {"f": [{"v": "us-gaap:Revenues"}, {"v": [{"v": "clean"}]}]}},
        ]
    }
    decoded = decode_bq_row(wire_row, schema_fields)
    assert decoded == {
        "ticker": "NVDA",
        "fiscal_year": 2026,
        "revenue": 130500000000.5,
        "verified": True,
        "tags": ["ai", "gpu"],
        "prov": {"concept": "us-gaap:Revenues", "notes": ["clean"]},
    }
    assert decode_bq_cell(None, {"type": "STRING", "mode": "REPEATED"}) == []
    assert decode_bq_rows({"schema": {"fields": schema_fields}, "rows": [wire_row]}) == [decoded]

    s_list = tmp_path / "list.json"
    s_list.write_text(json.dumps([{"name": "a", "type": "STRING"}]), encoding="utf-8")
    assert load_bigquery_schema_fields(s_list) == [{"name": "a", "type": "STRING"}]
    s_dict = tmp_path / "dict.json"
    s_dict.write_text(json.dumps({"fields": [{"name": "b", "type": "INT64"}]}), encoding="utf-8")
    assert load_bigquery_schema_fields(s_dict) == [{"name": "b", "type": "INT64"}]
    assert len(load_bigquery_schema_fields("financial_metrics")) > 0
    s_bad = tmp_path / "bad.json"
    s_bad.write_text(json.dumps({"fields": "not-a-list"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_bigquery_schema_fields(s_bad)

    calls: list[tuple[str, str]] = []

    def _bq_handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        if request.url.path.endswith("/queries") and request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "jobComplete": False,
                    "jobReference": {"jobId": "job-99", "location": "US"},
                },
            )
        if request.url.path.endswith("/queries/job-99") and "pageToken" not in request.url.params:
            return httpx.Response(
                200,
                json={
                    "jobComplete": True,
                    "schema": {"fields": [{"name": "ticker", "type": "STRING"}]},
                    "rows": [{"f": [{"v": "NVDA"}]}],
                    "pageToken": "pg2",
                },
            )
        if request.url.path.endswith("/queries/job-99"):
            return httpx.Response(
                200,
                json={"jobComplete": True, "rows": [{"f": [{"v": "AAPL"}]}]},
            )
        if request.url.path.endswith("/insertAll"):
            return httpx.Response(200, json={"kind": "bigquery#tableDataInsertAllResponse"})
        return httpx.Response(404)

    mock_http = httpx.Client(transport=httpx.MockTransport(_bq_handler))
    bq = BigQueryRestClient(
        project_id="proj-1",
        dataset_id="sec_edgar",
        http_client=mock_http,
        credentials=_DummyCreds("tok"),
    )
    rows = bq.run_query(
        "SELECT ticker FROM t WHERE fy = @fy",
        query_parameters=[bq_scalar_param("fy", "INT64", 2026)],
        poll_interval_seconds=0.0,
    )
    assert rows == [{"ticker": "NVDA"}, {"ticker": "AAPL"}]
    assert bq.table_ref("financial_metrics") == "`proj-1.sec_edgar.financial_metrics`"
    res = bq.insert_all("financial_metrics", [{"insertId": "1", "json": {"ticker": "NVDA"}}])
    assert res["kind"] == "bigquery#tableDataInsertAllResponse"
    assert len(calls) == 4
    mock_http.close()

    stall_http = httpx.Client(
        transport=httpx.MockTransport(
            lambda _req: httpx.Response(
                200,
                json={"jobComplete": False, "jobReference": {"jobId": "job-stall"}},
            )
        )
    )
    stall_bq = BigQueryRestClient(
        project_id="proj-1",
        dataset_id="sec_edgar",
        http_client=stall_http,
        credentials=_DummyCreds("tok"),
    )
    with pytest.raises(GcpRestError):
        stall_bq.run_query("SELECT 1", max_polls=1, poll_interval_seconds=0.0)
    stall_http.close()


def test_gcp_lro_poll_operation_retries_fail_fast_and_timeout() -> None:
    from edgar_capstone.gcp.lro import OperationError, OperationTimeoutError, poll_operation

    tokens_used: list[str] = []
    seq = iter(
        [
            httpx.Response(429, text="rate limit"),
            httpx.Response(503, text="unavailable"),
            httpx.Response(200, json={"done": False}),
            httpx.Response(200, json={"done": True, "result": "ok"}),
        ]
    )

    def _poll_retry() -> httpx.Response:
        tokens_used.append(f"tok-{len(tokens_used) + 1}")
        return next(seq)

    out = poll_operation(
        _poll_retry,
        lambda payload: bool(payload.get("done")),
        timeout_seconds=10.0,
        initial_interval_seconds=0.01,
        sleep_fn=lambda _s: None,
    )
    assert out == {"done": True, "result": "ok"}
    assert tokens_used == ["tok-1", "tok-2", "tok-3", "tok-4"]

    for fatal_code in (403, 404):
        attempts = [0]

        def _fatal_poll(code: int = fatal_code, counter: list[int] = attempts) -> httpx.Response:
            counter[0] += 1
            return httpx.Response(code, text="denied")

        with pytest.raises(OperationError):
            poll_operation(
                _fatal_poll,
                lambda payload: bool(payload.get("done")),
                timeout_seconds=10.0,
                sleep_fn=lambda _s: None,
            )
        assert attempts[0] == 1

    timeout_calls = [0]

    def _timeout_once() -> httpx.Response:
        timeout_calls[0] += 1
        if timeout_calls[0] == 1:
            raise httpx.ReadTimeout("temporary stall")
        return httpx.Response(200, json={"done": True})

    assert poll_operation(
        _timeout_once,
        lambda payload: bool(payload.get("done")),
        timeout_seconds=10.0,
        initial_interval_seconds=0.0,
        sleep_fn=lambda _s: None,
    ) == {"done": True}
    assert timeout_calls[0] == 2

    with pytest.raises(OperationTimeoutError):
        poll_operation(
            lambda: httpx.Response(200, json={"done": False}),
            lambda payload: bool(payload.get("done")),
            timeout_seconds=0.0,
            initial_interval_seconds=0.0,
            sleep_fn=lambda _s: None,
        )


def test_gcp_discovery_engine_and_paths_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from edgar_capstone.gcp import paths as paths_mod
    from edgar_capstone.gcp.discovery_engine import (
        discovery_engine_host,
        discovery_engine_root,
        vertex_search_datastore_path,
    )

    assert discovery_engine_host("global") == "discoveryengine.googleapis.com"
    assert discovery_engine_host("us") == "us-discoveryengine.googleapis.com"
    assert discovery_engine_root("eu") == "https://eu-discoveryengine.googleapis.com/v1"
    assert (
        vertex_search_datastore_path("my-proj", data_store_id="ds-1", location="global")
        == "projects/my-proj/locations/global/collections/default_collection/dataStores/ds-1"
    )

    # Wheel-safe fallback resolution for resolve_repo_root() and resolve_asset_dir()
    fake_site = tmp_path / "site-packages" / "edgar_capstone" / "gcp" / "paths.py"
    fake_site.parent.mkdir(parents=True)
    fake_site.write_text("", encoding="utf-8")
    monkeypatch.setattr(paths_mod, "__file__", str(fake_site))

    cwd_with_prompts = tmp_path / "app_cwd"
    (cwd_with_prompts / "prompts").mkdir(parents=True)
    monkeypatch.chdir(cwd_with_prompts)
    assert paths_mod.resolve_repo_root() == cwd_with_prompts.resolve()

    empty_cwd = tmp_path / "empty_cwd"
    empty_cwd.mkdir()
    monkeypatch.chdir(empty_cwd)
    assert paths_mod.resolve_repo_root() == empty_cwd.resolve()

    override_dir = tmp_path / "custom_assets"
    override_dir.mkdir()
    monkeypatch.setenv("CUSTOM_ASSET_DIR", str(override_dir))
    assert paths_mod.resolve_asset_dir("nonexistent", env_var="CUSTOM_ASSET_DIR") == (
        override_dir.resolve()
    )
    monkeypatch.delenv("CUSTOM_ASSET_DIR")
    assert paths_mod.resolve_asset_dir("nonexistent_dir_xyz").name == "nonexistent_dir_xyz"


def test_gcp_firestore_codec_and_collection_store() -> None:
    from edgar_capstone.gcp.firestore import (
        FirestoreCollectionStore,
        decode_firestore_document,
        encode_firestore_fields,
    )

    sample = {
        "nothing": None,
        "flag": True,
        "count": 42,
        "ratio": 3.25,
        "created_at": "2026-11-17T12:00:00Z",
        "label": "NVDA",
        "tags": ["ai", 7, False],
        "nested": {"updated_at": "2026-11-17T12:05:00Z", "inner": "ok"},
    }
    encoded = encode_firestore_fields(sample)
    assert "timestampValue" in encoded["created_at"]
    assert "timestampValue" in encoded["nested"]["mapValue"]["fields"]["updated_at"]

    decoded = decode_firestore_document({"name": "projects/p/doc/1", "fields": encoded})
    assert decoded["nothing"] is None
    assert decoded["flag"] is True
    assert decoded["count"] == 42
    assert decoded["ratio"] == 3.25
    assert decoded["created_at"] == "2026-11-17T12:00:00Z"
    assert decoded["label"] == "NVDA"
    assert decoded["tags"] == ["ai", 7, False]
    assert decoded["nested"] == {"updated_at": "2026-11-17T12:05:00Z", "inner": "ok"}
    assert decoded["document_name"] == "projects/p/doc/1"

    def _fs_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/missing"):
            return httpx.Response(404, text="not found")
        if request.url.path.endswith(":runQuery"):
            return httpx.Response(
                200,
                json=[
                    {"readTime": "2026-11-17T12:00:00Z"},
                    {
                        "document": {
                            "name": "projects/p/databases/(default)/documents/my%20coll/d1",
                            "fields": encode_firestore_fields({"label": "hit"}),
                        }
                    },
                ],
            )
        return httpx.Response(200, json={"fields": encode_firestore_fields({"label": "found"})})

    mock_http = httpx.Client(transport=httpx.MockTransport(_fs_handler))
    store = FirestoreCollectionStore(
        project_id="my-proj",
        database_id="(default)",
        collection_id="my coll",
        http_client=mock_http,
        credentials=_DummyCreds("tok"),
    )
    assert "/databases/(default)/documents/my%20coll/doc%2F1" in store.document_url("doc/1")
    assert store.get_document("missing") is None
    assert store.get_document("doc/1") == {"label": "found"}
    assert store.run_structured_query({"from": [{"collectionId": "my coll"}]}) == [
        {
            "label": "hit",
            "document_name": "projects/p/databases/(default)/documents/my%20coll/d1",
        }
    ]
    mock_http.close()
