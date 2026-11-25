"""Unit tests for `edgar_capstone.gcp` SDK wrappers, error translation, defaults, and paths."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from google.api_core import exceptions as gcp_exceptions
from google.auth.exceptions import DefaultCredentialsError

from edgar_capstone.gcp.bigquery import (
    BigQueryClientWrapper,
    load_bigquery_schema_fields,
    make_array_param,
    make_scalar_param,
    normalize_bq_row,
    normalize_bq_value,
    schema_fields_to_bigquery_schema,
    validate_bq_identifier,
)
from edgar_capstone.gcp.discovery_engine import (
    discovery_engine_host,
    vertex_search_datastore_path,
    vertex_search_engine_path,
)
from edgar_capstone.gcp.errors import (
    GcpClientError,
    GcpHttpError,
    GcpNotFoundError,
    GcpRateLimitError,
    GcpTimeoutError,
    build_client_options,
    parse_retry_after,
    resolve_quota_project_id,
    translate_gcp_sdk_error,
)


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
        "EDGAR_GCP_PROJECT_ID",
        "GCP_REGION",
        "GOOGLE_CLOUD_LOCATION",
        "BIGQUERY_LOCATION",
        "EDGAR_GCP_REGION",
        "SEC_CORPUS_BUCKET",
        "BIGQUERY_DATASET_ID",
        "SEC_EDGAR_USER_AGENT",
    ):
        monkeypatch.delenv(key, raising=False)

    assert resolve_gcp_project_id() == DEFAULT_PROJECT_ID
    monkeypatch.setenv("EDGAR_GCP_PROJECT_ID", "p-legacy")
    assert resolve_gcp_project_id() == "p-legacy"
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p-google")
    assert resolve_gcp_project_id() == "p-google"
    monkeypatch.setenv("GCP_PROJECT_ID", "p-canonical")
    assert resolve_gcp_project_id() == "p-canonical"
    assert resolve_gcp_project_id("p-explicit") == "p-explicit"

    monkeypatch.setenv("GCP_REGION", "global")
    assert resolve_gcp_location(allow_global=True) == "global"
    assert resolve_gcp_location(allow_global=False) == DEFAULT_LOCATION

    assert resolve_corpus_bucket("gs://my-bucket/") == "my-bucket"
    monkeypatch.setenv("SEC_CORPUS_BUCKET", "canonical-bucket")
    assert resolve_corpus_bucket() == "canonical-bucket"
    monkeypatch.delenv("SEC_CORPUS_BUCKET")
    assert resolve_corpus_bucket() == DEFAULT_BUCKET_NAME

    assert resolve_bq_dataset_id() == DEFAULT_BIGQUERY_DATASET_ID
    monkeypatch.setenv("BIGQUERY_DATASET_ID", "canon_ds")
    assert resolve_bq_dataset_id() == "canon_ds"

    assert resolve_sec_user_agent() == DEFAULT_SEC_USER_AGENT
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "CanonicalUA test@example.com")
    assert resolve_sec_user_agent() == "CanonicalUA test@example.com"


def test_gcp_sdk_error_translation_and_client_options() -> None:
    resp_429 = SimpleNamespace(status_code=429, headers={"Retry-After": "12.5"})
    err_429 = gcp_exceptions.TooManyRequests("quota", response=resp_429)
    t_429 = translate_gcp_sdk_error(err_429, context="BigQuery")
    assert isinstance(t_429, GcpRateLimitError)
    assert t_429.retry_after_seconds == 12.5

    t_404 = translate_gcp_sdk_error(gcp_exceptions.NotFound("missing"), context="GCS")
    assert isinstance(t_404, GcpNotFoundError)
    assert t_404.status_code == 404

    t_503 = translate_gcp_sdk_error(gcp_exceptions.ServiceUnavailable("down"), context="Firestore")
    assert isinstance(t_503, GcpTimeoutError)

    t_to = translate_gcp_sdk_error(TimeoutError("deadline"), context="ModelArmor")
    assert isinstance(t_to, GcpTimeoutError)

    t_400 = translate_gcp_sdk_error(gcp_exceptions.BadRequest("bad sql"), context="BigQuery")
    assert isinstance(t_400, GcpHttpError)
    assert t_400.status_code == 400

    assert parse_retry_after({"Retry-After": "3.5"}) == 3.5
    assert parse_retry_after({"retry-after": "bad"}) is None
    assert parse_retry_after(None) is None

    assert build_client_options() is None
    opts = build_client_options(
        api_endpoint="modelarmor.us-central1.rep.googleapis.com",
        quota_project_id="my-proj",
    )
    assert opts.api_endpoint == "modelarmor.us-central1.rep.googleapis.com"
    assert opts.quota_project_id == "my-proj"


def test_resolve_quota_project_id_only_applies_to_user_oauth_adc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only end-user OAuth ADC receives quota_project_id; WIF/SA/GCE credentials omit it."""
    import google.auth
    from google.auth import compute_engine, identity_pool
    from google.oauth2 import credentials as oauth2_creds
    from google.oauth2 import service_account

    user_creds = oauth2_creds.Credentials(token="tok")
    user_access_creds = object.__new__(oauth2_creds.UserAccessTokenCredentials)
    gce_creds = compute_engine.Credentials()
    wif_creds = object.__new__(identity_pool.Credentials)
    sa_creds = object.__new__(service_account.Credentials)

    assert resolve_quota_project_id("", credentials=user_creds) is None
    assert resolve_quota_project_id(None, credentials=user_creds) is None
    assert resolve_quota_project_id("my-proj", credentials=user_creds) == "my-proj"
    assert resolve_quota_project_id("my-proj", credentials=user_access_creds) == "my-proj"
    assert resolve_quota_project_id("my-proj", credentials=wif_creds) is None
    assert resolve_quota_project_id("my-proj", credentials=gce_creds) is None
    assert resolve_quota_project_id("my-proj", credentials=sa_creds) is None

    # Ambient ADC lookup when credentials=None
    monkeypatch.setattr(google.auth, "default", lambda: (wif_creds, "my-proj"))
    assert resolve_quota_project_id("my-proj") is None

    monkeypatch.setattr(google.auth, "default", lambda: (user_creds, "my-proj"))
    assert resolve_quota_project_id("my-proj") == "my-proj"

    def _no_adc() -> Any:
        raise DefaultCredentialsError("missing adc")

    monkeypatch.setattr(google.auth, "default", _no_adc)
    assert resolve_quota_project_id("my-proj") is None

    # Regression check: Discovery Engine and Model Armor SDK clients omit quota_project_id under WIF
    from google.cloud import discoveryengine_v1, modelarmor_v1

    from edgar_capstone.agents.adapters.registry import GcpToolConfig
    from edgar_capstone.agents.adapters.vertex_search import VertexAISearchAdapter
    from edgar_capstone.security.model_armor.client import ModelArmorClient
    from edgar_capstone.security.model_armor.config import ModelArmorConfig

    captured_opts: list[tuple[str, Any]] = []

    class _SpySearchClient:
        def __init__(self, **kwargs: Any) -> None:
            captured_opts.append(("vais", kwargs.get("client_options")))

    class _SpyModelArmorClient:
        def __init__(self, **kwargs: Any) -> None:
            captured_opts.append(("model_armor", kwargs.get("client_options")))

    monkeypatch.setattr(discoveryengine_v1, "SearchServiceClient", _SpySearchClient)
    monkeypatch.setattr(modelarmor_v1, "ModelArmorClient", _SpyModelArmorClient)

    monkeypatch.setattr(google.auth, "default", lambda: (wif_creds, "iron-radio-514217-q7"))
    VertexAISearchAdapter(GcpToolConfig(project_id="iron-radio-514217-q7"))._get_client()
    ModelArmorClient(ModelArmorConfig(project_id="iron-radio-514217-q7"))._get_sdk_client()

    monkeypatch.setattr(google.auth, "default", lambda: (user_creds, "iron-radio-514217-q7"))
    VertexAISearchAdapter(GcpToolConfig(project_id="iron-radio-514217-q7"))._get_client()
    ModelArmorClient(ModelArmorConfig(project_id="iron-radio-514217-q7"))._get_sdk_client()

    assert captured_opts[0][0] == "vais" and captured_opts[0][1].quota_project_id is None
    assert captured_opts[1][0] == "model_armor" and captured_opts[1][1].quota_project_id is None
    assert (
        captured_opts[2][0] == "vais"
        and captured_opts[2][1].quota_project_id == "iron-radio-514217-q7"
    )
    assert (
        captured_opts[3][0] == "model_armor"
        and captured_opts[3][1].quota_project_id == "iron-radio-514217-q7"
    )


def test_translate_gcp_sdk_error_auth_plugin_401_403_is_non_retryable() -> None:
    """gRPC 503 / RetryError wrapping AuthMetadataPlugin 403/401 maps to GcpHttpError, not retryable timeout."""
    denied = gcp_exceptions.ServiceUnavailable(
        'Getting metadata from plugin failed with error: ("Unable to acquire impersonated '
        'credentials", \'{"error": {"code": 403, "status": "PERMISSION_DENIED"}}\')'
    )
    t_denied = translate_gcp_sdk_error(denied, context="Vertex AI Search")
    assert isinstance(t_denied, GcpHttpError)
    assert t_denied.status_code == 403

    wrapped_retry = gcp_exceptions.RetryError(
        "Timeout of 60.0s exceeded, last exception: 503 Getting metadata from plugin failed "
        'with error: {"error": {"code": 403, "status": "PERMISSION_DENIED"}}',
        cause=denied,
    )
    t_retry = translate_gcp_sdk_error(wrapped_retry, context="Model Armor")
    assert isinstance(t_retry, GcpHttpError)
    assert t_retry.status_code == 403

    unauth = gcp_exceptions.ServiceUnavailable(
        'Getting metadata from plugin failed with error: {"error": {"code": 401, "status": "UNAUTHENTICATED"}}'
    )
    t_unauth = translate_gcp_sdk_error(unauth, context="Model Armor")
    assert isinstance(t_unauth, GcpHttpError)
    assert t_unauth.status_code == 401

    transient_plugin = gcp_exceptions.ServiceUnavailable(
        "Getting metadata from plugin failed with error: connection reset by peer"
    )
    t_transient = translate_gcp_sdk_error(transient_plugin, context="Model Armor")
    assert isinstance(t_transient, GcpTimeoutError)


def test_grpc_rpc_error_status_code_translation() -> None:
    """Raw gRPC StatusCode enums (0..16) map to canonical HTTP codes and retryable types."""
    import grpc

    class _FakeGrpcRpcError(RuntimeError):
        def __init__(self, status: Any, msg: str) -> None:
            super().__init__(msg)
            self._status = status

        def code(self) -> Any:
            return self._status

    err_rl = translate_gcp_sdk_error(
        _FakeGrpcRpcError(grpc.StatusCode.RESOURCE_EXHAUSTED, "quota"),
        context="Firestore",
    )
    assert isinstance(err_rl, GcpRateLimitError)
    assert err_rl.status_code == 429

    err_nf = translate_gcp_sdk_error(
        _FakeGrpcRpcError(grpc.StatusCode.NOT_FOUND, "missing"),
        context="DiscoveryEngine",
    )
    assert isinstance(err_nf, GcpNotFoundError)
    assert err_nf.status_code == 404

    for transient_code, expected_http in (
        (grpc.StatusCode.DEADLINE_EXCEEDED, 504),
        (grpc.StatusCode.UNAVAILABLE, 503),
        (grpc.StatusCode.ABORTED, 409),
    ):
        err_tr = translate_gcp_sdk_error(
            _FakeGrpcRpcError(transient_code, "transient"),
            context="ModelArmor",
        )
        assert isinstance(err_tr, GcpTimeoutError)
        assert err_tr.status_code == expected_http

    err_perm = translate_gcp_sdk_error(
        _FakeGrpcRpcError(grpc.StatusCode.PERMISSION_DENIED, "denied"),
        context="Firestore",
    )
    assert isinstance(err_perm, GcpHttpError)
    assert err_perm.status_code == 403

    err_bad = translate_gcp_sdk_error(
        _FakeGrpcRpcError(grpc.StatusCode.INVALID_ARGUMENT, "invalid"),
        context="DiscoveryEngine",
    )
    assert isinstance(err_bad, GcpHttpError)
    assert err_bad.status_code == 400

    # Also verify tuple-only gRPC StatusCode duck types without `.name`
    tuple_only = SimpleNamespace(value=(8, "resource exhausted"))
    err_tuple = translate_gcp_sdk_error(
        _FakeGrpcRpcError(tuple_only, "quota"),
        context="Firestore",
    )
    assert isinstance(err_tuple, GcpRateLimitError)
    assert err_tuple.status_code == 429


def test_bigquery_normalization_params_and_schema_loading(tmp_path: Path) -> None:
    assert validate_bq_identifier("sec_edgar-01", "dataset") == "sec_edgar-01"
    with pytest.raises(ValueError):
        validate_bq_identifier("bad; DROP TABLE", "table")

    # Native TIMESTAMP, DATE, TIME, Decimal, nested RECORD, and REPEATED normalization
    raw_row = {
        "ticker": "NVDA",
        "fiscal_year": 2026,
        "revenue": Decimal("215938000000.50"),
        "filing_date": date(2026, 2, 26),
        "ingested_at": datetime(2026, 2, 27, 14, 30, 0, tzinfo=UTC),
        "cutoff_time": time(9, 30, 0),
        "metric_provenance": {"revenue": "us-gaap:Revenues", "notes": ["audited"]},
        "missing": None,
    }
    norm = normalize_bq_row(raw_row)
    assert norm["revenue"] == pytest.approx(215938000000.50)
    assert isinstance(norm["revenue"], float)
    assert norm["filing_date"] == "2026-02-26"
    assert norm["ingested_at"] == "2026-02-27T14:30:00Z"
    assert norm["cutoff_time"] == "09:30:00"
    assert norm["metric_provenance"] == {"revenue": "us-gaap:Revenues", "notes": ["audited"]}
    assert norm["missing"] is None
    assert normalize_bq_value(datetime(2026, 1, 1, 0, 0, 0)) == "2026-01-01T00:00:00Z"

    sp = make_scalar_param("fy", "INT64", 2026)
    ap = make_array_param("tickers", "STRING", ["NVDA", "AAPL"])
    assert sp.name == "fy" and sp.value == 2026
    assert ap.name == "tickers" and ap.values == ["NVDA", "AAPL"]

    schema_path = tmp_path / "sample.json"
    schema_path.write_text(
        json.dumps([{"name": "ticker", "type": "STRING", "mode": "REQUIRED"}]),
        encoding="utf-8",
    )
    fields = load_bigquery_schema_fields(schema_path)
    bq_schema = schema_fields_to_bigquery_schema(fields)
    assert len(bq_schema) == 1
    assert bq_schema[0].name == "ticker" and bq_schema[0].field_type == "STRING"

    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps({"fields": "nope"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid BigQuery schema JSON"):
        load_bigquery_schema_fields(bad_path)


def test_bigquery_client_wrapper_query_insert_and_error_handling() -> None:
    calls: list[tuple[str, Any]] = []

    class _FakeJob:
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self._rows = rows

        def result(self, **kwargs: Any) -> list[dict[str, Any]]:
            calls.append(("result", kwargs))
            return self._rows

    class _FakeClient:
        def __init__(self) -> None:
            self.closed = False
            self.fail_with: Exception | None = None

        def query(self, sql: str, **kwargs: Any) -> _FakeJob:
            if self.fail_with:
                raise self.fail_with
            calls.append(("query", (sql, kwargs)))
            return _FakeJob([{"ticker": "NVDA", "ingested_at": datetime(2026, 1, 1, tzinfo=UTC)}])

        def insert_rows_json(
            self, table: str, rows: list[dict[str, Any]], **kwargs: Any
        ) -> list[Any]:
            if self.fail_with:
                raise self.fail_with
            calls.append(("insert_rows_json", (table, rows, kwargs)))
            return []

        def close(self) -> None:
            self.closed = True

    fake = _FakeClient()
    with BigQueryClientWrapper(
        project_id="proj-1",
        dataset_id="sec_edgar",
        location="us-central1",
        bq_client=fake,
    ) as wrapper:
        assert wrapper.table_ref("financial_metrics") == "`proj-1.sec_edgar.financial_metrics`"
        rows = wrapper.run_query(
            "SELECT * FROM t WHERE fy = @fy",
            query_parameters=[make_scalar_param("fy", "INT64", 2026)],
        )
        assert rows == [{"ticker": "NVDA", "ingested_at": "2026-01-01T00:00:00Z"}]
        assert wrapper.insert_rows_json("events", [{"id": "1"}], row_ids=["1"]) == []
        assert wrapper.insert_rows_json("events", []) == []

        fake.fail_with = gcp_exceptions.TooManyRequests("slow down")
        with pytest.raises(GcpRateLimitError):
            wrapper.run_query("SELECT 1")

        fake.fail_with = DefaultCredentialsError("no adc")
        with pytest.raises(DefaultCredentialsError):
            wrapper.insert_rows_json("events", [{"id": "2"}])

    # Injected client must NOT be closed on context exit
    assert fake.closed is False


def test_discovery_engine_host_and_paths() -> None:
    assert discovery_engine_host("global") == "discoveryengine.googleapis.com"
    assert discovery_engine_host("us") == "us-discoveryengine.googleapis.com"
    assert (
        vertex_search_datastore_path("proj-1", data_store_id="ds-1", location="global")
        == "projects/proj-1/locations/global/collections/default_collection/dataStores/ds-1"
    )
    assert (
        vertex_search_engine_path("proj-1", engine_id="eng-1", location="us")
        == "projects/proj-1/locations/us/collections/default_collection/engines/eng-1"
    )
    assert issubclass(GcpHttpError, GcpClientError)
