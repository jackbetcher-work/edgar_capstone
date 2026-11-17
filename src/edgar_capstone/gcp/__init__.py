"""Shared lightweight GCP REST, ADC auth, BigQuery, Firestore, and LRO polling primitives.

Zero heavy SDK dependencies (`edgartools`, `pandas`, `google-cloud-storage`).
"""

from __future__ import annotations

from edgar_capstone.gcp.bigquery import (
    BIGQUERY_API_ROOT,
    BIGQUERY_UPLOAD_ROOT,
    SAFE_BQ_IDENTIFIER_RE,
    BigQueryRestClient,
    bq_array_param,
    bq_scalar_param,
    decode_bq_cell,
    decode_bq_row,
    decode_bq_rows,
    load_bigquery_schema_fields,
    validate_bq_identifier,
)
from edgar_capstone.gcp.defaults import (
    CLOUD_PLATFORM_SCOPE,
    DEFAULT_BIGQUERY_DATASET_ID,
    DEFAULT_BUCKET_NAME,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
    DEFAULT_PROJECT_NAME,
    DEFAULT_SEC_USER_AGENT,
    resolve_bq_dataset_id,
    resolve_corpus_bucket,
    resolve_gcp_location,
    resolve_gcp_project_id,
    resolve_sec_user_agent,
)
from edgar_capstone.gcp.discovery_engine import (
    DEFAULT_VERTEX_AI_SEARCH_COLLECTION,
    DEFAULT_VERTEX_AI_SEARCH_DATASTORE_ID,
    DEFAULT_VERTEX_AI_SEARCH_ENGINE_ID,
    DEFAULT_VERTEX_AI_SEARCH_LOCATION,
    discovery_engine_host,
    discovery_engine_root,
    vertex_search_datastore_path,
)
from edgar_capstone.gcp.firestore import (
    DEFAULT_FIRESTORE_DATABASE_ID,
    FirestoreCollectionStore,
    decode_firestore_document,
    decode_firestore_value,
    encode_firestore_fields,
    encode_firestore_value,
)
from edgar_capstone.gcp.lro import OperationError, OperationTimeoutError, poll_operation
from edgar_capstone.gcp.paths import resolve_asset_dir, resolve_repo_root
from edgar_capstone.gcp.rest_client import (
    GcpHttpError,
    GcpNotFoundError,
    GcpRateLimitError,
    GcpRestClient,
    GcpRestError,
    GcpTimeoutError,
    parse_retry_after,
)

__all__ = [
    "BIGQUERY_API_ROOT",
    "BIGQUERY_UPLOAD_ROOT",
    "BigQueryRestClient",
    "CLOUD_PLATFORM_SCOPE",
    "DEFAULT_BIGQUERY_DATASET_ID",
    "DEFAULT_BUCKET_NAME",
    "DEFAULT_FIRESTORE_DATABASE_ID",
    "DEFAULT_LOCATION",
    "DEFAULT_PROJECT_ID",
    "DEFAULT_PROJECT_NAME",
    "DEFAULT_SEC_USER_AGENT",
    "DEFAULT_VERTEX_AI_SEARCH_COLLECTION",
    "DEFAULT_VERTEX_AI_SEARCH_DATASTORE_ID",
    "DEFAULT_VERTEX_AI_SEARCH_ENGINE_ID",
    "DEFAULT_VERTEX_AI_SEARCH_LOCATION",
    "FirestoreCollectionStore",
    "GcpHttpError",
    "GcpNotFoundError",
    "GcpRateLimitError",
    "GcpRestClient",
    "GcpRestError",
    "GcpTimeoutError",
    "OperationError",
    "OperationTimeoutError",
    "SAFE_BQ_IDENTIFIER_RE",
    "bq_array_param",
    "bq_scalar_param",
    "decode_bq_cell",
    "decode_bq_row",
    "decode_bq_rows",
    "decode_firestore_document",
    "decode_firestore_value",
    "discovery_engine_host",
    "discovery_engine_root",
    "encode_firestore_fields",
    "encode_firestore_value",
    "load_bigquery_schema_fields",
    "parse_retry_after",
    "poll_operation",
    "resolve_asset_dir",
    "resolve_bq_dataset_id",
    "resolve_corpus_bucket",
    "resolve_gcp_location",
    "resolve_gcp_project_id",
    "resolve_repo_root",
    "resolve_sec_user_agent",
    "validate_bq_identifier",
    "vertex_search_datastore_path",
]
