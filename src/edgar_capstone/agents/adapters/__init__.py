"""GCP tool adapters package (BigQuery, Vertex AI Search, GCS, and registry)."""

from edgar_capstone.agents.adapters.bigquery import BigQueryAdapter
from edgar_capstone.agents.adapters.gcs import GcsCorpusAdapter
from edgar_capstone.agents.adapters.registry import (
    DEFAULT_BQ_CHUNK_HEALTH_VIEW,
    DEFAULT_BQ_FILING_CHUNKS_TABLE,
    DEFAULT_BQ_FINANCIAL_METRICS_TABLE,
    DEFAULT_BQ_YOY_VARIANCES_VIEW,
    GcpAdapterErrorTranslationMixin,
    GcpAdapterHttpError,
    GcpEndpointNotReadyError,
    GcpToolAdapters,
    GcpToolConfig,
    _gcp_adapters,
    configure_gcp_tool_adapters,
    get_active_gcp_tool_adapters,
    get_gcp_adapters_status_metadata,
    register_adapter_reset_hook,
    reset_gcp_tool_adapters,
)
from edgar_capstone.agents.adapters.vertex_search import (
    VertexAISearchAdapter,
    _default_section_title,
    _normalize_section_id,
)

__all__ = [
    "BigQueryAdapter",
    "DEFAULT_BQ_CHUNK_HEALTH_VIEW",
    "DEFAULT_BQ_FILING_CHUNKS_TABLE",
    "DEFAULT_BQ_FINANCIAL_METRICS_TABLE",
    "DEFAULT_BQ_YOY_VARIANCES_VIEW",
    "GcpAdapterErrorTranslationMixin",
    "GcpAdapterHttpError",
    "GcpEndpointNotReadyError",
    "GcpToolAdapters",
    "GcpToolConfig",
    "GcsCorpusAdapter",
    "VertexAISearchAdapter",
    "_default_section_title",
    "_gcp_adapters",
    "_normalize_section_id",
    "configure_gcp_tool_adapters",
    "get_active_gcp_tool_adapters",
    "get_gcp_adapters_status_metadata",
    "register_adapter_reset_hook",
    "reset_gcp_tool_adapters",
]
