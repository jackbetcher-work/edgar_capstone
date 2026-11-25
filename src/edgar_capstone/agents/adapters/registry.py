"""Unified GCP tool adapter configuration, SDK error translation, and thread-safe registry."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any

from edgar_capstone.agents.core.resilience import (
    RateLimitExceededError,
    ToolExecutionError,
    UpstreamTimeoutError,
)
from edgar_capstone.common.cache import BoundedTTLCache
from edgar_capstone.gcp.defaults import (
    DEFAULT_BIGQUERY_DATASET_ID,
    DEFAULT_BUCKET_NAME,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
    resolve_bq_dataset_id,
    resolve_corpus_bucket,
    resolve_gcp_location,
    resolve_gcp_project_id,
)
from edgar_capstone.gcp.discovery_engine import (
    DEFAULT_VERTEX_AI_SEARCH_COLLECTION,
    DEFAULT_VERTEX_AI_SEARCH_DATASTORE_ID,
    DEFAULT_VERTEX_AI_SEARCH_ENGINE_ID,
    DEFAULT_VERTEX_AI_SEARCH_LOCATION,
)
from edgar_capstone.gcp.errors import (
    GcpHttpError,
    GcpNotFoundError,
    GcpRateLimitError,
    GcpTimeoutError,
    translate_gcp_sdk_error,
)
from edgar_capstone.vector_db.embedder import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    VertexAIEmbedder,
)

if TYPE_CHECKING:
    from edgar_capstone.agents.adapters.bigquery import BigQueryAdapter
    from edgar_capstone.agents.adapters.gcs import GcsCorpusAdapter
    from edgar_capstone.agents.adapters.vertex_search import VertexAISearchAdapter

DEFAULT_BQ_FINANCIAL_METRICS_TABLE = "financial_metrics"
DEFAULT_BQ_FILING_CHUNKS_TABLE = "filing_chunks"
DEFAULT_BQ_YOY_VARIANCES_VIEW = "v_financial_variances_yoy"
DEFAULT_BQ_CHUNK_HEALTH_VIEW = "v_chunk_health_metrics"


class GcpAdapterHttpError(ToolExecutionError):
    """Raised when a GCP API returns a non-retryable 4xx/API error."""

    def __init__(self, *, status_code: int, url: str, response_text: str) -> None:
        super().__init__(f"GCP API request failed (HTTP {status_code}) for {url}: {response_text}")
        self.status_code = status_code
        self.url = url
        self.response_text = response_text


class GcpEndpointNotReadyError(ToolExecutionError):
    """Raised when a required Discovery Engine / GCP endpoint is not configured or ready."""


def translate_gcp_adapter_error(
    exc: BaseException,
    *,
    context: str = "GCP API",
) -> ToolExecutionError:
    """Translate any GCP SDK or ``GcpClientError`` into an ADK tool resilience exception."""
    if isinstance(exc, ToolExecutionError):
        return exc
    translated = translate_gcp_sdk_error(exc, context=context)
    if isinstance(translated, GcpRateLimitError):
        return RateLimitExceededError(
            str(translated), retry_after_seconds=translated.retry_after_seconds
        )
    if isinstance(translated, GcpTimeoutError):
        return UpstreamTimeoutError(str(translated))
    if isinstance(translated, GcpNotFoundError):
        return GcpAdapterHttpError(
            status_code=404,
            url=context,
            response_text=str(translated),
        )
    if isinstance(translated, GcpHttpError):
        return GcpAdapterHttpError(
            status_code=translated.status_code or 400,
            url=translated.url,
            response_text=translated.response_text,
        )
    return ToolExecutionError(str(translated))


@dataclass(frozen=True, slots=True)
class GcpToolConfig:
    """Runtime configuration for BigQuery, Vertex AI Search, and GCS adapters."""

    project_id: str = DEFAULT_PROJECT_ID
    location: str = DEFAULT_LOCATION
    bq_dataset_id: str = DEFAULT_BIGQUERY_DATASET_ID
    bq_financial_metrics_table: str = DEFAULT_BQ_FINANCIAL_METRICS_TABLE
    bq_filing_chunks_table: str = DEFAULT_BQ_FILING_CHUNKS_TABLE
    bq_yoy_view: str = DEFAULT_BQ_YOY_VARIANCES_VIEW
    bq_chunk_health_view: str = DEFAULT_BQ_CHUNK_HEALTH_VIEW
    sec_corpus_bucket: str = DEFAULT_BUCKET_NAME
    vais_location: str = DEFAULT_VERTEX_AI_SEARCH_LOCATION
    vais_collection_id: str = DEFAULT_VERTEX_AI_SEARCH_COLLECTION
    vais_engine_id: str = DEFAULT_VERTEX_AI_SEARCH_ENGINE_ID
    vais_datastore_id: str = DEFAULT_VERTEX_AI_SEARCH_DATASTORE_ID
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS

    @classmethod
    def from_env(cls) -> GcpToolConfig:
        """Resolve live GCP tool adapter settings from environment variables."""
        raw_dims = os.environ.get("VERTEX_EMBEDDING_DIMENSIONS", "").strip()
        dims = int(raw_dims) if raw_dims.isdigit() else DEFAULT_EMBEDDING_DIMENSIONS

        return cls(
            project_id=resolve_gcp_project_id(),
            location=resolve_gcp_location(),
            bq_dataset_id=resolve_bq_dataset_id(),
            bq_financial_metrics_table=(
                os.environ.get("BIGQUERY_FINANCIAL_METRICS_TABLE")
                or DEFAULT_BQ_FINANCIAL_METRICS_TABLE
            ).strip(),
            bq_filing_chunks_table=(
                os.environ.get("BIGQUERY_FILING_CHUNKS_TABLE") or DEFAULT_BQ_FILING_CHUNKS_TABLE
            ).strip(),
            bq_yoy_view=(
                os.environ.get("BIGQUERY_YOY_VARIANCE_VIEW") or DEFAULT_BQ_YOY_VARIANCES_VIEW
            ).strip(),
            bq_chunk_health_view=(
                os.environ.get("BIGQUERY_CHUNK_HEALTH_VIEW") or DEFAULT_BQ_CHUNK_HEALTH_VIEW
            ).strip(),
            sec_corpus_bucket=resolve_corpus_bucket(),
            vais_location=(
                os.environ.get("VERTEX_AI_SEARCH_LOCATION") or DEFAULT_VERTEX_AI_SEARCH_LOCATION
            ).strip(),
            vais_collection_id=(
                os.environ.get("VERTEX_AI_SEARCH_COLLECTION") or DEFAULT_VERTEX_AI_SEARCH_COLLECTION
            ).strip(),
            # Intentionally use dict.get default (not `or`) so setting either env var to `""`
            # explicitly disables that Discovery Engine servingConfig target (see
            # `tests/test_live_gcp_tool_adapters.py`).
            vais_engine_id=os.environ.get(
                "VERTEX_AI_SEARCH_ENGINE_ID", DEFAULT_VERTEX_AI_SEARCH_ENGINE_ID
            ).strip(),
            vais_datastore_id=os.environ.get(
                "VERTEX_AI_SEARCH_DATASTORE_ID", DEFAULT_VERTEX_AI_SEARCH_DATASTORE_ID
            ).strip(),
            embedding_model=(
                os.environ.get("VERTEX_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
            ).strip(),
            embedding_dimensions=dims,
        )


class GcpToolAdapters:
    """Bundle of live GCP SDK tool adapters shared across ADK tool invocations."""

    def __init__(
        self,
        config: GcpToolConfig | None = None,
        *,
        bigquery: BigQueryAdapter | None = None,
        vertex_ai_search: VertexAISearchAdapter | None = None,
        gcs_corpus: GcsCorpusAdapter | None = None,
        embedder: VertexAIEmbedder | None = None,
        bq_client: Any | None = None,
        search_client: Any | None = None,
        storage_client: Any | None = None,
        genai_client: Any | None = None,
        credentials: Any | None = None,
    ) -> None:
        from edgar_capstone.agents.adapters.bigquery import BigQueryAdapter as _BQ
        from edgar_capstone.agents.adapters.gcs import GcsCorpusAdapter as _GCS
        from edgar_capstone.agents.adapters.vertex_search import VertexAISearchAdapter as _VAIS

        self.config = config or GcpToolConfig.from_env()
        self.bigquery: BigQueryAdapter = bigquery or _BQ(
            self.config, bq_client=bq_client, credentials=credentials
        )
        self.vertex_ai_search: VertexAISearchAdapter = vertex_ai_search or _VAIS(
            self.config, search_client=search_client, credentials=credentials
        )
        self.gcs_corpus: GcsCorpusAdapter = gcs_corpus or _GCS(
            self.config, storage_client=storage_client, credentials=credentials
        )
        self.embedder = embedder or VertexAIEmbedder(
            project_id=self.config.project_id,
            location=self.config.location,
            model_name=self.config.embedding_model,
            dimensions=self.config.embedding_dimensions,
            max_retries=1,
            max_workers=4,
            genai_client=genai_client,
            credentials=credentials,
        )
        self._query_embedding_cache: BoundedTTLCache[tuple[str, str], list[float]] = (
            BoundedTTLCache(max_entries=256, ttl_seconds=0.0)
        )

    def close(self) -> None:
        """Close all internally managed SDK client resources."""
        self.bigquery.close()
        self.vertex_ai_search.close()
        self.gcs_corpus.close()
        embedder_close = getattr(self.embedder, "close", None)
        if callable(embedder_close):
            embedder_close()

    def __enter__(self) -> GcpToolAdapters:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def embed_texts(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_QUERY",
    ) -> list[list[float]]:
        """Generate 768-d `text-embedding-005` vectors via `google-genai`, translating SDK errors."""
        if not texts:
            return []
        if len(texts) == 1:
            cache_key = (task_type, texts[0])
            cached = self._query_embedding_cache.get(cache_key)
            if cached is not None:
                return [cached]

        try:
            vectors = self.embedder.embed_texts(texts, task_type=task_type)
        except Exception as exc:
            raise translate_gcp_adapter_error(exc, context="Vertex AI embeddings") from exc

        if len(texts) == 1 and vectors:
            self._query_embedding_cache[(task_type, texts[0])] = vectors[0]
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query with `task_type='RETRIEVAL_QUERY'`."""
        return self.embed_texts([query], task_type="RETRIEVAL_QUERY")[0]


_ADAPTERS_LOCK = threading.Lock()
_CUSTOM_GCP_ADAPTERS: GcpToolAdapters | None = None
_ENV_GCP_ADAPTERS_CACHE: tuple[GcpToolConfig, GcpToolAdapters] | None = None
_RESET_HOOKS: list[Callable[[], None]] = []


def register_adapter_reset_hook(hook: Callable[[], None]) -> None:
    """Register a callback invoked whenever active GCP adapters are replaced or reset."""
    with _ADAPTERS_LOCK:
        if hook not in _RESET_HOOKS:
            _RESET_HOOKS.append(hook)


def _run_reset_hooks() -> None:
    with _ADAPTERS_LOCK:
        hooks = list(_RESET_HOOKS)
    for hook in hooks:
        hook()


def configure_gcp_tool_adapters(adapters: GcpToolAdapters | None) -> None:
    """Explicitly inject a `GcpToolAdapters` bundle (used for typed fakes or integration tests)."""
    global _CUSTOM_GCP_ADAPTERS
    with _ADAPTERS_LOCK:
        _CUSTOM_GCP_ADAPTERS = adapters
    _run_reset_hooks()


def reset_gcp_tool_adapters() -> None:
    """Clear any injected or cached `GcpToolAdapters` instance and close replaced SDK clients."""
    global _CUSTOM_GCP_ADAPTERS, _ENV_GCP_ADAPTERS_CACHE
    old_env: GcpToolAdapters | None = None
    with _ADAPTERS_LOCK:
        _CUSTOM_GCP_ADAPTERS = None
        if _ENV_GCP_ADAPTERS_CACHE is not None:
            old_env = _ENV_GCP_ADAPTERS_CACHE[1]
            _ENV_GCP_ADAPTERS_CACHE = None
    if old_env is not None:
        old_env.close()
    _run_reset_hooks()


def _gcp_adapters() -> GcpToolAdapters:
    """Return the live GCP adapter bundle: an injected one, or one built from the environment.

    There is no local data backend: every tool reads BigQuery, Vertex AI Search,
    Vertex AI embeddings, and the GCS corpus bucket.
    """
    global _ENV_GCP_ADAPTERS_CACHE
    old_env: GcpToolAdapters | None = None
    with _ADAPTERS_LOCK:
        if _CUSTOM_GCP_ADAPTERS is not None:
            return _CUSTOM_GCP_ADAPTERS
        cfg = GcpToolConfig.from_env()
        if _ENV_GCP_ADAPTERS_CACHE is None or _ENV_GCP_ADAPTERS_CACHE[0] != cfg:
            if _ENV_GCP_ADAPTERS_CACHE is not None:
                old_env = _ENV_GCP_ADAPTERS_CACHE[1]
            _ENV_GCP_ADAPTERS_CACHE = (cfg, GcpToolAdapters(cfg))
        active = _ENV_GCP_ADAPTERS_CACHE[1]
    if old_env is not None:
        old_env.close()
    return active


def get_active_gcp_tool_adapters() -> GcpToolAdapters:
    """Return the active live `GcpToolAdapters` bundle."""
    return _gcp_adapters()


def get_gcp_adapters_status_metadata() -> dict[str, Any]:
    """Return secret-free diagnostic metadata describing configured GCP tool adapters."""
    with _ADAPTERS_LOCK:
        custom = _CUSTOM_GCP_ADAPTERS
    cfg = custom.config if custom is not None else GcpToolConfig.from_env()
    return {
        "data_backend": "gcp",
        "project_id": cfg.project_id,
        "location": cfg.location,
        "hybrid_retrieval_backends": ["bigquery_vector_search", "vertex_ai_search"],
        "bigquery": {
            "dataset_id": cfg.bq_dataset_id,
            "financial_metrics_table": cfg.bq_financial_metrics_table,
            "filing_chunks_table": cfg.bq_filing_chunks_table,
            "yoy_variance_view": cfg.bq_yoy_view,
            "chunk_health_view": cfg.bq_chunk_health_view,
        },
        "vertex_ai_search": {
            "location": cfg.vais_location,
            "collection_id": cfg.vais_collection_id,
            "engine_id": cfg.vais_engine_id,
            "datastore_id": cfg.vais_datastore_id,
        },
        "gcs_corpus_bucket": cfg.sec_corpus_bucket,
        "embedding_model": cfg.embedding_model,
        "embedding_dimensions": cfg.embedding_dimensions,
    }
