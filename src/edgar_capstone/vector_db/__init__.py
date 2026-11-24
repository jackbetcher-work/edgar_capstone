"""Metadata-tagged vector database creation, population, and chunk health monitoring.

Uses PEP 562 lazy attribute loading so importing lightweight submodules does not
eagerly pull in ``google-cloud-storage`` or pipeline dependencies.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from edgar_capstone.vector_db.chunker import (
        chunk_filing_document,
        estimate_token_count,
        extract_sections_from_combined_markdown,
        split_section_into_passages,
    )
    from edgar_capstone.vector_db.embedder import (
        DEFAULT_EMBEDDING_DIMENSIONS,
        DEFAULT_EMBEDDING_MODEL,
        VertexAIEmbedder,
        populate_chunk_embeddings,
    )
    from edgar_capstone.vector_db.models import (
        ChunkAnomalyCounts,
        ChunkHealthReport,
        FilingChunk,
        SectionHealthBreakdown,
        VectorDBBuildSummary,
    )
    from edgar_capstone.vector_db.monitoring import (
        CloudMonitoringHealthPublisher,
        build_cloud_monitoring_timeseries,
        evaluate_chunk_health,
        evaluate_chunk_health_from_bigquery,
        format_structured_health_log,
    )
    from edgar_capstone.vector_db.pipeline import (
        VectorDBBuildConfig,
        run_vector_db_pipeline,
        trigger_bigquery_chunks_load_job,
        upload_vector_db_artifacts_to_gcs,
    )

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_EMBEDDING_DIMENSIONS": (
        "edgar_capstone.vector_db.embedder",
        "DEFAULT_EMBEDDING_DIMENSIONS",
    ),
    "DEFAULT_EMBEDDING_MODEL": ("edgar_capstone.vector_db.embedder", "DEFAULT_EMBEDDING_MODEL"),
    "ChunkAnomalyCounts": ("edgar_capstone.vector_db.models", "ChunkAnomalyCounts"),
    "ChunkHealthReport": ("edgar_capstone.vector_db.models", "ChunkHealthReport"),
    "CloudMonitoringHealthPublisher": (
        "edgar_capstone.vector_db.monitoring",
        "CloudMonitoringHealthPublisher",
    ),
    "FilingChunk": ("edgar_capstone.vector_db.models", "FilingChunk"),
    "SectionHealthBreakdown": ("edgar_capstone.vector_db.models", "SectionHealthBreakdown"),
    "VectorDBBuildConfig": ("edgar_capstone.vector_db.pipeline", "VectorDBBuildConfig"),
    "VectorDBBuildSummary": ("edgar_capstone.vector_db.models", "VectorDBBuildSummary"),
    "VertexAIEmbedder": ("edgar_capstone.vector_db.embedder", "VertexAIEmbedder"),
    "build_cloud_monitoring_timeseries": (
        "edgar_capstone.vector_db.monitoring",
        "build_cloud_monitoring_timeseries",
    ),
    "chunk_filing_document": ("edgar_capstone.vector_db.chunker", "chunk_filing_document"),
    "estimate_token_count": ("edgar_capstone.vector_db.chunker", "estimate_token_count"),
    "evaluate_chunk_health": ("edgar_capstone.vector_db.monitoring", "evaluate_chunk_health"),
    "evaluate_chunk_health_from_bigquery": (
        "edgar_capstone.vector_db.monitoring",
        "evaluate_chunk_health_from_bigquery",
    ),
    "extract_sections_from_combined_markdown": (
        "edgar_capstone.vector_db.chunker",
        "extract_sections_from_combined_markdown",
    ),
    "format_structured_health_log": (
        "edgar_capstone.vector_db.monitoring",
        "format_structured_health_log",
    ),
    "populate_chunk_embeddings": ("edgar_capstone.vector_db.embedder", "populate_chunk_embeddings"),
    "run_vector_db_pipeline": ("edgar_capstone.vector_db.pipeline", "run_vector_db_pipeline"),
    "split_section_into_passages": (
        "edgar_capstone.vector_db.chunker",
        "split_section_into_passages",
    ),
    "trigger_bigquery_chunks_load_job": (
        "edgar_capstone.vector_db.pipeline",
        "trigger_bigquery_chunks_load_job",
    ),
    "upload_vector_db_artifacts_to_gcs": (
        "edgar_capstone.vector_db.pipeline",
        "upload_vector_db_artifacts_to_gcs",
    ),
}

__all__ = [
    "DEFAULT_EMBEDDING_DIMENSIONS",
    "DEFAULT_EMBEDDING_MODEL",
    "ChunkAnomalyCounts",
    "ChunkHealthReport",
    "CloudMonitoringHealthPublisher",
    "FilingChunk",
    "SectionHealthBreakdown",
    "VectorDBBuildConfig",
    "VectorDBBuildSummary",
    "VertexAIEmbedder",
    "build_cloud_monitoring_timeseries",
    "chunk_filing_document",
    "estimate_token_count",
    "evaluate_chunk_health",
    "evaluate_chunk_health_from_bigquery",
    "extract_sections_from_combined_markdown",
    "format_structured_health_log",
    "populate_chunk_embeddings",
    "run_vector_db_pipeline",
    "split_section_into_passages",
    "trigger_bigquery_chunks_load_job",
    "upload_vector_db_artifacts_to_gcs",
]


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
