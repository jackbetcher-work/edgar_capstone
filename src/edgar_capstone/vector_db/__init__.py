"""Metadata-tagged vector database creation, population, and chunk health monitoring."""

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
    generate_deterministic_embedding,
    populate_chunk_embeddings,
)
from edgar_capstone.vector_db.models import (
    ChunkAnomalyCounts,
    ChunkHealthReport,
    FilingChunk,
    SectionHealthBreakdown,
    VectorDBBuildSummary,
    VertexVectorNumericRestrict,
    VertexVectorRestrict,
)
from edgar_capstone.vector_db.monitoring import (
    CloudMonitoringHealthPublisher,
    build_cloud_monitoring_timeseries,
    evaluate_chunk_health,
    format_structured_health_log,
)
from edgar_capstone.vector_db.pipeline import (
    VectorDBBuildConfig,
    run_vector_db_pipeline,
    trigger_bigquery_chunks_load_job,
    upload_vector_db_artifacts_to_gcs,
)

__all__ = [
    "ChunkAnomalyCounts",
    "ChunkHealthReport",
    "CloudMonitoringHealthPublisher",
    "DEFAULT_EMBEDDING_DIMENSIONS",
    "DEFAULT_EMBEDDING_MODEL",
    "FilingChunk",
    "SectionHealthBreakdown",
    "VectorDBBuildConfig",
    "VectorDBBuildSummary",
    "VertexAIEmbedder",
    "VertexVectorNumericRestrict",
    "VertexVectorRestrict",
    "build_cloud_monitoring_timeseries",
    "chunk_filing_document",
    "estimate_token_count",
    "evaluate_chunk_health",
    "extract_sections_from_combined_markdown",
    "format_structured_health_log",
    "generate_deterministic_embedding",
    "populate_chunk_embeddings",
    "run_vector_db_pipeline",
    "split_section_into_passages",
    "trigger_bigquery_chunks_load_job",
    "upload_vector_db_artifacts_to_gcs",
]
