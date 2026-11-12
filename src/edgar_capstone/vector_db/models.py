"""Pydantic models for metadata-tagged 10-K RAG chunks and chunk health metrics."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class VertexVectorRestrict(BaseModel):
    """String namespace token filter for Vertex AI Vector Search (Matching Engine)."""

    namespace: str = Field(..., description="Metadata attribute name (e.g., ticker, gics_sector)")
    allow: list[str] = Field(default_factory=list, description="Allowed token values")
    deny: list[str] = Field(default_factory=list, description="Denied token values")


class VertexVectorNumericRestrict(BaseModel):
    """Numeric attribute filter for Vertex AI Vector Search (Matching Engine)."""

    namespace: str = Field(..., description="Numeric metadata attribute name (e.g., fiscal_year)")
    value_int: int | None = Field(default=None, description="Integer value for filtering")
    value_float: float | None = Field(default=None, description="Float value for filtering")


class FilingChunk(BaseModel):
    """Single metadata-tagged 10-K text chunk and dense vector embedding.

    Aligns 1:1 with `terraform/schemas/filing_chunks.json` for BigQuery `VECTOR_SEARCH`
    and serializes to Vertex AI Vector Search batch/stream JSONL format.
    """

    chunk_id: str = Field(
        ..., description="Unique chunk identifier (e.g., NVDA_2026_item_1a_0000)"
    )
    profile_id: str = Field(
        ..., description="Foreign key to financial_metrics.profile_id ({TICKER}_{FY})"
    )
    ticker: str = Field(..., description="Canonical SEC EDGAR company ticker symbol")
    cik: str = Field(..., description="10-digit zero-padded SEC Central Index Key")
    company_name: str = Field(..., description="Official SEC registrant name")
    gics_sector: str = Field(
        default="Unknown",
        description="GICS Sector classification for metadata-filtered RAG",
    )
    gics_sub_industry: str = Field(
        default="Unknown", description="GICS Sub-Industry classification"
    )
    form_type: str = Field(default="10-K", description="SEC form type (10-K)")
    fiscal_year: int = Field(
        ..., description="Reporting fiscal year for longitudinal RAG filtering"
    )
    period_of_report: str | None = Field(
        default=None, description="Fiscal period end date (YYYY-MM-DD)"
    )
    filing_date: str | None = Field(
        default=None, description="SEC EDGAR filing submission date (YYYY-MM-DD)"
    )
    accession_number: str = Field(..., description="SEC EDGAR accession number")
    section_id: Literal["item_1a", "item_7"] = Field(
        ..., description="Normalized 10-K section identifier (item_1a or item_7)"
    )
    section_title: str = Field(
        ..., description="Human-readable section & subsection heading for grounded citations"
    )
    chunk_index: int = Field(
        ..., ge=0, description="0-based sequential chunk index within the section"
    )
    chunk_text: str = Field(
        ..., description="Markdown passage text for RAG synthesis and grounding"
    )
    char_count: int = Field(..., ge=0, description="Character length of chunk_text")
    token_estimate: int = Field(
        ..., ge=0, description="Approximate token count of chunk_text"
    )
    embedding: list[float] = Field(
        default_factory=list,
        description="Dense vector embedding (ARRAY<FLOAT64>) for vector search",
    )
    embedding_model: str = Field(
        default="text-embedding-005",
        description="Vertex AI embedding model identifier",
    )
    unstructured_gcs_uri: str = Field(
        ..., description="GCS URI of the parent unstructured 10-K Markdown document"
    )
    structured_gcs_uri: str = Field(
        ..., description="GCS URI of the paired structured financial metrics JSON"
    )
    sec_filing_url: str = Field(
        ..., description="Direct SEC EDGAR archive URL for split-pane citation viewing"
    )
    ingested_at: str = Field(
        default_factory=utc_now_iso,
        description="UTC timestamp when the chunk and embedding were indexed",
    )

    def to_bigquery_record(self) -> dict[str, Any]:
        """Serialize chunk to a dict matching `terraform/schemas/filing_chunks.json`."""
        return self.model_dump(mode="json")

    def to_vertex_vector_search_record(self) -> dict[str, Any]:
        """Serialize chunk to Vertex AI Vector Search (Matching Engine) JSONL format.

        Includes metadata `restricts` for pre-filtered semantic retrieval by ticker,
        CIK, GICS sector, GICS sub-industry, section ID, form type, and profile ID,
        plus `numeric_restricts` for fiscal year range queries and `crowding_tag`
        to ensure multi-company diversity in top-k results.
        """
        restricts: list[dict[str, Any]] = [
            {"namespace": "ticker", "allow": [self.ticker]},
            {"namespace": "cik", "allow": [self.cik]},
            {"namespace": "gics_sector", "allow": [self.gics_sector]},
            {"namespace": "gics_sub_industry", "allow": [self.gics_sub_industry]},
            {"namespace": "section_id", "allow": [self.section_id]},
            {"namespace": "form_type", "allow": [self.form_type]},
            {"namespace": "profile_id", "allow": [self.profile_id]},
        ]
        numeric_restricts: list[dict[str, Any]] = [
            {"namespace": "fiscal_year", "value_int": int(self.fiscal_year)},
        ]
        return {
            "id": self.chunk_id,
            "embedding": self.embedding,
            "restricts": restricts,
            "numeric_restricts": numeric_restricts,
            "crowding_tag": self.ticker,
            "embedding_metadata": {
                "profile_id": self.profile_id,
                "ticker": self.ticker,
                "company_name": self.company_name,
                "fiscal_year": self.fiscal_year,
                "section_id": self.section_id,
                "section_title": self.section_title,
                "chunk_index": self.chunk_index,
                "char_count": self.char_count,
                "token_estimate": self.token_estimate,
                "sec_filing_url": self.sec_filing_url,
                "unstructured_gcs_uri": self.unstructured_gcs_uri,
            },
        }


class ChunkAnomalyCounts(BaseModel):
    """Counts of chunk quality and indexing anomalies detected across the vector DB."""

    missing_embedding: int = Field(
        default=0, description="Chunks with empty or null embedding arrays"
    )
    dimension_mismatch: int = Field(
        default=0,
        description="Chunks whose embedding dimension != expected_embedding_dim",
    )
    non_finite_embedding: int = Field(
        default=0,
        description="Chunks containing NaN, Inf, or zero-norm embedding vectors",
    )
    undersized_chunk: int = Field(
        default=0,
        description="Chunks below the minimum character threshold (<100 chars)",
    )
    oversized_chunk: int = Field(
        default=0,
        description="Chunks exceeding the maximum character/token threshold (>8000 chars or >2048 tokens)",
    )
    missing_metadata: int = Field(
        default=0,
        description="Chunks missing required filter metadata (sector, ticker, section, or citation URL)",
    )
    missing_section_filings: int = Field(
        default=0,
        description="Filings missing either item_1a or item_7 chunks",
    )

    @property
    def total_anomalies(self) -> int:
        """Total count of all detected chunk anomalies."""
        return (
            self.missing_embedding
            + self.dimension_mismatch
            + self.non_finite_embedding
            + self.undersized_chunk
            + self.oversized_chunk
            + self.missing_metadata
            + self.missing_section_filings
        )


class SectionHealthBreakdown(BaseModel):
    """Chunk health metrics grouped by 10-K section (`item_1a` vs `item_7`) or GICS sector."""

    group_key: str = Field(..., description="Section ID or GICS Sector name")
    total_chunks: int = 0
    distinct_profiles: int = 0
    distinct_tickers: int = 0
    valid_embedding_count: int = 0
    embedding_coverage_ratio: float = 0.0
    avg_char_count: float = 0.0
    avg_token_estimate: float = 0.0
    min_token_estimate: int = 0
    max_token_estimate: int = 0
    p50_token_estimate: float = 0.0
    p95_token_estimate: float = 0.0
    undersized_chunk_count: int = 0
    oversized_chunk_count: int = 0
    missing_metadata_count: int = 0


class ChunkHealthReport(BaseModel):
    """Comprehensive health telemetry report for the SEC 10-K RAG vector database."""

    project_id: str
    dataset_id: str = "sec_edgar"
    table_id: str = "filing_chunks"
    evaluated_at: str = Field(default_factory=utc_now_iso)
    expected_embedding_dim: int = 768
    embedding_model: str = "text-embedding-005"

    # Filing & section coverage
    total_profiles_expected: int = 0
    total_profiles_indexed: int = 0
    profiles_with_both_sections: int = 0
    filing_coverage_ratio: float = 0.0
    full_section_coverage_ratio: float = 0.0
    missing_profile_ids: list[str] = Field(default_factory=list)
    incomplete_section_profile_ids: list[str] = Field(default_factory=list)

    # Chunk & embedding health
    total_chunks: int = 0
    valid_embedding_count: int = 0
    embedding_coverage_ratio: float = 0.0

    # Size & token distribution
    avg_char_count: float = 0.0
    avg_token_estimate: float = 0.0
    min_token_estimate: int = 0
    max_token_estimate: int = 0
    p50_token_estimate: float = 0.0
    p95_token_estimate: float = 0.0

    # Anomalies & breakdowns
    anomalies: ChunkAnomalyCounts = Field(default_factory=ChunkAnomalyCounts)
    by_section: dict[str, SectionHealthBreakdown] = Field(default_factory=dict)
    by_sector: dict[str, SectionHealthBreakdown] = Field(default_factory=dict)

    # Composite health score & status
    health_score: float = Field(
        default=0.0, description="Composite 0-100 health score of the chunk database"
    )
    health_status: Literal["HEALTHY", "DEGRADED", "UNHEALTHY"] = "UNHEALTHY"
    published_to_cloud_monitoring: bool = False
    published_timeseries_count: int = 0


class VectorDBBuildSummary(BaseModel):
    """Summary artifact written after building and populating the RAG vector database."""

    project_id: str
    bucket_name: str
    dataset_id: str
    table_id: str
    location: str
    generated_at: str = Field(default_factory=utc_now_iso)
    embedding_model: str
    embedding_backend: str
    embedding_dimensions: int
    total_profiles_processed: int
    total_chunks_generated: int
    item_1a_chunks: int
    item_7_chunks: int
    bigquery_manifest_path: str
    vertex_vector_manifest_path: str
    health_report_path: str
    uploaded_to_gcs: bool = False
    loaded_to_bigquery: bool = False
    published_monitoring_metrics: bool = False
    health_score: float = 0.0
    health_status: Literal["HEALTHY", "DEGRADED", "UNHEALTHY"] = "UNHEALTHY"
    warnings: list[str] = Field(default_factory=list)
