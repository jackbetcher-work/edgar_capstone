"""Shared Pydantic base models for SEC EDGAR filing identity and provenance."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FilingMetadataBase(BaseModel):
    """Shared SEC 10-K filing identity fields across dataset, vector_db, and agent schemas."""

    profile_id: str = Field(..., description="Unique company-year ID (e.g., NVDA_2026)")
    ticker: str = Field(..., description="Canonical company ticker symbol (e.g., NVDA, BRK-B)")
    company_name: str = Field(..., description="Official SEC registrant name")
    gics_sector: str = Field(..., description="GICS Sector classification")
    fiscal_year: int = Field(..., description="Reporting fiscal year")
    sec_filing_url: str = Field(..., description="Direct SEC EDGAR archive URL")
    unstructured_gcs_uri: str = Field(
        ..., description="GCS URI of the unstructured 10-K Markdown document"
    )
