"""Pydantic schemas for curated SEC 10-K Data Profiles and ingestion manifests."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class CompanyConstituent(BaseModel):
    """Metadata for an S&P 500 or requested public company."""

    model_config = ConfigDict(frozen=True)

    ticker: str = Field(..., description="Canonical ticker symbol (e.g., NVDA, BRK.B)")
    edgar_ticker: str = Field(
        ..., description="SEC EDGAR lookup ticker symbol (e.g., NVDA, BRK-B)"
    )
    company_name: str = Field(..., description="Company display or SEC legal name")
    cik: str | None = Field(
        default=None, description="10-digit zero-padded SEC Central Index Key"
    )
    gics_sector: str = Field(default="Unknown", description="GICS Sector classification")
    gics_sub_industry: str = Field(
        default="Unknown", description="GICS Sub-Industry classification"
    )


class MetricProvenance(BaseModel):
    """Tracks exact XBRL / extraction provenance for each structured financial metric."""

    revenue: str | None = Field(
        default=None, description="XBRL concept or method used to extract Revenue"
    )
    operating_income: str | None = Field(
        default=None,
        description="XBRL concept or method used to extract Operating Income",
    )
    net_income: str | None = Field(
        default=None, description="XBRL concept or method used to extract Net Income"
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Accounting notes (e.g., pre-tax income proxy for banking/energy filers)",
    )


class FinancialDataProfile(BaseModel):
    """Structured financial profile for a single company-year 10-K filing.

    Serialized to `structured/{ticker}/{fy}_financials.json` and aggregated into
    `manifests/bigquery_financials.jsonl` for BigQuery metadata-filtered RAG and
    deterministic variance calculations.
    """

    profile_id: str = Field(..., description="Unique identifier: {TICKER}_{FY}")
    ticker: str = Field(..., description="Canonical company ticker symbol")
    cik: str = Field(..., description="10-digit zero-padded SEC CIK")
    company_name: str = Field(..., description="Official SEC registrant name")
    gics_sector: str = Field(default="Unknown", description="GICS Sector")
    gics_sub_industry: str = Field(default="Unknown", description="GICS Sub-Industry")
    form_type: str = Field(default="10-K", description="SEC form type")
    fiscal_year: int = Field(..., description="Reporting fiscal year (e.g., 2026)")
    fiscal_period: str = Field(default="FY", description="Fiscal period focus")
    period_of_report: str = Field(
        ..., description="Fiscal period end date (YYYY-MM-DD)"
    )
    filing_date: str = Field(..., description="SEC EDGAR filing date (YYYY-MM-DD)")
    accession_number: str = Field(
        ..., description="SEC accession number (e.g., 0001045810-26-000021)"
    )
    currency: str = Field(default="USD", description="Reporting currency ISO code")
    revenue: float | None = Field(
        default=None, description="Total Revenue in raw currency units (USD)"
    )
    operating_income: float | None = Field(
        default=None,
        description="Operating Income (or pre-tax income for single-step filers) in USD",
    )
    net_income: float | None = Field(
        default=None, description="Net Income attributable to parent in USD"
    )
    metric_provenance: MetricProvenance = Field(
        default_factory=MetricProvenance,
        description="Provenance details for Revenue, Operating Income, and Net Income",
    )
    extracted_sections: list[str] = Field(
        default_factory=list,
        description="10-K sections included in the paired unstructured Markdown file",
    )
    unstructured_blob_path: str = Field(
        ..., description="Relative GCS object path for unstructured 10-K Markdown"
    )
    structured_blob_path: str = Field(
        ..., description="Relative GCS object path for structured financials JSON"
    )
    unstructured_gcs_uri: str = Field(
        ..., description="Full gs:// URI for unstructured 10-K Markdown"
    )
    structured_gcs_uri: str = Field(
        ..., description="Full gs:// URI for structured financials JSON"
    )
    sec_filing_url: str = Field(
        ..., description="Public SEC EDGAR URL for the source filing"
    )
    ingested_at: str = Field(
        default_factory=utc_now_iso, description="UTC ingestion timestamp (ISO-8601)"
    )


class VertexAISearchContent(BaseModel):
    """Content reference block for Vertex AI Search unstructured JSONL import."""

    mimeType: str = Field(
        default="text/plain",
        description="Discovery Engine supported MIME type for Markdown/text documents",
    )
    uri: str = Field(..., description="GCS URI pointing to the unstructured 10-K file")


class VertexAISearchManifestRecord(BaseModel):
    """Single JSONL record for Vertex AI Search (GEAP Agent Search) ingestion."""

    id: str = Field(..., description="Unique document ID (e.g., NVDA_2026_10K)")
    structData: dict[str, Any] = Field(
        ..., description="Filterable structured metadata attached to the 10-K document"
    )
    content: VertexAISearchContent

    @classmethod
    def from_profile(cls, profile: FinancialDataProfile) -> VertexAISearchManifestRecord:
        """Build a Vertex AI Search manifest entry directly from a FinancialDataProfile."""
        struct_data: dict[str, Any] = {
            "profile_id": profile.profile_id,
            "ticker": profile.ticker,
            "cik": profile.cik,
            "company_name": profile.company_name,
            "gics_sector": profile.gics_sector,
            "gics_sub_industry": profile.gics_sub_industry,
            "form_type": profile.form_type,
            "fiscal_year": profile.fiscal_year,
            "period_of_report": profile.period_of_report,
            "filing_date": profile.filing_date,
            "accession_number": profile.accession_number,
            "currency": profile.currency,
            "extracted_sections": profile.extracted_sections,
            "structured_gcs_uri": profile.structured_gcs_uri,
            "sec_filing_url": profile.sec_filing_url,
        }
        if profile.revenue is not None:
            struct_data["revenue"] = profile.revenue
        if profile.operating_income is not None:
            struct_data["operating_income"] = profile.operating_income
        if profile.net_income is not None:
            struct_data["net_income"] = profile.net_income

        safe_doc_id = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{profile.profile_id}_10K")
        return cls(
            id=safe_doc_id,
            structData=struct_data,
            content=VertexAISearchContent(
                mimeType="text/plain",
                uri=profile.unstructured_gcs_uri,
            ),
        )


class ExtractedFilingBundle(BaseModel):
    """Pairs the structured FinancialDataProfile with the unstructured 10-K Markdown."""

    profile: FinancialDataProfile
    markdown_content: str


class DatasetBuildSummary(BaseModel):
    """Catalog summary written to `manifests/dataset_catalog.json`."""

    project_id: str
    bucket_name: str
    location: str
    start_year: int
    end_year: int
    generated_at: str = Field(default_factory=utc_now_iso)
    requested_tickers: list[str]
    processed_tickers: list[str]
    failed_tickers: dict[str, str] = Field(default_factory=dict)
    total_profiles: int = 0
    uploaded_to_gcs: bool = False
    uploaded_blob_count: int = 0
    warnings: list[str] = Field(default_factory=list)
