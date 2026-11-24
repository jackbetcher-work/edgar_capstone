"""Pydantic schemas for curated SEC 10-K Data Profiles and ingestion manifests."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from edgar_capstone.common.domain import SECTION_TITLES, SectionId
from edgar_capstone.common.formatting import utc_now_iso
from edgar_capstone.common.models import FilingMetadataBase

_INDEXABLE_SECTIONS: tuple[SectionId, ...] = ("item_1a", "item_7")


class CompanyConstituent(BaseModel):
    """Metadata for an S&P 500 or requested public company."""

    model_config = ConfigDict(frozen=True)

    ticker: str = Field(..., description="Canonical ticker symbol (e.g., NVDA, BRK.B)")
    edgar_ticker: str = Field(..., description="SEC EDGAR lookup ticker symbol (e.g., NVDA, BRK-B)")
    company_name: str = Field(..., description="Company display or SEC legal name")
    cik: str | None = Field(default=None, description="10-digit zero-padded SEC Central Index Key")
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
    research_and_development: str | None = Field(
        default=None,
        description="XBRL concept or method used to extract Research & Development expense",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Accounting notes (e.g., pre-tax income proxy for banking/energy filers)",
    )


class FinancialDataProfile(FilingMetadataBase):
    """Structured financial profile for a single company-year 10-K filing.

    Serialized to `structured/{ticker}/{fy}_financials.json` and aggregated into
    `manifests/bigquery_financials.jsonl` for BigQuery metadata-filtered RAG and
    deterministic variance calculations.
    """

    cik: str = Field(..., description="10-digit zero-padded SEC CIK")
    gics_sector: str = Field(default="Unknown", description="GICS Sector")
    gics_sub_industry: str = Field(default="Unknown", description="GICS Sub-Industry")
    form_type: str = Field(default="10-K", description="SEC form type")
    fiscal_period: str = Field(default="FY", description="Fiscal period focus")
    period_of_report: str = Field(..., description="Fiscal period end date (YYYY-MM-DD)")
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
    research_and_development: float | None = Field(
        default=None,
        description=(
            "Research & Development expense in USD. None (not 0) when the issuer does not "
            "report an R&D line in its 10-K XBRL."
        ),
    )
    metric_provenance: MetricProvenance = Field(
        default_factory=MetricProvenance,
        description="Provenance details for Revenue, Operating Income, Net Income, and R&D",
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
    structured_gcs_uri: str = Field(
        ..., description="Full gs:// URI for structured financials JSON"
    )
    ingested_at: str = Field(
        default_factory=utc_now_iso, description="UTC ingestion timestamp (ISO-8601)"
    )

    def section_blob_path(self, section_id: SectionId) -> str:
        """Relative GCS object path for a single extracted 10-K section Markdown file."""
        if self.unstructured_blob_path.endswith("_10k.md"):
            return f"{self.unstructured_blob_path[: -len('_10k.md')]}_{section_id}.md"
        return f"unstructured/{self.ticker}/{self.fiscal_year}_{section_id}.md"

    def section_gcs_uri(self, section_id: SectionId) -> str:
        """Full `gs://` URI for a single extracted 10-K section Markdown file."""
        if self.unstructured_gcs_uri.endswith("_10k.md"):
            return f"{self.unstructured_gcs_uri[: -len('_10k.md')]}_{section_id}.md"
        base = self.unstructured_gcs_uri.rsplit("/", 1)[0]
        return f"{base}/{self.fiscal_year}_{section_id}.md"


class VertexAISearchContent(BaseModel):
    """Content reference block for Vertex AI Search unstructured JSONL import."""

    mimeType: str = Field(
        default="text/plain",
        description="Discovery Engine supported MIME type for Markdown/text documents",
    )
    uri: str = Field(..., description="GCS URI pointing to the unstructured 10-K file")


class VertexAISearchManifestRecord(BaseModel):
    """Single JSONL record for Vertex AI Search (GEAP Agent Search) ingestion."""

    id: str = Field(..., description="Unique document ID (e.g., NVDA_2026_item_1a)")
    structData: dict[str, Any] = Field(
        ..., description="Filterable structured metadata attached to the 10-K section document"
    )
    content: VertexAISearchContent

    @classmethod
    def from_profile(
        cls,
        profile: FinancialDataProfile,
        *,
        section_id: SectionId | None = None,
        section_gcs_uri: str | None = None,
    ) -> VertexAISearchManifestRecord:
        """Build a Vertex AI Search manifest entry from a FinancialDataProfile.

        When `section_id` is supplied (`'item_1a'` or `'item_7'`), emits a section-scoped
        document (`<profile_id>_<section_id>`) pointing at the per-section Markdown blob with
        filterable `section_id` and `section_title` metadata so Discovery Engine filters natively
        at the section boundary.
        """
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
            "extracted_sections": [section_id] if section_id else list(profile.extracted_sections),
            "structured_gcs_uri": profile.structured_gcs_uri,
            "sec_filing_url": profile.sec_filing_url,
        }
        if section_id is not None:
            struct_data["section_id"] = section_id
            struct_data["section_title"] = SECTION_TITLES[section_id]
        if profile.revenue is not None:
            struct_data["revenue"] = profile.revenue
        if profile.operating_income is not None:
            struct_data["operating_income"] = profile.operating_income
        if profile.net_income is not None:
            struct_data["net_income"] = profile.net_income
        if profile.research_and_development is not None:
            struct_data["research_and_development"] = profile.research_and_development

        doc_suffix = section_id if section_id is not None else "10K"
        safe_doc_id = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{profile.profile_id}_{doc_suffix}")
        content_uri = (
            section_gcs_uri
            if section_gcs_uri is not None
            else (
                profile.section_gcs_uri(section_id)
                if section_id is not None
                else profile.unstructured_gcs_uri
            )
        )
        return cls(
            id=safe_doc_id,
            structData=struct_data,
            content=VertexAISearchContent(
                mimeType="text/plain",
                uri=content_uri,
            ),
        )

    @classmethod
    def records_from_profile(
        cls,
        profile: FinancialDataProfile,
        *,
        section_ids: Sequence[SectionId] | None = None,
    ) -> list[VertexAISearchManifestRecord]:
        """Build one Vertex AI Search manifest record per extracted section (`item_1a`, `item_7`)."""
        sections: list[SectionId] = (
            list(section_ids)
            if section_ids is not None
            else [s for s in _INDEXABLE_SECTIONS if s in profile.extracted_sections]
        )
        if not sections:
            return [cls.from_profile(profile)]
        return [cls.from_profile(profile, section_id=sec) for sec in sections]


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
