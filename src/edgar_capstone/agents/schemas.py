"""Pydantic type schemas for ADK tools, variance calculations, and grounded 10-K reports."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MetricRecord(BaseModel):
    """Structured financial metric record for a single company and fiscal year."""

    profile_id: str = Field(..., description="Unique profile ID ({TICKER}_{FY})")
    ticker: str = Field(..., description="Canonical company ticker symbol")
    cik: str = Field(..., description="10-digit zero-padded SEC CIK")
    company_name: str = Field(..., description="Official SEC registrant name")
    gics_sector: str = Field(..., description="GICS Sector classification")
    gics_sub_industry: str = Field(..., description="GICS Sub-Industry classification")
    fiscal_year: int = Field(..., description="Reporting fiscal year")
    period_of_report: str = Field(..., description="Fiscal period end date (YYYY-MM-DD)")
    filing_date: str = Field(..., description="SEC 10-K filing date (YYYY-MM-DD)")
    accession_number: str = Field(..., description="SEC accession number")
    currency: str = Field(default="USD", description="Reporting currency")
    revenue: float | None = Field(default=None, description="Total Revenue in USD")
    operating_income: float | None = Field(
        default=None, description="Operating Income in USD"
    )
    net_income: float | None = Field(default=None, description="Net Income in USD")
    operating_margin_pct: float | None = Field(
        default=None, description="Operating Margin percentage (Operating Income / Revenue * 100)"
    )
    net_margin_pct: float | None = Field(
        default=None, description="Net Margin percentage (Net Income / Revenue * 100)"
    )
    sec_filing_url: str = Field(..., description="Public SEC EDGAR URL for the 10-K filing")
    unstructured_gcs_uri: str = Field(..., description="GCS URI for unstructured 10-K Markdown")
    structured_gcs_uri: str = Field(..., description="GCS URI for structured financials JSON")
    provenance: dict[str, Any] = Field(
        default_factory=dict, description="XBRL concept provenance metadata"
    )


class MetricVariance(BaseModel):
    """Deterministic period-over-period variance for a single financial metric."""

    metric_name: Literal["revenue", "operating_income", "net_income"]
    start_year: int
    end_year: int
    start_value: float | None = None
    end_value: float | None = None
    absolute_delta: float | None = Field(
        default=None, description="Deterministic USD change (end_value - start_value)"
    )
    percentage_delta: float | None = Field(
        default=None,
        description="Deterministic percentage change ((end_value - start_value) / |start_value| * 100)",
    )
    cagr_pct: float | None = Field(
        default=None,
        description="Compound Annual Growth Rate percentage across multi-year spans",
    )
    direction: Literal["increase", "decrease", "flat", "unavailable"] = "unavailable"


class AnnualStepVariance(BaseModel):
    """Year-over-year step change between consecutive fiscal years."""

    fiscal_year: int
    prev_fiscal_year: int | None = None
    revenue: float | None = None
    revenue_yoy_delta: float | None = None
    revenue_yoy_pct: float | None = None
    operating_income: float | None = None
    operating_income_yoy_delta: float | None = None
    operating_income_yoy_pct: float | None = None
    net_income: float | None = None
    net_income_yoy_delta: float | None = None
    net_income_yoy_pct: float | None = None
    sec_filing_url: str | None = None


class CompanyVarianceReport(BaseModel):
    """Deterministic multi-period financial variance summary for a single company."""

    ticker: str
    company_name: str
    cik: str
    gics_sector: str
    gics_sub_industry: str
    start_year: int
    end_year: int
    currency: str = "USD"
    revenue_variance: MetricVariance
    operating_income_variance: MetricVariance
    net_income_variance: MetricVariance
    operating_margin_bps_delta: float | None = Field(
        default=None,
        description="Change in Operating Margin in basis points (100 bps = 1.0%)",
    )
    net_margin_bps_delta: float | None = Field(
        default=None,
        description="Change in Net Margin in basis points (100 bps = 1.0%)",
    )
    annual_series: list[AnnualStepVariance] = Field(default_factory=list)
    source_urls: dict[int, str] = Field(
        default_factory=dict, description="Mapping of fiscal_year -> sec_filing_url"
    )


class LookerExploreQuery(BaseModel):
    """Input parameters for querying the Looker BI semantic layer explore."""

    explore_name: Literal[
        "yoy_variances",
        "sector_benchmarks",
        "peer_comparison",
        "chunk_health",
    ] = Field(
        ...,
        description="Target Looker semantic explore view to query",
    )
    tickers: list[str] = Field(
        default_factory=list, description="Optional ticker symbols filter"
    )
    gics_sector: str | None = Field(
        default=None, description="Optional GICS Sector filter"
    )
    start_year: int = Field(default=2020, description="Start fiscal year (inclusive)")
    end_year: int = Field(default=2026, description="End fiscal year (inclusive)")
    limit: int = Field(default=25, ge=1, le=200, description="Max rows to return")


class GroundedCitation(BaseModel):
    """Grounded citation linking a narrative claim to an exact 10-K section chunk."""

    chunk_id: str = Field(..., description="Unique chunk ID (e.g., NVDA_2026_item_7_0001)")
    profile_id: str = Field(..., description="Company-year profile ID (e.g., NVDA_2026)")
    ticker: str = Field(..., description="Company ticker symbol")
    company_name: str = Field(..., description="Official company name")
    gics_sector: str = Field(..., description="GICS Sector")
    fiscal_year: int = Field(..., description="Reporting fiscal year")
    section_id: Literal["item_1a", "item_7"] = Field(
        ..., description="10-K section identifier"
    )
    section_title: str = Field(
        ..., description="Human-readable section & subsection breadcrumb"
    )
    excerpt: str = Field(..., description="Grounded 10-K Markdown passage text")
    sec_filing_url: str = Field(
        ..., description="Direct SEC EDGAR archive URL for split-pane source viewing"
    )
    unstructured_gcs_uri: str = Field(
        ..., description="GCS URI of the unstructured 10-K Markdown document"
    )
    score: float = Field(default=0.0, description="Hybrid retrieval relevance score")


class FactCheckVerification(BaseModel):
    """Post-generation verification comparing narrative numbers against calculation outputs."""

    passed: bool = True
    checked_metrics_count: int = 0
    matched_citations_count: int = 0
    has_required_citations: bool = True
    discrepancies: list[str] = Field(default_factory=list)
    model_latency_ms: float = 0.0
