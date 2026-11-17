"""Pydantic type schemas for ADK tools, variance calculations, and grounded 10-K reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, Field

from edgar_capstone.common.domain import SECTION_TITLES, MetricName, SectionId
from edgar_capstone.common.models import FilingMetadataBase


class AgentReportTelemetryMixin(BaseModel):
    """Shared context/latency telemetry fields attached to agent evaluation & report outputs."""

    context_tokens: int = 0
    context_latency_ms: float = 0.0
    total_latency_ms: float = 0.0


class MetricRecord(FilingMetadataBase):
    """Structured financial metric record for a single company and fiscal year."""

    cik: str = Field(..., description="10-digit zero-padded SEC CIK")
    gics_sub_industry: str = Field(..., description="GICS Sub-Industry classification")
    period_of_report: str = Field(..., description="Fiscal period end date (YYYY-MM-DD)")
    filing_date: str = Field(..., description="SEC 10-K filing date (YYYY-MM-DD)")
    accession_number: str = Field(..., description="SEC accession number")
    currency: str = Field(default="USD", description="Reporting currency")
    revenue: float | None = Field(default=None, description="Total Revenue in USD")
    operating_income: float | None = Field(default=None, description="Operating Income in USD")
    net_income: float | None = Field(default=None, description="Net Income in USD")
    research_and_development: float | None = Field(
        default=None,
        description="Research & Development expense in USD (None when the issuer reports no R&D line)",
    )
    operating_margin_pct: float | None = Field(
        default=None, description="Operating Margin percentage (Operating Income / Revenue * 100)"
    )
    net_margin_pct: float | None = Field(
        default=None, description="Net Margin percentage (Net Income / Revenue * 100)"
    )
    rd_intensity_pct: float | None = Field(
        default=None, description="R&D intensity percentage (R&D expense / Revenue * 100)"
    )
    structured_gcs_uri: str = Field(..., description="GCS URI for structured financials JSON")
    provenance: dict[str, Any] = Field(
        default_factory=dict, description="XBRL concept provenance metadata"
    )


class NumericCitation(BaseModel):
    """Traceable SEC XBRL filing citation for a reported numeric value."""

    ticker: str
    fiscal_year: int
    period_of_report: str | None = Field(
        default=None, description="Fiscal period end date (YYYY-MM-DD)"
    )
    filing_date: str | None = Field(default=None, description="SEC 10-K filing date (YYYY-MM-DD)")
    accession_number: str | None = Field(default=None, description="SEC EDGAR accession number")
    sec_filing_url: str | None = Field(default=None, description="Direct SEC EDGAR filing URL")
    metric: str = Field(..., description="Metric name (e.g., revenue, research_and_development)")
    xbrl_concept: str | None = Field(
        default=None, description="XBRL concept from metric_provenance (e.g., us-gaap:Revenues)"
    )
    value: float | None = Field(default=None, description="Reported numeric figure in USD")


class CoverageNotice(BaseModel):
    """Deterministic system notice documenting corpus coverage, missing years, or calendar drift."""

    code: Literal[
        "missing_fiscal_year",
        "missing_ticker",
        "fiscal_calendar_misalignment",
        "missing_metric",
        "unsupported_metric",
        "unsupported_filing_type",
        "out_of_range_year",
        "non_sp500_issuer",
        "sector_coverage_summary",
    ]
    severity: Literal["info", "warning"] = "info"
    message: str
    tickers: list[str] = Field(default_factory=list)
    missing_years: list[int] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class MetricVariance(BaseModel):
    """Deterministic period-over-period variance for a single financial metric."""

    metric_name: MetricName
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
    start_period_of_report: str | None = Field(
        default=None, description="Fiscal period end date (YYYY-MM-DD) for start_year"
    )
    end_period_of_report: str | None = Field(
        default=None, description="Fiscal period end date (YYYY-MM-DD) for end_year"
    )
    start_accession_number: str | None = Field(
        default=None, description="SEC accession number for start_year 10-K"
    )
    end_accession_number: str | None = Field(
        default=None, description="SEC accession number for end_year 10-K"
    )
    start_sec_filing_url: str | None = Field(
        default=None, description="SEC EDGAR URL for start_year 10-K"
    )
    end_sec_filing_url: str | None = Field(
        default=None, description="SEC EDGAR URL for end_year 10-K"
    )
    start_xbrl_concept: str | None = Field(
        default=None, description="XBRL concept provenance for start_value"
    )
    end_xbrl_concept: str | None = Field(
        default=None, description="XBRL concept provenance for end_value"
    )
    derived_from: list[NumericCitation] = Field(
        default_factory=list,
        description="Reported start and end XBRL inputs from which this variance was derived",
    )


class AnnualStepVariance(BaseModel):
    """Year-over-year step change between consecutive fiscal years."""

    fiscal_year: int
    prev_fiscal_year: int | None = None
    period_of_report: str | None = None
    filing_date: str | None = None
    accession_number: str | None = None
    prior_period_of_report: str | None = None
    prior_accession_number: str | None = None
    prior_sec_filing_url: str | None = None
    revenue: float | None = None
    revenue_yoy_delta: float | None = None
    revenue_yoy_pct: float | None = None
    operating_income: float | None = None
    operating_income_yoy_delta: float | None = None
    operating_income_yoy_pct: float | None = None
    net_income: float | None = None
    net_income_yoy_delta: float | None = None
    net_income_yoy_pct: float | None = None
    research_and_development: float | None = None
    research_and_development_yoy_delta: float | None = None
    research_and_development_yoy_pct: float | None = None
    metric_provenance: dict[str, str | None] = Field(
        default_factory=dict,
        description="XBRL concept per reported metric for this fiscal year",
    )
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
    requested_start_year: int | None = None
    requested_end_year: int | None = None
    available_fiscal_years: list[int] = Field(default_factory=list)
    missing_fiscal_years: list[int] = Field(default_factory=list)
    currency: str = "USD"
    revenue_variance: MetricVariance
    operating_income_variance: MetricVariance
    net_income_variance: MetricVariance
    research_and_development_variance: MetricVariance | None = Field(
        default=None,
        description=(
            "R&D expense variance; direction 'unavailable' when the issuer reports no R&D line"
        ),
    )
    operating_margin_bps_delta: float | None = Field(
        default=None,
        description="Change in Operating Margin in basis points (100 bps = 1.0%)",
    )
    net_margin_bps_delta: float | None = Field(
        default=None,
        description="Change in Net Margin in basis points (100 bps = 1.0%)",
    )
    rd_intensity_bps_delta: float | None = Field(
        default=None,
        description="Change in R&D intensity (R&D / Revenue) in basis points",
    )
    annual_series: list[AnnualStepVariance] = Field(default_factory=list)
    source_urls: dict[int, str] = Field(
        default_factory=dict, description="Mapping of fiscal_year -> sec_filing_url"
    )
    period_end_dates: dict[int, str] = Field(
        default_factory=dict, description="Mapping of fiscal_year -> period_of_report (YYYY-MM-DD)"
    )
    filing_dates: dict[int, str] = Field(
        default_factory=dict, description="Mapping of fiscal_year -> filing_date (YYYY-MM-DD)"
    )
    accession_numbers: dict[int, str] = Field(
        default_factory=dict, description="Mapping of fiscal_year -> SEC accession_number"
    )
    metric_provenance: dict[int, dict[str, str | None]] = Field(
        default_factory=dict,
        description="Mapping of fiscal_year -> {metric -> XBRL concept}",
    )
    derived_margin_inputs: dict[str, list[NumericCitation]] = Field(
        default_factory=dict,
        description="Reported numerator and revenue denominator XBRL inputs for margin/intensity deltas",
    )


# TODO(Wave 3b): Delete dead `LookerExploreQuery` atomically with `agents/__init__.py` lines 131 & 214.
class LookerExploreQuery(BaseModel):
    """DEPRECATED: Retained temporarily until `agents/__init__.py` removes its re-export in Wave 3b."""

    explore_name: Literal[
        "yoy_variances",
        "sector_benchmarks",
        "peer_comparison",
        "chunk_health",
        "llm_telemetry",
        "session_analytics",
        "audit_trail",
    ] = Field(..., description="Target Looker semantic explore view to query")
    tickers: list[str] = Field(default_factory=list, description="Optional ticker symbols filter")
    gics_sector: str | None = Field(default=None, description="Optional GICS Sector filter")
    start_year: int = Field(default=2020, description="Start fiscal year (inclusive)")
    end_year: int = Field(default=2026, description="End fiscal year (inclusive)")
    limit: int = Field(default=25, ge=1, le=200, description="Max rows to return")


class GroundedCitation(FilingMetadataBase):
    """Grounded citation linking a narrative claim to an exact 10-K section chunk."""

    chunk_id: str = Field(..., description="Unique chunk ID (e.g., NVDA_2026_item_7_0001)")
    period_of_report: str | None = Field(
        default=None, description="Fiscal period end date (YYYY-MM-DD)"
    )
    filing_date: str | None = Field(default=None, description="SEC 10-K filing date (YYYY-MM-DD)")
    accession_number: str | None = Field(default=None, description="SEC EDGAR accession number")
    section_id: SectionId = Field(..., description="10-K section identifier")
    section_title: str = Field(..., description="Human-readable section & subsection breadcrumb")
    excerpt: str = Field(..., description="Grounded 10-K Markdown passage text")
    score: float = Field(default=0.0, description="Hybrid retrieval relevance score")

    @classmethod
    def from_row(
        cls,
        row: Mapping[str, Any],
        *,
        score: float | None = None,
        excerpt: str | None = None,
        fallback_ticker: str | None = None,
        fallback_fiscal_year: int | None = None,
        fallback_section_id: SectionId = "item_1a",
        clamp_score: bool = False,
    ) -> GroundedCitation:
        """Build a `GroundedCitation` from a BigQuery, Discovery Engine, or fused chunk row."""
        ticker = str(row.get("ticker") or fallback_ticker or "")
        fy_raw = row.get("fiscal_year")
        if fy_raw is None:
            fy_raw = fallback_fiscal_year if fallback_fiscal_year is not None else 0
        fy = int(fy_raw)
        raw_sec = str(row.get("section_id") or fallback_section_id).strip().lower()
        sec_id: SectionId = (
            "item_7"
            if raw_sec == "item_7"
            else ("item_1a" if raw_sec == "item_1a" else fallback_section_id)
        )
        text = (
            excerpt
            if excerpt is not None
            else str(row.get("chunk_text") or row.get("excerpt") or "")
        )
        raw_score = (
            float(score)
            if score is not None
            else float(
                row.get("score") if row.get("score") is not None else (row.get("similarity") or 0.0)
            )
        )
        if clamp_score:
            raw_score = max(0.0, min(1.0, raw_score))
        return cls(
            chunk_id=str(row["chunk_id"]),
            profile_id=str(row.get("profile_id") or f"{ticker}_{fy}"),
            ticker=ticker,
            company_name=str(row.get("company_name") or ticker),
            gics_sector=str(row.get("gics_sector") or "Unknown"),
            fiscal_year=fy,
            period_of_report=str(row.get("period_of_report") or "") or None,
            filing_date=str(row.get("filing_date") or "") or None,
            accession_number=str(row.get("accession_number") or "") or None,
            section_id=sec_id,
            section_title=str(row.get("section_title") or SECTION_TITLES[sec_id]),
            excerpt=text,
            sec_filing_url=str(row.get("sec_filing_url") or ""),
            unstructured_gcs_uri=str(row.get("unstructured_gcs_uri") or ""),
            score=round(raw_score, 4),
        )


class FactCheckVerification(BaseModel):
    """Post-generation verification comparing narrative numbers against calculation outputs."""

    passed: bool = True
    checked_metrics_count: int = 0
    matched_citations_count: int = 0
    has_required_citations: bool = True
    discrepancies: list[str] = Field(default_factory=list)
    model_latency_ms: float = 0.0


ModelTaskCategory = Literal[
    "financial_reasoning",
    "variance_explanation",
    "peer_comparison",
    "thematic_synthesis",
    "orchestration",
    "evaluation",
    "llm_as_judge",
    "coding_assistance",
    "context_compaction",
    "context_cache_warmup",
]

ReActPhaseType = Literal[
    "safety_guardrail",
    "scope_and_intent_routing",
    "evaluate_variances",
    "query_cached_10k_context",
    "hybrid_rag_grounding",
    "format_grounded_output",
    "fact_check_and_eval",
]


class ContextBreakdown(BaseModel):
    """Granular token breakdown of prompt, memory, variance, and cached 10-K context sent to an LLM."""

    system_instruction_tokens: int = Field(
        default=0, ge=0, description="Tokens in system instructions and ReAct protocol headers"
    )
    memory_and_history_tokens: int = Field(
        default=0,
        ge=0,
        description="Tokens from conversational session history and recalled memory",
    )
    variance_payload_tokens: int = Field(
        default=0,
        ge=0,
        description="Tokens representing deterministic variance calculation payloads",
    )
    cached_10k_tokens: int = Field(
        default=0, ge=0, description="Tokens served from cached 10-K filing sections"
    )
    uncached_10k_and_tool_tokens: int = Field(
        default=0, ge=0, description="Uncached RAG chunks and tool response tokens"
    )
    user_prompt_tokens: int = Field(
        default=0, ge=0, description="Tokens in the active analyst prompt"
    )
    total_context_tokens: int = Field(
        default=0, ge=0, description="Total input context tokens supplied to the model call"
    )
    cached_context_tokens: int = Field(
        default=0, ge=0, description="Input tokens served via Vertex AI / session context cache"
    )
    uncached_context_tokens: int = Field(
        default=0, ge=0, description="Input tokens requiring full uncached prefill processing"
    )
    cache_hit: bool = Field(
        default=False, description="Whether context cache hit accelerated this model call"
    )


class ModelContextLatencyRecord(BaseModel):
    """Per-call telemetry isolating latency attributable to context vs token generation."""

    call_id: str = Field(..., description="Unique telemetry identifier for the model call")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp")
    agent_name: str = Field(..., description="ADK agent executing the model call")
    model_name: str = Field(
        ..., description="Gemini model identifier (e.g., gemini-3.8-flash or gemini-3.5-flash)"
    )
    task_category: ModelTaskCategory = Field(
        default="financial_reasoning",
        description="Workload classification routed across Pro vs Flash tiers",
    )
    react_phase: str = Field(
        default="general", description="Active ReAct loop phase during this model invocation"
    )
    context_breakdown: ContextBreakdown = Field(default_factory=ContextBreakdown)
    output_tokens: int = Field(default=0, ge=0, description="Generated response tokens")
    thoughts_tokens: int = Field(
        default=0, ge=0, description="Internal reasoning / thinking tokens"
    )
    total_model_latency_ms: float = Field(
        default=0.0, ge=0.0, description="End-to-end model call latency including context prep"
    )
    context_prep_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock latency spent loading memory/cache and assembling prompt context",
    )
    context_prefill_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Model prefill latency attributable to processing input context tokens",
    )
    context_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total latency due to context (context_prep_latency_ms + context_prefill_latency_ms)",
    )
    generation_latency_ms: float = Field(
        default=0.0, ge=0.0, description="Output token decode / generation latency"
    )
    ttft_ms: float = Field(
        default=0.0, ge=0.0, description="Time to First Thought / Token in milliseconds"
    )
    context_latency_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of total model call latency attributable to context",
    )
    latency_per_1k_context_tokens_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Normalized context latency per 1,000 input context tokens",
    )
    estimated_cache_latency_saved_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated prefill latency saved via cached 10-K context tokens",
    )
    within_ttft_slo: bool = Field(
        default=True, description="Whether TTFT satisfied the <= 3000ms p95 SLO"
    )
    input_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="USD cost of input context tokens after applying cached-token discount",
    )
    output_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="USD cost of generated output and reasoning tokens",
    )
    total_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Total USD cost for this LLM invocation (input_cost_usd + output_cost_usd)",
    )
    uncached_baseline_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Hypothetical USD cost had all cached tokens been billed at uncached prefill rate",
    )
    cached_token_savings_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="USD cost saved on this call via context-cached input tokens",
    )
    session_id: str | None = Field(
        default=None, description="Optional conversational session ID associated with the call"
    )
    user_id: str | None = Field(
        default=None, description="Optional authenticated principal associated with the call"
    )


class ContextLatencySummary(BaseModel):
    """Aggregated context-induced latency and USD cost metrics across model tiers and sessions."""

    total_calls: int = 0
    total_context_tokens: int = 0
    total_cached_context_tokens: int = 0
    total_uncached_context_tokens: int = 0
    total_output_tokens: int = 0
    total_thoughts_tokens: int = 0
    mean_context_latency_ms: float = 0.0
    p50_context_latency_ms: float = 0.0
    p95_context_latency_ms: float = 0.0
    mean_total_model_latency_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    ttft_slo_target_ms: float = 3000.0
    ttft_slo_compliance_ratio: float = 1.0
    mean_context_latency_ratio: float = 0.0
    mean_latency_per_1k_context_tokens_ms: float = 0.0
    total_estimated_cache_latency_saved_ms: float = 0.0
    total_cost_usd: float = 0.0
    total_uncached_baseline_cost_usd: float = 0.0
    total_cached_token_savings_usd: float = 0.0
    cache_cost_savings_pct: float = 0.0
    by_model: dict[str, dict[str, Any]] = Field(default_factory=dict)


class VarianceEvaluationItem(BaseModel):
    """Structured ReAct evaluation of deterministic variances for a single issuer."""

    ticker: str
    company_name: str
    start_year: int
    end_year: int
    primary_metric_mover: MetricName = "revenue"
    revenue_pct_delta: float | None = None
    operating_income_pct_delta: float | None = None
    net_income_pct_delta: float | None = None
    research_and_development_pct_delta: float | None = None
    operating_margin_bps_delta: float | None = None
    net_margin_bps_delta: float | None = None
    material_findings: list[str] = Field(default_factory=list)
    divergences: list[str] = Field(default_factory=list)
    targeted_10k_queries: list[dict[str, Any]] = Field(default_factory=list)


class Cached10KExplanation(BaseModel):
    """Grounded explanation extracted from cached 10-K section context during ReAct execution."""

    cache_key: str
    cache_hit: bool
    ticker: str
    company_name: str
    fiscal_year: int
    section_id: SectionId
    section_title: str
    variance_driver_focus: str
    extracted_explanations: list[str] = Field(default_factory=list)
    matched_chunk_ids: list[str] = Field(default_factory=list)
    token_estimate: int = 0
    lookup_latency_ms: float = 0.0
    sec_filing_url: str = ""


# ---------------------------------------------------------------------------
# Deterministic anomaly flagging (`flag_material_variances`)
# ---------------------------------------------------------------------------

MaterialFlagRule = Literal[
    "yoy_change", "sign_reversal", "margin_shift", "period_change", "sector_divergence"
]
MaterialFlagMetric = Literal[
    "revenue",
    "operating_income",
    "net_income",
    "research_and_development",
    "operating_margin",
    "net_margin",
    "rd_intensity",
]


class MaterialFlagSource(BaseModel):
    """Filing provenance for a flagged figure (current and comparison filings)."""

    accession_number: str = ""
    sec_filing_url: str = ""
    period_of_report: str = ""
    xbrl_concept: str | None = Field(
        default=None, description="XBRL concept the current-year value was read from"
    )
    prior_accession_number: str | None = None
    prior_sec_filing_url: str | None = None
    prior_period_of_report: str | None = None
    prior_xbrl_concept: str | None = Field(
        default=None, description="XBRL concept the prior-year value was read from"
    )
    denominator_xbrl_concept: str | None = Field(
        default=None,
        description="Denominator XBRL concept (e.g. Revenue) for margin/intensity flags",
    )
    prior_denominator_xbrl_concept: str | None = Field(
        default=None, description="Prior-year denominator XBRL concept for margin/intensity flags"
    )


class MaterialFlagExplanation(BaseModel):
    """MD&A passage retrieved to explain a flagged variance."""

    chunk_id: str
    section_id: SectionId = "item_7"
    section_title: str = ""
    excerpt: str
    sec_filing_url: str = ""


class MaterialVarianceFlag(BaseModel):
    """A deterministically detected material variance (anomaly) that needs a 10-K explanation."""

    flag_id: str
    ticker: str
    company_name: str
    gics_sector: str = "Unknown"
    fiscal_year: int = Field(..., description="Fiscal year in which the variance occurred")
    prior_fiscal_year: int | None = Field(
        default=None, description="Comparison fiscal year (YoY prior year or period start)"
    )
    metric: MaterialFlagMetric
    rule: MaterialFlagRule
    severity: Literal["high", "medium"] = "medium"
    value: float = Field(..., description="Measured change: % (pct), bps, pp, or USD")
    unit: Literal["pct", "bps", "pp", "usd"]
    threshold: float = Field(..., description="Materiality threshold the value crossed")
    prior_value: float | None = None
    current_value: float | None = None
    benchmark_value: float | None = Field(
        default=None, description="Sector median for sector_divergence flags"
    )
    peer_count: int | None = None
    description: str = Field(..., description="Deterministic one-sentence description")
    explanation_query: str = Field(
        default="", description="Suggested MD&A query for explaining the variance"
    )
    source: MaterialFlagSource = Field(default_factory=MaterialFlagSource)
    explanation: MaterialFlagExplanation | None = None


# ---------------------------------------------------------------------------
# Disclosure Delta pipeline (`compare_disclosures_across_years`)
# ---------------------------------------------------------------------------


class DisclosureDelta(BaseModel):
    """One theme-relevant passage compared across consecutive fiscal years."""

    delta_id: str
    ticker: str
    company_name: str
    section_id: SectionId
    section_title: str = ""
    change_type: Literal["added", "removed", "modified", "unchanged"]
    prior_fiscal_year: int
    current_fiscal_year: int
    prior_chunk_id: str | None = None
    current_chunk_id: str | None = None
    prior_excerpt: str | None = None
    current_excerpt: str | None = None
    similarity: float = Field(
        default=0.0,
        description="Share of the passage's word trigrams found in the other year's section",
    )
    added_phrases: list[str] = Field(default_factory=list)
    removed_phrases: list[str] = Field(default_factory=list)
    theme_relevance: float = 0.0


class DisclosureYearPair(BaseModel):
    """Deltas for one company between two consecutive available fiscal years."""

    prior_fiscal_year: int
    current_fiscal_year: int
    counts: dict[str, int] = Field(default_factory=dict)
    deltas: list[DisclosureDelta] = Field(default_factory=list)


class CompanyDisclosureEvolution(BaseModel):
    """Year-over-year evolution of one company's disclosures for a theme."""

    ticker: str
    company_name: str
    gics_sector: str = "Unknown"
    fiscal_years_available: list[int] = Field(default_factory=list)
    missing_fiscal_years: list[int] = Field(default_factory=list)
    theme_mentions: dict[str, int] = Field(
        default_factory=dict,
        description="Theme-term mentions per fiscal year across the compared section(s)",
    )
    year_pairs: list[DisclosureYearPair] = Field(default_factory=list)


class ReActStep(BaseModel):
    """Single structured Thought -> Action -> Observation step in the ReAct execution loop."""

    iteration: int = Field(..., ge=1)
    phase: ReActPhaseType
    thought: str = Field(..., description="Agent reasoning before selecting the next action")
    action: str = Field(..., description="Tool or sub-agent action invoked")
    action_input: dict[str, Any] = Field(default_factory=dict)
    observation: str = Field(..., description="Structured observation from executing the action")
    agent: str
    model_used: str | None = None
    context_tokens: int = 0
    cached_context_tokens: int = 0
    context_latency_ms: float = 0.0
    duration_ms: float = 0.0
    status: Literal["completed", "warning", "blocked"] = "completed"


class NarrativeEvaluationReport(AgentReportTelemetryMixin):
    """LLM-as-a-Judge / rubric evaluation output powered by Gemini 3.5 Flash."""

    evaluator_model: str = "gemini-3.5-flash"
    task_category: Literal["evaluation", "llm_as_judge"] = "evaluation"
    grade: Literal["pass", "needs_revision", "fail", "pending"] = "pass"
    overall_score: float | None = Field(default=None, ge=0.0, le=1.0)
    numeric_fidelity_score: float | None = Field(default=None, ge=0.0, le=1.0)
    citation_grounding_score: float | None = Field(default=None, ge=0.0, le=1.0)
    citation_precision_score: float | None = Field(default=None, ge=0.0, le=1.0)
    explanation_quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    thematic_consistency_score: float | None = Field(default=None, ge=0.0, le=1.0)
    unsupported_claims_count: int = Field(default=0, ge=0)
    rationale: str = ""


# TODO(Wave 3b): Delete `CodingAssistanceReport` atomically with `evaluation_and_coding.py` & `agents/__init__.py`.
class CodingAssistanceReport(AgentReportTelemetryMixin):
    """Analytical code / query generation output powered by Gemini 3.5 Flash."""

    assistant_model: str = "gemini-3.5-flash"
    task_category: Literal["coding_assistance"] = "coding_assistance"
    language: Literal["sql", "python", "lookml"] = "sql"
    artifact_title: str
    code: str
    explanation: str
    tickers_referenced: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# FinancialAnalystAgent structured output (constrained decoding via ADK output_schema)
# ---------------------------------------------------------------------------
# Length limits are stated in field descriptions rather than enforced with max_length:
# a model that overshoots would fail ADK's schema validation and lose the whole answer.
# `structured_report.py` trims and verifies after parsing instead.


class AnalystMetricReference(BaseModel):
    """A headline metric to display. Values and XBRL citations are filled in from the deterministic engine."""

    ticker: str = Field(
        ..., description="Ticker of a company in the variance payload, e.g. 'NVDA'."
    )
    metric: MetricName = Field(
        ...,
        description=(
            "Metric to show; its figures and XBRL citations come from the deterministic engine. "
            "Use 'research_and_development' for R&D spend questions."
        ),
    )


class AnalystFinding(BaseModel):
    """One 10-K-grounded claim that can be checked against its cited passage."""

    statement: str = Field(
        ...,
        description=(
            "One sentence (at most 40 words) answering part of the question. Any figure it "
            "contains must appear verbatim in evidence_quote; do not restate variance "
            "percentages, which are shown from key_metrics."
        ),
    )
    ticker: str = Field(..., description="Company ticker the finding is about.")
    fiscal_year: int = Field(..., description="Fiscal year of the cited filing.")
    chunk_ids: list[str] = Field(
        ...,
        description=(
            "1-3 chunk_ids returned by your retrieval tools or listed as evidence leads. "
            "Never invent IDs."
        ),
    )
    evidence_quote: str = Field(
        ...,
        description=(
            "A short verbatim quote (10-40 words) copied exactly from one of the cited chunks "
            "that supports the statement."
        ),
    )


class AnalystReport(BaseModel):
    """Structured, verifiable answer produced by FinancialAnalystAgent.

    Every qualitative claim lives in `findings` with verbatim quote & chunk_id grounding;
    uncited free-text notes are disallowed. System-generated coverage caveats and missing-year
    notices are rendered deterministically outside the LLM schema.
    """

    headline: str = Field(
        ...,
        description="One-sentence direct answer to the analyst's question (at most 35 words).",
    )
    key_metrics: list[AnalystMetricReference] = Field(
        default_factory=list,
        description="Up to 6 metrics to display, most relevant first.",
    )
    findings: list[AnalystFinding] = Field(
        default_factory=list,
        description="Up to 6 grounded findings, most important first.",
    )
