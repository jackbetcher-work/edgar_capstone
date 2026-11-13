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
        default=0, ge=0, description="Tokens from conversational session history and recalled memory"
    )
    variance_payload_tokens: int = Field(
        default=0, ge=0, description="Tokens representing deterministic variance calculation payloads"
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
        ..., description="Gemini model identifier (e.g., gemini-2.5-pro or gemini-3.5-flash)"
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


class ContextLatencySummary(BaseModel):
    """Aggregated context-induced latency metrics across model tiers and sessions."""

    total_calls: int = 0
    total_context_tokens: int = 0
    total_cached_context_tokens: int = 0
    total_uncached_context_tokens: int = 0
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
    by_model: dict[str, dict[str, Any]] = Field(default_factory=dict)


class VarianceEvaluationItem(BaseModel):
    """Structured ReAct evaluation of deterministic variances for a single issuer."""

    ticker: str
    company_name: str
    start_year: int
    end_year: int
    primary_metric_mover: Literal["revenue", "operating_income", "net_income"] = "revenue"
    revenue_pct_delta: float | None = None
    operating_income_pct_delta: float | None = None
    net_income_pct_delta: float | None = None
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
    section_id: Literal["item_1a", "item_7"]
    section_title: str
    variance_driver_focus: str
    extracted_explanations: list[str] = Field(default_factory=list)
    matched_chunk_ids: list[str] = Field(default_factory=list)
    token_estimate: int = 0
    lookup_latency_ms: float = 0.0
    sec_filing_url: str = ""


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


class NarrativeEvaluationReport(BaseModel):
    """LLM-as-a-Judge / rubric evaluation output powered by Gemini 3.5 Flash."""

    evaluator_model: str = "gemini-3.5-flash"
    task_category: Literal["evaluation", "llm_as_judge"] = "evaluation"
    grade: Literal["pass", "needs_revision", "fail"] = "pass"
    overall_score: float = Field(default=1.0, ge=0.0, le=1.0)
    numeric_fidelity_score: float = Field(default=1.0, ge=0.0, le=1.0)
    citation_grounding_score: float = Field(default=1.0, ge=0.0, le=1.0)
    explanation_quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    thematic_consistency_score: float = Field(default=1.0, ge=0.0, le=1.0)
    rationale: str = ""
    context_tokens: int = 0
    context_latency_ms: float = 0.0
    total_latency_ms: float = 0.0


class CodingAssistanceReport(BaseModel):
    """Analytical code / query generation output powered by Gemini 3.5 Flash."""

    assistant_model: str = "gemini-3.5-flash"
    task_category: Literal["coding_assistance"] = "coding_assistance"
    language: Literal["sql", "python", "lookml"] = "sql"
    artifact_title: str
    code: str
    explanation: str
    tickers_referenced: list[str] = Field(default_factory=list)
    context_tokens: int = 0
    context_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
