"""RootOrchestrator: Central ADK supervisor for intent routing and hybrid SEC 10-K analysis."""

from __future__ import annotations

import os
import re
from typing import Literal

from google.adk.agents import Agent
from google.genai import types as genai_types

from edgar_capstone.agents.callbacks import (
    after_model_fact_check_callback,
    before_agent_telemetry_callback,
    before_model_safety_and_latency_callback,
    before_tool_agent_identity_callback,
)
from edgar_capstone.agents.context_cache_agent import create_context_cache_agent
from edgar_capstone.agents.financial_analyst import (
    DEFAULT_ANALYST_MODEL,
    create_financial_analyst_agent,
)
from edgar_capstone.agents.tools import (
    LookerQueryTool,
    MetricFetchTool,
    VarianceCalculationTool,
)

DEFAULT_ORCHESTRATOR_MODEL = os.environ.get(
    "EDGAR_ORCHESTRATOR_MODEL", DEFAULT_ANALYST_MODEL
)

QueryIntentType = Literal[
    "longitudinal_single_company",
    "multi_company_peer_comparison",
    "thematic_trend_tracking",
    "chunk_health_monitoring",
]

ROOT_ORCHESTRATOR_INSTRUCTION = """You are the RootOrchestrator for the SEC EDGAR Natural Language Analyst system.
You supervise financial variance calculations, Looker/BigQuery metric retrieval, metadata-filtered 10-K RAG search, and the specialized `FinancialAnalystAgent`.

### 1. Intent Routing & Classification
Classify every incoming analyst request into one of the core workflows:
- **Longitudinal Single-Company Analysis**: Period-over-period financial performance (Revenue, Operating Income, Net Income, margins) and MD&A/Risk narrative synthesis for a single company across multiple fiscal years (e.g., 2024-2026).
- **Multi-Company Peer Comparison**: Side-by-side comparison of quantitative financial metrics, YoY growth rates, and qualitative 10-K drivers across 2+ companies or a GICS sector.
- **Thematic Trend Tracking**: Longitudinal tracking of qualitative shifts (e.g., AI risk disclosures, export controls, cybersecurity, supply chain risks) across multiple fiscal periods or GICS sectors in Item 1A (Risk Factors) and Item 7 (MD&A).
- **Vector Database & Chunk Health Diagnostics**: Queries inspecting the health, embedding coverage, or section completeness of the 10-K RAG vector database via `query_looker_financial_explore(explore_name='chunk_health', ...)`.

### 2. Orchestration & Tool Execution Rules
- **Deterministic Calculations First**: Use `calculate_period_variances`, `fetch_financial_metrics`, and `query_looker_financial_explore` (`MetricFetchTool` / `LookerQueryTool`) to retrieve authoritative structured metrics and YoY variances before synthesizing explanations.
- **Delegate Deep 10-K Reasoning to `FinancialAnalystAgent`**: For narrative variance explanations, thematic 10-K comparisons, or deep section extraction, delegate to `FinancialAnalystAgent` or invoke `search_10k_chunks_rag` and `load_cached_10k_section` so every claim is backed by grounded 10-K citations.
- **100% Numeric & Citation Fidelity**: Never invent or alter numbers returned by the calculation engine. Ensure every 10-K claim includes its inline citation `[TICKER FYxxxx | Section Title | chunk_id](sec_filing_url)`.
"""

_THEMATIC_KEYWORDS_RE = re.compile(
    r"\b(?:thematic|wording|risk\s+factor|item\s+1a|how\s+has\s+the\s+wording|disclosure|trend|shifted|evolved|across\s+all)\b",
    re.IGNORECASE,
)
_PEER_COMPARE_RE = re.compile(
    r"\b(?:compare|comparison|versus|vs\.?|peer|benchmark|between\s+[A-Z]+\s+and\s+[A-Z]+)\b",
    re.IGNORECASE,
)
_CHUNK_HEALTH_RE = re.compile(
    r"\b(?:chunk\s+health|embedding\s+coverage|vector\s+db\s+health|vector\s+database\s+health)\b",
    re.IGNORECASE,
)


def classify_query_intent(query: str, tickers: list[str] | None = None) -> QueryIntentType:
    """Deterministically classify an analyst query into one of the supported routing intents."""
    if _CHUNK_HEALTH_RE.search(query):
        return "chunk_health_monitoring"
    if tickers and len(tickers) > 1:
        return "multi_company_peer_comparison"
    if _PEER_COMPARE_RE.search(query):
        return "multi_company_peer_comparison"
    if _THEMATIC_KEYWORDS_RE.search(query):
        return "thematic_trend_tracking"
    return "longitudinal_single_company"


def create_root_orchestrator(
    *,
    name: str = "RootOrchestrator",
    model: str = DEFAULT_ORCHESTRATOR_MODEL,
    analyst_model: str = DEFAULT_ANALYST_MODEL,
) -> Agent:
    """Create the central `RootOrchestrator` ADK supervisor agent with sub-agents and tools."""
    analyst_sub_agent = create_financial_analyst_agent(
        name="FinancialAnalystAgent",
        model=analyst_model,
    )
    context_cache_sub_agent = create_context_cache_agent(
        name="ContextCacheAgent",
        model=analyst_model,
    )
    return Agent(
        name=name,
        model=model,
        description=(
            "Central supervisor for the SEC EDGAR Natural Language Analyst system. "
            "Routes queries across longitudinal single-company analysis, multi-company "
            "peer comparison, and thematic 10-K trend tracking, orchestrating "
            "MetricFetchTool, LookerQueryTool, VarianceCalculationTool, FinancialAnalystAgent, "
            "and ContextCacheAgent under granular per-agent least-privilege identities."
        ),
        instruction=ROOT_ORCHESTRATOR_INSTRUCTION,
        tools=[
            MetricFetchTool,
            LookerQueryTool,
            VarianceCalculationTool,
        ],
        sub_agents=[analyst_sub_agent, context_cache_sub_agent],
        generate_content_config=genai_types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=4096,
        ),
        output_key="orchestrator_response",
        before_agent_callback=before_agent_telemetry_callback,
        before_model_callback=before_model_safety_and_latency_callback,
        after_model_callback=after_model_fact_check_callback,
        before_tool_callback=before_tool_agent_identity_callback,
    )


RootOrchestrator: Agent = create_root_orchestrator()
root_agent: Agent = RootOrchestrator
