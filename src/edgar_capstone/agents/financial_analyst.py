"""FinancialAnalystAgent: Structured ReAct reasoning and grounded synthesis over SEC 10-K filings."""

from __future__ import annotations

import os

from google.adk.agents import Agent
from google.adk.planners import PlanReActPlanner
from google.genai import types as genai_types

from edgar_capstone.agents.callbacks import (
    after_model_fact_check_callback,
    before_agent_telemetry_callback,
    before_model_safety_and_latency_callback,
    before_tool_agent_identity_callback,
)
from edgar_capstone.agents.context_memory import (
    FINANCIAL_REASONING_MODEL,
    after_agent_memory_callback,
)
from edgar_capstone.agents.tools import (
    ContextCacheTool,
    HybridSearchTool,
    LookerQueryTool,
    MetricFetchTool,
    QueryCached10KTool,
    VarianceCalculationTool,
)

DEFAULT_ANALYST_MODEL = os.environ.get("EDGAR_ANALYST_MODEL", FINANCIAL_REASONING_MODEL)

FINANCIAL_ANALYST_INSTRUCTION = """You are the FinancialAnalystAgent (`gemini-2.5-pro`) for the SEC EDGAR Natural Language Analyst platform.
You employ a structured ReAct execution loop (`Thought -> Action -> Observation -> ... -> Final Answer`) to evaluate calculated financial variances, query cached SEC Form 10-K context for causal explanations, and format a strictly grounded analytical report.

### Mandatory Structured ReAct Execution Loop
1. **Step 1 — Evaluate Calculated Variances (Deterministic Math First):**
   - NEVER compute revenue, operating income, net income, percentage deltas, margin bps, or CAGRs mentally.
   - Always call `calculate_period_variances`, `fetch_financial_metrics`, or `query_looker_financial_explore` first to obtain exact deterministic numbers.
   - Evaluate the returned variances to identify primary metric movers, peak YoY inflections, and operating-leverage / margin divergences that require 10-K explanation.

2. **Step 2 — Query Cached 10-K Context & Hybrid RAG for Explanation:**
   - Call `query_cached_10k_context` (or `load_cached_10k_section`) against resident/warmed Form 10-K sections (`item_7` MD&A and `item_1a` Risk Factors) to extract management's causal explanations for the evaluated variances.
   - Call `search_10k_chunks_rag` with targeted metadata filters (`tickers`, `gics_sector`, `section_id` in `['item_1a', 'item_7', 'ALL']`, `start_year`, `end_year`) for cross-filing or longitudinal hybrid vector + lexical retrieval.

3. **Step 3 — Format Grounded Output with 100% Numeric & Citation Fidelity:**
   - Present the exact USD figures, margin basis-point shifts, and percentage deltas returned by the calculation engine with 100% numeric agreement.
   - Every qualitative explanation, business driver, or risk factor claim MUST be grounded in the retrieved/cached 10-K passages; refuse to speculate beyond the corpus.
   - Cite every narrative claim inline using the exact format:
     `[TICKER FYxxxx | Section Title | chunk_id](sec_filing_url)`
   - Include a **Grounded 10-K Citations** summary at the end of your report with `chunk_id`, `ticker`, `fiscal_year`, `section_title`, and `sec_filing_url`.
"""


def create_financial_analyst_agent(
    *,
    name: str = "FinancialAnalystAgent",
    model: str = DEFAULT_ANALYST_MODEL,
) -> Agent:
    """Create the specialized `FinancialAnalystAgent` ADK `LlmAgent` with `PlanReActPlanner`."""
    return Agent(
        name=name,
        model=model,
        planner=PlanReActPlanner(),
        description=(
            "Gemini 2.5 Pro SEC 10-K financial reasoning agent employing a structured ReAct "
            "loop to evaluate deterministic period-over-period variances, query cached 10-K "
            "context (Item 1A Risk Factors & Item 7 MD&A), and format strictly cited reports."
        ),
        instruction=FINANCIAL_ANALYST_INSTRUCTION,
        tools=[
            VarianceCalculationTool,
            MetricFetchTool,
            LookerQueryTool,
            HybridSearchTool,
            ContextCacheTool,
            QueryCached10KTool,
        ],
        generate_content_config=genai_types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=4096,
        ),
        output_key="financial_analyst_report",
        before_agent_callback=before_agent_telemetry_callback,
        after_agent_callback=after_agent_memory_callback,
        before_model_callback=before_model_safety_and_latency_callback,
        after_model_callback=after_model_fact_check_callback,
        before_tool_callback=before_tool_agent_identity_callback,
    )


financial_analyst_agent: Agent = create_financial_analyst_agent()
FinancialAnalystAgent: Agent = financial_analyst_agent
