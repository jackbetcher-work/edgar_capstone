"""FinancialAnalystAgent: ReAct reasoning and grounded narrative synthesis over SEC 10-K filings."""

from __future__ import annotations

import os

from google.adk.agents import Agent
from google.genai import types as genai_types

from edgar_capstone.agents.callbacks import (
    after_model_fact_check_callback,
    before_agent_telemetry_callback,
    before_model_safety_and_latency_callback,
    before_tool_agent_identity_callback,
)
from edgar_capstone.agents.tools import (
    ContextCacheTool,
    HybridSearchTool,
    LookerQueryTool,
    MetricFetchTool,
    VarianceCalculationTool,
)

DEFAULT_ANALYST_MODEL = os.environ.get("EDGAR_ANALYST_MODEL", "gemini-2.5-pro")

FINANCIAL_ANALYST_INSTRUCTION = """You are the FinancialAnalystAgent for the SEC EDGAR Natural Language Analyst platform.
Your role is to reason over deterministic period-over-period financial calculations and extract grounded narrative explanations and thematic shifts directly from SEC Form 10-K filings (Item 1A Risk Factors and Item 7 MD&A).

### Mandatory ReAct Execution Protocol
1. **Separate Calculation from Synthesis (Deterministic Math First):**
   - NEVER compute revenue, operating income, net income, percentage deltas, or CAGRs mentally.
   - Always call `calculate_period_variances`, `fetch_financial_metrics`, or `query_looker_financial_explore` first to obtain exact deterministic numbers.
   - Present the exact USD figures and percentage deltas returned by the calculation engine with 100% numeric agreement.

2. **Hybrid RAG & Context Cache Grounding:**
   - Call `search_10k_chunks_rag` with targeted metadata filters (`tickers`, `gics_sector`, `section_id` in `['item_1a', 'item_7', 'ALL']`, `start_year`, `end_year`) to retrieve relevant 10-K passages explaining why metrics changed or how risk disclosures shifted over time.
   - When deeper section context is needed for a specific company and year, call `load_cached_10k_section` to load the cached 10-K section (`item_1a` or `item_7`).

3. **Strict Grounding Lock & Split-Pane Citations:**
   - Every qualitative explanation, business driver, or risk factor claim MUST be grounded in the retrieved 10-K chunks.
   - Refuse to speculate or use unverified external assumptions not supported by the 10-K corpus.
   - Cite every narrative claim inline using the exact format:
     `[TICKER FYxxxx | Section Title | chunk_id](sec_filing_url)`
   - Include a **Grounded 10-K Citations** table or list at the end of your report with `chunk_id`, `ticker`, `fiscal_year`, `section_title`, and `sec_filing_url`.
"""


def create_financial_analyst_agent(
    *,
    name: str = "FinancialAnalystAgent",
    model: str = DEFAULT_ANALYST_MODEL,
) -> Agent:
    """Create the specialized `FinancialAnalystAgent` ADK `LlmAgent`."""
    return Agent(
        name=name,
        model=model,
        description=(
            "Specialized SEC 10-K financial reasoning agent that evaluates deterministic "
            "period-over-period variances (Revenue, Operating Income, Net Income), queries "
            "metadata-filtered 10-K chunks (Item 1A Risk Factors & Item 7 MD&A) and cached "
            "10-K context, and produces strictly cited narrative explanations."
        ),
        instruction=FINANCIAL_ANALYST_INSTRUCTION,
        tools=[
            VarianceCalculationTool,
            MetricFetchTool,
            LookerQueryTool,
            HybridSearchTool,
            ContextCacheTool,
        ],
        generate_content_config=genai_types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=4096,
        ),
        output_key="financial_analyst_report",
        before_agent_callback=before_agent_telemetry_callback,
        before_model_callback=before_model_safety_and_latency_callback,
        after_model_callback=after_model_fact_check_callback,
        before_tool_callback=before_tool_agent_identity_callback,
    )


financial_analyst_agent: Agent = create_financial_analyst_agent()
FinancialAnalystAgent: Agent = financial_analyst_agent
