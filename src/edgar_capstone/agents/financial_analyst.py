"""FinancialAnalystAgent: Structured ReAct reasoning and grounded synthesis over SEC 10-K filings."""

from __future__ import annotations

import os

from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.planners import BuiltInPlanner
from google.genai import types as genai_types

from edgar_capstone.agents.callbacks import (
    after_model_fact_check_callback,
    before_agent_telemetry_callback,
    before_model_safety_and_latency_callback,
    before_tool_agent_identity_callback,
    retrieval_budget_before_tool_callback,
)
from edgar_capstone.agents.context_memory import (
    FINANCIAL_REASONING_MODEL,
    after_agent_memory_callback,
)
from edgar_capstone.agents.prompt_store import (
    load_agent_prompt,
    make_adk_instruction_provider,
)
from edgar_capstone.agents.schemas import AnalystReport
from edgar_capstone.agents.tools import (
    HybridSearchTool,
    LookerQueryTool,
    MetricFetchTool,
    QueryCached10KTool,
    VarianceCalculationTool,
)

DEFAULT_ANALYST_MODEL = os.environ.get("EDGAR_ANALYST_MODEL", FINANCIAL_REASONING_MODEL)

# The analyst follows the registry's `latest` prompt (v2: targeted 10-K retrieval) rather than
# the store-wide default version used by the other agents.
DEFAULT_ANALYST_PROMPT_VERSION = "latest"

FINANCIAL_ANALYST_INSTRUCTION: str = load_agent_prompt(
    "financial_analyst", version=DEFAULT_ANALYST_PROMPT_VERSION
)


def create_financial_analyst_agent(
    *,
    name: str = "FinancialAnalystAgent",
    model: str | BaseLlm = DEFAULT_ANALYST_MODEL,
    prompt_version: str | None = None,
) -> Agent:
    """Create the specialized `FinancialAnalystAgent` ADK `LlmAgent`.

    Uses the model's native thinking (`BuiltInPlanner`) rather than `PlanReActPlanner`.
    PlanReActPlanner asks the model to write tool calls as text under `/*ACTION*/`, and
    Gemini 3.x sometimes did exactly that ("default_api.query_cached_10k_context(...)")
    instead of issuing a real function call, so retrieval never ran. The ReAct structure
    lives in the system prompt and the deterministic pre-pass instead.
    """
    return Agent(
        name=name,
        model=model,
        planner=BuiltInPlanner(thinking_config=genai_types.ThinkingConfig(include_thoughts=True)),
        description=(
            "Gemini SEC 10-K financial reasoning agent employing a structured ReAct "
            "loop to evaluate deterministic period-over-period variances, retrieve only the 10-K "
            "evidence each question needs (Item 1A Risk Factors & Item 7 MD&A passages), and "
            "format strictly cited reports."
        ),
        instruction=make_adk_instruction_provider(
            "financial_analyst", version=prompt_version or DEFAULT_ANALYST_PROMPT_VERSION
        ),
        tools=[
            VarianceCalculationTool,
            MetricFetchTool,
            LookerQueryTool,
            # Targeted retrieval only: ranked passages and hybrid search, each returning
            # chunk_ids. Whole-section loading (`load_cached_10k_section`) stays with
            # ContextCacheAgent and the source-viewer API, not the synthesis agent.
            HybridSearchTool,
            QueryCached10KTool,
        ],
        generate_content_config=genai_types.GenerateContentConfig(
            temperature=0.1,
            # Includes thinking tokens on Gemini 3.x; a 4096 cap truncated full reports.
            max_output_tokens=16384,
        ),
        # Constrained JSON answer (headline, metric references, quote-backed findings, notes),
        # verified claim by claim in structured_report.py before it is rendered.
        output_schema=AnalystReport,
        output_key="financial_analyst_report",
        before_agent_callback=before_agent_telemetry_callback,
        after_agent_callback=after_agent_memory_callback,
        before_model_callback=before_model_safety_and_latency_callback,
        after_model_callback=after_model_fact_check_callback,
        before_tool_callback=[
            before_tool_agent_identity_callback,
            retrieval_budget_before_tool_callback,
        ],
    )


financial_analyst_agent: Agent = create_financial_analyst_agent()
FinancialAnalystAgent: Agent = financial_analyst_agent
