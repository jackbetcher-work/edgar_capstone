"""FinancialAnalystAgent: Structured ReAct reasoning and grounded synthesis over SEC 10-K filings."""

from __future__ import annotations

import os

from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
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
from edgar_capstone.agents.prompt_store import (
    load_agent_prompt,
    make_adk_instruction_provider,
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

FINANCIAL_ANALYST_INSTRUCTION: str = load_agent_prompt("financial_analyst")


def create_financial_analyst_agent(
    *,
    name: str = "FinancialAnalystAgent",
    model: str | BaseLlm = DEFAULT_ANALYST_MODEL,
    prompt_version: str | None = None,
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
        instruction=make_adk_instruction_provider("financial_analyst", version=prompt_version),
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
