"""RootOrchestrator: Central ADK supervisor for intent routing and hybrid SEC 10-K analysis."""

from __future__ import annotations

import os
import re
from typing import Literal

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
from edgar_capstone.agents.context_cache_agent import create_context_cache_agent
from edgar_capstone.agents.context_memory import after_agent_memory_callback
from edgar_capstone.agents.evaluation_and_coding import (
    DEFAULT_FLASH_MODEL,
    create_coding_assistant_agent,
    create_narrative_evaluator_agent,
)
from edgar_capstone.agents.financial_analyst import (
    DEFAULT_ANALYST_MODEL,
    create_financial_analyst_agent,
)
from edgar_capstone.agents.prompt_store import (
    load_agent_prompt,
    make_adk_instruction_provider,
)
from edgar_capstone.agents.tools import (
    LookerQueryTool,
    MetricFetchTool,
    VarianceCalculationTool,
)
from edgar_capstone.observability.tracing import traced

DEFAULT_ORCHESTRATOR_MODEL = os.environ.get("EDGAR_ORCHESTRATOR_MODEL", DEFAULT_ANALYST_MODEL)

QueryIntentType = Literal[
    "longitudinal_single_company",
    "multi_company_peer_comparison",
    "thematic_trend_tracking",
    "chunk_health_monitoring",
]

ROOT_ORCHESTRATOR_INSTRUCTION: str = load_agent_prompt("root_orchestrator")

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


@traced("orchestrator.classify_intent")
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
    model: str | BaseLlm = DEFAULT_ORCHESTRATOR_MODEL,
    analyst_model: str | BaseLlm = DEFAULT_ANALYST_MODEL,
    flash_model: str | BaseLlm = DEFAULT_FLASH_MODEL,
    prompt_version: str | None = None,
) -> Agent:
    """Create the central `RootOrchestrator` ADK supervisor agent with ReAct planner and Pro/Flash sub-agents."""
    analyst_sub_agent = create_financial_analyst_agent(
        name="FinancialAnalystAgent",
        model=analyst_model,
        prompt_version=prompt_version,
    )
    context_cache_sub_agent = create_context_cache_agent(
        name="ContextCacheAgent",
        model=flash_model,
        prompt_version=prompt_version,
    )
    evaluator_sub_agent = create_narrative_evaluator_agent(
        name="NarrativeEvaluatorAgent",
        model=flash_model,
        prompt_version=prompt_version,
    )
    coding_sub_agent = create_coding_assistant_agent(
        name="CodingAssistantAgent",
        model=flash_model,
        prompt_version=prompt_version,
    )
    return Agent(
        name=name,
        model=model,
        planner=PlanReActPlanner(),
        description=(
            "Central Gemini ReAct supervisor for the SEC EDGAR Natural Language Analyst system. "
            "Routes queries across longitudinal single-company analysis, multi-company peer comparison, "
            "and thematic 10-K trend tracking, orchestrating FinancialAnalystAgent (financial reasoning model) alongside "
            "ContextCacheAgent, NarrativeEvaluatorAgent, and CodingAssistantAgent (gemini-3.5-flash) "
            "under granular per-agent least-privilege identities."
        ),
        instruction=make_adk_instruction_provider("root_orchestrator", version=prompt_version),
        tools=[
            MetricFetchTool,
            LookerQueryTool,
            VarianceCalculationTool,
        ],
        sub_agents=[
            analyst_sub_agent,
            context_cache_sub_agent,
            evaluator_sub_agent,
            coding_sub_agent,
        ],
        generate_content_config=genai_types.GenerateContentConfig(
            temperature=0.1,
            # Includes thinking tokens on Gemini 3.x; a 4096 cap truncated full reports.
            max_output_tokens=16384,
        ),
        output_key="orchestrator_response",
        before_agent_callback=before_agent_telemetry_callback,
        after_agent_callback=after_agent_memory_callback,
        before_model_callback=before_model_safety_and_latency_callback,
        after_model_callback=after_model_fact_check_callback,
        before_tool_callback=before_tool_agent_identity_callback,
    )


RootOrchestrator: Agent = create_root_orchestrator()
root_agent: Agent = RootOrchestrator
