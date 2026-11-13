"""Standard ADK entrypoint module exposing `root_agent` (`RootOrchestrator`) and `app` (`App`)."""

from __future__ import annotations

from google.adk.apps import App

from edgar_capstone.agents.context_cache_agent import (
    ContextCacheAgent,
    context_cache_agent,
)
from edgar_capstone.agents.context_memory import create_edgar_adk_app
from edgar_capstone.agents.evaluation_and_coding import (
    CodingAssistantAgent,
    NarrativeEvaluatorAgent,
    coding_assistant_agent,
    narrative_evaluator_agent,
)
from edgar_capstone.agents.financial_analyst import (
    FinancialAnalystAgent,
    financial_analyst_agent,
)
from edgar_capstone.agents.root_orchestrator import RootOrchestrator, root_agent

edgar_adk_app: App = create_edgar_adk_app(root_agent)
app: App = edgar_adk_app

__all__ = [
    "CodingAssistantAgent",
    "ContextCacheAgent",
    "FinancialAnalystAgent",
    "NarrativeEvaluatorAgent",
    "RootOrchestrator",
    "app",
    "coding_assistant_agent",
    "context_cache_agent",
    "edgar_adk_app",
    "financial_analyst_agent",
    "narrative_evaluator_agent",
    "root_agent",
]
