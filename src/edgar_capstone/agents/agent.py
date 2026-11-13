"""Standard ADK entrypoint module exposing `root_agent` (`RootOrchestrator`)."""

from __future__ import annotations

from edgar_capstone.agents.context_cache_agent import (
    ContextCacheAgent,
    context_cache_agent,
)
from edgar_capstone.agents.financial_analyst import (
    FinancialAnalystAgent,
    financial_analyst_agent,
)
from edgar_capstone.agents.root_orchestrator import RootOrchestrator, root_agent

__all__ = [
    "ContextCacheAgent",
    "FinancialAnalystAgent",
    "RootOrchestrator",
    "context_cache_agent",
    "financial_analyst_agent",
    "root_agent",
]
