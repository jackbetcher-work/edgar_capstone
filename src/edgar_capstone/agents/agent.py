"""Standard ADK entrypoint module exposing `root_agent` (`RootOrchestrator`)."""

from __future__ import annotations

from edgar_capstone.agents.financial_analyst import (
    FinancialAnalystAgent,
    financial_analyst_agent,
)
from edgar_capstone.agents.root_orchestrator import RootOrchestrator, root_agent

__all__ = [
    "FinancialAnalystAgent",
    "RootOrchestrator",
    "financial_analyst_agent",
    "root_agent",
]
