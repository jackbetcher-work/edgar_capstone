"""FinancialAnalystAgent: Structured ReAct reasoning and grounded synthesis over SEC 10-K filings."""

from __future__ import annotations

import os

from edgar_capstone.agents.runtime.factories import (
    DEFAULT_ANALYST_PROMPT_VERSION,
    create_financial_analyst_agent,
)
from edgar_capstone.agents.runtime.model_config import FINANCIAL_REASONING_MODEL

DEFAULT_ANALYST_MODEL: str = os.environ.get("EDGAR_ANALYST_MODEL", FINANCIAL_REASONING_MODEL)

__all__ = [
    "DEFAULT_ANALYST_MODEL",
    "DEFAULT_ANALYST_PROMPT_VERSION",
    "create_financial_analyst_agent",
]
