"""Deterministic query intent classification for SEC 10-K conversational routing."""

from __future__ import annotations

import re
from typing import Literal

from edgar_capstone.observability.tracing import traced

QueryIntentType = Literal[
    "longitudinal_single_company",
    "multi_company_peer_comparison",
    "thematic_trend_tracking",
    "chunk_health_monitoring",
]

TurnIntentType = Literal[
    "longitudinal_single_company",
    "multi_company_peer_comparison",
    "thematic_trend_tracking",
    "chunk_health_monitoring",
    "clarification_needed",
    "out_of_corpus",
    "projection_compliance_block",
    "safety_guardrail_block",
]

_THEMATIC_KEYWORDS_RE = re.compile(
    r"\b(?:thematic|wording|risk\s+factor|item\s+1a|how\s+has\s+the\s+wording|disclosure|trend|shifted|evolved|across\s+all)\b",
    re.IGNORECASE,
)
_PEER_COMPARE_RE = re.compile(
    r"\b(?:compare|comparison|versus|vs\.?|peer|benchmark|between\s+[A-Z]+\s+and\s+[A-Z]+)\b",
    re.IGNORECASE,
)
CHUNK_HEALTH_RE = re.compile(
    r"\b(?:chunk\s+health|embedding\s+coverage|vector\s+db\s+health|vector\s+database\s+health)\b",
    re.IGNORECASE,
)
_CHUNK_HEALTH_RE = CHUNK_HEALTH_RE


@traced("orchestrator.classify_intent")
def classify_query_intent(query: str, tickers: list[str] | None = None) -> QueryIntentType:
    """Deterministically classify an analyst query into one of the supported routing intents."""
    if CHUNK_HEALTH_RE.search(query):
        return "chunk_health_monitoring"
    if tickers and len(tickers) > 1:
        return "multi_company_peer_comparison"
    if _PEER_COMPARE_RE.search(query):
        return "multi_company_peer_comparison"
    if _THEMATIC_KEYWORDS_RE.search(query):
        return "thematic_trend_tracking"
    return "longitudinal_single_company"


__all__ = [
    "CHUNK_HEALTH_RE",
    "QueryIntentType",
    "TurnIntentType",
    "classify_query_intent",
]
