"""Canonical query intent types and agent-driven intent resolution.

Intent is no longer classified with keyword regexes. `FinancialAnalystAgent` emits the canonical
`QueryIntentType` in its structured `AnalystReport.intent`, and `resolve_turn_intent` falls back
to a deterministic, non-lexical inference from the deterministic tools the agent actually
invoked plus the resolved scope structure when the agent omits it (or its answer is withheld).

The enum values are unchanged so API schemas, Looker telemetry, golden evals, and UI badges keep
working. Lexical matching that is NOT intent classification (scope / out-of-corpus gates, Model
Armor, hybrid RAG reranking, delta term counting, parsing, citation verification) lives in its
own subsystem and is intentionally untouched.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, get_args

from edgar_capstone.agents.schemas import QueryIntentType
from edgar_capstone.observability.tracing import traced

IntentSource = Literal["agent", "tool_inference", "scope_gate"]

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

QUERY_INTENT_VALUES: tuple[str, ...] = tuple(get_args(QueryIntentType))

_THEMATIC_TOOLS = frozenset({"compare_disclosures_across_years", "sector_thematic_map_reduce"})


def is_sector_wide_scope(scope: dict[str, Any]) -> bool:
    """True when the resolved scope names a GICS sector but no specific companies."""
    return not (scope.get("tickers") or []) and str(scope.get("gics_sector") or "ALL") not in (
        "ALL",
        "",
        "*",
    )


def _tool_names(tool_calls: Iterable[dict[str, Any]] | None) -> list[str]:
    return [str(c.get("name") or "") for c in tool_calls or [] if isinstance(c, dict)]


def _explores(tool_calls: Iterable[dict[str, Any]] | None) -> set[str]:
    return {
        str((c.get("args") or {}).get("explore_name") or "").strip().lower()
        for c in tool_calls or []
        if isinstance(c, dict) and c.get("name") == "query_looker_financial_explore"
    }


@traced("orchestrator.infer_intent_from_tools")
def infer_intent_from_tool_calls(
    scope: dict[str, Any],
    tool_calls: Iterable[dict[str, Any]] | None = None,
) -> QueryIntentType:
    """Deterministic (non-lexical) intent fallback from invoked tools and scope structure."""
    calls = list(tool_calls or [])
    names = set(_tool_names(calls))
    explores = _explores(calls)
    if "chunk_health" in explores:
        return "chunk_health_monitoring"
    if names & _THEMATIC_TOOLS:
        return "thematic_trend_tracking"
    if explores & {"sector_benchmarks", "peer_comparison"}:
        return "multi_company_peer_comparison"
    if is_sector_wide_scope(scope) and "calculate_period_variances" not in names:
        return "thematic_trend_tracking"
    if len(scope.get("tickers") or []) > 1:
        return "multi_company_peer_comparison"
    return "longitudinal_single_company"


def resolve_turn_intent(
    *,
    agent_intent: Any,
    scope: dict[str, Any],
    tool_calls: Iterable[dict[str, Any]] | None = None,
) -> tuple[QueryIntentType, IntentSource]:
    """Prefer the agent's structured `AnalystReport.intent`; else infer from tools + scope."""
    if isinstance(agent_intent, str) and agent_intent in QUERY_INTENT_VALUES:
        return agent_intent, "agent"  # type: ignore[return-value]
    return infer_intent_from_tool_calls(scope, tool_calls), "tool_inference"


__all__ = [
    "QUERY_INTENT_VALUES",
    "IntentSource",
    "QueryIntentType",
    "TurnIntentType",
    "infer_intent_from_tool_calls",
    "is_sector_wide_scope",
    "resolve_turn_intent",
]
