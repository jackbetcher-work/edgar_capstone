"""Fast `gemini-3.5-flash-lite` scope & subject-continuity resolver with deterministic fallback.

Invoked ONLY on unresolved or ambiguous turns (when deterministic company, GICS sector, GICS
sub-industry, known non-S&P 500 entity, and broad-corpus resolution found nothing). When L1
session memory and the entity catalog are warm, this runs concurrently with Model Armor inside
`chat_engine._run_parallel_prepass` so its ~150–300 ms latency is hidden behind prompt screening.

Every ticker, GICS sector, and GICS sub-industry emitted by the model is strictly validated
against `EntityCatalog` before entering the RAG or deterministic variance pipeline.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from typing import Any, Literal

from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.genai import types as genai_types
from pydantic import BaseModel, Field

from edgar_capstone.agents.orchestration.scope_validation import EntityCatalog
from edgar_capstone.agents.runtime.callbacks import ModelTelemetryPlugin
from edgar_capstone.agents.runtime.live_runtime import (
    build_adk_llm_for_task,
    resolve_live_adk_runtime,
    run_adk_agent_stream,
)
from edgar_capstone.agents.runtime.model_config import minimal_thinking_config_for_model
from edgar_capstone.common.domain import (
    CANONICAL_GICS_SECTORS,
    canonicalize_gics_sector,
    normalize_edgar_ticker,
)
from edgar_capstone.common.formatting import parse_model_output
from edgar_capstone.observability.tracing import set_span_attributes, traced
from edgar_capstone.security.model_armor.projection_filter import _detect_sensitive_spans

logger = logging.getLogger(__name__)

ScopeResolutionAction = Literal[
    "continue_prior",
    "new_tickers",
    "new_sector",
    "new_sub_industry",
    "broad_corpus",
    "unknown_entity",
    "clarify",
]

ScopeLlmMode = Literal["auto", "live", "deterministic"]

# Conservative deterministic signals indicating the user introduced a NEW subject rather than
# continuing the prior company/sector (used when offline or if Flash-Lite is unavailable).
_EXPLICIT_SWITCH_RE = re.compile(
    r"\b(?:switch\s+to|change\s+(?:the\s+)?(?:topic|subject)|moving\s+on\s+to|instead\s+of|"
    r"forget\s+about|new\s+topic|different\s+company|another\s+company)\b",
    re.IGNORECASE,
)
_NEW_INDUSTRY_PHRASE_RE = re.compile(
    r"\b(?P<mod>[a-z][a-z\-]{2,})\s+"
    r"(?:industry|sector|space|vertical|stocks?|contractors?|makers?|manufacturers?|"
    r"carriers?|insurers?|retailers?|producers?|market)\b",
    re.IGNORECASE,
)
_ANAPHORIC_INDUSTRY_MODIFIERS: frozenset[str] = frozenset(
    {
        "this",
        "that",
        "the",
        "their",
        "its",
        "same",
        "overall",
        "broader",
        "end",
        "core",
        "primary",
        "key",
        "domestic",
        "global",
        "international",
        "addressable",
        "capital",
        "stock",
        "equity",
        "credit",
        "bond",
        "labor",
        "data",
    }
)
_CROSS_MARKET_COMPANIES_RE = re.compile(
    r"\b(?:which|what|top|best|worst)\s+(?:s&p\s*500\s+)?(?:companies|firms|issuers|stocks)\b",
    re.IGNORECASE,
)
_UNKNOWN_PROPER_ENTITY_RE = re.compile(
    r"\b(?:about|analyze|check|evaluate|review|compare|for)\s+"
    r"(?P<entity>[A-Z][A-Za-z0-9&.\-]{2,}(?:\s+[A-Z][A-Za-z0-9&.\-]{2,})?)\b"
)
_SAFE_CAPITALIZED_TERMS: frozenset[str] = frozenset(
    {
        "item",
        "risk",
        "risks",
        "factors",
        "revenue",
        "revenues",
        "operating",
        "income",
        "margin",
        "margins",
        "net",
        "profit",
        "profits",
        "gaap",
        "xbrl",
        "sec",
        "usa",
        "china",
        "europe",
        "asia",
        "japan",
        "india",
        "americas",
        "emea",
        "apac",
        "north",
        "south",
        "united",
        "states",
        "fiscal",
        "year",
        "years",
        "quarter",
        "annual",
        "md&a",
        "r&d",
        "ebitda",
        "eps",
        "capex",
        "cagr",
        "genai",
        "llm",
        "cloud",
        "covid",
        "tariffs",
        "inflation",
        "supply",
        "chain",
        "export",
        "controls",
        "management",
        "discussion",
        "analysis",
        "segment",
        "segments",
        "products",
        "services",
        "part",
        "section",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    }
)


class ScopeResolutionDecision(BaseModel):
    """Structured output emitted by `ScopeContinuityAgent` (`gemini-3.5-flash-lite`)."""

    action: ScopeResolutionAction
    tickers: list[str] = Field(
        default_factory=list,
        description="S&P 500 ticker symbols when action='new_tickers' (uppercase, e.g. ['RTX']).",
    )
    gics_sector: str | None = Field(
        default=None,
        description="Exact canonical GICS sector name when action='new_sector'.",
    )
    gics_sub_industry: str | None = Field(
        default=None,
        description="Exact GICS sub-industry name from the provided list when action='new_sub_industry'.",
    )
    unknown_entities: list[str] = Field(
        default_factory=list,
        description="Non-S&P 500 / private company names when action='unknown_entity'.",
    )
    reason: str = Field(
        default="",
        description="Brief 5-15 word justification for the scope/continuity decision.",
    )


def resolve_scope_llm_mode() -> ScopeLlmMode:
    raw = os.environ.get("EDGAR_SCOPE_LLM_MODE", "auto").strip().lower()
    if raw in ("deterministic", "off", "false", "0", "disabled"):
        return "deterministic"
    if raw in ("live", "on", "true", "1", "enabled"):
        return "live"
    return "auto"


def is_scope_llm_enabled(
    *,
    scope_llm: BaseLlm | None = None,
    reasoning_llm: BaseLlm | None = None,
) -> bool:
    """Return True when semantic Flash-Lite scope classification should execute."""
    if scope_llm is not None:
        return True
    mode = resolve_scope_llm_mode()
    if mode == "deterministic":
        return False
    if mode == "live":
        return True
    # Auto mode: run live Flash-Lite in production (`reasoning_llm is None`), never fire
    # surprise network calls in scripted tests that inject a fake `reasoning_llm`.
    return reasoning_llm is None


def has_local_sensitive_pii(query: str) -> bool:
    """True when local SDP detectors see sensitive PII (skip speculative pre-redaction LLM calls)."""
    return bool(_detect_sensitive_spans(query or ""))


def _extract_prior_scope(
    memory_context: dict[str, Any] | None,
) -> tuple[list[str], str, str | None, str]:
    if not isinstance(memory_context, dict) or not memory_context.get("has_prior_memory"):
        return [], "ALL", None, ""
    tickers = [
        str(t).strip() for t in (memory_context.get("remembered_tickers") or []) if str(t).strip()
    ]
    raw_sector = str(memory_context.get("last_gics_sector") or "ALL").strip()
    sector = raw_sector if raw_sector and raw_sector.upper() not in ("ALL", "*") else "ALL"
    raw_sub = memory_context.get("last_gics_sub_industry")
    sub = str(raw_sub).strip() if raw_sub else None
    summary = str(memory_context.get("memory_summary_text") or "").strip()
    return tickers, sector, sub, summary


def _detect_unrecognized_proper_entity(query: str) -> str | None:
    for m in _UNKNOWN_PROPER_ENTITY_RE.finditer(query or ""):
        cand = m.group("entity").strip()
        words = [w.lower() for w in cand.split()]
        if not words or all(w in _SAFE_CAPITALIZED_TERMS for w in words):
            continue
        if any(w.startswith("fy") and w[2:].isdigit() for w in words):
            continue
        return cand
    return None


def deterministic_continuity_fallback(
    query: str,
    *,
    memory_context: dict[str, Any] | None,
    catalog: EntityCatalog,
) -> ScopeResolutionDecision:
    """Conservative deterministic fallback when Flash-Lite is disabled or unavailable.

    Protects against silent prior-company carryover when a follow-up introduces an unrecognized
    industry, explicit subject switch, cross-market question, or unknown proper-noun entity.
    """
    _ = catalog
    prior_tickers, prior_sector, _prior_sub, _ = _extract_prior_scope(memory_context)
    has_prior_tickers = bool(prior_tickers)
    has_prior_sector = prior_sector != "ALL"
    has_prior_scope = has_prior_tickers or has_prior_sector

    if not has_prior_scope:
        return ScopeResolutionDecision(action="clarify", reason="No company or sector detected")

    if _EXPLICIT_SWITCH_RE.search(query):
        return ScopeResolutionDecision(
            action="clarify", reason="Explicit topic switch without recognized target"
        )

    for m in _NEW_INDUSTRY_PHRASE_RE.finditer(query):
        if m.group("mod").lower() not in _ANAPHORIC_INDUSTRY_MODIFIERS:
            return ScopeResolutionDecision(
                action="clarify",
                reason=f"New industry phrase '{m.group(0)}' does not match prior scope",
            )

    if has_prior_tickers and not has_prior_sector and _CROSS_MARKET_COMPANIES_RE.search(query):
        return ScopeResolutionDecision(
            action="broad_corpus",
            reason="Cross-market question asking across companies rather than prior issuer",
        )

    unknown_entity = _detect_unrecognized_proper_entity(query)
    if unknown_entity:
        return ScopeResolutionDecision(
            action="clarify",
            reason=f"Unrecognized entity '{unknown_entity}' appears to change subject",
        )

    return ScopeResolutionDecision(
        action="continue_prior",
        reason="Follow-up query without new-subject signals continues active scope",
    )


def _create_scope_continuity_agent(model: str | BaseLlm) -> Agent:
    return Agent(
        name="ScopeContinuityAgent",
        model=model,
        description=(
            "Fast structured classifier deciding whether an SEC 10-K analyst query continues the "
            "prior conversation subject or switches to a new S&P 500 company, GICS sector, GICS "
            "sub-industry, broad corpus question, or non-S&P 500 entity."
        ),
        instruction=(
            "You resolve conversational scope for an S&P 500 SEC Form 10-K analyst. "
            "Given the prior conversation scope (if any), the analyst's new query, and the exact "
            "lists of canonical GICS sectors and GICS sub-industries, output ONLY valid JSON "
            "matching `ScopeResolutionDecision`.\n"
            "Rules:\n"
            "1. `continue_prior`: ONLY when Prior Scope is present AND the new query genuinely "
            "continues or asks a follow-up about that exact prior company/sector/sub-industry "
            "(e.g. 'What about Item 1A risks?', 'How did margins trend?', 'What about FY2022?'). "
            "NEVER choose `continue_prior` if the user asks about a different industry, sector, "
            "market segment, or company.\n"
            "2. `new_sub_industry`: when the user asks about a specific industry or sub-industry "
            "(e.g. 'defense industry' or 'military contractors' -> 'Aerospace & Defense'; "
            "'airlines' -> 'Passenger Airlines'; 'automakers' -> 'Automobile Manufacturers'). "
            "You MUST copy the exact string from the provided GICS Sub-Industries list.\n"
            "3. `new_sector`: when the user asks about one of the 11 top-level GICS sectors "
            "(copy the exact name from the GICS Sectors list).\n"
            "4. `new_tickers`: when the user refers to specific S&P 500 companies by colloquial "
            "name, former name, nickname, or ticker (return uppercase S&P 500 tickers).\n"
            "5. `broad_corpus`: when the user asks a cross-market question across the S&P 500 "
            "(e.g. 'Which companies are most exposed to tariffs?').\n"
            "6. `unknown_entity`: when the user asks about a specific private, foreign, or "
            "non-S&P 500 company (e.g. 'Anduril', 'SpaceX', 'OpenAI'); put its name in "
            "`unknown_entities`.\n"
            "7. `clarify`: when the query is genuinely vague and cannot be scoped."
        ),
        output_schema=ScopeResolutionDecision,
        output_key="scope_resolution_decision",
        generate_content_config=genai_types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=256,
            thinking_config=minimal_thinking_config_for_model(model),
        ),
    )


def _build_scope_prompt(
    *,
    query: str,
    memory_context: dict[str, Any] | None,
    catalog: EntityCatalog,
) -> str:
    prior_tickers, prior_sector, prior_sub, summary = _extract_prior_scope(memory_context)
    if prior_tickers:
        labeled = [f"{tk} ({catalog.ticker_to_name.get(tk, tk)})" for tk in prior_tickers[:8]]
        extra = f" (+{len(prior_tickers) - 8} more)" if len(prior_tickers) > 8 else ""
        prior_desc = (
            f"Sub-Industry `{prior_sub}` ({', '.join(labeled)}{extra})"
            if prior_sub
            else f"Companies: {', '.join(labeled)}{extra}"
        )
    elif prior_sector != "ALL":
        prior_desc = f"GICS Sector: `{prior_sector}`"
    else:
        prior_desc = "None (first turn / no active prior scope)"

    last_turn_line = ""
    if summary:
        last_line = summary.splitlines()[-1][:240]
        last_turn_line = f"\nLast Turn Summary: {last_line}"

    sectors_csv = ", ".join(CANONICAL_GICS_SECTORS)
    subs_csv = ", ".join(catalog.sub_industries) if catalog.sub_industries else "(none loaded)"
    return (
        f"Prior Active Scope: {prior_desc}{last_turn_line}\n"
        f"New User Query: {query.strip()}\n\n"
        f"Canonical GICS Sectors: {sectors_csv}\n"
        f"Available GICS Sub-Industries: {subs_csv}\n\n"
        "Return JSON matching `ScopeResolutionDecision`."
    )


def validate_scope_decision(
    decision: ScopeResolutionDecision,
    *,
    catalog: EntityCatalog,
    has_prior_scope: bool,
) -> ScopeResolutionDecision:
    """Strictly validate Flash-Lite's output against `EntityCatalog` (zero hallucinated tickers)."""
    action = decision.action
    reason = (decision.reason or "").strip()[:180]

    if action == "continue_prior":
        if has_prior_scope:
            return ScopeResolutionDecision(action="continue_prior", reason=reason)
        return ScopeResolutionDecision(action="clarify", reason="No prior scope to continue")

    if action == "new_tickers":
        valid: list[str] = []
        unknown: list[str] = []
        for raw_tk in decision.tickers:
            tk = normalize_edgar_ticker(str(raw_tk or ""))
            if not tk:
                continue
            if not catalog.tickers or catalog.contains_ticker(tk):
                if tk not in valid:
                    valid.append(tk)
            elif tk not in unknown:
                unknown.append(tk)
        if valid:
            return ScopeResolutionDecision(
                action="new_tickers",
                tickers=valid,
                unknown_entities=unknown,
                reason=reason,
            )
        if unknown:
            return ScopeResolutionDecision(
                action="unknown_entity",
                unknown_entities=unknown,
                reason=reason or "Tickers not in S&P 500 catalog",
            )
        return ScopeResolutionDecision(action="clarify", reason="No valid S&P 500 tickers resolved")

    if action == "new_sub_industry":
        canon_sub, sub_tickers = catalog.tickers_for_sub_industry(decision.gics_sub_industry)
        if canon_sub and sub_tickers:
            return ScopeResolutionDecision(
                action="new_sub_industry",
                tickers=sub_tickers,
                gics_sub_industry=canon_sub,
                reason=reason,
            )
        # Fallback if the model picked a sector name instead of a sub-industry name
        canon_sec = canonicalize_gics_sector(decision.gics_sector or decision.gics_sub_industry)
        if canon_sec:
            return ScopeResolutionDecision(
                action="new_sector",
                gics_sector=canon_sec,
                reason=reason,
            )
        return ScopeResolutionDecision(
            action="clarify", reason="Unrecognized GICS sub-industry returned"
        )

    if action == "new_sector":
        canon_sec = canonicalize_gics_sector(decision.gics_sector)
        if canon_sec:
            return ScopeResolutionDecision(
                action="new_sector",
                gics_sector=canon_sec,
                reason=reason,
            )
        return ScopeResolutionDecision(action="clarify", reason="Unrecognized GICS sector returned")

    if action == "unknown_entity":
        cleaned_unknown = [str(u).strip() for u in decision.unknown_entities if str(u).strip()]
        if cleaned_unknown:
            return ScopeResolutionDecision(
                action="unknown_entity",
                unknown_entities=cleaned_unknown,
                reason=reason,
            )
        return ScopeResolutionDecision(action="clarify", reason=reason)

    if action == "broad_corpus":
        return ScopeResolutionDecision(action="broad_corpus", reason=reason)

    return ScopeResolutionDecision(action="clarify", reason=reason)


@traced("chat.scope_continuity_llm")
def classify_scope_and_continuity(
    *,
    query: str,
    memory_context: dict[str, Any] | None,
    catalog: EntityCatalog,
    scope_llm: BaseLlm | None = None,
    reasoning_llm: BaseLlm | None = None,
    parallel_prepass: bool = False,
) -> tuple[ScopeResolutionDecision, str]:
    """Classify an unresolved/ambiguous query via `gemini-3.5-flash-lite` with deterministic fallback.

    Returns `(validated_decision, source)` where `source` is one of:
      - `"flash_lite_parallel"` (ran concurrently with Model Armor in `_run_parallel_prepass`)
      - `"flash_lite"` (ran synchronously after memory recall / catalog fetch)
      - `"deterministic_fallback"` (offline/test mode or upstream Flash-Lite error)
    """
    prior_tickers, prior_sector, _, _ = _extract_prior_scope(memory_context)
    has_prior_scope = bool(prior_tickers) or prior_sector != "ALL"

    if not is_scope_llm_enabled(scope_llm=scope_llm, reasoning_llm=reasoning_llm):
        fallback = deterministic_continuity_fallback(
            query, memory_context=memory_context, catalog=catalog
        )
        set_span_attributes(
            **{
                "edgar.scope_llm.action": fallback.action,
                "edgar.scope_llm.source": "deterministic_fallback",
                "edgar.scope_llm.parallel_prepass": parallel_prepass,
            }
        )
        return fallback, "deterministic_fallback"

    prompt = _build_scope_prompt(query=query, memory_context=memory_context, catalog=catalog)
    try:
        resolved = resolve_live_adk_runtime(custom_llm=scope_llm)
        model_obj: BaseLlm = (
            scope_llm
            if scope_llm is not None
            else build_adk_llm_for_task("context_compaction", resolution=resolved)
        )
        stream_res = run_adk_agent_stream(
            agent=_create_scope_continuity_agent(model_obj),
            prompt_text=prompt,
            user_id="scope_resolver",
            session_id=f"scope-resolve-{uuid.uuid4().hex[:10]}",
            app_name="edgar_scope_app",
            use_adk_app=False,
            backend=resolved.backend,
            max_llm_calls=2,
            plugins=[ModelTelemetryPlugin()],
            ephemeral=True,
        )
        raw_obj = stream_res.session_state.get("scope_resolution_decision")
        parsed = parse_model_output(
            raw_obj,
            ScopeResolutionDecision,
            fallback_text=stream_res.final_text,
        )
        if parsed is None:
            raise ValueError("ScopeContinuityAgent returned unparseable decision JSON")
        validated = validate_scope_decision(
            parsed, catalog=catalog, has_prior_scope=has_prior_scope
        )
        source = "flash_lite_parallel" if parallel_prepass else "flash_lite"
        set_span_attributes(
            **{
                "edgar.scope_llm.action": validated.action,
                "edgar.scope_llm.source": source,
                "edgar.scope_llm.parallel_prepass": parallel_prepass,
            }
        )
        return validated, source
    except Exception as exc:
        logger.warning(
            "Flash-Lite scope/continuity classifier unavailable (%s); using deterministic fallback.",
            exc,
        )
        fallback = deterministic_continuity_fallback(
            query, memory_context=memory_context, catalog=catalog
        )
        set_span_attributes(
            **{
                "edgar.scope_llm.action": fallback.action,
                "edgar.scope_llm.source": "deterministic_fallback",
                "edgar.scope_llm.parallel_prepass": parallel_prepass,
                "edgar.scope_llm.fallback_error": type(exc).__name__,
            }
        )
        return fallback, "deterministic_fallback"


__all__ = [
    "ScopeLlmMode",
    "ScopeResolutionAction",
    "ScopeResolutionDecision",
    "classify_scope_and_continuity",
    "deterministic_continuity_fallback",
    "has_local_sensitive_pii",
    "is_scope_llm_enabled",
    "resolve_scope_llm_mode",
    "validate_scope_decision",
]
