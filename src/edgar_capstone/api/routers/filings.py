"""SEC 10-K filings catalog and split-pane section viewer routes."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from edgar_capstone.agents.orchestration.chat_engine import (
    list_curated_companies,
    lookup_cached_profile,
)
from edgar_capstone.agents.tools import (
    get_active_gcp_tool_adapters,
    load_cached_10k_section,
)
from edgar_capstone.agents.tools.runtime import tool_turn_context
from edgar_capstone.api.dependencies import audited_permissions
from edgar_capstone.common.domain import normalize_edgar_ticker
from edgar_capstone.security.auth import (
    AuthenticatedUser,
    record_security_audit_event,
    require_permissions,
)
from edgar_capstone.security.rbac import Permission

router = APIRouter(tags=["SEC Filings"])

_SECTION_PROFILE_META_FIELDS: tuple[str, ...] = (
    "cik",
    "accession_number",
    "filing_date",
    "period_of_report",
    "gics_sector",
    "gics_sub_industry",
)


def _enrich_section_response(
    ticker: str,
    fiscal_year: int,
    section_id: str,
    max_chars: int,
) -> dict[str, Any]:
    norm_ticker = normalize_edgar_ticker(ticker)
    if not norm_ticker:
        raise HTTPException(
            status_code=422,
            detail="ticker must contain at least one non-whitespace character",
        )

    cached_prof = lookup_cached_profile(norm_ticker, fiscal_year)
    if cached_prof is not None:
        with tool_turn_context(cached_profiles=[cached_prof]):
            res = load_cached_10k_section(
                ticker=norm_ticker,
                fiscal_year=fiscal_year,
                section_id=section_id,
                max_chars=max_chars,
            )
    else:
        res = load_cached_10k_section(
            ticker=norm_ticker,
            fiscal_year=fiscal_year,
            section_id=section_id,
            max_chars=max_chars,
        )
    if res.get("status") == "error":
        raise HTTPException(status_code=404, detail=res.get("message", "Section not found"))

    if any(field not in res for field in _SECTION_PROFILE_META_FIELDS):
        matched_prof = cached_prof
        if matched_prof is None:
            profiles = get_active_gcp_tool_adapters().bigquery.fetch_profiles(
                tickers={norm_ticker},
                start_year=fiscal_year,
                end_year=fiscal_year,
            )
            matched_prof = profiles[0] if profiles else None
        if matched_prof is not None:
            for field in _SECTION_PROFILE_META_FIELDS:
                res.setdefault(field, getattr(matched_prof, field))
    return res


@router.get("/api/v1/companies")
def get_companies(
    _user: AuthenticatedUser = Depends(
        audited_permissions(
            Permission.FILINGS_READ.value,
            event_type="companies_catalog_read",
        )
    ),
) -> dict[str, Any]:
    """List all curated S&P 500 issuers, GICS sectors, and available fiscal years."""
    return list_curated_companies()


@router.get("/api/v1/section")
def get_cached_section(
    ticker: str = Query("NVDA", min_length=1, max_length=16),
    fiscal_year: int = Query(2026, ge=2015, le=2035),
    section_id: Literal["item_1a", "item_7"] = Query("item_7"),
    max_chars: int = Query(32000, ge=500, le=50000),
    user: AuthenticatedUser = Depends(require_permissions(Permission.FILINGS_READ.value)),
) -> dict[str, Any]:
    """Load a cached 10-K filing section (Item 1A Risk Factors or Item 7 MD&A) for the split-pane viewer."""
    norm_ticker = normalize_edgar_ticker(ticker)
    if not norm_ticker:
        raise HTTPException(
            status_code=422,
            detail="ticker must contain at least one non-whitespace character",
        )
    record_security_audit_event(
        event_type="filing_10k_section_read",
        user=user,
        resource="/api/v1/section",
        tickers=[norm_ticker],
        detail=f"FY{fiscal_year} {section_id}",
    )
    return _enrich_section_response(
        ticker=norm_ticker,
        fiscal_year=fiscal_year,
        section_id=section_id,
        max_chars=max_chars,
    )
