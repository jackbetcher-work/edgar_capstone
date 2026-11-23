"""Canonical SEC EDGAR domain constants, identifiers, and normalizers."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import date
from typing import Literal, TypeVar

_T = TypeVar("_T")

MetricName = Literal["revenue", "operating_income", "net_income", "research_and_development"]
SectionId = Literal["item_1a", "item_7"]

CORE_METRICS: tuple[MetricName, ...] = (
    "revenue",
    "operating_income",
    "net_income",
    "research_and_development",
)

RATIO_METRICS: tuple[tuple[str, MetricName, str], ...] = (
    ("operating_margin_pct", "operating_income", "operating_margin_bps_delta"),
    ("net_margin_pct", "net_income", "net_margin_bps_delta"),
    ("rd_intensity_pct", "research_and_development", "rd_intensity_bps_delta"),
)

METRIC_LABELS: dict[str, str] = {
    "revenue": "Revenue",
    "operating_income": "Operating Income",
    "net_income": "Net Income",
    "research_and_development": "R&D Expense",
    "operating_margin_pct": "Operating Margin",
    "net_margin_pct": "Net Margin",
    "rd_intensity_pct": "R&D Intensity",
}

SECTION_TITLES: dict[str, str] = {
    "item_1a": "Item 1A. Risk Factors",
    "item_7": "Item 7. MD&A",
}

CANONICAL_GICS_SECTORS: dict[str, str] = {
    "information technology": "Information Technology",
    "financials": "Financials",
    "health care": "Health Care",
    "consumer discretionary": "Consumer Discretionary",
    "communication services": "Communication Services",
    "industrials": "Industrials",
    "consumer staples": "Consumer Staples",
    "energy": "Energy",
    "utilities": "Utilities",
    "real estate": "Real Estate",
    "materials": "Materials",
}

# Authoritative set of ADK retrieval tools counted toward grounding / retrieval budgets
RETRIEVAL_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "search_10k_chunks_rag",
        "query_cached_10k_context",
        "load_cached_10k_section",
        "compare_disclosures_across_years",
        "flag_material_variances",
        "sector_thematic_map_reduce",
    }
)

# Authoritative set of ADK structured variance / quantitative tools for telemetry & ReAct phase tagging
VARIANCE_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "calculate_period_variances",
        "fetch_financial_metrics",
        "query_analytics_view",
        "query_looker_financial_explore",
        "flag_material_variances",
    }
)


def normalize_edgar_ticker(raw_ticker: str) -> str:
    """Normalize ticker symbol to SEC EDGAR & Discovery Engine safe format (e.g., BRK.B -> BRK-B)."""
    return raw_ticker.strip().upper().replace(".", "-")


def format_cik(raw_cik: str | int | None) -> str | None:
    """Format a numeric or string CIK as a 10-digit zero-padded string."""
    if raw_cik is None:
        return None
    cleaned = str(raw_cik).strip()
    if not cleaned:
        return None
    digits = "".join(ch for ch in cleaned if ch.isdigit())
    return digits.zfill(10) if digits else None


def canonical_xbrl_tag(raw_tag: str) -> str:
    """Convert ``us-gaap_Revenues`` style tags to canonical ``us-gaap:Revenues`` form."""
    cleaned = raw_tag.strip()
    if ":" not in cleaned and "_" in cleaned:
        return cleaned.replace("_", ":", 1)
    return cleaned


def parse_iso_date(raw: object) -> date | None:
    """Parse a ``YYYY-MM-DD`` prefix into a ``datetime.date`` (returns ``None`` if invalid)."""
    if raw is None:
        return None
    text = str(raw).strip()[:10]
    if len(text) != 10:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def canonicalize_gics_sector(raw_sector: str | None) -> str | None:
    """Normalize a GICS sector string, returning ``None`` for ``ALL``/``*``/blank."""
    if not raw_sector:
        return None
    cleaned = raw_sector.strip()
    if not cleaned or cleaned.upper() in ("ALL", "*"):
        return None
    return CANONICAL_GICS_SECTORS.get(cleaned.lower(), cleaned)


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def lookup_year_key(mapping: Mapping[int | str, _T] | None, year: int) -> _T | None:
    """Look up a fiscal year key in dicts whose keys may be ``int`` or JSON-stringified ``str``."""
    if not mapping:
        return None
    if year in mapping:
        return mapping[year]
    year_str = str(year)
    if year_str in mapping:
        return mapping[year_str]
    return None
