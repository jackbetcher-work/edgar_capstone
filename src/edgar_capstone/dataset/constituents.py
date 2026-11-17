"""S&P 500 constituent metadata loader and curated starter company set."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import httpx

from edgar_capstone.common.domain import format_cik, normalize_edgar_ticker
from edgar_capstone.dataset.models import CompanyConstituent
from edgar_capstone.dataset.paths import resolve_sp500_constituents_csv

logger = logging.getLogger(__name__)

SP500_UPSTREAM_CSV_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
)

# Retained for backward compatibility; call sites passing `None` resolve dynamically via
# `resolve_sp500_constituents_csv()` so runtime `EDGAR_DATA_ROOT` changes are honored.
DEFAULT_SP500_CSV_PATH = resolve_sp500_constituents_csv()

# 18 flagship S&P 500 companies spanning key GICS sectors (including NVDA & AAPL
# highlighted in the project reference, tech peers, banks, healthcare, energy, retail).
CURATED_STARTER_TICKERS: tuple[str, ...] = (
    "NVDA",
    "AAPL",
    "MSFT",
    "AVGO",
    "CRM",
    "GOOGL",
    "META",
    "AMZN",
    "TSLA",
    "WMT",
    "PG",
    "JPM",
    "V",
    "UNH",
    "LLY",
    "JNJ",
    "XOM",
    "CAT",
)


def refresh_sp500_csv(csv_path: Path | None = None, timeout: float = 20.0) -> Path:
    """Fetch the latest S&P 500 constituents CSV and persist it locally."""
    target_path = csv_path if csv_path is not None else resolve_sp500_constituents_csv()
    logger.info("Refreshing S&P 500 constituents CSV from %s", SP500_UPSTREAM_CSV_URL)
    response = httpx.get(SP500_UPSTREAM_CSV_URL, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(response.text, encoding="utf-8")
    return target_path


def load_sp500_catalog(
    csv_path: Path | None = None,
    refresh: bool = False,
) -> dict[str, CompanyConstituent]:
    """Load the S&P 500 constituent catalog keyed by normalized SEC EDGAR ticker."""
    target_path = csv_path if csv_path is not None else resolve_sp500_constituents_csv()
    if refresh or not target_path.exists():
        refresh_sp500_csv(csv_path=target_path)

    catalog: dict[str, CompanyConstituent] = {}
    with target_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_symbol = (row.get("Symbol") or "").strip()
            if not raw_symbol:
                continue
            edgar_ticker = normalize_edgar_ticker(raw_symbol)
            constituent = CompanyConstituent(
                ticker=edgar_ticker,
                edgar_ticker=edgar_ticker,
                company_name=(row.get("Security") or edgar_ticker).strip(),
                cik=format_cik(row.get("CIK")),
                gics_sector=(row.get("GICS Sector") or "Unknown").strip(),
                gics_sub_industry=(row.get("GICS Sub-Industry") or "Unknown").strip(),
            )
            catalog[edgar_ticker] = constituent

    return catalog


def load_constituents(
    *,
    full_sp500: bool = False,
    tickers: list[str] | None = None,
    sector_filter: str | None = None,
    sp500_csv_path: Path | None = None,
    refresh_sp500: bool = False,
) -> list[CompanyConstituent]:
    """Resolve the target list of companies for dataset curation.

    Priority:
    1. Explicit `tickers` list if provided.
    2. Full S&P 500 catalog if `full_sp500=True`.
    3. Default `CURATED_STARTER_TICKERS` subset.
    """
    target_csv_path = (
        sp500_csv_path if sp500_csv_path is not None else resolve_sp500_constituents_csv()
    )
    catalog: dict[str, CompanyConstituent] = {}
    if target_csv_path.exists() or full_sp500 or refresh_sp500:
        try:
            catalog = load_sp500_catalog(csv_path=target_csv_path, refresh=refresh_sp500)
        except Exception as exc:
            if full_sp500:
                raise RuntimeError(
                    f"Unable to load S&P 500 catalog from {target_csv_path}: {exc}"
                ) from exc
            logger.warning("Proceeding without S&P 500 CSV metadata enrichment: %s", exc)

    if tickers:
        target_symbols: list[str] = []
        seen: set[str] = set()
        for item in tickers:
            for part in item.split(","):
                norm = normalize_edgar_ticker(part)
                if norm and norm not in seen:
                    seen.add(norm)
                    target_symbols.append(norm)
        resolved = [
            catalog.get(
                sym,
                CompanyConstituent(
                    ticker=sym,
                    edgar_ticker=sym,
                    company_name=sym,
                ),
            )
            for sym in target_symbols
        ]
    elif full_sp500:
        resolved = list(catalog.values())
    else:
        resolved = [
            catalog.get(
                sym,
                CompanyConstituent(
                    ticker=sym,
                    edgar_ticker=sym,
                    company_name=sym,
                ),
            )
            for sym in CURATED_STARTER_TICKERS
        ]

    if sector_filter:
        needle = sector_filter.strip().lower()
        resolved = [comp for comp in resolved if comp.gics_sector.strip().lower() == needle]

    return resolved


__all__ = [
    "CURATED_STARTER_TICKERS",
    "DEFAULT_SP500_CSV_PATH",
    "SP500_UPSTREAM_CSV_URL",
    "format_cik",
    "load_constituents",
    "load_sp500_catalog",
    "normalize_edgar_ticker",
    "refresh_sp500_csv",
]
