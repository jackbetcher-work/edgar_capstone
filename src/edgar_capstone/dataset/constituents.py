"""S&P 500 constituent metadata loader and curated starter company set."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import httpx

from edgar_capstone.dataset.models import CompanyConstituent

logger = logging.getLogger(__name__)

SP500_UPSTREAM_CSV_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
)

DEFAULT_SP500_CSV_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "sp500_constituents.csv"
)

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


def refresh_sp500_csv(csv_path: Path = DEFAULT_SP500_CSV_PATH, timeout: float = 20.0) -> Path:
    """Fetch the latest S&P 500 constituents CSV and persist it locally."""
    logger.info("Refreshing S&P 500 constituents CSV from %s", SP500_UPSTREAM_CSV_URL)
    response = httpx.get(SP500_UPSTREAM_CSV_URL, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(response.text, encoding="utf-8")
    return csv_path


def load_sp500_catalog(
    csv_path: Path = DEFAULT_SP500_CSV_PATH,
    refresh: bool = False,
) -> dict[str, CompanyConstituent]:
    """Load the S&P 500 constituent catalog keyed by normalized SEC EDGAR ticker."""
    if refresh or not csv_path.exists():
        refresh_sp500_csv(csv_path=csv_path)

    catalog: dict[str, CompanyConstituent] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
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
    sp500_csv_path: Path = DEFAULT_SP500_CSV_PATH,
    refresh_sp500: bool = False,
) -> list[CompanyConstituent]:
    """Resolve the target list of companies for dataset curation.

    Priority:
    1. Explicit `tickers` list if provided.
    2. Full S&P 500 catalog if `full_sp500=True`.
    3. Default `CURATED_STARTER_TICKERS` subset.
    """
    catalog: dict[str, CompanyConstituent] = {}
    if sp500_csv_path.exists() or full_sp500 or refresh_sp500:
        try:
            catalog = load_sp500_catalog(csv_path=sp500_csv_path, refresh=refresh_sp500)
        except Exception as exc:
            if full_sp500:
                raise RuntimeError(
                    f"Unable to load S&P 500 catalog from {sp500_csv_path}: {exc}"
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
        resolved = [
            comp for comp in resolved if comp.gics_sector.strip().lower() == needle
        ]

    return resolved
