"""SEC XBRL `companyfacts` lookups used to backfill annual metrics onto staged 10-K profiles.

`extract_company_10k_bundles` reads metrics from each filing's income statement while it parses
the 10-K. Re-parsing all ~3,000 staged filings just to add one metric would take hours, so this
module reads the same as-filed XBRL facts from SEC's `companyfacts` API instead: one request per
company returns every fact it has ever filed, tagged with the accession number of the filing.

Facts are matched to a profile by accession number and reporting period, so the backfilled value
is the one the issuer reported in that 10-K. Companies that don't report the concept get `None`.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from edgar_capstone.dataset.extractor import (
    DEFAULT_SEC_USER_AGENT,
    RESEARCH_AND_DEVELOPMENT_CONCEPTS,
)
from edgar_capstone.dataset.models import FinancialDataProfile

logger = logging.getLogger(__name__)

SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# SEC fair-access policy allows 10 requests/second; stay comfortably below it.
DEFAULT_REQUESTS_PER_SECOND = 8.0
_ANNUAL_DURATION_DAYS = (300, 400)
_PERIOD_END_TOLERANCE_DAYS = 10
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

RD_PROVENANCE_PREFIX = "sec-companyfacts:"


class _RateLimiter:
    """Thread-safe minimum-interval limiter."""

    def __init__(self, requests_per_second: float) -> None:
        self._interval = 1.0 / max(0.1, requests_per_second)
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next_at - now
            self._next_at = max(now, self._next_at) + self._interval
        if delay > 0:
            time.sleep(delay)


def fetch_company_facts(
    cik: str,
    *,
    client: httpx.Client,
    user_agent: str = DEFAULT_SEC_USER_AGENT,
    rate_limiter: _RateLimiter | None = None,
    max_attempts: int = 4,
) -> dict[str, Any] | None:
    """Fetch SEC `companyfacts` JSON for a CIK. Returns None when SEC has no XBRL facts (404)."""
    cik_10 = str(int(str(cik).strip())).zfill(10)
    url = SEC_COMPANYFACTS_URL.format(cik=cik_10)
    headers = {"User-Agent": user_agent, "Accept": "application/json"}
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        if rate_limiter is not None:
            rate_limiter.wait()
        try:
            resp = client.get(url, headers=headers, timeout=60.0)
        except httpx.HTTPError as exc:
            last_error = exc
        else:
            if resp.status_code == 404:
                return None
            if resp.status_code not in _RETRYABLE_STATUS:
                resp.raise_for_status()
                return resp.json()
            last_error = httpx.HTTPStatusError(
                f"SEC companyfacts HTTP {resp.status_code} for CIK {cik_10}",
                request=resp.request,
                response=resp,
            )
        if attempt < max_attempts:
            time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
    assert last_error is not None
    raise last_error


def _parse_date(raw: Any) -> date | None:
    try:
        return date.fromisoformat(str(raw)[:10])
    except (TypeError, ValueError):
        return None


def _annual_facts_for_period(
    facts: Iterable[dict[str, Any]],
    period_end: date,
) -> list[dict[str, Any]]:
    """Keep full-year duration facts from 10-K filings whose period ends at `period_end`."""
    low, high = _ANNUAL_DURATION_DAYS
    matched: list[dict[str, Any]] = []
    for fact in facts:
        form = str(fact.get("form") or "").upper()
        if not form.startswith("10-K"):
            continue
        start, end = _parse_date(fact.get("start")), _parse_date(fact.get("end"))
        if start is None or end is None:
            continue
        if not (low <= (end - start).days <= high):
            continue
        if abs((end - period_end).days) > _PERIOD_END_TOLERANCE_DAYS:
            continue
        if not isinstance(fact.get("val"), int | float):
            continue
        matched.append(fact)
    return matched


def select_annual_usd_fact(
    company_facts: dict[str, Any] | None,
    concepts: tuple[str, ...],
    *,
    accession_number: str,
    period_of_report: str,
) -> tuple[float | None, str | None, str | None]:
    """Pick the annual USD value a 10-K reported for the first matching concept.

    Concepts use the extractor's `us-gaap_Name` form. Pass 1 only accepts facts filed in the
    profile's own 10-K (`accn == accession_number`); pass 2 falls back to the same fiscal period
    as reported in a later 10-K's comparative column, earliest filing first.

    Returns `(value, provenance, note)`; `(None, None, None)` when no concept matches.
    """
    period_end = _parse_date(period_of_report)
    if not company_facts or period_end is None:
        return None, None, None

    taxonomy_facts: dict[str, Any] = company_facts.get("facts") or {}
    candidates_by_concept: list[tuple[str, list[dict[str, Any]]]] = []
    for concept in concepts:
        prefix, _, name = concept.partition("_")
        usd_facts = ((taxonomy_facts.get(prefix) or {}).get(name) or {}).get("units", {}).get(
            "USD"
        ) or []
        annual = _annual_facts_for_period(usd_facts, period_end)
        if annual:
            candidates_by_concept.append((concept, annual))

    target_accn = accession_number.strip()
    for concept, annual in candidates_by_concept:
        same_filing = [f for f in annual if str(f.get("accn") or "").strip() == target_accn]
        if same_filing:
            tag = concept.replace("_", ":", 1)
            return float(same_filing[0]["val"]), f"{RD_PROVENANCE_PREFIX}{tag}", None

    for concept, annual in candidates_by_concept:
        later = sorted(annual, key=lambda f: str(f.get("filed") or ""))
        fact = later[0]
        tag = concept.replace("_", ":", 1)
        note = (
            f"{tag} for period ending {period_of_report} was not tagged in 10-K "
            f"{target_accn}; value taken from 10-K {fact.get('accn')} (filed {fact.get('filed')})."
        )
        return float(fact["val"]), f"{RD_PROVENANCE_PREFIX}{tag}", note

    return None, None, None


@dataclass
class MetricBackfillSummary:
    """Outcome of `backfill_research_and_development`."""

    profiles_scanned: int = 0
    profiles_updated: int = 0
    profiles_with_value: int = 0
    profiles_without_value: int = 0
    companies_queried: int = 0
    companies_without_facts: list[str] = field(default_factory=list)
    companies_failed: dict[str, str] = field(default_factory=dict)
    updated_paths: list[str] = field(default_factory=list)
    scanned_paths: list[str] = field(default_factory=list)


def _apply_rd_value(
    profile: FinancialDataProfile,
    value: float | None,
    provenance: str | None,
    note: str | None,
) -> bool:
    """Set R&D fields on `profile`; return True when anything changed."""
    prov = profile.metric_provenance
    notes = [n for n in prov.notes if not n.startswith("R&D: ")]
    if note:
        notes.append(f"R&D: {note}")
    changed = (
        profile.research_and_development != value
        or prov.research_and_development != provenance
        or notes != prov.notes
    )
    profile.research_and_development = value
    prov.research_and_development = provenance
    prov.notes = notes
    return changed


def backfill_research_and_development(
    output_dir: Path,
    *,
    tickers: Iterable[str] | None = None,
    user_agent: str = DEFAULT_SEC_USER_AGENT,
    requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND,
    client: httpx.Client | None = None,
    overwrite_extracted: bool = False,
) -> MetricBackfillSummary:
    """Backfill `research_and_development` onto staged `structured/<T>/<FY>_financials.json`.

    Profiles whose R&D value came from the filing itself (provenance not prefixed with
    `sec-companyfacts:`) are left alone unless `overwrite_extracted` is set.
    """
    struct_root = output_dir / "structured"
    wanted = {t.strip().upper() for t in tickers or [] if t.strip()}
    summary = MetricBackfillSummary()

    by_cik: dict[str, list[tuple[Path, FinancialDataProfile]]] = defaultdict(list)
    for json_path in sorted(struct_root.rglob("*_financials.json")):
        profile = FinancialDataProfile.model_validate_json(json_path.read_text(encoding="utf-8"))
        if wanted and profile.ticker.upper() not in wanted:
            continue
        summary.profiles_scanned += 1
        summary.scanned_paths.append(json_path.relative_to(output_dir).as_posix())
        by_cik[profile.cik].append((json_path, profile))

    limiter = _RateLimiter(requests_per_second)
    owns_client = client is None
    http = client or httpx.Client(follow_redirects=True)
    try:
        for cik, entries in sorted(by_cik.items()):
            ticker = entries[0][1].ticker
            summary.companies_queried += 1
            try:
                facts = fetch_company_facts(
                    cik, client=http, user_agent=user_agent, rate_limiter=limiter
                )
            except Exception as exc:
                logger.warning("[%s] SEC companyfacts lookup failed: %s", ticker, exc)
                summary.companies_failed[ticker] = str(exc)
                continue
            if facts is None:
                summary.companies_without_facts.append(ticker)

            for json_path, profile in entries:
                existing = profile.metric_provenance.research_and_development
                extracted_in_filing = (
                    profile.research_and_development is not None
                    and existing is not None
                    and not existing.startswith(RD_PROVENANCE_PREFIX)
                )
                if extracted_in_filing and not overwrite_extracted:
                    summary.profiles_with_value += 1
                    continue

                value, provenance, note = select_annual_usd_fact(
                    facts,
                    RESEARCH_AND_DEVELOPMENT_CONCEPTS,
                    accession_number=profile.accession_number,
                    period_of_report=profile.period_of_report,
                )
                if value is None:
                    summary.profiles_without_value += 1
                else:
                    summary.profiles_with_value += 1
                if _apply_rd_value(profile, value, provenance, note):
                    json_path.write_text(
                        json.dumps(profile.model_dump(mode="json"), indent=2) + "\n",
                        encoding="utf-8",
                    )
                    summary.profiles_updated += 1
                    summary.updated_paths.append(json_path.relative_to(output_dir).as_posix())
    finally:
        if owns_client:
            http.close()

    logger.info(
        "R&D backfill: %d profiles scanned, %d with R&D, %d without, %d updated, %d companies failed.",
        summary.profiles_scanned,
        summary.profiles_with_value,
        summary.profiles_without_value,
        summary.profiles_updated,
        len(summary.companies_failed),
    )
    return summary
