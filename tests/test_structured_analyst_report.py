"""Structured AnalystReport output: claim-level verification, rendering, and guardrail safety.

Verification runs against a live NVIDIA FY2026 Item 7 chunk from BigQuery `filing_chunks`
(found via the live hybrid search) and the deterministic variance engine's output for the
same live `financial_metrics` rows. Rejected findings are built by perturbing that live
evidence (wrong figure, wrong chunk id, quote absent from the passage).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from edgar_capstone.agents import calculate_period_variances
from edgar_capstone.agents.callbacks import verify_narrative_against_calculations
from edgar_capstone.agents.schemas import AnalystReport
from edgar_capstone.agents.structured_report import (
    extract_figures,
    parse_analyst_report,
    render_analyst_report_markdown,
    verify_analyst_report,
)
from edgar_capstone.agents.tools import search_10k_chunks_rag
from edgar_capstone.security.model_armor import schema_safe_block_text
from tests.structured_report_helpers import verbatim_quote

# Figures that can't be traced to NVIDIA's payload (its amounts are in the billions) or a quote.
_UNTRACEABLE_AMOUNT = "$3.21 trillion"
_ABSENT_QUOTE = "this sentence was never written in the cited filing passage"


def _body_words(excerpt: str) -> list[str]:
    """Words of a live chunk body, skipping its `[TICKER | FY | Section]` header line."""
    return verbatim_quote(excerpt, words=10_000).split()


@pytest.fixture(scope="module")
def live_evidence() -> dict[str, Any]:
    """A live NVDA FY2026 MD&A chunk containing a dollar amount, plus a verbatim quote of it."""
    res = search_10k_chunks_rag(
        query="charge for excess inventory and purchase obligations export license billion",
        tickers="NVDA",
        gics_sector="ALL",
        section_id="item_7",
        start_year=2026,
        end_year=2026,
        top_k=8,
    )
    assert res["status"] == "success" and res["citations"], res
    for cit in res["citations"]:
        words = _body_words(cit["excerpt"])
        for i, word in enumerate(words):
            if not word.startswith("$"):
                continue
            quote = " ".join(words[max(0, i - 6) : i + 6])
            amounts = [f for f in extract_figures(quote) if f.kind == "amount" and "$" in f.text]
            if amounts and len(quote) >= 40:
                return {
                    "citation": cit,
                    "amount_quote": quote,
                    "amount_text": amounts[0].text,
                    "plain_quote": " ".join(words[:14]),
                }
    pytest.fail("No live NVDA FY2026 Item 7 chunk with a dollar amount was retrieved")


@pytest.fixture(scope="module")
def reports() -> list[dict[str, Any]]:
    res = calculate_period_variances(tickers="NVDA", start_year=2024, end_year=2026)
    assert res["status"] == "success" and res["variance_reports"], res
    return res["variance_reports"]


def _finding(citation: dict[str, Any], statement: str, quote: str, chunk_id: str | None = None):
    return {
        "statement": statement,
        "ticker": citation["ticker"],
        "fiscal_year": citation["fiscal_year"],
        "chunk_ids": [chunk_id or citation["chunk_id"]],
        "evidence_quote": quote,
    }


def test_figure_extraction_ignores_identifiers_and_years() -> None:
    figs = extract_figures(
        "FY2026 H20 sales fell 41% after a $4.5 billion charge, down 120 bps in Q1."
    )
    assert [(f.kind, f.value) for f in figs] == [("pct", 41.0), ("amount", 4.5e9), ("bps", 120.0)]
    # Markdown-escaped corpus text ("$4\\.5 billion") parses to the same figure
    assert [(f.kind, f.value) for f in extract_figures("we incurred a $4\\.5 billion charge")] == [
        ("amount", 4.5e9)
    ]


def test_findings_are_kept_only_when_quote_chunk_and_figures_verify(
    live_evidence: dict[str, Any], reports: list[dict[str, Any]]
) -> None:
    cit = live_evidence["citation"]
    citations = [cit]
    amount_quote = live_evidence["amount_quote"]
    amount_text = live_evidence["amount_text"]
    rev_pct = reports[0]["revenue_variance"]["percentage_delta"]
    unretrieved_id = f"{cit['chunk_id'].rsplit('_', 1)[0]}_9999"

    report = AnalystReport(
        headline="NVIDIA's FY2026 MD&A explains the drivers behind its revenue change.",
        findings=[
            # verified: quote is in the live chunk and the figure appears in the quote
            _finding(cit, f"The filing discloses a {amount_text} item.", amount_quote),
            # rejected: figure appears in neither the quote nor the deterministic payload
            _finding(cit, f"The filing discloses a {_UNTRACEABLE_AMOUNT} item.", amount_quote),
            # rejected: quote does not appear in the cited passage (or its section)
            _finding(cit, "The filing discusses demand.", _ABSENT_QUOTE),
            # rejected: chunk not retrieved this turn
            _finding(cit, "The filing discusses demand.", amount_quote, chunk_id=unretrieved_id),
            # verified: the deterministic payload percentage is allowed in the statement
            _finding(
                cit,
                f"Revenue changed {rev_pct:.2f}% over the period.",
                live_evidence["plain_quote"],
            ),
        ],
        notes=(
            "Management discussed export licensing in the filing. "
            f"A {_UNTRACEABLE_AMOUNT} rebound is expected next year."
        ),
    )
    verified = verify_analyst_report(report, variance_reports=reports, citations=citations)
    v = verified.verification
    assert (v.findings_total, v.findings_verified) == (5, 2), v.model_dump()
    reasons = {f.index: " ".join(f.reasons) for f in v.findings if not f.verified}
    assert set(reasons) == {1, 2, 3}
    assert "figures not in the quote" in reasons[1] and _UNTRACEABLE_AMOUNT in reasons[1]
    assert "not found in the cited passages" in reasons[2]
    assert "not retrieved this turn" in reasons[3]
    assert v.usable is True
    # The unsupported figure in the notes is removed; the qualitative sentence stays
    assert v.notes_removed_sentences == [f"A {_UNTRACEABLE_AMOUNT} rebound is expected next year."]
    assert "export licensing" in verified.report.notes

    md = render_analyst_report_markdown(
        verified, variance_reports=reports, citations=citations, scope={"tickers_csv": "NVDA"}
    )
    assert _UNTRACEABLE_AMOUNT not in md and _ABSENT_QUOTE not in md
    tag = (
        f"[{cit['ticker']} FY{cit['fiscal_year']} | {cit['section_title']} | {cit['chunk_id']}]"
        f"({cit['sec_filing_url']})"
    )
    assert tag in md
    assert f'*Evidence:* "{amount_quote}"' in md
    # Key metrics default to the deterministic figures, so the numeric fact-check passes
    assert f"{rev_pct:+.2f}%" in md
    assert "#### Notes (not source-verified)" in md
    calc_map = {f"{r['ticker']}_{r['start_year']}_{r['end_year']}": r for r in reports}
    check = verify_narrative_against_calculations(
        response_text=md, calculated_variances=calc_map, retrieved_citations=citations
    )
    assert check.passed, check.discrepancies


def test_unverifiable_headline_is_replaced_and_all_rejected_findings_mark_unusable(
    live_evidence: dict[str, Any], reports: list[dict[str, Any]]
) -> None:
    cit = live_evidence["citation"]
    bad = AnalystReport(
        headline="NVIDIA will reach a $5 trillion valuation.",
        findings=[_finding(cit, "Sales tripled.", _ABSENT_QUOTE)],
    )
    verified = verify_analyst_report(bad, variance_reports=reports, citations=[cit])
    assert verified.verification.headline_verified is False
    assert verified.verification.usable is False
    assert "could be verified" in str(verified.verification.unusable_reason)
    md = render_analyst_report_markdown(
        verified,
        variance_reports=reports,
        citations=[cit],
        scope={"tickers_csv": "NVDA", "start_year": 2024, "end_year": 2026},
    )
    assert "$5 trillion" not in md
    assert md.startswith("### Grounded analysis for NVDA, FY2024–FY2026")


def test_report_is_trimmed_and_parsed_from_adk_state(
    live_evidence: dict[str, Any], reports: list[dict[str, Any]]
) -> None:
    cit = live_evidence["citation"]
    raw = {
        "headline": "Short answer.",
        "key_metrics": [
            {"ticker": "NVDA", "metric": "revenue"},
            {"ticker": "ZZZZ", "metric": "net_income"},
        ],
        "findings": [_finding(cit, "The filing discusses demand.", live_evidence["plain_quote"])]
        * 9,
        "notes": "",
    }
    report = parse_analyst_report(json.dumps(raw))
    assert report is not None and parse_analyst_report(raw) == report
    assert parse_analyst_report("not json") is None
    verified = verify_analyst_report(report, variance_reports=reports, citations=[cit])
    assert verified.verification.findings_total == 6  # capped
    assert verified.verification.findings_verified == 6
    assert verified.verification.metrics_dropped == ["ZZZZ:net_income"]


def test_guardrail_block_keeps_structured_output_parseable() -> None:
    original = AnalystReport(headline="x", notes="price target of $260").model_dump_json()
    replaced = schema_safe_block_text(original, "Blocked by policy.")
    assert AnalystReport.model_validate_json(replaced).headline == "Blocked by policy."
    assert schema_safe_block_text("plain markdown", "Blocked by policy.") == "Blocked by policy."
