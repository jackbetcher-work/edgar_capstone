"""Structured AnalystReport output: claim-level verification, rendering, and guardrail safety."""

from __future__ import annotations

import json

from edgar_capstone.agents import calculate_period_variances
from edgar_capstone.agents.callbacks import verify_narrative_against_calculations
from edgar_capstone.agents.schemas import AnalystReport
from edgar_capstone.agents.structured_report import (
    extract_figures,
    parse_analyst_report,
    render_analyst_report_markdown,
    verify_analyst_report,
)
from edgar_capstone.security.model_armor import schema_safe_block_text

CHUNK_TEXT = (
    "Revenue from the Data Center segment was driven by strong demand for our Hopper and "
    "Blackwell platforms. In the first quarter, we recorded a $1.2 billion charge for excess "
    "inventory associated with export restrictions on shipments to China."
)
CITATIONS = [
    {
        "chunk_id": "NVDA_2026_item_7_0042",
        "ticker": "NVDA",
        "fiscal_year": 2026,
        "section_id": "item_7",
        "section_title": "Item 7. MD&A",
        "sec_filing_url": "https://www.sec.gov/nvda-2026",
        "excerpt": CHUNK_TEXT,
    }
]


def _reports() -> list[dict]:
    return calculate_period_variances(tickers="NVDA", start_year=2024, end_year=2026)[
        "variance_reports"
    ]


def _finding(statement: str, quote: str, chunk_id: str = "NVDA_2026_item_7_0042") -> dict:
    return {
        "statement": statement,
        "ticker": "NVDA",
        "fiscal_year": 2026,
        "chunk_ids": [chunk_id],
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


def test_findings_are_kept_only_when_quote_chunk_and_figures_verify() -> None:
    reports = _reports()
    rev_pct = reports[0]["revenue_variance"]["percentage_delta"]
    report = AnalystReport(
        headline="Data Center demand drove NVIDIA's growth while export rules cost inventory.",
        findings=[
            # verified: quote in chunk, figure appears in quote
            _finding(
                "Export restrictions forced a $1.2 billion inventory charge.",
                "we recorded a $1.2 billion charge for excess inventory",
            ),
            # rejected: outside-corpus figure (the live '$4.5 billion' failure mode)
            _finding(
                "Export controls caused a $4.5 billion H20 charge.",
                "we recorded a $1.2 billion charge for excess inventory",
            ),
            # rejected: quote not in the cited passage
            _finding("Gaming drove growth.", "Gaming revenue more than doubled on RTX 50 demand"),
            # rejected: chunk not retrieved this turn
            _finding(
                "Hopper demand drove revenue.",
                "driven by strong demand for our Hopper and Blackwell platforms",
                chunk_id="NVDA_2026_item_7_9999",
            ),
            # verified: deterministic payload percentage is allowed in the statement
            _finding(
                f"Revenue rose {rev_pct:.2f}% on Hopper and Blackwell demand.",
                "driven by strong demand for our Hopper and Blackwell platforms",
            ),
        ],
        notes=(
            "Wording on China shifted from licensing to outright restrictions. "
            "Management expects a $9 billion rebound next year."
        ),
    )
    verified = verify_analyst_report(report, variance_reports=reports, citations=CITATIONS)
    v = verified.verification
    assert (v.findings_total, v.findings_verified) == (5, 2)
    reasons = {f.index: " ".join(f.reasons) for f in v.findings if not f.verified}
    assert "figures not in the quote" in reasons[1] and "$4.5 billion" in reasons[1]
    assert "not found in the cited passages" in reasons[2]
    assert "not retrieved this turn" in reasons[3]
    assert v.usable is True
    # Unsupported figure in notes is removed; the qualitative sentence stays
    assert v.notes_removed_sentences == ["Management expects a $9 billion rebound next year."]
    assert "licensing to outright restrictions" in verified.report.notes

    md = render_analyst_report_markdown(
        verified, variance_reports=reports, citations=CITATIONS, scope={"tickers_csv": "NVDA"}
    )
    assert "$4.5 billion" not in md and "$9 billion" not in md and "Gaming drove" not in md
    assert (
        "[NVDA FY2026 | Item 7. MD&A | NVDA_2026_item_7_0042](https://www.sec.gov/nvda-2026)" in md
    )
    assert '*Evidence:* "we recorded a $1.2 billion charge for excess inventory"' in md
    # Key metrics default to the deterministic figures, so the numeric fact-check passes
    assert f"{rev_pct:+.2f}%" in md
    assert "#### Notes (not source-verified)" in md
    calc_map = {f"{r['ticker']}_{r['start_year']}_{r['end_year']}": r for r in reports}
    check = verify_narrative_against_calculations(
        response_text=md, calculated_variances=calc_map, retrieved_citations=CITATIONS
    )
    assert check.passed, check.discrepancies


def test_unverifiable_headline_is_replaced_and_all_rejected_findings_mark_unusable() -> None:
    reports = _reports()
    bad = AnalystReport(
        headline="NVIDIA will reach a $5 trillion valuation.",
        findings=[_finding("Sales tripled.", "this sentence is not in the passage at all")],
    )
    verified = verify_analyst_report(bad, variance_reports=reports, citations=CITATIONS)
    assert verified.verification.headline_verified is False
    assert verified.verification.usable is False
    assert "could be verified" in str(verified.verification.unusable_reason)
    md = render_analyst_report_markdown(
        verified,
        variance_reports=reports,
        citations=CITATIONS,
        scope={"tickers_csv": "NVDA", "start_year": 2024, "end_year": 2026},
    )
    assert "$5 trillion" not in md
    assert md.startswith("### Grounded analysis for NVDA, FY2024–FY2026")


def test_report_is_trimmed_and_parsed_from_adk_state() -> None:
    raw = {
        "headline": "Short answer.",
        "key_metrics": [
            {"ticker": "NVDA", "metric": "revenue"},
            {"ticker": "ZZZZ", "metric": "net_income"},
        ],
        "findings": [
            _finding(
                "Hopper demand drove revenue.",
                "driven by strong demand for our Hopper and Blackwell platforms",
            )
        ]
        * 9,
        "notes": "",
    }
    report = parse_analyst_report(json.dumps(raw))
    assert report is not None and parse_analyst_report(raw) == report
    assert parse_analyst_report("not json") is None
    verified = verify_analyst_report(report, variance_reports=_reports(), citations=CITATIONS)
    assert verified.verification.findings_total == 6  # capped
    assert verified.verification.metrics_dropped == ["ZZZZ:net_income"]


def test_guardrail_block_keeps_structured_output_parseable() -> None:
    original = AnalystReport(headline="x", notes="price target of $260").model_dump_json()
    replaced = schema_safe_block_text(original, "Blocked by policy.")
    assert AnalystReport.model_validate_json(replaced).headline == "Blocked by policy."
    assert schema_safe_block_text("plain markdown", "Blocked by policy.") == "Blocked by policy."
