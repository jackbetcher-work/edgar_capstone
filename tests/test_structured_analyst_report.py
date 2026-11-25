"""Structured AnalystReport output: claim-level verification, rendering, and guardrail safety."""

from __future__ import annotations

import json
from typing import Any

import pytest

from edgar_capstone.agents.core.schemas import AnalystFinding, AnalystReport
from edgar_capstone.agents.evaluation.structured_report import (
    extract_figures,
    parse_analyst_report,
    render_analyst_report_markdown,
    verify_analyst_report,
)
from edgar_capstone.agents.runtime.callbacks import verify_narrative_against_calculations
from edgar_capstone.agents.tools.variance_engine import _build_variance_reports_from_profiles
from edgar_capstone.dataset.models import FinancialDataProfile
from tests.fake_gcp import use_fake_gcp_adapters
from tests.structured_report_helpers import verbatim_quote


@pytest.fixture(autouse=True)
def _mock_gcp_adapters():
    """Keep `get_full_10k_section()` fallbacks inside `verify_analyst_report()` strictly offline."""
    with use_fake_gcp_adapters():
        yield


# Figures that can't be traced to NVIDIA's payload (its amounts are in the billions) or a quote.
_UNTRACEABLE_AMOUNT = "$3.21 trillion"
_ABSENT_QUOTE = "this sentence was never written in the cited filing passage"


def _body_words(excerpt: str) -> list[str]:
    """Words of a chunk body, skipping its `[TICKER | FY | Section]` header line."""
    return verbatim_quote(excerpt, words=10_000).split()


@pytest.fixture(scope="module")
def live_evidence() -> dict[str, Any]:
    """Deterministic NVDA FY2026 MD&A chunk containing a dollar amount, plus verbatim quotes."""
    excerpt = (
        "[NVDA | FY2026 | Item 7. MD&A] "
        "During fiscal year 2026, Data Center revenue expanded sharply on hyperscale cloud "
        "adoption of Blackwell systems. We incurred a $4.5 billion charge for excess inventory "
        "and purchase obligations related to H20 export license requirements."
    )
    cit = {
        "chunk_id": "NVDA_2026_item_7_0001",
        "ticker": "NVDA",
        "company_name": "NVIDIA CORP",
        "gics_sector": "Information Technology",
        "fiscal_year": 2026,
        "section_id": "item_7",
        "section_title": "Item 7. MD&A",
        "accession_number": "0001045810-26-000021",
        "sec_filing_url": "https://www.sec.gov/Archives/edgar/data/1045810/NVDA_2026_10k.htm",
        "excerpt": excerpt,
        "score": 0.94,
    }
    words = _body_words(excerpt)
    dollar_idx = next(i for i, w in enumerate(words) if w.startswith("$"))
    quote = " ".join(words[max(0, dollar_idx - 6) : dollar_idx + 6])
    return {
        "citation": cit,
        "amount_quote": quote,
        "amount_text": "$4.5 billion",
        "plain_quote": " ".join(words[:14]),
    }


@pytest.fixture(scope="module")
def reports() -> list[dict[str, Any]]:
    profiles = [
        FinancialDataProfile(
            profile_id=f"NVDA_{fy}",
            ticker="NVDA",
            cik="0001045810",
            company_name="NVIDIA CORP",
            gics_sector="Information Technology",
            fiscal_year=fy,
            period_of_report=f"{fy}-01-25",
            filing_date=f"{fy}-02-25",
            accession_number=f"0001045810-{str(fy)[-2:]}-000021",
            revenue=rev,
            operating_income=op,
            net_income=net,
            unstructured_blob_path=f"unstructured/NVDA/{fy}_10k.md",
            structured_blob_path=f"structured/NVDA/{fy}_financials.json",
            unstructured_gcs_uri=f"gs://bucket/unstructured/NVDA/{fy}_10k.md",
            structured_gcs_uri=f"gs://bucket/structured/NVDA/{fy}_financials.json",
            sec_filing_url=f"https://www.sec.gov/Archives/edgar/data/1045810/NVDA_{fy}_10k.htm",
        )
        for fy, rev, op, net in (
            (2024, 60.922e9, 32.972e9, 29.760e9),
            (2025, 130.497e9, 81.453e9, 72.880e9),
            (2026, 215.938e9, 130.387e9, 120.067e9),
        )
    ]
    variance_reports, missing = _build_variance_reports_from_profiles(
        profiles, ticker_filter={"NVDA"}, start_year=2024, end_year=2026
    )
    assert not missing and variance_reports
    return variance_reports


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
    )
    verified = verify_analyst_report(report, variance_reports=reports, citations=citations)
    v = verified.verification
    assert (v.findings_total, v.findings_verified) == (5, 2), v.model_dump()
    assert v.findings_rejected == 3
    assert v.unsupported_claims_count == 0
    reasons = {f.index: " ".join(f.reasons) for f in v.findings if not f.verified}
    assert set(reasons) == {1, 2, 3}
    assert "figures not in the quote" in reasons[1] and _UNTRACEABLE_AMOUNT in reasons[1]
    assert "not found in the cited passages" in reasons[2]
    assert "not retrieved this turn" in reasons[3]
    assert v.usable is True
    assert not hasattr(verified.report, "notes")

    md = render_analyst_report_markdown(
        verified,
        variance_reports=reports,
        citations=citations,
        scope={"tickers_csv": "NVDA"},
        coverage_notices=[
            {
                "code": "fiscal_calendar_alignment",
                "severity": "info",
                "message": "NVDA fiscal year ends in late January.",
            }
        ],
    )
    assert _UNTRACEABLE_AMOUNT not in md and _ABSENT_QUOTE not in md
    assert "#### Notes (not source-verified)" not in md
    assert "> **Note (Coverage):** NVDA fiscal year ends in late January." in md
    tag = (
        f"[{cit['ticker']} FY{cit['fiscal_year']} | {cit['section_title']} | {cit['chunk_id']}]"
        f"({cit['sec_filing_url']})"
    )
    assert tag in md
    assert f'*Evidence:* "{amount_quote}"' in md
    # No forced Key Metrics fallback: the agent selected none, so none are rendered even though
    # the deterministic payload is available.
    assert "#### Key Metrics" not in md
    assert f"{rev_pct:+.2f}%" not in md
    calc_map = {f"{r['ticker']}_{r['start_year']}_{r['end_year']}": r for r in reports}
    check = verify_narrative_against_calculations(
        response_text=md, calculated_variances=calc_map, retrieved_citations=citations
    )
    assert check.passed, check.discrepancies

    # Explicitly selected key metrics render the deterministic figures with period_of_report
    # and XBRL provenance (and only the selected metric).
    selected = verify_analyst_report(
        report.model_copy(update={"key_metrics": [{"ticker": "NVDA", "metric": "revenue"}]}),
        variance_reports=reports,
        citations=citations,
    )
    md_metrics = render_analyst_report_markdown(
        selected, variance_reports=reports, citations=citations, scope={"tickers_csv": "NVDA"}
    )
    assert "#### Key Metrics (deterministic engine)" in md_metrics
    assert f"{rev_pct:+.2f}%" in md_metrics
    assert "XBRL" in md_metrics and "acc:" in md_metrics
    assert "2024-01-" in md_metrics or "2026-01-" in md_metrics
    assert "Net Income:" not in md_metrics and "Operating Income:" not in md_metrics
    check_metrics = verify_narrative_against_calculations(
        response_text=md_metrics, calculated_variances=calc_map, retrieved_citations=citations
    )
    assert check_metrics.passed, check_metrics.discrepancies


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


def test_table_aware_figure_verification_honors_chunk_scale_and_rejects_ungrounded() -> None:
    table_quote = "iPhone | $ | 254,098 | 21 | % | $ | 209,586 | 4 | % | $ | 201,183"
    cit = {
        "chunk_id": "AAPL_2026_item_7_0005",
        "ticker": "AAPL",
        "fiscal_year": 2026,
        "section_id": "item_7",
        "section_title": "Item 7. MD&A",
        "sec_filing_url": "https://www.sec.gov/aapl-2026.htm",
        "excerpt": (f"Net sales by category (dollars in millions):\n| {table_quote} |\n"),
    }
    ok_report = AnalystReport(
        headline="Apple iPhone revenue grew 21% to $254,098 million in FY2026.",
        findings=[
            AnalystFinding(
                ticker="AAPL",
                fiscal_year=2026,
                statement="iPhone net sales reached $254,098 million in FY2026, up 21% from $209,586 million.",
                chunk_ids=["AAPL_2026_item_7_0005"],
                evidence_quote=table_quote,
            )
        ],
    )
    verified = verify_analyst_report(ok_report, variance_reports=[], citations=[cit])
    assert verified.verification.usable is True
    assert verified.verification.findings_verified == 1
    assert verified.verification.findings_rejected == 0
    assert verified.verification.headline_verified is True

    # A fabricated number (not in the table quote) MUST still be rejected
    bad_report = AnalystReport(
        headline="Apple revenue dropped sharply.",
        findings=[
            AnalystFinding(
                ticker="AAPL",
                fiscal_year=2026,
                statement="iPhone net sales dropped to $999,999 million, down 88%.",
                chunk_ids=["AAPL_2026_item_7_0005"],
                evidence_quote=table_quote,
            )
        ],
    )
    bad_verified = verify_analyst_report(bad_report, variance_reports=[], citations=[cit])
    assert bad_verified.verification.usable is False
    assert bad_verified.verification.findings_verified == 0
    assert bad_verified.verification.findings_rejected == 1


def test_figure_extraction_ignores_3m_and_bare_short_suffix_identifiers() -> None:
    """Company names like '3M' and bare identifiers ('10K', '4K', '2B') are not monetary amounts."""
    assert extract_figures("In FY2021, 3M's 10K cited 4K displays and 2B standards.") == []
    figs = extract_figures(
        "3M recorded a $3M adjustment, $4.5B charge, and 12 million units (up 5%)."
    )
    assert [(f.kind, f.value) for f in figs] == [
        ("amount", 3e6),
        ("amount", 4.5e9),
        ("amount", 12e6),
        ("pct", 5.0),
    ]

    cit = {
        "chunk_id": "MMM_2021_item_1a_0004",
        "ticker": "MMM",
        "fiscal_year": 2021,
        "section_id": "item_1a",
        "section_title": "Item 1A. Risk Factors",
        "sec_filing_url": "https://www.sec.gov/mmm-2021.htm",
        "excerpt": (
            "As previously reported, the Company has been voluntarily cooperating with various "
            "local, state, federal, and international agencies reviewing PFAS."
        ),
    }
    report = AnalystReport(
        headline="3M framed PFAS around voluntary regulatory cooperation in FY2021.",
        findings=[
            AnalystFinding(
                ticker="MMM",
                fiscal_year=2021,
                statement="In FY2021, 3M framed PFAS risk around voluntary cooperation with regulators.",
                chunk_ids=["MMM_2021_item_1a_0004"],
                evidence_quote=(
                    "the Company has been voluntarily cooperating with various local, state, "
                    "federal, and international agencies"
                ),
            )
        ],
    )
    verified = verify_analyst_report(report, variance_reports=[], citations=[cit])
    assert verified.verification.findings_verified == 1
    assert verified.verification.findings_rejected == 0
    assert verified.verification.headline_verified is True


def test_quote_normalization_handles_markdown_tables_html_tags_and_vais_chunk_ids() -> None:
    """Markdown table pipes, empty cells, inline HTML tags, and headings do not cause false drops."""
    from edgar_capstone.agents.evaluation.structured_report import _CHUNK_SECTION_RE

    assert _CHUNK_SECTION_RE.match("XOM_2022_item_7_vais_0000") is not None
    assert _CHUNK_SECTION_RE.match("XOM_2022_item_7_vais_0000").group("key") == "XOM_2022_item_7"

    table_cit = {
        "chunk_id": "UAL_2024_item_7_0007",
        "ticker": "UAL",
        "fiscal_year": 2024,
        "section_id": "item_7",
        "section_title": "Item 7. MD&A",
        "sec_filing_url": "https://www.sec.gov/ual-2024.htm",
        "excerpt": (
            "### Results of Operations\n"
            "<u>Table of Contents</u>\n"
            "| (in millions) | 2024 |  | 2023 |  | 2022 |  |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            "| Operating revenue | $ | 57,063 | $ | 53,717 | $ | 44,955 |\n"
            "| Operating income | 5,096 |  | 4,211 |  | 2,337 |  |\n"
            "• Net income rose on stronger passenger demand.\n"
        ),
    }
    report = AnalystReport(
        headline="United Airlines grew operating income from 2022 to 2024.",
        findings=[
            AnalystFinding(
                ticker="UAL",
                fiscal_year=2024,
                statement="United reported higher operating income across 2022-2024.",
                chunk_ids=["UAL_2024_item_7_0007"],
                evidence_quote="Operating income | 5,096 | 4,211 | 2,337",
            ),
            AnalystFinding(
                ticker="UAL",
                fiscal_year=2024,
                statement="United reported higher operating revenue and net income.",
                chunk_ids=["UAL_2024_item_7_0007"],
                evidence_quote=("Operating revenue $57,063 $53,717 $44,955"),
            ),
            AnalystFinding(
                ticker="UAL",
                fiscal_year=2024,
                statement="Passenger demand drove stronger net income.",
                chunk_ids=["UAL_2024_item_7_0007"],
                evidence_quote="Net income rose on stronger passenger demand.",
            ),
            # Fabricated quote must still be rejected
            AnalystFinding(
                ticker="UAL",
                fiscal_year=2024,
                statement="Invented claim.",
                chunk_ids=["UAL_2024_item_7_0007"],
                evidence_quote="Operating income collapsed due to imaginary strikes.",
            ),
        ],
    )
    verified = verify_analyst_report(report, variance_reports=[], citations=[table_cit])
    assert verified.verification.findings_verified == 3
    assert verified.verification.findings_rejected == 1
