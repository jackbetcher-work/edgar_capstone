"""Deterministic anomaly flagging (`flag_material_variances`) and the Disclosure Delta pipeline
(`compare_disclosures_across_years`): rules, alignment, rendering, and chat-turn integration."""

from __future__ import annotations

import re
from typing import Any

import pytest

from edgar_capstone.agents import create_financial_analyst_agent
from edgar_capstone.agents.chat_engine import run_conversational_rag_turn
from edgar_capstone.agents.disclosure_deltas import (
    SectionMaterial,
    align_year_pair,
    build_company_evolution,
    compare_disclosures_across_years,
    count_theme_mentions,
    extract_theme_terms,
    strip_chunk_header,
    wording_changes,
)
from edgar_capstone.agents.live_runtime import _build_financial_reasoning_prompt
from edgar_capstone.agents.material_variances import (
    MaterialityThresholds,
    compute_material_variance_flags,
    flag_material_variances,
)
from edgar_capstone.agents.schemas import AnalystReport
from edgar_capstone.agents.structured_report import verify_analyst_report
from edgar_capstone.agents.tools import (
    _build_variance_reports_from_profiles,
    calculate_period_variances,
)
from edgar_capstone.dataset.models import FinancialDataProfile
from edgar_capstone.security.agent_identity import authorize_agent_tool_invocation

_CITATION_RE = re.compile(r"\[[^\]|]+\|[^\]|]+\|\s*([^\]|]+?)\s*\]\(")


def _profile(
    ticker: str,
    fy: int,
    *,
    revenue: float | None,
    operating_income: float | None,
    net_income: float | None,
    rd: float | None = None,
    sector: str = "Information Technology",
) -> FinancialDataProfile:
    return FinancialDataProfile(
        profile_id=f"{ticker}_{fy}",
        ticker=ticker,
        cik="0000000001",
        company_name=f"{ticker} Corp",
        gics_sector=sector,
        fiscal_year=fy,
        period_of_report=f"{fy}-12-31",
        filing_date=f"{fy + 1}-02-15",
        accession_number=f"0000000001-{(fy + 1) % 100:02d}-000001",
        revenue=revenue,
        operating_income=operating_income,
        net_income=net_income,
        research_and_development=rd,
        metric_provenance={
            "revenue": "us-gaap:Revenues",
            "operating_income": "us-gaap:OperatingIncomeLoss",
            "net_income": "us-gaap:NetIncomeLoss",
            "research_and_development": "us-gaap:ResearchAndDevelopmentExpense",
        },
        unstructured_blob_path=f"unstructured/{ticker}/{fy}_10k.md",
        structured_blob_path=f"structured/{ticker}/{fy}_financials.json",
        unstructured_gcs_uri=f"gs://bucket/unstructured/{ticker}/{fy}_10k.md",
        structured_gcs_uri=f"gs://bucket/structured/{ticker}/{fy}_financials.json",
        sec_filing_url=f"https://www.sec.gov/{ticker}/{fy}.htm",
    )


def _flags_by_id(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {f["flag_id"]: f for f in result["flags"]}


# ---------------------------------------------------------------------------
# Material variance flags
# ---------------------------------------------------------------------------


def test_yoy_sign_reversal_and_margin_rules_fire_with_provenance() -> None:
    profiles = [
        _profile("ACME", 2023, revenue=100e9, operating_income=20e9, net_income=15e9, rd=10e9),
        # +40% revenue, operating income flat-ish -> margin contracts 571 bps
        _profile("ACME", 2024, revenue=140e9, operating_income=20e9, net_income=16e9, rd=11e9),
        # Net income swings to a loss
        _profile("ACME", 2025, revenue=141e9, operating_income=21e9, net_income=-5e9, rd=11.5e9),
    ]
    result = compute_material_variance_flags(
        profiles, tickers={"ACME"}, start_year=2024, end_year=2025, max_flags=None
    )
    flags = _flags_by_id(result)

    rev = flags["ACME_2024_revenue_yoy_change"]
    assert rev["value"] == 40.0 and rev["unit"] == "pct" and rev["severity"] == "medium"
    assert rev["source"]["accession_number"] == "0000000001-25-000001"
    assert rev["source"]["prior_accession_number"] == "0000000001-24-000001"
    assert rev["source"]["xbrl_concept"] == "us-gaap:Revenues"

    reversal = flags["ACME_2025_net_income_sign_reversal"]
    assert reversal["severity"] == "high" and reversal["rule"] == "sign_reversal"
    assert "profit to a loss" in reversal["description"]
    # A sign reversal replaces the (meaningless) percentage flag for the same metric/year.
    assert "ACME_2025_net_income_yoy_change" not in flags

    margin = flags["ACME_2024_operating_margin_margin_shift"]
    assert margin["unit"] == "bps" and margin["value"] == pytest.approx(-571.0, abs=0.1)
    assert margin["prior_value"] == 20.0 and margin["current_value"] == 14.29

    # Ranked: high severity first
    assert result["flags"][0]["severity"] == "high"
    assert result["thresholds"]["yoy_pct"] == 25.0


def test_flag_percentages_match_the_variance_engine_payload() -> None:
    profiles = [
        _profile("ACME", 2023, revenue=61.0e9, operating_income=4.2e9, net_income=4.4e9),
        _profile("ACME", 2024, revenue=130.5e9, operating_income=81.5e9, net_income=72.9e9),
    ]
    flags = compute_material_variance_flags(
        profiles, tickers={"ACME"}, start_year=2024, end_year=2024
    )["flags"]
    reports, _ = _build_variance_reports_from_profiles(
        profiles, ticker_filter={"ACME"}, start_year=2024, end_year=2024
    )
    step = reports[0]["annual_series"][0]
    by_metric = {f["metric"]: f for f in flags if f["rule"] == "yoy_change"}
    assert by_metric["revenue"]["value"] == step["revenue_yoy_pct"]
    assert by_metric["operating_income"]["value"] == step["operating_income_yoy_pct"]


def test_tiny_profit_base_is_not_flagged_but_period_change_is() -> None:
    profiles = [
        _profile("ACME", 2023, revenue=100e9, operating_income=0.10e9, net_income=10e9),
        # Operating income x5 but the change ($0.4B) is < 1% of revenue -> suppressed
        _profile("ACME", 2024, revenue=110e9, operating_income=0.50e9, net_income=11e9),
        _profile("ACME", 2025, revenue=130e9, operating_income=0.55e9, net_income=12e9),
        _profile("ACME", 2026, revenue=180e9, operating_income=0.60e9, net_income=13e9),
    ]
    result = compute_material_variance_flags(
        profiles, tickers={"ACME"}, start_year=2024, end_year=2026, max_flags=None
    )
    flags = _flags_by_id(result)
    assert "ACME_2024_operating_income_yoy_change" not in flags
    assert "ACME_2024_2026_operating_income_period_change" not in flags
    period = flags["ACME_2024_2026_revenue_period_change"]
    assert period["value"] == 63.64 and period["prior_fiscal_year"] == 2024
    assert period["fiscal_year"] == 2026


def test_sector_divergence_uses_median_of_peers() -> None:
    profiles = [
        _profile("ACME", 2023, revenue=100e9, operating_income=20e9, net_income=15e9),
        _profile("ACME", 2024, revenue=180e9, operating_income=36e9, net_income=27e9),
    ]
    peers = list(profiles)
    for i in range(6):
        peers.append(_profile(f"P{i}", 2023, revenue=10e9, operating_income=2e9, net_income=1e9))
        peers.append(
            _profile(f"P{i}", 2024, revenue=11e9, operating_income=2.2e9, net_income=1.1e9)
        )
    result = compute_material_variance_flags(
        profiles,
        tickers={"ACME"},
        start_year=2024,
        end_year=2024,
        sector_profiles=peers,
        max_flags=None,
    )
    div = _flags_by_id(result)["ACME_2024_revenue_sector_divergence"]
    assert div["benchmark_value"] == 10.0 and div["peer_count"] == 6
    assert div["value"] == 70.0 and div["unit"] == "pp" and div["severity"] == "high"


def test_thresholds_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDGAR_MATERIALITY_YOY_PCT", "50")
    monkeypatch.setenv("EDGAR_MATERIALITY_MARGIN_BPS", "not-a-number")
    thr = MaterialityThresholds.from_env()
    assert thr.yoy_pct == 50.0
    assert thr.margin_bps == 500.0


def test_flag_tool_reads_bigquery_and_matches_the_variance_engine() -> None:
    res = flag_material_variances(tickers="NVDA,ZZZZ", start_year=2024, end_year=2026)
    assert res["status"] == "success"
    assert res["missing_tickers"] == ["ZZZZ"]
    assert res["returned_flags"] <= 10 and res["total_flags"] >= res["returned_flags"]
    variance = calculate_period_variances(tickers="NVDA", start_year=2024, end_year=2026)
    yoy_pcts = {
        (step["fiscal_year"], metric): step[f"{metric}_yoy_pct"]
        for rep in variance["variance_reports"]
        for step in rep["annual_series"]
        for metric in ("revenue", "operating_income", "net_income", "research_and_development")
    }
    for flag in res["flags"]:
        assert flag["ticker"] == "NVDA"
        assert flag["source"]["accession_number"]
        assert 2024 <= flag["fiscal_year"] <= 2026
        if flag["rule"] == "yoy_change":
            # Flags are computed from the same BigQuery rows as the variance engine.
            assert flag["value"] == yoy_pcts[(flag["fiscal_year"], flag["metric"])]
    assert flag_material_variances(tickers="", start_year=2024, end_year=2026)["status"] == "error"


def test_structured_report_accepts_figures_from_flags() -> None:
    cit = {
        "chunk_id": "ACME_2024_item_7_0001",
        "ticker": "ACME",
        "fiscal_year": 2024,
        "section_id": "item_7",
        "excerpt": "Gross margin declined primarily due to higher inventory provisions and mix.",
    }
    report = AnalystReport(
        headline="ACME's margin contracted.",
        findings=[
            {
                "statement": "Operating margin contracted 571.0 bps on inventory provisions.",
                "ticker": "ACME",
                "fiscal_year": 2024,
                "chunk_ids": [cit["chunk_id"]],
                "evidence_quote": "Gross margin declined primarily due to higher inventory provisions",
            }
        ],
    )
    flag = {"metric": "operating_margin", "unit": "bps", "value": -571.0}
    without = verify_analyst_report(report, variance_reports=[], citations=[cit])
    assert without.verification.findings_verified == 0
    with_flags = verify_analyst_report(
        report, variance_reports=[], citations=[cit], material_flags=[flag]
    )
    assert with_flags.verification.findings_verified == 1


# ---------------------------------------------------------------------------
# Disclosure deltas: text primitives
# ---------------------------------------------------------------------------

_BASE = (
    "We face risks related to export controls that restrict our ability to sell advanced "
    "data center products to customers in China and other regions, which could harm our "
    "business, financial condition and results of operations."
)
_MODIFIED = (
    "We face risks related to export controls and the AI Diffusion rule that restrict our ability "
    "to sell advanced data center products to customers in China and other regions, which could "
    "harm our business, financial condition and results of operations."
)
_NEW = (
    "Generative AI models we deploy internally may produce inaccurate outputs, raise novel "
    "intellectual property claims, and expose us to emerging regulation such as the EU AI Act."
)
_OLD_ONLY = (
    "Our gaming segment depends on a small number of add-in board partners whose financial "
    "difficulties could disrupt the distribution of our consumer graphics cards."
)


def test_theme_terms_and_mentions() -> None:
    terms = extract_theme_terms("How has the wording on AI risk changed across all tech companies?")
    assert terms[:2] == ["ai", "artificial intelligence"]
    assert "risk" not in terms and "wording" not in terms and "companies" not in terms
    text = "Artificial intelligence and AI models; generative AI. Maintain trail."
    # 'artificial intelligence', 'AI', 'generative AI' (longest match wins); 'maintain' is not 'ai'
    assert count_theme_mentions(text, terms) == 3


def test_wording_changes_ignore_boundary_runs() -> None:
    prior = "Unrelated tail of previous paragraph. " + _BASE
    added, removed = wording_changes(prior, _MODIFIED + " Start of next paragraph here.")
    assert added == ["and the AI Diffusion rule"]
    assert removed == []


def test_strip_chunk_header() -> None:
    text = "[NVDA (NVIDIA CORP) | FY2025 Form 10-K | Item 1A. Risk Factors — Export]\n\nBody text."
    assert strip_chunk_header(text) == "Body text."
    assert strip_chunk_header("No header here.") == "No header here."


def _material(fy: int, paragraphs: list[str]) -> SectionMaterial:
    chunks = [
        {
            "chunk_id": f"ACME_{fy}_item_1a_{i:04d}",
            "profile_id": f"ACME_{fy}",
            "ticker": "ACME",
            "company_name": "ACME Corp",
            "gics_sector": "Information Technology",
            "fiscal_year": fy,
            "section_id": "item_1a",
            "section_title": "Item 1A. Risk Factors",
            "chunk_index": i,
            "chunk_text": f"[ACME (ACME Corp) | FY{fy} Form 10-K | Item 1A. Risk Factors]\n\n{p}",
            "sec_filing_url": f"https://sec.gov/acme/{fy}",
        }
        for i, p in enumerate(paragraphs)
    ]
    return SectionMaterial(
        ticker="ACME",
        company_name="ACME Corp",
        fiscal_year=fy,
        section_id="item_1a",
        full_text="\n\n".join(paragraphs),
        chunks=chunks,
    )


def _as_passage(material: SectionMaterial, idx: int, score: float = 1.0) -> dict[str, Any]:
    row = material.chunks[idx]
    return {
        **row,
        "excerpt": row["chunk_text"],
        "score": score,
    }


def test_align_year_pair_classifies_added_removed_modified_unchanged() -> None:
    stable = (
        "Our operations could be disrupted by natural disasters, pandemics, and other events "
        "outside of our control, including power outages and telecommunications failures."
    )
    prior = _material(2024, [_BASE, stable, _OLD_ONLY])
    current = _material(2025, [_MODIFIED, stable, _NEW])
    pair, cited = align_year_pair(
        ticker="ACME",
        company_name="ACME Corp",
        section_id="item_1a",
        prior=prior,
        current=current,
        prior_passages=[_as_passage(prior, 2, 0.5)],
        current_passages=[
            _as_passage(current, 0, 0.9),
            _as_passage(current, 1, 0.4),
            _as_passage(current, 2, 0.8),
        ],
    )
    assert pair.counts == {"modified": 1, "added": 1, "removed": 1, "unchanged": 1}
    kinds = [d.change_type for d in pair.deltas]
    assert kinds == ["modified", "added", "removed", "unchanged"]

    modified = pair.deltas[0]
    assert modified.prior_chunk_id == "ACME_2024_item_1a_0000"
    assert modified.current_chunk_id == "ACME_2025_item_1a_0000"
    assert modified.added_phrases == ["and the AI Diffusion rule"]
    assert "FY2025" not in " ".join(modified.added_phrases)  # chunk header stripped

    added = pair.deltas[1]
    assert added.current_chunk_id == "ACME_2025_item_1a_0002" and added.prior_chunk_id is None
    removed = pair.deltas[2]
    assert removed.prior_chunk_id == "ACME_2024_item_1a_0002" and removed.current_chunk_id is None

    cited_ids = {c["chunk_id"] for c in cited}
    assert {"ACME_2024_item_1a_0000", "ACME_2025_item_1a_0000"} <= cited_ids
    assert all(c["section_id"] == "item_1a" for c in cited)


def test_build_company_evolution_reports_missing_years_and_mentions() -> None:
    m24 = _material(2024, [_BASE])
    m25 = _material(2025, [_MODIFIED, _NEW])
    evolution, cited = build_company_evolution(
        ticker="ACME",
        company_name="ACME Corp",
        gics_sector="Information Technology",
        window_years=[2024, 2025, 2026],
        available_years=[2024, 2025],
        theme_terms=extract_theme_terms("AI risk"),
        sections={(2024, "item_1a"): m24, (2025, "item_1a"): m25},
        theme_passages={2024: [_as_passage(m24, 0)], 2025: [_as_passage(m25, 1)]},
    )
    assert evolution.missing_fiscal_years == [2026]
    assert evolution.theme_mentions == {"2024": 0, "2025": 3}
    assert len(evolution.year_pairs) == 1
    pair = evolution.year_pairs[0]
    cited_ids = {c["chunk_id"] for c in cited}
    for d in pair.deltas:
        for cid in (d.prior_chunk_id, d.current_chunk_id):
            if cid:
                assert cid in cited_ids


def test_compare_disclosures_tool_on_live_corpus() -> None:
    res = compare_disclosures_across_years(
        theme="export controls China AI",
        tickers="NVDA",
        gics_sector="ALL",
        section_id="item_1a",
        start_year=2024,
        end_year=2026,
    )
    assert res["status"] == "success"
    assert res["companies_selected_by"] == "explicit"
    comp = res["companies"][0]
    assert comp["ticker"] == "NVDA"
    cit_ids = {c["chunk_id"] for c in res["citations"]}
    for pair in comp["year_pairs"]:
        assert pair["current_fiscal_year"] == pair["prior_fiscal_year"] + 1 or (
            pair["current_fiscal_year"] > pair["prior_fiscal_year"]
        )
        for d in pair["deltas"]:
            assert d["section_id"] == "item_1a"
            for cid in (d["prior_chunk_id"], d["current_chunk_id"]):
                if cid:
                    assert cid in cit_ids
            if d["change_type"] == "modified":
                assert d["prior_chunk_id"] and d["current_chunk_id"]

    bad = compare_disclosures_across_years(
        theme="AI",
        tickers="NVDA",
        gics_sector="ALL",
        section_id="ALL",
        start_year=2025,
        end_year=2025,
    )
    assert bad["status"] == "error"


# ---------------------------------------------------------------------------
# Agent wiring, prompt, and chat-turn integration
# ---------------------------------------------------------------------------


def test_analyst_agent_registers_new_tools_with_least_privilege() -> None:
    analyst = create_financial_analyst_agent()
    names = {getattr(t, "name", "") for t in analyst.tools}
    assert {"flag_material_variances", "compare_disclosures_across_years"} <= names
    for tool in ("flag_material_variances", "compare_disclosures_across_years"):
        claims = authorize_agent_tool_invocation(
            agent_name="FinancialAnalystAgent", tool_name=tool, caller="RootOrchestrator"
        )
        assert claims.agent_name == "FinancialAnalystAgent"


def test_prompt_carries_flags_and_disclosure_deltas() -> None:
    prompt = _build_financial_reasoning_prompt(
        query="How has AI risk wording changed?",
        intent="thematic_trend_tracking",
        scope={"tickers_csv": "NVDA", "start_year": 2024, "end_year": 2026},
        variance_reports=[],
        variance_evaluations=[],
        cached_explanations=[],
        citations=[],
        sector_benchmarks=[],
        chunk_health=None,
        material_variance_flags=[
            {"flag_id": "NVDA_2025_revenue_yoy_change", "severity": "high", "value": 114.2}
        ],
        disclosure_deltas={
            "status": "success",
            "theme": "AI risk",
            "summary": {"modified": 1},
            "companies": [
                {
                    "ticker": "NVDA",
                    "year_pairs": [
                        {
                            "prior_fiscal_year": 2024,
                            "current_fiscal_year": 2025,
                            "counts": {"modified": 1},
                            "deltas": [
                                {
                                    "change_type": "modified",
                                    "prior_chunk_id": "NVDA_2024_item_1a_0001",
                                    "current_chunk_id": "NVDA_2025_item_1a_0002",
                                    "current_excerpt": "x",
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    )
    assert "## Authoritative Material Variance Flags" in prompt
    assert "NVDA_2025_revenue_yoy_change" in prompt
    assert "## Authoritative Year-over-Year Disclosure Deltas" in prompt
    assert "NVDA_2024_item_1a_0001" in prompt and "cite BOTH" in prompt


def _assert_citations_resolve(resp: dict[str, Any]) -> None:
    known = {c["chunk_id"] for c in resp["citations"]}
    cited = _CITATION_RE.findall(resp["answer_markdown"])
    text_citations = [c for c in cited if not c.startswith("acc:")]
    assert [c for c in text_citations if c not in known] == []
    # Inline XBRL numeric provenance tags should also be present when variance reports exist
    if resp.get("variance_reports") and "#### Key Metrics" in resp.get("answer_markdown", ""):
        assert any(c.startswith("acc:") for c in cited)


def test_live_anomaly_turn_attaches_explained_material_flags() -> None:
    """Anomaly / inflection question: the live agent invokes variances AND anomaly flags."""
    resp = run_conversational_rag_turn(
        query="Why did Intel's net income turn negative between 2023 and 2025?",
        tickers="INTC",
        start_year=2023,
        end_year=2025,
    )
    assert resp["status"] == "success"
    assert resp["execution_mode"] == "live_adk"
    invoked = set(resp["adk_runtime"]["agent_invoked_tools"])
    assert "flag_material_variances" in invoked, resp["adk_runtime"]["adk_tool_calls"]
    summary = resp["material_variance_summary"]
    assert summary["total_flags"] >= len(resp["material_variance_flags"]) > 0
    assert any(
        f["ticker"] == "INTC" and f["metric"] == "net_income"
        for f in resp["material_variance_flags"]
    )
    assert any(s["tool"] == "flag_material_variances" for s in resp["agent_trace"])
    known = {c["chunk_id"] for c in resp["citations"]}
    for flag in resp["material_variance_flags"]:
        if flag.get("explanation"):
            # MD&A explanations are cited passages from the live corpus.
            assert flag["explanation"]["chunk_id"] in known
    _assert_citations_resolve(resp)
    if resp["answer_verified"]:
        assert resp["fact_check"]["passed"] is True
    else:
        assert resp["answer_markdown"].startswith("### Answer withheld")


def test_live_chat_turn_runs_disclosure_delta_pipeline_for_thematic_queries() -> None:
    resp = run_conversational_rag_turn(
        query="How has the wording on AI export control risk changed for NVIDIA since 2024?",
        tickers="NVDA",
        section_id="item_1a",
        start_year=2024,
        end_year=2026,
    )
    assert resp["intent"] == "thematic_trend_tracking"
    deltas = resp["disclosure_deltas"]
    assert deltas is not None and deltas["status"] == "success"
    assert deltas["summary"]["year_pairs"] >= 1
    assert any(s["tool"] == "compare_disclosures_across_years" for s in resp["agent_trace"])
    known = {c["chunk_id"] for c in resp["citations"]}
    for cit in deltas["citations"]:
        assert cit["chunk_id"] in known
    _assert_citations_resolve(resp)
