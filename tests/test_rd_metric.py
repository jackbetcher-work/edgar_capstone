"""Tests for the Research & Development metric: extraction, companyfacts backfill, and variances."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pandas as pd

from edgar_capstone.agents.tools import _build_variance_reports_from_profiles
from edgar_capstone.dataset.extractor import (
    RESEARCH_AND_DEVELOPMENT_CONCEPTS,
    extract_research_and_development_expense,
)
from edgar_capstone.dataset.models import FinancialDataProfile, VertexAISearchManifestRecord
from edgar_capstone.dataset.xbrl_facts import (
    backfill_research_and_development,
    select_annual_usd_fact,
)

NVDA_FY2026_ACCN = "0001045810-26-000021"


def _profile(
    ticker: str,
    fy: int,
    *,
    cik: str = "0001045810",
    period: str | None = None,
    accn: str | None = None,
    revenue: float = 100.0,
    rd: float | None = None,
) -> FinancialDataProfile:
    return FinancialDataProfile(
        profile_id=f"{ticker}_{fy}",
        ticker=ticker,
        cik=cik,
        company_name=f"{ticker} Corp",
        gics_sector="Information Technology",
        fiscal_year=fy,
        period_of_report=period or f"{fy}-01-25",
        filing_date=f"{fy}-02-25",
        accession_number=accn or f"{cik}-{fy % 100:02d}-000001",
        revenue=revenue,
        operating_income=revenue / 2,
        net_income=revenue / 3,
        research_and_development=rd,
        unstructured_blob_path=f"unstructured/{ticker}/{fy}_10k.md",
        structured_blob_path=f"structured/{ticker}/{fy}_financials.json",
        unstructured_gcs_uri=f"gs://bucket/unstructured/{ticker}/{fy}_10k.md",
        structured_gcs_uri=f"gs://bucket/structured/{ticker}/{fy}_financials.json",
        sec_filing_url=f"https://www.sec.gov/{ticker}/{fy}.htm",
    )


def _fact(val: float, *, start: str, end: str, accn: str, filed: str, form: str = "10-K") -> dict:
    return {"start": start, "end": end, "val": val, "accn": accn, "filed": filed, "form": form}


def _companyfacts(concept_facts: dict[str, list[dict]]) -> dict:
    return {
        "facts": {
            "us-gaap": {name: {"units": {"USD": facts}} for name, facts in concept_facts.items()}
        }
    }


def test_extract_rd_from_income_statement_concept_and_label() -> None:
    inc_df = pd.DataFrame(
        [
            {
                "concept": "us-gaap_ResearchAndDevelopmentExpense",
                "label": "Research and development",
                "2026-01-25 (FY)": 18497000000.0,
                "dimension": False,
                "is_breakdown": False,
                "abstract": False,
            }
        ]
    )
    tenk = SimpleNamespace(
        financials=SimpleNamespace(
            income_statement=lambda: SimpleNamespace(to_dataframe=lambda: inc_df)
        )
    )
    val, tag = extract_research_and_development_expense(tenk, report_date_str="2026-01-25")
    assert val == 18497000000.0
    assert tag == "us-gaap:ResearchAndDevelopmentExpense"

    # Company-extension concept matched by its line label.
    ext_df = inc_df.assign(concept="abc_TechnologyAndDevelopment", label="Research & development")
    tenk_ext = SimpleNamespace(
        financials=SimpleNamespace(
            income_statement=lambda: SimpleNamespace(to_dataframe=lambda: ext_df)
        )
    )
    assert extract_research_and_development_expense(tenk_ext, report_date_str="2026-01-25") == (
        18497000000.0,
        "abc:TechnologyAndDevelopment",
    )

    # Banks and other non-reporters: None, never 0.
    no_rd_df = inc_df.assign(concept="us-gaap_Revenues", label="Total revenues")
    tenk_none = SimpleNamespace(
        financials=SimpleNamespace(
            income_statement=lambda: SimpleNamespace(to_dataframe=lambda: no_rd_df)
        )
    )
    assert extract_research_and_development_expense(tenk_none, report_date_str="2026-01-25") == (
        None,
        None,
    )
    assert extract_research_and_development_expense(SimpleNamespace(financials=None)) == (
        None,
        None,
    )


def test_select_annual_usd_fact_prefers_same_filing_and_skips_quarters() -> None:
    facts = _companyfacts(
        {
            "ResearchAndDevelopmentExpense": [
                # Quarterly fact from the same 10-K: ignored.
                _fact(
                    5.0e9,
                    start="2025-10-27",
                    end="2026-01-25",
                    accn=NVDA_FY2026_ACCN,
                    filed="2026-02-25",
                ),
                # Comparative value restated in a later 10-K: only used as a fallback.
                _fact(
                    18.6e9,
                    start="2025-01-27",
                    end="2026-01-25",
                    accn="0001045810-27-000010",
                    filed="2027-02-24",
                ),
                _fact(
                    18.497e9,
                    start="2025-01-27",
                    end="2026-01-25",
                    accn=NVDA_FY2026_ACCN,
                    filed="2026-02-25",
                ),
            ]
        }
    )
    val, prov, note = select_annual_usd_fact(
        facts,
        RESEARCH_AND_DEVELOPMENT_CONCEPTS,
        accession_number=NVDA_FY2026_ACCN,
        period_of_report="2026-01-25",
    )
    assert val == 18.497e9
    assert prov == "sec-companyfacts:us-gaap:ResearchAndDevelopmentExpense"
    assert note is None


def test_select_annual_usd_fact_falls_back_to_later_filing_and_secondary_concept() -> None:
    facts = _companyfacts(
        {
            "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost": [
                _fact(
                    7.0e9,
                    start="2023-01-01",
                    end="2023-12-31",
                    accn="0000310158-25-000005",
                    filed="2025-02-25",
                ),
                _fact(
                    7.1e9,
                    start="2023-01-01",
                    end="2023-12-31",
                    accn="0000310158-26-000004",
                    filed="2026-02-24",
                ),
            ]
        }
    )
    val, prov, note = select_annual_usd_fact(
        facts,
        RESEARCH_AND_DEVELOPMENT_CONCEPTS,
        accession_number="0000310158-24-000003",
        period_of_report="2023-12-31",
    )
    assert val == 7.0e9  # earliest later filing wins
    assert prov.endswith("ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost")
    assert note and "0000310158-25-000005" in note

    assert select_annual_usd_fact(
        None, RESEARCH_AND_DEVELOPMENT_CONCEPTS, accession_number="x", period_of_report="2023-12-31"
    ) == (None, None, None)


def test_backfill_writes_rd_and_leaves_non_reporters_null(tmp_path: Path) -> None:
    nvda = _profile("NVDA", 2026, accn=NVDA_FY2026_ACCN)
    bank = _profile("JPM", 2025, cik="0000019617", period="2025-12-31")
    for prof in (nvda, bank):
        path = tmp_path / prof.structured_blob_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(prof.model_dump(mode="json")), encoding="utf-8")

    nvda_facts = _companyfacts(
        {
            "ResearchAndDevelopmentExpense": [
                _fact(
                    18.497e9,
                    start="2025-01-27",
                    end="2026-01-25",
                    accn=NVDA_FY2026_ACCN,
                    filed="2026-02-25",
                )
            ]
        }
    )
    bank_facts = _companyfacts({"Revenues": []})
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        assert request.headers["User-Agent"]
        if request.url.path.endswith("CIK0001045810.json"):
            return httpx.Response(200, json=nvda_facts)
        return httpx.Response(200, json=bank_facts)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        summary = backfill_research_and_development(
            tmp_path, client=client, requests_per_second=1000
        )

    assert sorted(requested) == [
        "/api/xbrl/companyfacts/CIK0000019617.json",
        "/api/xbrl/companyfacts/CIK0001045810.json",
    ]
    assert summary.profiles_scanned == 2
    assert summary.profiles_with_value == 1
    assert summary.profiles_without_value == 1
    assert summary.updated_paths == ["structured/NVDA/2026_financials.json"]

    nvda_saved = FinancialDataProfile.model_validate_json(
        (tmp_path / nvda.structured_blob_path).read_text(encoding="utf-8")
    )
    bank_saved = FinancialDataProfile.model_validate_json(
        (tmp_path / bank.structured_blob_path).read_text(encoding="utf-8")
    )
    assert nvda_saved.research_and_development == 18.497e9
    assert nvda_saved.metric_provenance.research_and_development.startswith("sec-companyfacts:")
    assert bank_saved.research_and_development is None


def test_variance_reports_include_rd_variance_and_intensity() -> None:
    profiles = [
        _profile("NVDA", 2024, revenue=60.922e9, rd=8.675e9),
        _profile("NVDA", 2025, revenue=130.497e9, rd=12.914e9),
        _profile("NVDA", 2026, revenue=215.938e9, rd=18.497e9),
        _profile("JPM", 2024, cik="0000019617", revenue=1.0e11),
        _profile("JPM", 2026, cik="0000019617", revenue=1.2e11),
    ]
    reports, missing = _build_variance_reports_from_profiles(
        profiles, ticker_filter={"NVDA", "JPM"}, start_year=2024, end_year=2026
    )
    assert missing == []
    by_ticker = {r["ticker"]: r for r in reports}

    rd = by_ticker["NVDA"]["research_and_development_variance"]
    assert rd["metric_name"] == "research_and_development"
    assert rd["start_value"] == 8.675e9 and rd["end_value"] == 18.497e9
    assert rd["percentage_delta"] == round((18.497e9 - 8.675e9) / 8.675e9 * 100, 2)
    assert rd["direction"] == "increase"
    assert by_ticker["NVDA"]["rd_intensity_bps_delta"] is not None
    fy25 = next(s for s in by_ticker["NVDA"]["annual_series"] if s["fiscal_year"] == 2025)
    assert fy25["research_and_development"] == 12.914e9
    assert fy25["research_and_development_yoy_pct"] == round((12.914 - 8.675) / 8.675 * 100, 2)

    jpm_rd = by_ticker["JPM"]["research_and_development_variance"]
    assert jpm_rd["direction"] == "unavailable"
    assert jpm_rd["start_value"] is None and jpm_rd["end_value"] is None


def test_vertex_ai_search_record_carries_rd_only_when_reported() -> None:
    with_rd = VertexAISearchManifestRecord.from_profile(_profile("NVDA", 2026, rd=1.0e9))
    without_rd = VertexAISearchManifestRecord.from_profile(_profile("JPM", 2026))
    assert with_rd.structData["research_and_development"] == 1.0e9
    assert "research_and_development" not in without_rd.structData


def test_xbrl_facts_dotted_ticker_matches_brk_b_profile(tmp_path: Path) -> None:
    """Relocated regression test: dotted ticker `BRK.B` matches hyphenated `BRK-B` staged directory."""
    prof_dir = tmp_path / "structured" / "BRK-B"
    prof_dir.mkdir(parents=True)
    profile = _profile(
        "BRK-B",
        2025,
        cik="0001067983",
        period="2025-12-31",
        accn="0000950170-26-000123",
        revenue=370_000_000_000.0,
    )
    prof_path = prof_dir / "2025_financials.json"
    prof_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")

    company_facts = _companyfacts(
        {
            "ResearchAndDevelopmentExpense": [
                _fact(
                    420_000_000,
                    start="2025-01-01",
                    end="2025-12-31",
                    accn="0000950170-26-000123",
                    filed="2026-02-22",
                )
            ]
        }
    )
    with httpx.Client(
        transport=httpx.MockTransport(lambda _req: httpx.Response(200, json=company_facts))
    ) as mock_http:
        summary = backfill_research_and_development(
            tmp_path,
            tickers=["BRK.B"],
            client=mock_http,
            requests_per_second=1000.0,
        )
    assert summary.profiles_scanned == 1
    assert summary.profiles_updated == 1
    assert summary.profiles_with_value == 1
    reloaded = FinancialDataProfile.model_validate_json(prof_path.read_text(encoding="utf-8"))
    assert reloaded.research_and_development == 420_000_000.0

    # Idempotent re-run with `overwrite_extracted=False` does not rewrite unchanged files
    with httpx.Client(
        transport=httpx.MockTransport(lambda _req: httpx.Response(200, json=company_facts))
    ) as mock_http2:
        again = backfill_research_and_development(
            tmp_path,
            tickers=["BRK.B"],
            client=mock_http2,
            overwrite_extracted=False,
            requests_per_second=1000.0,
        )
    assert again.updated_paths == []


def test_fetch_company_facts_retry_404_and_invalid_cik(monkeypatch) -> None:
    """Covers `fetch_company_facts` invalid CIK, 404 -> None, 429/503 retry recovery, and exhaustion."""
    import time

    import pytest

    from edgar_capstone.dataset.xbrl_facts import fetch_company_facts

    monkeypatch.setattr(time, "sleep", lambda _s: None)

    with httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={}))) as c:
        with pytest.raises(ValueError, match="Invalid CIK"):
            fetch_company_facts("not-a-cik", client=c)

    with httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(404))) as c404:
        assert fetch_company_facts("1045810", client=c404) is None

    calls = {"n": 0}

    def _retry_handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("reset", request=req)
        if calls["n"] == 2:
            return httpx.Response(429, request=req)
        return httpx.Response(200, json={"facts": {}}, request=req)

    with httpx.Client(transport=httpx.MockTransport(_retry_handler)) as c_retry:
        assert fetch_company_facts("1045810", client=c_retry, max_attempts=3) == {"facts": {}}
        assert calls["n"] == 3

    with httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(503, request=req))
    ) as c_503:
        with pytest.raises(httpx.HTTPStatusError):
            fetch_company_facts("1045810", client=c_503, max_attempts=2)
