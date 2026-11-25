"""Automated unit tests for the SEC 10-K dataset curation & GCS upload pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from edgar_capstone.dataset.constituents import (
    CURATED_STARTER_TICKERS,
    format_cik,
    load_constituents,
    normalize_edgar_ticker,
)
from edgar_capstone.dataset.extractor import (
    _extract_single_section,
    _filter_non_dimensional_income_df,
    clean_markdown_section,
    extract_financial_metrics,
    format_combined_10k_markdown,
    is_substantive_section_markdown,
    reconcile_company_fiscal_years,
    resolve_fiscal_year,
    truncate_at_section_boundary,
)
from edgar_capstone.dataset.gcs_storage import (
    GCSAccessError,
    ensure_gcs_bucket,
    normalize_bucket_name,
)
from edgar_capstone.dataset.pipeline import DatasetBuildConfig, run_dataset_pipeline
from scripts.build_sec_dataset import parse_args


def test_ticker_and_cik_normalization() -> None:
    assert normalize_edgar_ticker(" brk.b ") == "BRK-B"
    assert normalize_edgar_ticker("bf.b") == "BF-B"
    assert normalize_edgar_ticker("nvda") == "NVDA"
    assert format_cik(320193) == "0000320193"
    assert format_cik("0001045810") == "0001045810"
    assert format_cik("") is None


def test_load_constituents_modes() -> None:
    starter = load_constituents()
    assert len(starter) == len(CURATED_STARTER_TICKERS)
    starter_map = {c.ticker: c for c in starter}
    assert starter_map["NVDA"].gics_sector == "Information Technology"
    assert starter_map["NVDA"].cik == "0001045810"
    assert starter_map["AAPL"].cik == "0000320193"
    assert starter_map["JPM"].gics_sector == "Financials"

    full_sp500 = load_constituents(full_sp500=True)
    assert len(full_sp500) >= 500
    assert any(c.ticker == "BRK-B" for c in full_sp500)

    tech_only = load_constituents(full_sp500=True, sector_filter="Information Technology")
    assert len(tech_only) > 50
    assert all(c.gics_sector == "Information Technology" for c in tech_only)

    custom = load_constituents(tickers=["NVDA,AAPL", "BRK.B"])
    assert [c.ticker for c in custom] == ["NVDA", "AAPL", "BRK-B"]


def test_resolve_fiscal_year_conventions() -> None:
    # NVDA Jan 25, 2026 -> DEI says 2026
    assert resolve_fiscal_year(dei_fiscal_year="2026", report_date_str="2026-01-25") == 2026
    # Home Depot Feb 1, 2026 -> DEI says 2025
    assert resolve_fiscal_year(dei_fiscal_year="2025", report_date_str="2026-02-01") == 2025
    # Fallback when DEI is missing: 52/53-week spill-over into early Jan vs normal year
    assert resolve_fiscal_year(dei_fiscal_year=None, report_date_str="2021-01-02") == 2020
    assert resolve_fiscal_year(dei_fiscal_year=None, report_date_str="2025-12-31") == 2025


def test_clean_and_format_markdown() -> None:
    raw_item_7 = (
        "<u>Table of Contents</u>\n\n"
        "**Item 7. Management's Discussion and Analysis**\n\n"
        "Revenue increased due to Data Center compute demand.\n\n\n\n"
        "| Metric | FY2026 | FY2025 |\n"
        "| --- | --- | --- |\n"
        "| Revenue | $215,938 | $130,497 |\n"
    )
    cleaned = clean_markdown_section(raw_item_7)
    assert "Table of Contents" not in cleaned
    assert not cleaned.startswith("**Item 7")
    assert "| Revenue | $215,938 | $130,497 |" in cleaned

    combined = format_combined_10k_markdown(
        ticker="NVDA",
        company_name="NVIDIA CORP",
        cik="0001045810",
        gics_sector="Information Technology",
        gics_sub_industry="Semiconductors",
        fiscal_year=2026,
        period_of_report="2026-01-25",
        filing_date="2026-02-25",
        accession_number="0001045810-26-000021",
        sec_filing_url="https://www.sec.gov/Archives/edgar/data/1045810/nvda.htm",
        item_1a_markdown="AI semiconductor supply chain risks.",
        item_7_markdown=cleaned,
    )
    assert "## Item 1A. Risk Factors" in combined
    assert "## Item 7. Management's Discussion and Analysis" in combined
    assert "fiscal_year: 2026" in combined


def test_extract_financial_metrics_with_single_step_fallback() -> None:
    # Simulate JPM / XOM where get_operating_income() is None on single-step statements
    inc_df = pd.DataFrame(
        [
            {
                "concept": "us-gaap_Revenues",
                "label": "Revenues",
                "2025-12-31 (FY)": 332238000000.0,
                "dimension": False,
                "is_breakdown": False,
                "abstract": False,
            },
            {
                "concept": "us-gaap_IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                "label": "Income before income taxes",
                "2025-12-31 (FY)": 41268000000.0,
                "dimension": True,  # dimensional row should be ignored!
                "is_breakdown": True,
                "abstract": False,
            },
            {
                "concept": "us-gaap_IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                "label": "Income before income taxes",
                "2025-12-31 (FY)": 41268000000.0,
                "dimension": False,
                "is_breakdown": False,
                "abstract": False,
            },
        ]
    )
    fake_inc_stmt = SimpleNamespace(to_dataframe=lambda: inc_df)
    fake_fin = SimpleNamespace(
        get_revenue=lambda: float("nan"),  # NaN should fall back cleanly
        get_operating_income=lambda: None,
        get_net_income=lambda: 28844000000.0,
        income_statement=lambda: fake_inc_stmt,
    )
    fake_tenk = SimpleNamespace(financials=fake_fin)

    rev, op_inc, net_inc, prov = extract_financial_metrics(fake_tenk)
    assert rev == 332238000000.0
    assert op_inc == 41268000000.0
    assert net_inc == 28844000000.0
    assert prov.revenue == "us-gaap:Revenues"
    assert "IncomeLossFromContinuingOperationsBeforeIncomeTaxes" in (prov.operating_income or "")
    assert len(prov.notes) == 1


def _make_fake_filing(
    *,
    accession_no: str = "0001045810-26-000021",
    filing_date: str = "2026-02-25",
    report_date: str = "2026-01-25",
    dei_fy: str = "2026",
    revenue: float = 215938000000.0,
    operating_income: float = 130387000000.0,
    net_income: float = 120067000000.0,
) -> MagicMock:
    sec_1a = SimpleNamespace(
        markdown=lambda: "**Item 1A. Risk Factors**\n\n" + ("AI risk disclosure text. " * 15)
    )
    sec_7 = SimpleNamespace(
        markdown=lambda: "**Item 7. MD&A**\n\n" + ("Data center revenue grew rapidly. " * 15)
    )
    fake_xb = SimpleNamespace(entity_info={"fiscal_year": dei_fy})
    fake_fin = SimpleNamespace(
        xb=fake_xb,
        get_revenue=lambda: revenue,
        get_operating_income=lambda: operating_income,
        get_net_income=lambda: net_income,
        income_statement=lambda: None,
    )
    fake_tenk = SimpleNamespace(
        sections={"part_i_item_1a": sec_1a, "part_ii_item_7": sec_7},
        financials=fake_fin,
        risk_factors=None,
        management_discussion=None,
    )
    filing = MagicMock()
    filing.form = "10-K"
    filing.accession_no = accession_no
    filing.filing_date = filing_date
    filing.report_date = report_date
    filing.filing_url = "https://www.sec.gov/Archives/edgar/data/1045810/nvda-20260125.htm"
    filing.obj.return_value = fake_tenk
    return filing


def test_end_to_end_pipeline_local_and_mock_gcs(tmp_path: Path) -> None:
    fake_filing_2026 = _make_fake_filing()
    fake_filing_2019 = _make_fake_filing(
        accession_no="0001045810-20-000001",
        filing_date="2020-02-04",
        report_date="2019-12-31",
        dei_fy="2019",
    )

    fake_filings_collection = MagicMock()
    fake_filings_collection.filter.return_value = [fake_filing_2026, fake_filing_2019]

    fake_company = MagicMock()
    fake_company.not_found = False
    fake_company.cik = 1045810
    fake_company.name = "NVIDIA CORP"
    fake_company.get_filings.return_value = fake_filings_collection

    # Step 1: Run local-only pipeline
    local_cfg = DatasetBuildConfig(
        project_id="iron-radio-514217-q7",
        bucket_name="gs://fde-sec-edgar-sandbox-dev-sec-corpus/",
        location="us-central1",
        output_dir=tmp_path,
        start_year=2020,
        end_year=2026,
        tickers=["NVDA"],
        local_only=True,
    )

    with patch("edgar_capstone.dataset.extractor.edgar.Company", return_value=fake_company):
        summary_local = run_dataset_pipeline(local_cfg)

    assert summary_local.total_profiles == 1
    assert summary_local.uploaded_to_gcs is False
    # Verify 2019 filing was skipped BEFORE downloading .obj()
    fake_filing_2019.obj.assert_not_called()

    md_file = tmp_path / "unstructured" / "NVDA" / "2026_10k.md"
    item_1a_file = tmp_path / "unstructured" / "NVDA" / "2026_item_1a.md"
    item_7_file = tmp_path / "unstructured" / "NVDA" / "2026_item_7.md"
    json_file = tmp_path / "structured" / "NVDA" / "2026_financials.json"
    vais_manifest = tmp_path / "manifests" / "vertex_ai_search_metadata.jsonl"
    bq_manifest = tmp_path / "manifests" / "bigquery_financials.jsonl"

    assert md_file.exists()
    assert item_1a_file.exists()
    assert item_7_file.exists()
    assert json_file.exists()
    assert vais_manifest.exists()
    assert bq_manifest.exists()

    fin_payload = json.loads(json_file.read_text(encoding="utf-8"))
    assert fin_payload["revenue"] == 215938000000.0
    assert fin_payload["operating_income"] == 130387000000.0
    assert fin_payload["net_income"] == 120067000000.0
    assert (
        fin_payload["unstructured_gcs_uri"]
        == "gs://fde-sec-edgar-sandbox-dev-sec-corpus/unstructured/NVDA/2026_10k.md"
    )

    vais_lines = [
        json.loads(line)
        for line in vais_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(vais_lines) == 2
    by_sec = {rec["structData"]["section_id"]: rec for rec in vais_lines}
    assert set(by_sec) == {"item_1a", "item_7"}
    for sec_id, vais_payload in by_sec.items():
        assert re.match(r"^[a-zA-Z0-9_-]+$", vais_payload["id"])
        assert vais_payload["id"] == f"NVDA_2026_{sec_id}"
        assert vais_payload["content"]["mimeType"] == "text/plain"
        assert (
            vais_payload["content"]["uri"]
            == f"gs://fde-sec-edgar-sandbox-dev-sec-corpus/unstructured/NVDA/2026_{sec_id}.md"
        )
        assert vais_payload["structData"]["extracted_sections"] == [sec_id]
        assert vais_payload["structData"]["revenue"] == 215938000000.0

    # Step 2: Run --upload-only with a mocked GCS Client (verifying bucket creation + blob uploads)
    mock_client = MagicMock()
    mock_client.lookup_bucket.return_value = None  # Simulate missing bucket -> triggers creation

    mock_bucket = MagicMock()
    mock_bucket.iam_configuration = SimpleNamespace(uniform_bucket_level_access_enabled=False)
    uploaded_blobs: dict[str, MagicMock] = {}

    def _get_blob(blob_name: str) -> MagicMock:
        blob = MagicMock()
        blob.name = blob_name
        uploaded_blobs[blob_name] = blob
        return blob

    mock_bucket.blob.side_effect = _get_blob
    mock_client.bucket.return_value = mock_bucket
    mock_client.create_bucket.return_value = mock_bucket

    upload_cfg = DatasetBuildConfig(
        project_id="iron-radio-514217-q7",
        bucket_name="fde-sec-edgar-sandbox-dev-sec-corpus",
        location="us-central1",
        output_dir=tmp_path,
        upload_only=True,
    )
    summary_upload = run_dataset_pipeline(upload_cfg, storage_client=mock_client)

    assert summary_upload.uploaded_to_gcs is True
    assert (
        summary_upload.uploaded_blob_count == 7
    )  # 1 combined + 2 section mds + 1 json + 3 manifests
    assert mock_bucket.iam_configuration.uniform_bucket_level_access_enabled is True
    mock_client.create_bucket.assert_called_once_with(
        mock_bucket,
        project="iron-radio-514217-q7",
        location="us-central1",
    )
    assert "unstructured/NVDA/2026_10k.md" in uploaded_blobs
    assert "unstructured/NVDA/2026_item_1a.md" in uploaded_blobs
    assert "unstructured/NVDA/2026_item_7.md" in uploaded_blobs
    assert "structured/NVDA/2026_financials.json" in uploaded_blobs
    assert "manifests/vertex_ai_search_metadata.jsonl" in uploaded_blobs
    assert "manifests/bigquery_financials.jsonl" in uploaded_blobs
    assert "manifests/dataset_catalog.json" in uploaded_blobs


def test_normalize_bucket_name_and_missing_bucket_error() -> None:
    assert (
        normalize_bucket_name("gs://fde-sec-edgar-sandbox-dev-sec-corpus/")
        == "fde-sec-edgar-sandbox-dev-sec-corpus"
    )
    with pytest.raises(ValueError):
        normalize_bucket_name("gs://bucket/with/subpath")

    mock_client = MagicMock()
    mock_client.lookup_bucket.return_value = None
    with pytest.raises(GCSAccessError):
        ensure_gcs_bucket(
            mock_client,
            bucket_name="fde-sec-edgar-sandbox-dev-sec-corpus",
            create_if_missing=False,
        )


def test_cli_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure unrelated global GCP_PROJECT_ID does not hijack iron-radio-514217-q7
    monkeypatch.setenv("GCP_PROJECT_ID", "unrelated-other-project")
    monkeypatch.delenv("EDGAR_GCP_PROJECT_ID", raising=False)
    args = parse_args([])
    assert args.project_id == "iron-radio-514217-q7"
    assert args.bucket == "fde-sec-edgar-sandbox-dev-sec-corpus"
    assert args.location == "us-central1"
    assert args.start_year == 2020
    assert args.sp500 is False


def test_resolve_and_reconcile_fiscal_year_dei_typos() -> None:
    # June-Dec period ends must never adopt a stale prior-year DEI tag (ALB, ACGL, COHR bugs)
    assert resolve_fiscal_year(dei_fiscal_year="2021", report_date_str="2022-12-31") == 2022
    assert resolve_fiscal_year(dei_fiscal_year="2020", report_date_str="2021-12-31") == 2021
    assert resolve_fiscal_year(dei_fiscal_year="2020", report_date_str="2021-06-30") == 2021

    # Out-of-range Jan-May DEI tags fall back to report_date
    assert resolve_fiscal_year(dei_fiscal_year="2022", report_date_str="2025-01-31") == 2025

    # CRWD Jan 31 mixed DEI offsets (4 filings with offset 0, 3 filings with offset -1)
    crwd_series = [
        ("2026-01-31", 2026),
        ("2025-01-31", 2024),
        ("2024-01-31", 2023),
        ("2023-01-31", 2022),
        ("2022-01-31", 2022),
        ("2021-01-31", 2021),
        ("2020-01-31", 2020),
    ]
    assert reconcile_company_fiscal_years(crwd_series) == [
        2026,
        2025,
        2024,
        2023,
        2022,
        2021,
        2020,
    ]

    # Consistent -1 retail convention (e.g., Home Depot Feb year-end) stays untouched
    hd_series = [("2026-02-01", 2025), ("2025-02-02", 2024), ("2024-01-28", 2023)]
    assert reconcile_company_fiscal_years(hd_series) == [2025, 2024, 2023]


def test_extract_financial_metrics_dimensional_reit_and_bank_cases() -> None:
    # 1. APA case: fin.get_revenue() leaks a $35M dimensional breakdown row while the consolidated
    # us-gaap:Revenues row is $19.47B (that raw tag value is itself 2x the true top line; see
    # test_revenue_reconciled_against_total_revenues_and_other_top_line)
    apa_df = pd.DataFrame(
        [
            {
                "concept": "us-gaap_Revenues",
                "label": "Total revenues",
                "2024-12-31 (FY)": 19474000000.0,
                "dimension": False,
                "is_breakdown": False,
                "abstract": False,
            },
            {
                "concept": "us-gaap_Revenues",
                "label": "Kinetik",
                "2024-12-31 (FY)": 35000000.0,
                "dimension": True,
                "is_breakdown": True,
                "abstract": False,
            },
        ]
    )
    apa_tenk = SimpleNamespace(
        financials=SimpleNamespace(
            get_revenue=lambda: 35000000.0,
            get_operating_income=lambda: 1535000000.0,
            get_net_income=lambda: 1118000000.0,
            income_statement=lambda: SimpleNamespace(to_dataframe=lambda: apa_df),
        )
    )
    rev, op_inc, net_inc, _ = extract_financial_metrics(apa_tenk, report_date_str="2024-12-31")
    assert rev == 19474000000.0
    assert op_inc == 1535000000.0
    assert net_inc == 1118000000.0

    # 2. CPT Equity REIT case: ASC 842 OperatingLeaseLeaseIncome ($1.54B) vs minor ASC 606 fees ($7.14M)
    cpt_df = pd.DataFrame(
        [
            {
                "concept": "us-gaap_OperatingLeaseLeaseIncome",
                "label": "Property revenues",
                "2024-12-31 (FY)": 1543842000.0,
                "dimension": False,
                "is_breakdown": False,
                "abstract": False,
            },
            {
                "concept": "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax",
                "label": "Fee and asset management",
                "2024-12-31 (FY)": 7137000.0,
                "dimension": False,
                "is_breakdown": False,
                "abstract": False,
            },
            {
                "concept": "us-gaap_OperatingIncomeLoss",
                "label": "Operating income",
                "2024-12-31 (FY)": 173766000.0,
                "dimension": False,
                "is_breakdown": False,
                "abstract": False,
            },
            {
                "concept": "us-gaap_NetIncomeLoss",
                "label": "Net income",
                "2024-12-31 (FY)": 163293000.0,
                "dimension": False,
                "is_breakdown": False,
                "abstract": False,
            },
        ]
    )
    cpt_tenk = SimpleNamespace(
        financials=SimpleNamespace(
            get_revenue=lambda: 7137000.0,
            get_operating_income=lambda: 173766000.0,
            get_net_income=lambda: 163293000.0,
            income_statement=lambda: SimpleNamespace(to_dataframe=lambda: cpt_df),
        )
    )
    cpt_rev, _, _, cpt_prov = extract_financial_metrics(cpt_tenk, report_date_str="2024-12-31")
    assert cpt_rev == 1543842000.0
    assert cpt_prov.revenue == "us-gaap:OperatingLeaseLeaseIncome"

    # 3. BK / CASY company-extension total revenue & pre-tax income fallback
    bk_df = pd.DataFrame(
        [
            {
                "concept": "bk_TotalRevenuesIncludingRevenueGeneratedByVariableInterestEntities",
                "label": "Total revenue",
                "2025-12-31 (FY)": 20080000000.0,
                "dimension": False,
                "is_breakdown": False,
                "abstract": False,
            },
            {
                "concept": "bk_IncomeLossFromContinuingOperationsBeforeIncomeTaxes",
                "label": "Income before income taxes",
                "2025-12-31 (FY)": 7058000000.0,
                "dimension": False,
                "is_breakdown": False,
                "abstract": False,
            },
            {
                "concept": "us-gaap_NetIncomeLoss",
                "label": "Net income",
                "2025-12-31 (FY)": 5549000000.0,
                "dimension": False,
                "is_breakdown": False,
                "abstract": False,
            },
        ]
    )
    bk_tenk = SimpleNamespace(
        financials=SimpleNamespace(
            get_revenue=lambda: None,
            get_operating_income=lambda: None,
            get_net_income=lambda: 5549000000.0,
            income_statement=lambda: SimpleNamespace(to_dataframe=lambda: bk_df),
        )
    )
    bk_rev, bk_op, bk_ni, bk_prov = extract_financial_metrics(bk_tenk)
    assert bk_rev == 20080000000.0
    assert bk_op == 7058000000.0
    assert bk_ni == 5549000000.0
    assert bk_prov.notes


def test_section_validation_and_exhibit_fallbacks() -> None:
    # Reject cross-reference stubs and wrong footnote/item anchors
    stub_7 = (
        "The information required by Item 7 is set forth in Exhibit 13 "
        "and is incorporated herein by reference. " * 3
    )
    assert not is_substantive_section_markdown(stub_7, item_id="item_7")
    assert not is_substantive_section_markdown(
        "7. Property, Plant and Equipment, Net\n\n" + ("Depreciation schedule details. " * 120),
        item_id="item_7",
    )
    assert not is_substantive_section_markdown(
        "**Item 1. Business**\n\n" + ("Company overview and history. " * 120),
        item_id="item_1a",
    )

    # Full-filing Markdown fallback for `# ITEM 1.A. RISK FACTORS` + EX-13 exhibit fallback for MD&A
    stub_sec_7 = SimpleNamespace(markdown=lambda: stub_7)
    fake_tenk = SimpleNamespace(
        sections={"part_ii_item_7": stub_sec_7},
        risk_factors=None,
        management_discussion=stub_7,
    )
    full_10k_md = (
        "| Item 1.A. | Risk Factors | 12 |\n"
        "| Item 1.B. | Unresolved Staff Comments | 25 |\n\n"
        "# ITEM 1.A. RISK FACTORS\n\n"
        + ("Supply chain and raw material inflation risk analysis. " * 120)
        + "\n\n# ITEM 1.B. UNRESOLVED STAFF COMMENTS\n\nNone.\n"
    )
    ex13_md = (
        "## MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS\n\n"
        + ("Consolidated net sales increased 8% driven by volume and price mix. " * 120)
        + "\n\n## Report of Independent Registered Public Accounting Firm\n\nAudit opinion text.\n"
    )
    fake_ex13 = SimpleNamespace(
        document_type="EX-13.1",
        document="annual_report.htm",
        markdown=lambda: ex13_md,
    )
    fake_filing = SimpleNamespace(
        markdown=lambda: full_10k_md,
        attachments=[fake_ex13],
    )

    cache: dict[str, object] = {}
    extracted_1a = _extract_single_section(
        fake_tenk,
        section_keys=("part_i_item_1a", "item_1a"),
        fallback_attr="risk_factors",
        item_key="Item 1A",
        item_id="item_1a",
        filing=fake_filing,
        markdown_cache=cache,
    )
    extracted_7 = _extract_single_section(
        fake_tenk,
        section_keys=("part_ii_item_7", "item_7"),
        fallback_attr="management_discussion",
        item_key="Item 7",
        item_id="item_7",
        filing=fake_filing,
        markdown_cache=cache,
    )
    assert extracted_1a is not None and "Supply chain and raw material" in extracted_1a
    assert extracted_7 is not None and "Consolidated net sales increased 8%" in extracted_7


def test_unhealthy_cache_invalidation_and_stale_pruning(tmp_path: Path) -> None:
    # Pre-seed an unhealthy cached bundle where 2022-12-31 was mis-saved under FY2021
    struct_dir = tmp_path / "structured" / "ALB"
    unstruct_dir = tmp_path / "unstructured" / "ALB"
    struct_dir.mkdir(parents=True)
    unstruct_dir.mkdir(parents=True)

    bad_md = format_combined_10k_markdown(
        ticker="ALB",
        company_name="ALBEMARLE CORP",
        cik="0000915913",
        gics_sector="Materials",
        gics_sub_industry="Specialty Chemicals",
        fiscal_year=2021,
        period_of_report="2022-12-31",
        filing_date="2023-02-15",
        accession_number="0000915913-23-000039",
        sec_filing_url="https://www.sec.gov/Archives/edgar/data/915913/alb-20221231.htm",
        item_1a_markdown="Lithium market risk factors. " * 25,
        item_7_markdown="Lithium segment results and outlook. " * 25,
    )
    (unstruct_dir / "2021_10k.md").write_text(bad_md, encoding="utf-8")
    (struct_dir / "2021_financials.json").write_text(
        json.dumps(
            {
                "profile_id": "ALB_2021",
                "ticker": "ALB",
                "cik": "0000915913",
                "company_name": "ALBEMARLE CORP",
                "gics_sector": "Materials",
                "gics_sub_industry": "Specialty Chemicals",
                "form_type": "10-K",
                "fiscal_year": 2021,
                "fiscal_period": "FY",
                "period_of_report": "2022-12-31",
                "filing_date": "2023-02-15",
                "accession_number": "0000915913-23-000039",
                "currency": "USD",
                "revenue": 7320104000.0,
                "operating_income": 2470061000.0,
                "net_income": 2689816000.0,
                "extracted_sections": ["item_1a", "item_7"],
                "unstructured_blob_path": "unstructured/ALB/2021_10k.md",
                "structured_blob_path": "structured/ALB/2021_financials.json",
                "unstructured_gcs_uri": "gs://fde-sec-edgar-sandbox-dev-sec-corpus/unstructured/ALB/2021_10k.md",
                "structured_gcs_uri": "gs://fde-sec-edgar-sandbox-dev-sec-corpus/structured/ALB/2021_financials.json",
                "sec_filing_url": "https://www.sec.gov/Archives/edgar/data/915913/alb-20221231.htm",
            }
        ),
        encoding="utf-8",
    )

    filing_2022 = _make_fake_filing(
        accession_no="0000915913-23-000039",
        filing_date="2023-02-15",
        report_date="2022-12-31",
        dei_fy="2021",  # Filer DEI typo!
        revenue=7320104000.0,
    )
    fake_collection = MagicMock()
    fake_collection.filter.return_value = [filing_2022]
    fake_company = MagicMock()
    fake_company.not_found = False
    fake_company.cik = 915913
    fake_company.name = "ALBEMARLE CORP"
    fake_company.get_filings.return_value = fake_collection

    cfg = DatasetBuildConfig(
        output_dir=tmp_path,
        start_year=2021,
        end_year=2022,
        tickers=["ALB"],
        skip_existing=True,
        local_only=True,
    )
    with patch("edgar_capstone.dataset.extractor.edgar.Company", return_value=fake_company):
        summary = run_dataset_pipeline(cfg)

    assert summary.total_profiles == 1
    # Mis-keyed 2021 file was invalidated and pruned; 2022 file was created
    assert not (struct_dir / "2021_financials.json").exists()
    assert not (unstruct_dir / "2021_10k.md").exists()
    assert (struct_dir / "2022_financials.json").exists()
    assert (unstruct_dir / "2022_10k.md").exists()


def test_jan_spillover_fiscal_year_ignores_calendar_year_dei() -> None:
    # DPZ tags DEI fiscal_year with the calendar year of a Jan 1-10 52/53-week period end
    assert resolve_fiscal_year(dei_fiscal_year="2022", report_date_str="2022-01-02") == 2021
    assert resolve_fiscal_year(dei_fiscal_year="2023", report_date_str="2023-01-01") == 2022
    assert resolve_fiscal_year(dei_fiscal_year="2020", report_date_str="2021-01-03") == 2020


def test_income_df_never_falls_back_to_prior_year_column() -> None:
    # EXE FY2021 (fresh-start successor period column lacks the "(FY)" suffix)
    df = pd.DataFrame(
        {
            "concept": ["us-gaap_NetIncomeLoss"],
            "label": ["Net income"],
            "dimension": [False],
            "is_breakdown": [False],
            "abstract": [False],
            "2021-12-31": [6.3e9],
            "2020-12-31 (FY)": [-9.734e9],
            "2019-12-31 (FY)": [-3.0e8],
        }
    )
    _, col = _filter_non_dimensional_income_df(df, report_date_str="2021-12-31")
    assert col == "2021-12-31"
    # No column for the reporting period at all -> refuse rather than return FY2020 values
    _, col = _filter_non_dimensional_income_df(
        df.drop(columns=["2021-12-31"]), report_date_str="2021-12-31"
    )
    assert col is None


def test_section_boundary_truncation() -> None:
    body = "Liquidity and capital resources discussion. " * 80
    mdna = body + "\n\n**Report of Independent Registered Public Accounting Firm**\n\nOpinion text."
    assert truncate_at_section_boundary(mdna, item_id="item_7") == body.strip()
    bleed_7a = (
        body + "\n\n**Item 7A\\. ** <u>Quantitative and Qualitative Disclosures</u>\n\nRates."
    )
    assert truncate_at_section_boundary(bleed_7a, item_id="item_7") == body.strip()
    # Prose cross-references to Item 8 must not truncate MD&A
    prose = body + "\nItem 8 of this report\\.\n" + body
    assert truncate_at_section_boundary(prose, item_id="item_7") == prose
    note_ref = (
        body
        + "\nItem 8\\. Financial Statements and Supplementary Data—Notes to the Consolidated Financial Statements—Note 4\n"
        + body
    )
    assert truncate_at_section_boundary(note_ref, item_id="item_7") == note_ref
    # Item 1A bleeding into Item 1B / Item 2 (FDX "see pages 88-99" stub)
    fdx = (
        "We present information about our risk factors on pages 88 through 99 of this Annual Report\\.\n\n"
        "ITEM 1B\\. UNRESOLVED STAFF COMMENTS\n\nNone\\.\n\nITEM 2\\. PROPERTIES\n\n" + body
    )
    trimmed = truncate_at_section_boundary(fdx, item_id="item_1a")
    assert "PROPERTIES" not in trimmed
    assert not is_substantive_section_markdown(trimmed, item_id="item_1a")
    # "Refer to ..." cross-reference stubs are rejected
    assert not is_substantive_section_markdown(
        "### Refer to\n\n**MANAGEMENT'S FINANCIAL DISCUSSION AND ANALYSIS of each registrant** "
        * 6,
        item_id="item_7",
    )


_MDNA_BODY = (
    "**EXECUTIVE OVERVIEW**\n\nManagement's discussion and analysis of results. "
    + "Results of operations improved on higher volumes and pricing. " * 30
    + "\n\n**Liquidity and Capital Resources**\n\n"
    + "Cash flows from operations funded capital expenditures. " * 30
    + "\n\n**Critical Accounting Estimates**\n\nImpairment testing requires judgment.\n"
)


def test_section_boundary_keeps_body_after_leading_item_7a_heading() -> None:
    # RF FY2020: combined Item 7/7A MD&A whose text begins with the Item 7A heading
    rf = (
        "**Item 7A\\. ** **Quantitative and Qualitative Disclosures about Market Risk**\n\n"
        + _MDNA_BODY
        + "\n\n**Item 8\\. ** **Financial Statements and Supplementary Data**\n\nBalance sheet."
    )
    trimmed = truncate_at_section_boundary(rf, item_id="item_7")
    assert "Critical Accounting Estimates" in trimmed
    assert "Balance sheet" not in trimmed
    assert is_substantive_section_markdown(trimmed, item_id="item_7")
    # Idempotent, so the cached-bundle health check accepts the trimmed text
    assert truncate_at_section_boundary(trimmed, item_id="item_7") == trimmed


def test_section_boundary_keeps_body_after_item_7_index_and_7a_stub() -> None:
    # SO: Item 7 page index, one-line Item 7A stub, then the full combined MD&A
    so = (
        "**Item 7\\.** **MANAGEMENT'S DISCUSSION AND ANALYSIS**\n\n| | Page |\n| --- | --- |\n"
        "| Combined Management's Discussion and Analysis | II-3 |\n\n"
        "**Item 7A\\.** **QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK**\n\n"
        'See MANAGEMENT\'S DISCUSSION AND ANALYSIS – "Market Price Risk" in Item 7 herein\\.\n\n'
        "**COMBINED MANAGEMENT'S DISCUSSION AND ANALYSIS**\n\n" + _MDNA_BODY
    )
    trimmed = truncate_at_section_boundary(so, item_id="item_7")
    assert trimmed == so
    assert truncate_at_section_boundary(trimmed, item_id="item_7") == trimmed


def test_section_boundary_still_cuts_stubs_and_real_7a_sections() -> None:
    # ETR: cross-reference stub followed by Items 7A-9A stubs -> cut at Item 7A (stub rejected)
    etr = (
        "### Refer to\n\n**MANAGEMENT'S FINANCIAL DISCUSSION AND ANALYSIS of each registrant**\n\n"
        "**Item 7A\\. ** **Quantitative and Qualitative Disclosures About Market Risk**\n\nRefer to.\n\n"
        "**Item 8\\. ** **Financial Statements and Supplementary Data**\n\nRefer to.\n"
    )
    trimmed = truncate_at_section_boundary(etr, item_id="item_7")
    assert "Item 7A" not in trimmed
    assert not is_substantive_section_markdown(trimmed, item_id="item_7")
    # Short MD&A stub followed by a genuine (non-MD&A) Item 7A section -> still cut at Item 7A
    market_risk = "Interest rate risk and foreign currency exposure are hedged with swaps. " * 60
    stub_then_7a = (
        "See the Annual Report to shareholders\\.\n\n"
        "**Item 7A\\. Quantitative and Qualitative Disclosures About Market Risk**\n\n"
        + market_risk
    )
    assert "Interest rate risk" not in truncate_at_section_boundary(stub_then_7a, item_id="item_7")
    # Item 1A behaviour is unchanged (no content markers): leading boundary still cuts
    assert (
        truncate_at_section_boundary(
            "**Item 1B\\. Unresolved Staff Comments**\n\n" + _MDNA_BODY, item_id="item_1a"
        )
        == ""
    )


def _income_tenk(rows: list[tuple[str, str, float]], col: str) -> SimpleNamespace:
    df = pd.DataFrame(
        [
            {
                "concept": concept,
                "label": label,
                col: value,
                "dimension": False,
                "is_breakdown": False,
                "abstract": False,
            }
            for concept, label, value in rows
        ]
    )
    return SimpleNamespace(
        financials=SimpleNamespace(
            get_revenue=lambda: None,
            get_operating_income=lambda: None,
            get_net_income=lambda: None,
            income_statement=lambda: SimpleNamespace(to_dataframe=lambda: df),
        )
    )


def test_revenue_reconciled_against_total_revenues_and_other_top_line() -> None:
    # APA FY2021: us-gaap:Revenues tags a $1.08B sub-line; statement top line is $7.93B
    apa_2021 = _income_tenk(
        [
            ("us-gaap_Revenues", "Total revenues", 1.082e9),
            ("apa_RevenuesAndOther", "Total revenues and other", 7.928e9),
            ("us-gaap_CostsAndExpenses", "Total operating expenses", 6.037e9),
            ("us-gaap_ProfitLoss", "Net income including noncontrolling interests", 1.313e9),
        ],
        "2021-12-31 (FY)",
    )
    rev, _, _, prov = extract_financial_metrics(apa_2021, report_date_str="2021-12-31")
    assert rev == 7.928e9
    assert prov.revenue == "apa:RevenuesAndOther"
    assert any("outside 0.5x-1.5x" in n for n in prov.notes)

    # APA FY2024: us-gaap:Revenues reported at 2x the true total
    apa_2024 = _income_tenk(
        [
            ("us-gaap_Revenues", "Total revenues", 19.474e9),
            ("apa_RevenuesAndOther", "Total revenues and other", 9.737e9),
        ],
        "2024-12-31 (FY)",
    )
    assert extract_financial_metrics(apa_2024, report_date_str="2024-12-31")[0] == 9.737e9

    # APA FY2022: revenue within tolerance of the top line -> keep the revenue concept
    apa_2022 = _income_tenk(
        [
            ("us-gaap_Revenues", "Total revenues", 11.075e9),
            ("apa_RevenuesAndOther", "Total revenues and other", 12.132e9),
        ],
        "2022-12-31 (FY)",
    )
    rev, _, _, prov = extract_financial_metrics(apa_2022, report_date_str="2022-12-31")
    assert rev == 11.075e9
    assert prov.revenue == "us-gaap:Revenues"
    assert not any("outside 0.5x-1.5x" in n for n in prov.notes)


def test_section_parser_parenthesized_escaped_and_quoted_item_headings() -> None:
    risk_body = "Material business and market risks could adversely affect our operations. " * 40
    # Parenthesized `Item 1(a)` + inline quoted cross-ref heading + split bold + `| --- | --- |` glue
    hal_md = (
        "| Item 1(a). | Risk Factors | 9 |\n"
        "| Item 1(b). | Unresolved Staff Comments | 19 |\n\n"
        "See “\n\n### Item 1(a). Risk Factors.\u201d\n\n"
        "**Hydraulic fracturing** business description text. "
        * 40
        + "\n\n| 9 | Table of Contents | Item 1(a) \\| Risk Factors |\n"
        "| --- | --- | **Item 1\\(a\\)\\. Risk ** **Factors** **\\.**\n\n"
        + risk_body
        + "\n\n**Item 1\\(b\\)\\. Unresolved Staff Comments\\.**\n\nNone\\.\n"
    )
    empty_tenk = SimpleNamespace(sections={}, risk_factors=None, management_discussion=None)
    filing = SimpleNamespace(markdown=lambda: hal_md, attachments=[])
    extracted = _extract_single_section(
        empty_tenk,
        section_keys=("part_i_item_1a", "item_1a"),
        fallback_attr="risk_factors",
        item_key="Item 1A",
        item_id="item_1a",
        filing=filing,
        markdown_cache={},
    )
    assert extracted is not None
    assert "Material business and market risks" in extracted
    assert "Hydraulic fracturing" not in extracted
    assert "Unresolved Staff Comments" not in extracted

    # Boundary truncation also trims parenthesized `Item 1(b)` and `Item 7(a)` bleed
    bled_1a = risk_body + "\n\n**ITEM 1 \\(B\\)\\. UNRESOLVED STAFF COMMENTS**\n\nNone\\."
    assert "UNRESOLVED STAFF" not in truncate_at_section_boundary(bled_1a, item_id="item_1a")
    bled_7 = (
        _MDNA_BODY + "\n\n**ITEM 7 \\(A\\)\\. QUANTITATIVE AND QUALITATIVE DISCLOSURES**\n\nFX."
    )
    assert "QUANTITATIVE" not in truncate_at_section_boundary(bled_7, item_id="item_7")

    # Escaped periods (`Item 1\.A\.` / `Item 1\.B\.`)
    rol_md = (
        "| Item 1.A. | Risk Factors. | 10 |\n"
        "| Item 1.B. | Unresolved Staff Comments. | 15 |\n\n"
        + ("Company overview prose. " * 250)
        + "\n\nItem 1\\.A\\. Risk Factors\n\n"
        + risk_body
        + "\n\nItem 1\\.B\\. Unresolved Staff Comments\n\nNone\\.\n"
    )
    rol_extracted = _extract_single_section(
        empty_tenk,
        section_keys=("part_i_item_1a", "item_1a"),
        fallback_attr="risk_factors",
        item_key="Item 1A",
        item_id="item_1a",
        filing=SimpleNamespace(markdown=lambda: rol_md, attachments=[]),
        markdown_cache={},
    )
    assert rol_extracted is not None and "Material business and market risks" in rol_extracted


def test_section_parser_multiline_split_and_inline_glued_headings() -> None:
    empty_tenk = SimpleNamespace(sections={}, risk_factors=None, management_discussion=None)
    body = _MDNA_BODY * 2

    # 1. Apostrophe broken onto its own line (`# ITEM 7. MANAGEMENT\n\n<u>’</u>\n\n**S DISCUSSION...**`)
    jbht_md = (
        ("Intro text padding. " * 300) + "\n\n# ITEM 7. MANAGEMENT\n\n<u>\u2019</u>\n\n"
        "**S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS**\n\n"
        + body
        + "\n\n# ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK\n\nRates.\n"
    )
    got_jbht = _extract_single_section(
        empty_tenk,
        section_keys=("part_ii_item_7", "item_7"),
        fallback_attr="management_discussion",
        item_key="Item 7",
        item_id="item_7",
        filing=SimpleNamespace(markdown=lambda: jbht_md, attachments=[]),
        markdown_cache={},
    )
    assert got_jbht is not None and "Critical Accounting Estimates" in got_jbht
    assert "QUANTITATIVE" not in got_jbht

    # 2. Inline-glued heading after `<u>Index</u>` + wrapped Annual Report title + split bold words
    de_md = (
        ("Padding content before item 7. " * 200) + "\n\nTable of Contents\n\n"
        "**MANAGEMENT\u2019S** **DISCUSSION** **AND** **ANALYSIS** "
        "Management\u2019s Discussion and Analysis overview. "
        + body
        + "\n\nTable of Contents COMPANY\n\n**STATEMENTS**\n\n## OF\n\n"
        "**CONSOLIDATED**\n\n**INCOM**\n\n**E**\n\n| Net sales | $100 |\n"
    )
    got_de = _extract_single_section(
        empty_tenk,
        section_keys=("part_ii_item_7", "item_7"),
        fallback_attr="management_discussion",
        item_key="Item 7",
        item_id="item_7",
        filing=SimpleNamespace(markdown=lambda: de_md, attachments=[]),
        markdown_cache={},
    )
    assert got_de is not None and "Critical Accounting Estimates" in got_de
    assert "Net sales" not in got_de
    assert not got_de.rstrip().endswith("**")

    # 3. Reversed MD&A clause order + wrapped heading + bare plain-text RISK FACTORS & auditor headings
    risk_body = "Supply chain and fuel price volatility risks could affect our earnings. " * 60
    fdx_md = (
        ("Padding before financial section. " * 200)
        + "\n\n| Management\u2019s Discussion and Analysis of Results of Operations and Financial Condition |  |\n"
        "| Overview of Financial Section | 55 |\n"
        "| Risk Factors | 88 |\n\n"
        "MANAGEMENT\u2019S DISCUSSION AND ANALYSIS OF RESULTS OF OPERATIONS AND\n\n"
        "FINANCIAL CONDITION\n\n"
        + body
        + "\n\nRISK FACTORS\n\n"
        + risk_body
        + "\n\nMANAGEMENT\u2019S REPORT ON INTERNAL\n\nCONTROL OVER FINANCIAL REPORTING\n\nDone.\n"
    )
    cache: dict[str, object] = {}
    filing_fdx = SimpleNamespace(markdown=lambda: fdx_md, attachments=[])
    got_fdx_7 = _extract_single_section(
        empty_tenk,
        section_keys=("part_ii_item_7", "item_7"),
        fallback_attr="management_discussion",
        item_key="Item 7",
        item_id="item_7",
        filing=filing_fdx,
        markdown_cache=cache,
    )
    got_fdx_1a = _extract_single_section(
        empty_tenk,
        section_keys=("part_i_item_1a", "item_1a"),
        fallback_attr="risk_factors",
        item_key="Item 1A",
        item_id="item_1a",
        filing=filing_fdx,
        markdown_cache=cache,
    )
    assert got_fdx_7 is not None
    assert "Overview of Financial Section | 55" not in got_fdx_7
    assert "Critical Accounting Estimates" in got_fdx_7
    assert "Supply chain and fuel price" not in got_fdx_7
    assert got_fdx_1a is not None
    assert "Supply chain and fuel price" in got_fdx_1a
    assert "CONTROL OVER FINANCIAL REPORTING" not in got_fdx_1a


def test_section_parser_ex13_short_title_caption_and_inline_risk_heading() -> None:
    body = _MDNA_BODY * 2
    risk_body = "An investment in the Company involves substantial market risk. " * 60

    # 1. Cross-ref quoted caption ("Financial Review") in EX-13
    stub_wfc = (
        "Information in response to this Item 7 can be found in the 2025 Annual Report "
        "to Shareholders under \u201cFinancial Review.\u201d That information is incorporated by reference."
    )
    ex13_wfc = (
        "| 2 | Financial Review | 114 | Notes |\n| 62 | Risk Factors | 134 | Notes |\n\n"
        "**Financial Review**\n\n"
        + body
        + "\n\n| Risk Factors |\n\n"
        + risk_body
        + "\n\n| Controls and Procedures |\n\nEffective.\n"
    )
    tenk_wfc = SimpleNamespace(
        sections={"part_ii_item_7": SimpleNamespace(markdown=lambda: stub_wfc)},
        risk_factors=None,
        management_discussion=stub_wfc,
    )
    filing_wfc = SimpleNamespace(
        markdown=lambda: "Short 10-K shell. " * 100,
        attachments=[
            SimpleNamespace(document_type="EX-13", document="ar.htm", markdown=lambda: ex13_wfc)
        ],
    )
    got_wfc_7 = _extract_single_section(
        tenk_wfc,
        section_keys=("part_ii_item_7", "item_7"),
        fallback_attr="management_discussion",
        item_key="Item 7",
        item_id="item_7",
        filing=filing_wfc,
        markdown_cache={},
    )
    assert got_wfc_7 is not None and "Critical Accounting Estimates" in got_wfc_7
    assert "substantial market risk" not in got_wfc_7

    # 2. EX-13 running page-header table with page numbers ("Management Discussion" -> "Report of Management | 39 |")
    ex13_ibm = (
        "| MANAGEMENT DISCUSSION |  |\n| --- | --- |\n| Overview | 6 |\n| Report of Management | 39 |\n\n"
        "| 6 | Management Discussion Example Corp |\n\n" + body + "\n\n<u>Table of Contents</u>\n\n"
        "| Report of Management Example Corp | 39 |\n\n"
        "**Management Responsibility for Financial Information**\n\nAudit text.\n"
    )
    tenk_ibm = SimpleNamespace(
        sections={},
        risk_factors=None,
        management_discussion="Refer to pages 6 through 38 incorporated herein by reference.",
    )
    filing_ibm = SimpleNamespace(
        markdown=lambda: "Short 10-K shell. " * 100,
        attachments=[
            SimpleNamespace(
                document_type="EX-13", document="ibm_ex13.htm", markdown=lambda: ex13_ibm
            )
        ],
    )
    got_ibm_7 = _extract_single_section(
        tenk_ibm,
        section_keys=("part_ii_item_7", "item_7"),
        fallback_attr="management_discussion",
        item_key="Item 7",
        item_id="item_7",
        filing=filing_ibm,
        markdown_cache={},
    )
    assert got_ibm_7 is not None
    assert "Overview | 6" not in got_ibm_7
    assert "Critical Accounting Estimates" in got_ibm_7
    assert "Management Responsibility" not in got_ibm_7

    # 3. Inline-glued EX-13 `# Risk Factors  An investment...` ending at `Managing Committee`
    ex13_usb = (
        ("Padding before risk factors. " * 300)
        + "employed 68,108 people\\. # Risk Factors  "
        + risk_body
        + "\n\n| 150 |\n\nManaging Committee Jane Doe CEO bio.\n"
    )
    filing_usb = SimpleNamespace(
        markdown=lambda: "Short 10-K shell. " * 100,
        attachments=[
            SimpleNamespace(
                document_type="EX-13", document="usb_ex13.htm", markdown=lambda: ex13_usb
            )
        ],
    )
    got_usb_1a = _extract_single_section(
        SimpleNamespace(sections={}, risk_factors=None, management_discussion=None),
        section_keys=("part_i_item_1a", "item_1a"),
        fallback_attr="risk_factors",
        item_key="Item 1A",
        item_id="item_1a",
        filing=filing_usb,
        markdown_cache={},
    )
    assert got_usb_1a is not None and "substantial market risk" in got_usb_1a
    assert "CEO bio" not in got_usb_1a


def test_wrong_anchor_guards_bleed_and_mid_sentence_truncation() -> None:
    # Forward cross-ref opening inside Item 1 Business rejected as Item 1A
    bad_1a = (
        'Risk factors associated with our business are discussed in Item 1.A. "Risk Factors."\n\n'
        + ("Our strategic objectives and business overview. " * 150)
    )
    assert not is_substantive_section_markdown(bad_1a, item_id="item_1a")

    # Item 6 Selected Financial Data rejected as Item 7
    bad_7_item6 = (
        "The following selected historical consolidated financial data should be read in conjunction "
        + ("with our financial statements. " * 120)
    )
    assert not is_substantive_section_markdown(bad_7_item6, item_id="item_7")

    # Footnote body (`8. Income Taxes`) rejected when short Item 7
    bad_7_note = (
        "Company has operating and finance leases. " * 80
        + "\n\n8. Income Taxes\n\nDeferred tax asset details.\n"
    )
    assert not is_substantive_section_markdown(bad_7_note, item_id="item_7")

    # Bare plain-text auditor report trims Item 7 bleed; `# PROPERTIES` trims Item 1A bleed
    body = "Liquidity and capital resources discussion. " * 80
    assert (
        truncate_at_section_boundary(
            body + "\n\nReport of Independent Registered Public Accounting Firm\n\nWe audited...",
            item_id="item_7",
        )
        == body.strip()
    )
    assert (
        truncate_at_section_boundary(
            body + "\n\n# PROPERTIES\n\nStore counts.",
            item_id="item_1a",
        )
        == body.strip()
    )

    # Trailing comma mid-sentence truncation defers to full-document fallback
    truncated_1a = (
        "Investing in our securities involves risk. " * 80
    ) + "please see Part II, Item 7,"
    full_1a = (
        ("Padding before risk factors. " * 250)
        + "\n\n### ITEM 1A\u2014RISK FACTORS\n\n"
        + ("Complete risk factor discussion text. " * 160)
        + "\n\n### ITEM 1B\u2014UNRESOLVED STAFF COMMENTS\n\nNone.\n"
    )
    tenk_trunc = SimpleNamespace(
        sections={"risk_factors": SimpleNamespace(markdown=lambda: truncated_1a)},
        risk_factors=truncated_1a,
        management_discussion=None,
    )
    got_full_1a = _extract_single_section(
        tenk_trunc,
        section_keys=("part_i_item_1a", "item_1a"),
        fallback_attr="risk_factors",
        item_key="Item 1A",
        item_id="item_1a",
        filing=SimpleNamespace(markdown=lambda: full_1a, attachments=[]),
        markdown_cache={},
    )
    assert got_full_1a is not None and "Complete risk factor discussion text" in got_full_1a

    # Prefix-mangled `part_ii_item_7m` section key is discovered (while `part_ii_item_7a` is excluded)
    tenk_mangled = SimpleNamespace(
        sections={
            "part_ii_item_7a": SimpleNamespace(markdown=lambda: "Market risk table. " * 150),
            "part_ii_item_7m": SimpleNamespace(markdown=lambda: _MDNA_BODY),
        },
        risk_factors=None,
        management_discussion="Short intro.",
    )
    got_mangled = _extract_single_section(
        tenk_mangled,
        section_keys=("part_ii_item_7", "item_7"),
        fallback_attr="management_discussion",
        item_key="Item 7",
        item_id="item_7",
        filing=None,
    )
    assert got_mangled is not None and "Critical Accounting Estimates" in got_mangled


def test_section_parser_rewrite_markdown_fiscal_year() -> None:
    from edgar_capstone.dataset.section_parser import rewrite_markdown_fiscal_year

    md = format_combined_10k_markdown(
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        cik="0001045810",
        gics_sector="Information Technology",
        gics_sub_industry="Semiconductors",
        fiscal_year=2025,
        period_of_report="2026-01-25",
        filing_date="2026-02-26",
        accession_number="0001045810-26-000021",
        sec_filing_url="https://www.sec.gov/example",
        item_1a_markdown="Risk content",
        item_7_markdown="MD&A content",
    )
    rewritten = rewrite_markdown_fiscal_year(md, fiscal_year=2026)
    assert "fiscal_year: 2026\n" in rewritten
    assert "# NVIDIA Corporation (NVDA) — FY2026 Form 10-K\n" in rewritten
    assert "- **Fiscal Year:** 2026 (Period Ended 2026-01-25)\n" in rewritten
    assert "fiscal_year: 2025" not in rewritten


def test_cached_bundle_health_checks_and_corrupt_loader(tmp_path: Path) -> None:
    """Covers `_is_cached_bundle_healthy` rejection branches and `_load_existing_company_bundles`."""
    from edgar_capstone.dataset.models import ExtractedFilingBundle, FinancialDataProfile
    from edgar_capstone.dataset.pipeline import (
        _is_cached_bundle_healthy,
        _load_existing_company_bundles,
    )

    good_1a = (
        "Risk factors affecting our operations include competition, export licensing, and foundry supply. "
        * 35
    )
    good_7 = (
        "Management discussion of results of operations, revenue growth, and liquidity resources. "
        * 35
    )

    def _bundle(
        *,
        fy: int = 2025,
        period: str = "2025-12-31",
        sections: list[str] | None = None,
        item_1a: str = good_1a,
        item_7: str = good_7,
        revenue: float | None = 100e9,
        op_inc: float | None = 30e9,
        net_inc: float | None = 25e9,
        raw_md: str | None = None,
    ) -> ExtractedFilingBundle:
        prof = FinancialDataProfile(
            profile_id=f"ACME_{fy}",
            ticker="ACME",
            cik="0000000001",
            company_name="ACME Corp",
            gics_sector="Information Technology",
            fiscal_year=fy,
            period_of_report=period,
            filing_date=f"{fy + 1}-02-15",
            accession_number=f"0000000001-{fy % 100:02d}-000001",
            revenue=revenue,
            operating_income=op_inc,
            net_income=net_inc,
            extracted_sections=sections if sections is not None else ["item_1a", "item_7"],
            unstructured_blob_path=f"unstructured/ACME/{fy}_10k.md",
            structured_blob_path=f"structured/ACME/{fy}_financials.json",
            unstructured_gcs_uri=f"gs://b/unstructured/ACME/{fy}_10k.md",
            structured_gcs_uri=f"gs://b/structured/ACME/{fy}_financials.json",
            sec_filing_url="https://www.sec.gov/example",
        )
        md = (
            raw_md
            if raw_md is not None
            else format_combined_10k_markdown(
                ticker="ACME",
                company_name="ACME Corp",
                cik="0000000001",
                gics_sector="Information Technology",
                gics_sub_industry="Semiconductors",
                fiscal_year=fy,
                period_of_report=period,
                filing_date=f"{fy + 1}-02-15",
                accession_number=prof.accession_number,
                sec_filing_url=prof.sec_filing_url,
                item_1a_markdown=item_1a,
                item_7_markdown=item_7,
            )
        )
        return ExtractedFilingBundle(profile=prof, markdown_content=md)

    assert _is_cached_bundle_healthy(_bundle()) is True
    assert _is_cached_bundle_healthy(_bundle(fy=2024, period="2025-12-31")) is False
    assert _is_cached_bundle_healthy(_bundle(sections=["item_1a"])) is False
    assert _is_cached_bundle_healthy(_bundle(raw_md="no headers at all")) is False
    assert _is_cached_bundle_healthy(_bundle(item_1a="Too short")) is False
    assert _is_cached_bundle_healthy(_bundle(item_7="Too short")) is False
    assert (
        _is_cached_bundle_healthy(_bundle(item_1a=good_1a + "\n\n# PROPERTIES\n\nBleed")) is False
    )
    assert (
        _is_cached_bundle_healthy(
            _bundle(
                item_7=good_7
                + "\n\n**Report of Independent Registered Public Accounting Firm**\n\nOpinion text."
            )
        )
        is False
    )
    assert _is_cached_bundle_healthy(_bundle(item_1a=good_1a.strip() + ",")) is False
    assert _is_cached_bundle_healthy(_bundle(revenue=0.0)) is False
    assert _is_cached_bundle_healthy(_bundle(op_inc=200e9)) is False
    assert _is_cached_bundle_healthy(_bundle(net_inc=200e9)) is False

    # Missing dir, corrupt JSON, and fiscal year gap force clean reload
    assert _load_existing_company_bundles(tmp_path, "NONE", "b") == {}
    sdir = tmp_path / "structured" / "ACME"
    udir = tmp_path / "unstructured" / "ACME"
    sdir.mkdir(parents=True)
    udir.mkdir(parents=True)
    (sdir / "bad_financials.json").write_text("{not valid json", encoding="utf-8")
    assert _load_existing_company_bundles(tmp_path, "ACME", "b") == {}

    # Valid FY2022 + FY2024 with a FY2023 gap triggers series re-evaluation
    for fy in (2022, 2024):
        b = _bundle(fy=fy, period=f"{fy}-12-31")
        (sdir / f"{fy}_financials.json").write_text(b.profile.model_dump_json(), encoding="utf-8")
        (udir / f"{fy}_10k.md").write_text(b.markdown_content, encoding="utf-8")
    assert _load_existing_company_bundles(tmp_path, "ACME", "b") == {}


def test_extractor_retry_company_fallback_constituents_gcs_and_lazy_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Covers `_retry_sec_call`, `_resolve_edgar_company`, `configure_sec_identity`, `refresh_sp500_csv`, GCS helpers, and `dataset.__getattr__`."""
    import time

    import httpx

    import edgar_capstone.dataset as ds_pkg
    from edgar_capstone.dataset import constituents as const_mod
    from edgar_capstone.dataset import extractor as ext_mod
    from edgar_capstone.dataset import gcs_storage as gcs_mod
    from edgar_capstone.dataset.models import CompanyConstituent

    # Lazy package __getattr__ and __dir__
    assert ds_pkg.load_constituents is load_constituents
    assert "run_dataset_pipeline" in dir(ds_pkg)
    with pytest.raises(AttributeError):
        _ = ds_pkg.NoSuchDatasetSymbol

    # configure_sec_identity validation
    with pytest.raises(ValueError, match="Organization Name"):
        ext_mod.configure_sec_identity("no-email-or-space")

    # _retry_sec_call recovers from transient OSError / httpx.HTTPError and raises immediately on CompanyNotFoundError
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    attempts = {"n": 0}

    def _flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.ConnectError("transient")
        return "ok"

    assert ext_mod._retry_sec_call(_flaky, description="test", max_attempts=3) == "ok"
    assert attempts["n"] == 3

    with pytest.raises(OSError):
        ext_mod._retry_sec_call(
            lambda: (_ for _ in ()).throw(OSError("down")),
            description="exhaust",
            max_attempts=2,
        )

    # _resolve_edgar_company falls back to CIK when ticker lookup reports not_found
    calls: list[Any] = []

    def _fake_company(ident: Any) -> Any:
        calls.append(ident)
        if isinstance(ident, str):
            return SimpleNamespace(not_found=True)
        return SimpleNamespace(not_found=False, cik=ident)

    monkeypatch.setattr(ext_mod.edgar, "Company", _fake_company)
    comp = ext_mod._resolve_edgar_company(
        CompanyConstituent(
            ticker="BRK-B", edgar_ticker="BRK-B", company_name="Berkshire", cik="0001067983"
        )
    )
    assert comp.cik == 1067983 and calls == ["BRK-B", 1067983]

    with pytest.raises(ext_mod.CompanyNotFoundError):
        ext_mod._resolve_edgar_company(
            CompanyConstituent(ticker="NOPE", edgar_ticker="NOPE", company_name="None", cik=None)
        )

    # refresh_sp500_csv via mocked httpx.get
    csv_body = "Symbol,Security,GICS Sector,GICS Sub-Industry,CIK\nNVDA,NVIDIA Corp,Information Technology,Semiconductors,1045810\n"
    monkeypatch.setattr(
        const_mod.httpx,
        "get",
        lambda url, **_kw: httpx.Response(200, text=csv_body, request=httpx.Request("GET", url)),
    )
    out_csv = tmp_path / "sp500.csv"
    const_mod.refresh_sp500_csv(csv_path=out_csv)
    assert out_csv.exists()
    cat = const_mod.load_sp500_catalog(csv_path=out_csv)
    assert cat["NVDA"].cik == "0001045810"

    # gcs_storage helpers: content_type_for_path fallback
    assert gcs_mod.content_type_for_path(Path("x.bin")) == "application/octet-stream"
    assert gcs_mod.content_type_for_path(Path("x.jsonl")) == "application/x-ndjson; charset=utf-8"
