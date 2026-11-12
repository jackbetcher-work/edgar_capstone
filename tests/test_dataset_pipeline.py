"""Automated unit tests for the SEC 10-K dataset curation & GCS upload pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import re
from types import SimpleNamespace
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
    clean_markdown_section,
    extract_company_10k_bundles,
    extract_financial_metrics,
    format_combined_10k_markdown,
    resolve_fiscal_year,
)
from edgar_capstone.dataset.gcs_storage import (
    GCSAccessError,
    ensure_gcs_bucket,
    normalize_bucket_name,
)
from edgar_capstone.dataset.models import CompanyConstituent
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
    json_file = tmp_path / "structured" / "NVDA" / "2026_financials.json"
    vais_manifest = tmp_path / "manifests" / "vertex_ai_search_metadata.jsonl"
    bq_manifest = tmp_path / "manifests" / "bigquery_financials.jsonl"

    assert md_file.exists()
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

    vais_payload = json.loads(vais_manifest.read_text(encoding="utf-8").strip())
    assert re.match(r"^[a-zA-Z0-9_-]+$", vais_payload["id"])
    assert vais_payload["content"]["mimeType"] == "text/plain"
    assert (
        vais_payload["content"]["uri"]
        == "gs://fde-sec-edgar-sandbox-dev-sec-corpus/unstructured/NVDA/2026_10k.md"
    )
    assert vais_payload["structData"]["revenue"] == 215938000000.0

    # Step 2: Run --upload-only with a mocked GCS Client (verifying bucket creation + blob uploads)
    mock_client = MagicMock()
    mock_client.lookup_bucket.return_value = None  # Simulate missing bucket -> triggers creation

    mock_bucket = MagicMock()
    mock_bucket.iam_configuration = SimpleNamespace(
        uniform_bucket_level_access_enabled=False
    )
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
    assert summary_upload.uploaded_blob_count == 5  # 1 md + 1 json + 3 manifest files
    assert mock_bucket.iam_configuration.uniform_bucket_level_access_enabled is True
    mock_client.create_bucket.assert_called_once_with(
        mock_bucket,
        project="iron-radio-514217-q7",
        location="us-central1",
    )
    assert "unstructured/NVDA/2026_10k.md" in uploaded_blobs
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

