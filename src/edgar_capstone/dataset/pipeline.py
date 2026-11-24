"""End-to-end orchestration for curating SEC 10-K Data Profiles and syncing to GCS."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from edgar_capstone.common.domain import SECTION_TITLES, SectionId
from edgar_capstone.common.jsonl import write_json, write_jsonl
from edgar_capstone.dataset.constituents import load_constituents
from edgar_capstone.dataset.extractor import (
    DEFAULT_SEC_USER_AGENT,
    MIN_SUBSTANTIVE_SECTION_CHARS,
    PARSE_FAILURE_MARKER,
    configure_sec_identity,
    extract_company_10k_bundles,
    is_substantive_section_markdown,
    resolve_fiscal_year,
    truncate_at_section_boundary,
)
from edgar_capstone.dataset.gcs_storage import (
    DEFAULT_RETRY,
    create_storage_client,
    normalize_bucket_name,
    upload_files_to_bucket,
    upload_staged_dataset,
)
from edgar_capstone.dataset.models import (
    DatasetBuildSummary,
    ExtractedFilingBundle,
    FinancialDataProfile,
    VertexAISearchManifestRecord,
)
from edgar_capstone.dataset.paths import (
    resolve_curated_dataset_dir,
    resolve_sp500_constituents_csv,
)
from edgar_capstone.gcp.defaults import (
    DEFAULT_BUCKET_NAME,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
)
from edgar_capstone.vector_db.chunker import extract_sections_from_combined_markdown

logger = logging.getLogger(__name__)

# Retained for backward compatibility; DatasetBuildConfig resolves dynamically at instantiation.
DEFAULT_OUTPUT_DIR = resolve_curated_dataset_dir()

_CACHED_MD_SPLIT_RE = re.compile(
    r"^## Item 1A\. Risk Factors\n(.*?)\n---\n\n"
    r"## Item 7\. Management's Discussion and Analysis of Financial Condition and Results of Operations\n(.*)$",
    re.MULTILINE | re.DOTALL,
)

_STALE_FILE_SPECS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("structured", "*_financials.json", re.compile(r"^(\d{4})_financials\.json$")),
    ("unstructured", "*_10k.md", re.compile(r"^(\d{4})_10k\.md$")),
    ("unstructured", "*_item_*.md", re.compile(r"^(\d{4})_item_(?:1a|7)\.md$")),
)
_INDEXABLE_SECTIONS: tuple[SectionId, ...] = ("item_1a", "item_7")


@dataclass(slots=True)
class DatasetBuildConfig:
    """Configuration parameters for building and uploading the SEC 10-K dataset."""

    project_id: str = DEFAULT_PROJECT_ID
    bucket_name: str = DEFAULT_BUCKET_NAME
    location: str = DEFAULT_LOCATION
    output_dir: Path = field(default_factory=resolve_curated_dataset_dir)
    start_year: int = 2020
    end_year: int = field(default_factory=lambda: date.today().year)
    full_sp500: bool = False
    tickers: list[str] | None = None
    sector_filter: str | None = None
    sp500_csv_path: Path = field(default_factory=resolve_sp500_constituents_csv)
    refresh_sp500: bool = False
    sec_user_agent: str = DEFAULT_SEC_USER_AGENT
    skip_existing: bool = True
    local_only: bool = False
    upload_only: bool = False
    create_bucket_if_missing: bool = True
    max_upload_workers: int = 8


def _is_cached_bundle_healthy(bundle: ExtractedFilingBundle) -> bool:
    """Validate that a locally staged bundle has clean FY metadata, sections, and financials."""
    prof = bundle.profile

    # 1. Fiscal year must be consistent with period_of_report
    expected_fy = resolve_fiscal_year(
        dei_fiscal_year=prof.fiscal_year,
        report_date_str=prof.period_of_report,
    )
    if expected_fy != prof.fiscal_year:
        return False

    # 2. Both Item 1A and Item 7 must be present and free of cross-reference stubs / wrong anchors
    if "item_1a" not in prof.extracted_sections or "item_7" not in prof.extracted_sections:
        return False

    match = _CACHED_MD_SPLIT_RE.search(bundle.markdown_content)
    if not match:
        return False
    sec_1a = match.group(1).strip()
    sec_7 = match.group(2).strip()
    if not is_substantive_section_markdown(sec_1a, item_id="item_1a"):
        return False
    if not is_substantive_section_markdown(sec_7, item_id="item_7"):
        return False
    # Sections that bleed into the next 10-K item (Item 1B/1C/2, Item 7A/8, auditor's report)
    # were staged before boundary trimming existed; re-extract so the trimmed text is used.
    if truncate_at_section_boundary(sec_1a, item_id="item_1a") != sec_1a:
        return False
    if truncate_at_section_boundary(sec_7, item_id="item_7") != sec_7:
        return False
    # Mid-sentence truncation (trailing comma from edgartools cutting inside a cross-reference).
    if sec_1a.endswith(",") or sec_7.endswith(","):
        return False
    # If one section is full-sized SEC text (>=2k chars) and the other is a tiny fragment (<1.5k),
    # treat as truncated so full-document/exhibit fallback can recover it.
    if (
        max(len(sec_1a), len(sec_7)) >= MIN_SUBSTANTIVE_SECTION_CHARS
        and min(len(sec_1a), len(sec_7)) < 1500
    ):
        return False

    # 3. Structured financial metrics must be populated and free of dimensional/REIT scale inversions
    if (
        prof.revenue is None
        or prof.revenue <= 0
        or prof.operating_income is None
        or prof.net_income is None
    ):
        return False
    if prof.operating_income > 0 and prof.operating_income > 1.5 * prof.revenue:
        return False
    return not (prof.net_income > 0 and prof.net_income > 1.5 * prof.revenue)


def _load_existing_company_bundles(
    output_dir: Path,
    ticker: str,
    bucket_name: str,
) -> dict[str, ExtractedFilingBundle]:
    """Load validated already-staged bundles for `ticker` keyed by accession_number."""
    struct_dir = output_dir / "structured" / ticker
    if not struct_dir.exists():
        return {}

    raw_bundles: list[ExtractedFilingBundle] = []
    for json_file in sorted(struct_dir.glob("*_financials.json")):
        try:
            profile = FinancialDataProfile.model_validate_json(
                json_file.read_text(encoding="utf-8")
            )
            md_file = output_dir / profile.unstructured_blob_path
            if not md_file.exists():
                continue
            profile.unstructured_gcs_uri = f"gs://{bucket_name}/{profile.unstructured_blob_path}"
            profile.structured_gcs_uri = f"gs://{bucket_name}/{profile.structured_blob_path}"
            raw_bundles.append(
                ExtractedFilingBundle(
                    profile=profile,
                    markdown_content=md_file.read_text(encoding="utf-8"),
                )
            )
        except Exception as exc:
            logger.warning("Ignoring unreadable cached profile %s: %s", json_file, exc)

    if not raw_bundles:
        return {}

    # If the cached series has a fiscal-year gap (e.g., mixed Jan-May DEI offsets on CRWD),
    # force a clean series re-evaluation so no dropped year remains missing.
    fys = sorted({b.profile.fiscal_year for b in raw_bundles})
    if len(fys) >= 2 and (fys[-1] - fys[0] + 1) != len(fys):
        logger.info(
            "[%s] Cached fiscal years %s contain a gap; re-evaluating series from SEC EDGAR.",
            ticker,
            fys,
        )
        return {}

    by_accession: dict[str, ExtractedFilingBundle] = {}
    for bundle in raw_bundles:
        if _is_cached_bundle_healthy(bundle):
            by_accession[bundle.profile.accession_number] = bundle
        else:
            logger.info(
                "[%s] Invalidating unhealthy cached FY%d bundle (%s) for re-extraction.",
                ticker,
                bundle.profile.fiscal_year,
                bundle.profile.accession_number,
            )

    return by_accession


def _format_section_markdown(
    profile: FinancialDataProfile, section_id: SectionId, body: str
) -> str:
    """Format a standalone section Markdown document for Vertex AI Search section-level indexing."""
    heading = SECTION_TITLES[section_id]
    return (
        f"# {profile.company_name} ({profile.ticker}) — FY{profile.fiscal_year} Form 10-K — {heading}\n\n"
        f"- **Fiscal Year:** {profile.fiscal_year} (Period Ended {profile.period_of_report})\n"
        f"- **Section:** {heading}\n"
        f"- **Accession Number:** `{profile.accession_number}`\n\n"
        f"## {heading}\n\n"
        f"{body.strip()}\n"
    )


def _write_section_markdowns(
    output_dir: Path, profile: FinancialDataProfile, markdown_content: str
) -> list[SectionId]:
    """Write per-section Markdown files (`*_item_1a.md`, `*_item_7.md`) alongside `*_10k.md`."""
    sections = extract_sections_from_combined_markdown(markdown_content)
    written: list[SectionId] = []
    for section_id in _INDEXABLE_SECTIONS:
        body = (sections.get(section_id) or "").strip()
        if not body:
            continue
        sec_path = output_dir / profile.section_blob_path(section_id)
        sec_path.parent.mkdir(parents=True, exist_ok=True)
        sec_path.write_text(_format_section_markdown(profile, section_id, body), encoding="utf-8")
        written.append(section_id)
    return written


def materialize_section_markdowns(
    output_dir: Path,
    profile: FinancialDataProfile,
    *,
    force: bool = False,
) -> list[SectionId]:
    """Ensure per-section Markdown files exist for a staged profile; return present section IDs."""
    combined_path = output_dir / profile.unstructured_blob_path
    if not combined_path.exists():
        return [s for s in _INDEXABLE_SECTIONS if s in profile.extracted_sections]

    combined_mtime = combined_path.stat().st_mtime
    expected = [s for s in _INDEXABLE_SECTIONS if s in profile.extracted_sections] or list(
        _INDEXABLE_SECTIONS
    )
    needs_write = force or any(
        not (sec_path := output_dir / profile.section_blob_path(sid)).exists()
        or sec_path.stat().st_mtime < combined_mtime
        for sid in expected
    )
    if needs_write:
        return _write_section_markdowns(
            output_dir, profile, combined_path.read_text(encoding="utf-8")
        )
    return [
        sid for sid in _INDEXABLE_SECTIONS if (output_dir / profile.section_blob_path(sid)).exists()
    ]


def _write_bundle_to_disk(output_dir: Path, bundle: ExtractedFilingBundle) -> None:
    """Persist paired unstructured Markdown, per-section Markdowns, and structured JSON locally."""
    md_path = output_dir / bundle.profile.unstructured_blob_path
    json_path = output_dir / bundle.profile.structured_blob_path

    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    md_path.write_text(bundle.markdown_content, encoding="utf-8")
    _write_section_markdowns(output_dir, bundle.profile, bundle.markdown_content)
    json_path.write_text(
        json.dumps(bundle.profile.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )


def _prune_stale_company_files(
    output_dir: Path,
    ticker: str,
    *,
    start_year: int,
    end_year: int,
    valid_fiscal_years: set[int],
) -> None:
    """Remove any orphaned per-year files in `[start_year, end_year]` after fiscal-year repairs."""
    for subdir, pattern, filename_re in _STALE_FILE_SPECS:
        target_dir = output_dir / subdir / ticker
        if not target_dir.exists():
            continue
        for file_path in target_dir.glob(pattern):
            match = filename_re.match(file_path.name)
            if not match:
                continue
            fy = int(match.group(1))
            if start_year <= fy <= end_year and fy not in valid_fiscal_years:
                logger.info("[%s] Removing stale cached file %s", ticker, file_path)
                file_path.unlink(missing_ok=True)


def collect_staged_profiles(
    output_dir: Path,
    *,
    bucket_name: str,
    update_uris_on_disk: bool = False,
) -> list[FinancialDataProfile]:
    """Read all staged structured JSON profiles under `output_dir/structured`.

    By default this read is non-mutating on disk (`update_uris_on_disk=False`). In-memory
    GCS URIs are normalized against `bucket_name`; pass `update_uris_on_disk=True` to rewrite
    any JSON files whose stored URIs pointed at a different bucket.
    """
    struct_root = output_dir / "structured"
    if not struct_root.exists():
        return []

    profiles: list[FinancialDataProfile] = []
    for json_path in sorted(struct_root.rglob("*_financials.json")):
        profile = FinancialDataProfile.model_validate_json(json_path.read_text(encoding="utf-8"))
        expected_unstructured_uri = f"gs://{bucket_name}/{profile.unstructured_blob_path}"
        expected_structured_uri = f"gs://{bucket_name}/{profile.structured_blob_path}"
        if (
            profile.unstructured_gcs_uri != expected_unstructured_uri
            or profile.structured_gcs_uri != expected_structured_uri
        ):
            profile.unstructured_gcs_uri = expected_unstructured_uri
            profile.structured_gcs_uri = expected_structured_uri
            if update_uris_on_disk:
                json_path.write_text(
                    json.dumps(profile.model_dump(mode="json"), indent=2) + "\n",
                    encoding="utf-8",
                )
        profiles.append(profile)

    profiles.sort(key=lambda p: (p.ticker, p.fiscal_year))
    return profiles


def _collect_all_staged_profiles(
    output_dir: Path,
    bucket_name: str,
) -> list[FinancialDataProfile]:
    """Backward-compatible alias for `collect_staged_profiles(..., update_uris_on_disk=True)`."""
    return collect_staged_profiles(output_dir, bucket_name=bucket_name, update_uris_on_disk=True)


def write_dataset_catalog_only(output_dir: Path, summary: DatasetBuildSummary) -> Path:
    """Write only `manifests/dataset_catalog.json` (used after upload status flips)."""
    catalog_path = output_dir / "manifests" / "dataset_catalog.json"
    write_json(catalog_path, summary.model_dump(mode="json"), indent=2)
    return catalog_path


def write_manifests(
    output_dir: Path,
    profiles: list[FinancialDataProfile],
    summary: DatasetBuildSummary,
) -> None:
    """Write Vertex AI Search metadata JSONL, BigQuery financials JSONL, and catalog JSON."""
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    vais_path = manifests_dir / "vertex_ai_search_metadata.jsonl"
    bq_path = manifests_dir / "bigquery_financials.jsonl"

    vais_records: list[dict[str, Any]] = []
    for prof in profiles:
        present_sections = materialize_section_markdowns(output_dir, prof)
        for rec in VertexAISearchManifestRecord.records_from_profile(
            prof, section_ids=present_sections
        ):
            vais_records.append(rec.model_dump(mode="json"))

    write_jsonl(vais_path, vais_records)
    write_jsonl(
        bq_path,
        (prof.model_dump(mode="json") for prof in profiles),
    )
    write_dataset_catalog_only(output_dir, summary)


_write_manifests = write_manifests


def run_dataset_pipeline(
    config: DatasetBuildConfig,
    *,
    storage_client: Any | None = None,
) -> DatasetBuildSummary:
    """Execute extraction, manifest generation, and optional GCS upload."""
    bucket_name = normalize_bucket_name(config.bucket_name)
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_tickers: list[str] = []
    failed_tickers: dict[str, str] = {}
    all_warnings: list[str] = []
    requested_tickers: list[str] = []

    if not config.upload_only:
        configure_sec_identity(config.sec_user_agent)
        constituents = load_constituents(
            full_sp500=config.full_sp500,
            tickers=config.tickers,
            sector_filter=config.sector_filter,
            sp500_csv_path=config.sp500_csv_path,
            refresh_sp500=config.refresh_sp500,
        )
        requested_tickers = [c.ticker for c in constituents]
        logger.info(
            "Starting SEC 10-K curation for %d companies (fiscal years %d-%d) -> %s",
            len(constituents),
            config.start_year,
            config.end_year,
            output_dir,
        )

        for idx, constituent in enumerate(constituents, start=1):
            logger.info(
                "(%d/%d) Processing %s (%s - %s)...",
                idx,
                len(constituents),
                constituent.ticker,
                constituent.company_name,
                constituent.gics_sector,
            )
            existing_map = (
                _load_existing_company_bundles(output_dir, constituent.ticker, bucket_name)
                if config.skip_existing
                else None
            )
            try:
                bundles, comp_warnings = extract_company_10k_bundles(
                    constituent,
                    bucket_name=bucket_name,
                    start_year=config.start_year,
                    end_year=config.end_year,
                    existing_accessions=existing_map,
                )
                all_warnings.extend(comp_warnings)
                for bundle in bundles:
                    _write_bundle_to_disk(output_dir, bundle)
                had_parse_failure = any(PARSE_FAILURE_MARKER in w for w in comp_warnings)
                if had_parse_failure:
                    # A 10-K failed to download/parse (often transient). Its fiscal year is
                    # unknown, so pruning could delete a previously staged good file.
                    logger.warning(
                        "[%s] Skipping stale-file pruning because a 10-K failed to parse.",
                        constituent.ticker,
                    )
                else:
                    _prune_stale_company_files(
                        output_dir,
                        constituent.ticker,
                        start_year=config.start_year,
                        end_year=config.end_year,
                        valid_fiscal_years={b.profile.fiscal_year for b in bundles},
                    )
                processed_tickers.append(constituent.ticker)
            except Exception as exc:
                err_msg = str(exc)
                logger.error(
                    "[%s] Failed extracting company 10-K filings: %s",
                    constituent.ticker,
                    err_msg,
                )
                failed_tickers[constituent.ticker] = err_msg

    staged_profiles = collect_staged_profiles(
        output_dir, bucket_name=bucket_name, update_uris_on_disk=True
    )
    if config.upload_only:
        processed_tickers = sorted({p.ticker for p in staged_profiles})
        requested_tickers = list(processed_tickers)

    summary = DatasetBuildSummary(
        project_id=config.project_id,
        bucket_name=bucket_name,
        location=config.location,
        start_year=config.start_year,
        end_year=config.end_year,
        requested_tickers=requested_tickers,
        processed_tickers=processed_tickers,
        failed_tickers=failed_tickers,
        total_profiles=len(staged_profiles),
        uploaded_to_gcs=False,
        uploaded_blob_count=0,
        warnings=all_warnings,
    )
    write_manifests(output_dir, staged_profiles, summary)

    if not config.local_only:
        profiles_by_blob: dict[str, FinancialDataProfile] = {}
        for prof in staged_profiles:
            profiles_by_blob[prof.unstructured_blob_path] = prof
            profiles_by_blob[prof.structured_blob_path] = prof
            for sec_id in _INDEXABLE_SECTIONS:
                profiles_by_blob[prof.section_blob_path(sec_id)] = prof

        uploaded_count = upload_staged_dataset(
            output_dir=output_dir,
            project_id=config.project_id,
            bucket_name=bucket_name,
            location=config.location,
            create_bucket_if_missing=config.create_bucket_if_missing,
            profiles_by_blob=profiles_by_blob,
            max_workers=config.max_upload_workers,
            storage_client=storage_client,
        )
        summary.uploaded_to_gcs = True
        summary.uploaded_blob_count = uploaded_count
        # Update ONLY dataset_catalog.json locally & re-upload so GCS reflects uploaded_to_gcs=True
        # without rewriting the full Vertex AI Search and BigQuery JSONL manifests.
        catalog_path = write_dataset_catalog_only(output_dir, summary)
        catalog_client = storage_client or create_storage_client(project_id=config.project_id)
        catalog_bucket = catalog_client.bucket(bucket_name)
        upload_files_to_bucket(
            catalog_bucket,
            [(catalog_path, "manifests/dataset_catalog.json")],
            max_workers=1,
            timeout=120.0,
            retry=DEFAULT_RETRY,
        )

    return summary
