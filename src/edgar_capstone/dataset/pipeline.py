"""End-to-end orchestration for curating SEC 10-K Data Profiles and syncing to GCS."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import logging
from pathlib import Path

from google.cloud import storage

from edgar_capstone.dataset.constituents import (
    DEFAULT_SP500_CSV_PATH,
    load_constituents,
)
from edgar_capstone.dataset.extractor import (
    DEFAULT_SEC_USER_AGENT,
    configure_sec_identity,
    extract_company_10k_bundles,
)
from edgar_capstone.dataset.gcs_storage import (
    DEFAULT_BUCKET_NAME,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
    normalize_bucket_name,
    upload_staged_dataset,
)
from edgar_capstone.dataset.models import (
    DatasetBuildSummary,
    ExtractedFilingBundle,
    FinancialDataProfile,
    VertexAISearchManifestRecord,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[3] / "data" / "curated_dataset"
)


@dataclass(slots=True)
class DatasetBuildConfig:
    """Configuration parameters for building and uploading the SEC 10-K dataset."""

    project_id: str = DEFAULT_PROJECT_ID
    bucket_name: str = DEFAULT_BUCKET_NAME
    location: str = DEFAULT_LOCATION
    output_dir: Path = DEFAULT_OUTPUT_DIR
    start_year: int = 2020
    end_year: int = date.today().year
    full_sp500: bool = False
    tickers: list[str] | None = None
    sector_filter: str | None = None
    sp500_csv_path: Path = DEFAULT_SP500_CSV_PATH
    refresh_sp500: bool = False
    sec_user_agent: str = DEFAULT_SEC_USER_AGENT
    skip_existing: bool = True
    local_only: bool = False
    upload_only: bool = False
    create_bucket_if_missing: bool = True
    max_upload_workers: int = 8


def _load_existing_company_bundles(
    output_dir: Path,
    ticker: str,
    bucket_name: str,
) -> dict[str, ExtractedFilingBundle]:
    """Load already-staged bundles for `ticker` keyed by accession_number."""
    struct_dir = output_dir / "structured" / ticker
    if not struct_dir.exists():
        return {}

    by_accession: dict[str, ExtractedFilingBundle] = {}
    for json_file in sorted(struct_dir.glob("*_financials.json")):
        try:
            profile = FinancialDataProfile.model_validate_json(
                json_file.read_text(encoding="utf-8")
            )
            md_file = output_dir / profile.unstructured_blob_path
            if not md_file.exists():
                continue
            # Keep gs:// URIs aligned with the active target bucket
            profile.unstructured_gcs_uri = (
                f"gs://{bucket_name}/{profile.unstructured_blob_path}"
            )
            profile.structured_gcs_uri = (
                f"gs://{bucket_name}/{profile.structured_blob_path}"
            )
            by_accession[profile.accession_number] = ExtractedFilingBundle(
                profile=profile,
                markdown_content=md_file.read_text(encoding="utf-8"),
            )
        except Exception as exc:
            logger.warning("Ignoring unreadable cached profile %s: %s", json_file, exc)

    return by_accession


def _write_bundle_to_disk(output_dir: Path, bundle: ExtractedFilingBundle) -> None:
    """Persist paired unstructured Markdown and structured financial JSON files locally."""
    md_path = output_dir / bundle.profile.unstructured_blob_path
    json_path = output_dir / bundle.profile.structured_blob_path

    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    md_path.write_text(bundle.markdown_content, encoding="utf-8")
    json_path.write_text(
        json.dumps(bundle.profile.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )


def _collect_all_staged_profiles(
    output_dir: Path,
    bucket_name: str,
) -> list[FinancialDataProfile]:
    """Read all staged structured JSON profiles under `output_dir/structured`."""
    struct_root = output_dir / "structured"
    if not struct_root.exists():
        return []

    profiles: list[FinancialDataProfile] = []
    for json_path in sorted(struct_root.rglob("*_financials.json")):
        profile = FinancialDataProfile.model_validate_json(
            json_path.read_text(encoding="utf-8")
        )
        expected_unstructured_uri = (
            f"gs://{bucket_name}/{profile.unstructured_blob_path}"
        )
        expected_structured_uri = f"gs://{bucket_name}/{profile.structured_blob_path}"
        if (
            profile.unstructured_gcs_uri != expected_unstructured_uri
            or profile.structured_gcs_uri != expected_structured_uri
        ):
            profile.unstructured_gcs_uri = expected_unstructured_uri
            profile.structured_gcs_uri = expected_structured_uri
            json_path.write_text(
                json.dumps(profile.model_dump(mode="json"), indent=2) + "\n",
                encoding="utf-8",
            )
        profiles.append(profile)

    profiles.sort(key=lambda p: (p.ticker, p.fiscal_year))
    return profiles


def _write_manifests(
    output_dir: Path,
    profiles: list[FinancialDataProfile],
    summary: DatasetBuildSummary,
) -> None:
    """Write Vertex AI Search metadata JSONL, BigQuery financials JSONL, and catalog JSON."""
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    vais_path = manifests_dir / "vertex_ai_search_metadata.jsonl"
    bq_path = manifests_dir / "bigquery_financials.jsonl"
    catalog_path = manifests_dir / "dataset_catalog.json"

    vais_lines: list[str] = []
    bq_lines: list[str] = []
    for prof in profiles:
        vais_rec = VertexAISearchManifestRecord.from_profile(prof)
        vais_lines.append(json.dumps(vais_rec.model_dump(mode="json")))
        bq_lines.append(json.dumps(prof.model_dump(mode="json")))

    vais_path.write_text(
        ("\n".join(vais_lines) + "\n") if vais_lines else "",
        encoding="utf-8",
    )
    bq_path.write_text(
        ("\n".join(bq_lines) + "\n") if bq_lines else "",
        encoding="utf-8",
    )
    catalog_path.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )


def run_dataset_pipeline(
    config: DatasetBuildConfig,
    *,
    storage_client: storage.Client | None = None,
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
                processed_tickers.append(constituent.ticker)
            except Exception as exc:
                err_msg = str(exc)
                logger.error(
                    "[%s] Failed extracting company 10-K filings: %s",
                    constituent.ticker,
                    err_msg,
                )
                failed_tickers[constituent.ticker] = err_msg

    staged_profiles = _collect_all_staged_profiles(output_dir, bucket_name)
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
    _write_manifests(output_dir, staged_profiles, summary)

    if not config.local_only:
        profiles_by_blob: dict[str, FinancialDataProfile] = {}
        for prof in staged_profiles:
            profiles_by_blob[prof.unstructured_blob_path] = prof
            profiles_by_blob[prof.structured_blob_path] = prof

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
        # Update catalog locally & re-upload catalog so GCS catalog reflects uploaded_to_gcs=True
        _write_manifests(output_dir, staged_profiles, summary)
        catalog_only_client = storage_client or storage.Client(project=config.project_id)
        catalog_bucket = catalog_only_client.bucket(bucket_name)
        catalog_blob = catalog_bucket.blob("manifests/dataset_catalog.json")
        catalog_blob.upload_from_filename(
            str(output_dir / "manifests" / "dataset_catalog.json"),
            content_type="application/json; charset=utf-8",
        )

    return summary
