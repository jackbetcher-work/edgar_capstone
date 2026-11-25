"""Chunk health evaluation for the SEC 10-K RAG vector DB."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from edgar_capstone.dataset.models import FinancialDataProfile
from edgar_capstone.gcp.bigquery import BigQueryClientWrapper
from edgar_capstone.gcp.defaults import (
    DEFAULT_BIGQUERY_DATASET_ID,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
)
from edgar_capstone.vector_db.models import (
    ChunkAnomalyCounts,
    ChunkHealthReport,
    FilingChunk,
    SectionHealthBreakdown,
)

MIN_HEALTHY_CHARS = 100
MAX_HEALTHY_CHARS = 8000
MAX_HEALTHY_TOKENS = 2048


@dataclass(slots=True)
class _ChunkHealthFlags:
    """Precomputed per-chunk validity flags avoiding redundant NumPy norm/isfinite scans."""

    valid_embedding: bool
    undersized: bool
    oversized: bool
    missing_metadata: bool


def _is_valid_finite_embedding(vec: list[float], expected_dim: int) -> tuple[bool, bool, bool]:
    """Return `(is_non_empty, is_dim_match, is_finite_nonzero)`."""
    if not vec:
        return False, False, False
    if len(vec) != expected_dim:
        return True, False, False
    arr = np.asarray(vec, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return True, True, False
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9:
        return True, True, False
    return True, True, True


def _has_complete_metadata(chunk: FilingChunk) -> bool:
    """Verify required metadata fields for metadata-filtered RAG and grounded citations."""
    if not chunk.ticker or not chunk.cik or not chunk.profile_id:
        return False
    if not chunk.gics_sector or chunk.gics_sector == "Unknown":
        return False
    if chunk.section_id not in ("item_1a", "item_7"):
        return False
    if not chunk.section_title or not chunk.sec_filing_url or not chunk.unstructured_gcs_uri:
        return False
    return 1990 <= chunk.fiscal_year <= 2100


def _compute_group_breakdown(
    group_key: str,
    group_chunks: list[FilingChunk],
    expected_dim: int,
    *,
    flags: Sequence[_ChunkHealthFlags] | None = None,
) -> SectionHealthBreakdown:
    """Compute health breakdown stats for a slice of chunks (by section or GICS sector)."""
    if not group_chunks:
        return SectionHealthBreakdown(group_key=group_key)

    total = len(group_chunks)
    profiles = {c.profile_id for c in group_chunks}
    tickers = {c.ticker for c in group_chunks}

    valid_emb = 0
    undersized = 0
    oversized = 0
    missing_meta = 0
    char_counts: list[int] = []
    token_counts: list[int] = []

    if flags is not None:
        for c, f in zip(group_chunks, flags, strict=True):
            if f.valid_embedding:
                valid_emb += 1
            if f.undersized:
                undersized += 1
            if f.oversized:
                oversized += 1
            if f.missing_metadata:
                missing_meta += 1
            char_counts.append(c.char_count)
            token_counts.append(c.token_estimate)
    else:
        for c in group_chunks:
            has_emb, dim_ok, finite_ok = _is_valid_finite_embedding(c.embedding, expected_dim)
            if has_emb and dim_ok and finite_ok:
                valid_emb += 1
            if c.char_count < MIN_HEALTHY_CHARS or not c.chunk_text.strip():
                undersized += 1
            if c.char_count > MAX_HEALTHY_CHARS or c.token_estimate > MAX_HEALTHY_TOKENS:
                oversized += 1
            if not _has_complete_metadata(c):
                missing_meta += 1
            char_counts.append(c.char_count)
            token_counts.append(c.token_estimate)

    tok_arr = np.asarray(token_counts, dtype=np.float64)
    char_arr = np.asarray(char_counts, dtype=np.float64)

    return SectionHealthBreakdown(
        group_key=group_key,
        total_chunks=total,
        distinct_profiles=len(profiles),
        distinct_tickers=len(tickers),
        valid_embedding_count=valid_emb,
        embedding_coverage_ratio=round(valid_emb / total, 4),
        avg_char_count=round(float(np.mean(char_arr)), 2),
        avg_token_estimate=round(float(np.mean(tok_arr)), 2),
        min_token_estimate=int(np.min(tok_arr)),
        max_token_estimate=int(np.max(tok_arr)),
        p50_token_estimate=round(float(np.percentile(tok_arr, 50)), 2),
        p95_token_estimate=round(float(np.percentile(tok_arr, 95)), 2),
        undersized_chunk_count=undersized,
        oversized_chunk_count=oversized,
        missing_metadata_count=missing_meta,
    )


def _score_chunk_health(
    *,
    total_chunks: int,
    embedding_coverage_ratio: float,
    filing_coverage_ratio: float,
    full_section_coverage_ratio: float,
    anomalies: ChunkAnomalyCounts,
) -> tuple[float, str]:
    """Compute the composite 0-100 chunk health score and status badge.

    Weights: 45 pts embedding validity, 30 pts filing & dual-section completeness,
    15 pts chunk size compliance, 10 pts metadata completeness.
    """
    if total_chunks == 0:
        health_score = 0.0
    else:
        size_anomaly_ratio = min(
            1.0, (anomalies.undersized_chunk + anomalies.oversized_chunk) / total_chunks
        )
        meta_anomaly_ratio = min(1.0, anomalies.missing_metadata / total_chunks)
        raw_score = (
            45.0 * embedding_coverage_ratio
            + 15.0 * filing_coverage_ratio
            + 15.0 * full_section_coverage_ratio
            + 15.0 * (1.0 - size_anomaly_ratio)
            + 10.0 * (1.0 - meta_anomaly_ratio)
        )
        health_score = round(max(0.0, min(100.0, raw_score)), 2)

    if health_score >= 95.0 and embedding_coverage_ratio >= 0.99:
        health_status = "HEALTHY"
    elif health_score >= 80.0 and embedding_coverage_ratio >= 0.90:
        health_status = "DEGRADED"
    else:
        health_status = "UNHEALTHY"
    return health_score, health_status


def evaluate_chunk_health(
    chunks: list[FilingChunk],
    *,
    profiles: list[FinancialDataProfile] | None = None,
    project_id: str = DEFAULT_PROJECT_ID,
    dataset_id: str = DEFAULT_BIGQUERY_DATASET_ID,
    table_id: str = "filing_chunks",
    expected_embedding_dim: int = 768,
    embedding_model: str = "text-embedding-005",
) -> ChunkHealthReport:
    """Evaluate comprehensive health metrics across all chunks and expected 10-K filings."""
    expected_profile_ids = (
        sorted({p.profile_id for p in profiles})
        if profiles
        else sorted({c.profile_id for c in chunks})
    )
    total_profiles_expected = len(expected_profile_ids)

    chunks_by_profile: dict[str, set[str]] = defaultdict(set)
    for c in chunks:
        chunks_by_profile[c.profile_id].add(c.section_id)

    indexed_profile_ids = sorted(chunks_by_profile.keys())
    total_profiles_indexed = len(
        set(indexed_profile_ids).intersection(expected_profile_ids)
        if expected_profile_ids
        else indexed_profile_ids
    )
    missing_profile_ids = [pid for pid in expected_profile_ids if pid not in chunks_by_profile]
    incomplete_section_profile_ids = [
        pid
        for pid in expected_profile_ids
        if pid in chunks_by_profile and not {"item_1a", "item_7"}.issubset(chunks_by_profile[pid])
    ]
    profiles_with_both_sections = sum(
        1
        for pid in expected_profile_ids
        if {"item_1a", "item_7"}.issubset(chunks_by_profile.get(pid, set()))
    )

    filing_coverage_ratio = (
        round(total_profiles_indexed / total_profiles_expected, 4)
        if total_profiles_expected > 0
        else 0.0
    )
    full_section_coverage_ratio = (
        round(profiles_with_both_sections / total_profiles_expected, 4)
        if total_profiles_expected > 0
        else 0.0
    )

    anomalies = ChunkAnomalyCounts(
        missing_section_filings=len(missing_profile_ids) + len(incomplete_section_profile_ids)
    )

    valid_embedding_count = 0
    by_section_chunks: dict[str, list[FilingChunk]] = defaultdict(list)
    by_section_flags: dict[str, list[_ChunkHealthFlags]] = defaultdict(list)
    by_sector_chunks: dict[str, list[FilingChunk]] = defaultdict(list)
    by_sector_flags: dict[str, list[_ChunkHealthFlags]] = defaultdict(list)

    # Evaluate each chunk's embedding norm/finiteness and metadata ONCE, then reuse the
    # precomputed flags for both section and GICS-sector breakdowns (3x speedup on full corpus).
    for c in chunks:
        has_emb, dim_ok, finite_ok = _is_valid_finite_embedding(c.embedding, expected_embedding_dim)
        valid_emb = has_emb and dim_ok and finite_ok
        if not has_emb:
            anomalies.missing_embedding += 1
        elif not dim_ok:
            anomalies.dimension_mismatch += 1
        elif not finite_ok:
            anomalies.non_finite_embedding += 1
        else:
            valid_embedding_count += 1

        is_undersized = c.char_count < MIN_HEALTHY_CHARS or not c.chunk_text.strip()
        is_oversized = c.char_count > MAX_HEALTHY_CHARS or c.token_estimate > MAX_HEALTHY_TOKENS
        is_missing_meta = not _has_complete_metadata(c)

        if is_undersized:
            anomalies.undersized_chunk += 1
        if is_oversized:
            anomalies.oversized_chunk += 1
        if is_missing_meta:
            anomalies.missing_metadata += 1

        chunk_flags = _ChunkHealthFlags(
            valid_embedding=valid_emb,
            undersized=is_undersized,
            oversized=is_oversized,
            missing_metadata=is_missing_meta,
        )
        sector_key = c.gics_sector or "Unknown"
        by_section_chunks[c.section_id].append(c)
        by_section_flags[c.section_id].append(chunk_flags)
        by_sector_chunks[sector_key].append(c)
        by_sector_flags[sector_key].append(chunk_flags)

    total_chunks = len(chunks)
    embedding_coverage_ratio = (
        round(valid_embedding_count / total_chunks, 4) if total_chunks > 0 else 0.0
    )

    if total_chunks > 0:
        char_arr = np.asarray([c.char_count for c in chunks], dtype=np.float64)
        tok_arr = np.asarray([c.token_estimate for c in chunks], dtype=np.float64)
        avg_char = round(float(np.mean(char_arr)), 2)
        avg_tok = round(float(np.mean(tok_arr)), 2)
        min_tok = int(np.min(tok_arr))
        max_tok = int(np.max(tok_arr))
        p50_tok = round(float(np.percentile(tok_arr, 50)), 2)
        p95_tok = round(float(np.percentile(tok_arr, 95)), 2)
    else:
        avg_char = avg_tok = p50_tok = p95_tok = 0.0
        min_tok = max_tok = 0

    by_section = {
        sec: _compute_group_breakdown(
            sec, sec_chunks, expected_embedding_dim, flags=by_section_flags[sec]
        )
        for sec, sec_chunks in sorted(by_section_chunks.items())
    }
    by_sector = {
        sector: _compute_group_breakdown(
            sector, sec_chunks, expected_embedding_dim, flags=by_sector_flags[sector]
        )
        for sector, sec_chunks in sorted(by_sector_chunks.items())
    }

    health_score, health_status = _score_chunk_health(
        total_chunks=total_chunks,
        embedding_coverage_ratio=embedding_coverage_ratio,
        filing_coverage_ratio=filing_coverage_ratio,
        full_section_coverage_ratio=full_section_coverage_ratio,
        anomalies=anomalies,
    )

    return ChunkHealthReport(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
        expected_embedding_dim=expected_embedding_dim,
        embedding_model=embedding_model,
        total_profiles_expected=total_profiles_expected,
        total_profiles_indexed=total_profiles_indexed,
        profiles_with_both_sections=profiles_with_both_sections,
        filing_coverage_ratio=filing_coverage_ratio,
        full_section_coverage_ratio=full_section_coverage_ratio,
        missing_profile_ids=missing_profile_ids,
        incomplete_section_profile_ids=incomplete_section_profile_ids,
        total_chunks=total_chunks,
        valid_embedding_count=valid_embedding_count,
        embedding_coverage_ratio=embedding_coverage_ratio,
        avg_char_count=avg_char,
        avg_token_estimate=avg_tok,
        min_token_estimate=min_tok,
        max_token_estimate=max_tok,
        p50_token_estimate=p50_tok,
        p95_token_estimate=p95_tok,
        anomalies=anomalies,
        by_section=by_section,
        by_sector=by_sector,
        health_score=health_score,
        health_status=health_status,
    )


def _breakdown_from_bq_row(group_key: str, row: dict[str, Any]) -> SectionHealthBreakdown:
    """Convert a BigQuery `v_chunk_health_metrics` row into a `SectionHealthBreakdown`."""
    return SectionHealthBreakdown(
        group_key=group_key,
        total_chunks=int(row.get("total_chunks") or 0),
        distinct_profiles=int(row.get("distinct_profiles") or 0),
        distinct_tickers=int(row.get("distinct_tickers") or 0),
        valid_embedding_count=int(row.get("valid_embedding_count") or 0),
        embedding_coverage_ratio=round(float(row.get("embedding_coverage_ratio") or 0.0), 4),
        avg_char_count=round(float(row.get("avg_char_count") or 0.0), 2),
        avg_token_estimate=round(float(row.get("avg_token_estimate") or 0.0), 2),
        min_token_estimate=int(row.get("min_token_estimate") or 0),
        max_token_estimate=int(row.get("max_token_estimate") or 0),
        p50_token_estimate=round(float(row.get("p50_token_estimate") or 0.0), 2),
        p95_token_estimate=round(float(row.get("p95_token_estimate") or 0.0), 2),
        undersized_chunk_count=int(row.get("undersized_chunk_count") or 0),
        oversized_chunk_count=int(row.get("oversized_chunk_count") or 0),
        missing_metadata_count=int(row.get("missing_metadata_count") or 0),
    )


def _rollup_section_rows(
    sec_id: str,
    cells: list[dict[str, Any]],
) -> SectionHealthBreakdown:
    """Fallback rollup for legacy `ROLLUP`-based views that omit `(ALL_SECTORS, section_id)`."""
    tot_chunks = sum(int(c.get("total_chunks") or 0) for c in cells)
    valid_embs = sum(int(c.get("valid_embedding_count") or 0) for c in cells)
    weighted_chars = sum(
        float(c.get("avg_char_count") or 0.0) * int(c.get("total_chunks") or 0) for c in cells
    )
    weighted_tokens = sum(
        float(c.get("avg_token_estimate") or 0.0) * int(c.get("total_chunks") or 0) for c in cells
    )
    return SectionHealthBreakdown(
        group_key=sec_id,
        total_chunks=tot_chunks,
        distinct_profiles=sum(int(c.get("distinct_profiles") or 0) for c in cells),
        distinct_tickers=sum(int(c.get("distinct_tickers") or 0) for c in cells),
        valid_embedding_count=valid_embs,
        embedding_coverage_ratio=round(valid_embs / tot_chunks, 4) if tot_chunks > 0 else 0.0,
        avg_char_count=round(weighted_chars / tot_chunks, 2) if tot_chunks > 0 else 0.0,
        avg_token_estimate=round(weighted_tokens / tot_chunks, 2) if tot_chunks > 0 else 0.0,
        min_token_estimate=min((int(c.get("min_token_estimate") or 0) for c in cells), default=0),
        max_token_estimate=max((int(c.get("max_token_estimate") or 0) for c in cells), default=0),
        undersized_chunk_count=sum(int(c.get("undersized_chunk_count") or 0) for c in cells),
        oversized_chunk_count=sum(int(c.get("oversized_chunk_count") or 0) for c in cells),
        missing_metadata_count=sum(int(c.get("missing_metadata_count") or 0) for c in cells),
    )


def evaluate_chunk_health_from_bigquery(
    *,
    project_id: str = DEFAULT_PROJECT_ID,
    dataset_id: str = DEFAULT_BIGQUERY_DATASET_ID,
    table_id: str = "filing_chunks",
    location: str = DEFAULT_LOCATION,
    chunk_health_view: str = "v_chunk_health_metrics",
    filing_coverage_view: str = "v_filing_chunk_coverage",
    expected_embedding_dim: int = 768,
    embedding_model: str = "text-embedding-005",
    bq_client: Any | None = None,
    credentials: Any | None = None,
) -> ChunkHealthReport:
    """Evaluate a live `ChunkHealthReport` directly from BigQuery health & coverage views."""
    owns_wrapper = not callable(getattr(bq_client, "run_query", None))
    wrapper = (
        bq_client
        if callable(getattr(bq_client, "run_query", None))
        else BigQueryClientWrapper(
            project_id=project_id,
            dataset_id=dataset_id,
            location=location,
            bq_client=bq_client,
            credentials=credentials,
            default_timeout=60.0,
            quota_project_id=project_id,
        )
    )
    try:
        health_rows = wrapper.run_query(
            f"SELECT * FROM {wrapper.table_ref(chunk_health_view)} "
            "ORDER BY gics_sector ASC, section_id ASC",
            timeout=60.0,
        )
        coverage_rows = wrapper.run_query(
            f"SELECT profile_id, filing_chunk_status "
            f"FROM {wrapper.table_ref(filing_coverage_view)} "
            "WHERE filing_chunk_status IN ('UNINDEXED', 'PARTIAL_SECTIONS') "
            "ORDER BY profile_id ASC",
            timeout=60.0,
        )
    finally:
        if owns_wrapper:
            wrapper.close()

    if not health_rows:
        raise RuntimeError(
            f"BigQuery view {project_id}.{dataset_id}.{chunk_health_view} returned 0 rows."
        )

    overall: dict[str, Any] = {}
    by_sector: dict[str, SectionHealthBreakdown] = {}
    by_section: dict[str, SectionHealthBreakdown] = {}
    section_cells: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in health_rows:
        sector = str(row.get("gics_sector") or "")
        sec_id = str(row.get("section_id") or "")
        if sector == "ALL_SECTORS" and sec_id == "ALL_SECTIONS":
            overall = row
        elif sector != "ALL_SECTORS" and sec_id == "ALL_SECTIONS":
            by_sector[sector] = _breakdown_from_bq_row(sector, row)
        elif sector == "ALL_SECTORS" and sec_id != "ALL_SECTIONS":
            by_section[sec_id] = _breakdown_from_bq_row(sec_id, row)
        elif sector != "ALL_SECTORS" and sec_id != "ALL_SECTIONS":
            section_cells[sec_id].append(row)

    if not by_section and section_cells:
        for sec_id, cells in sorted(section_cells.items()):
            by_section[sec_id] = _rollup_section_rows(sec_id, cells)

    if not overall:
        overall = health_rows[0]

    missing_profile_ids = sorted(
        str(r["profile_id"])
        for r in coverage_rows
        if str(r.get("filing_chunk_status") or "") == "UNINDEXED" and r.get("profile_id")
    )
    incomplete_section_profile_ids = sorted(
        str(r["profile_id"])
        for r in coverage_rows
        if str(r.get("filing_chunk_status") or "") == "PARTIAL_SECTIONS" and r.get("profile_id")
    )

    total_profiles_expected = int(overall.get("expected_profiles") or 0)
    total_profiles_indexed = int(overall.get("distinct_profiles") or 0)
    missing_section_count = len(missing_profile_ids) + len(incomplete_section_profile_ids)
    profiles_with_both_sections = (
        int(overall["profiles_with_both_sections"])
        if overall.get("profiles_with_both_sections") is not None
        else max(0, total_profiles_expected - missing_section_count)
    )
    filing_coverage_ratio = (
        round(total_profiles_indexed / total_profiles_expected, 4)
        if total_profiles_expected > 0
        else 0.0
    )
    full_section_coverage_ratio = (
        round(profiles_with_both_sections / total_profiles_expected, 4)
        if total_profiles_expected > 0
        else 0.0
    )

    total_chunks = int(overall.get("total_chunks") or 0)
    valid_embedding_count = int(overall.get("valid_embedding_count") or 0)
    embedding_coverage_ratio = (
        round(valid_embedding_count / total_chunks, 4) if total_chunks > 0 else 0.0
    )

    anomalies = ChunkAnomalyCounts(
        missing_embedding=int(overall.get("missing_embedding_count") or 0),
        dimension_mismatch=int(overall.get("dimension_mismatch_count") or 0),
        non_finite_embedding=0,
        undersized_chunk=int(overall.get("undersized_chunk_count") or 0),
        oversized_chunk=int(overall.get("oversized_chunk_count") or 0),
        missing_metadata=int(overall.get("missing_metadata_count") or 0),
        missing_section_filings=missing_section_count,
    )

    health_score, health_status = _score_chunk_health(
        total_chunks=total_chunks,
        embedding_coverage_ratio=embedding_coverage_ratio,
        filing_coverage_ratio=filing_coverage_ratio,
        full_section_coverage_ratio=full_section_coverage_ratio,
        anomalies=anomalies,
    )

    return ChunkHealthReport(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
        expected_embedding_dim=expected_embedding_dim,
        embedding_model=embedding_model,
        total_profiles_expected=total_profiles_expected,
        total_profiles_indexed=total_profiles_indexed,
        profiles_with_both_sections=profiles_with_both_sections,
        filing_coverage_ratio=filing_coverage_ratio,
        full_section_coverage_ratio=full_section_coverage_ratio,
        missing_profile_ids=missing_profile_ids,
        incomplete_section_profile_ids=incomplete_section_profile_ids,
        total_chunks=total_chunks,
        valid_embedding_count=valid_embedding_count,
        embedding_coverage_ratio=embedding_coverage_ratio,
        avg_char_count=round(float(overall.get("avg_char_count") or 0.0), 2),
        avg_token_estimate=round(float(overall.get("avg_token_estimate") or 0.0), 2),
        min_token_estimate=int(overall.get("min_token_estimate") or 0),
        max_token_estimate=int(overall.get("max_token_estimate") or 0),
        p50_token_estimate=round(float(overall.get("p50_token_estimate") or 0.0), 2),
        p95_token_estimate=round(float(overall.get("p95_token_estimate") or 0.0), 2),
        anomalies=anomalies,
        by_section=by_section,
        by_sector=by_sector,
        health_score=health_score,
        health_status=health_status,
    )
