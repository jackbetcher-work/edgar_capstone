"""Chunk health evaluation and Google Cloud Monitoring metric exporter for the RAG vector DB."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import logging
import math
from typing import Any

import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest
import httpx
import numpy as np

from edgar_capstone.dataset.models import FinancialDataProfile
from edgar_capstone.vector_db.models import (
    ChunkAnomalyCounts,
    ChunkHealthReport,
    FilingChunk,
    SectionHealthBreakdown,
)

logger = logging.getLogger(__name__)

METRIC_PREFIX = "custom.googleapis.com/sec_edgar/chunk_health"
MIN_HEALTHY_CHARS = 100
MAX_HEALTHY_CHARS = 8000
MAX_HEALTHY_TOKENS = 2048


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
    if chunk.fiscal_year < 1990 or chunk.fiscal_year > 2100:
        return False
    return True


def _compute_group_breakdown(
    group_key: str,
    group_chunks: list[FilingChunk],
    expected_dim: int,
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


def evaluate_chunk_health(
    chunks: list[FilingChunk],
    *,
    profiles: list[FinancialDataProfile] | None = None,
    project_id: str = "iron-radio-514217-q7",
    dataset_id: str = "sec_edgar",
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
    missing_profile_ids = [
        pid for pid in expected_profile_ids if pid not in chunks_by_profile
    ]
    incomplete_section_profile_ids = [
        pid
        for pid in expected_profile_ids
        if pid in chunks_by_profile
        and not {"item_1a", "item_7"}.issubset(chunks_by_profile[pid])
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
    by_sector_chunks: dict[str, list[FilingChunk]] = defaultdict(list)

    for c in chunks:
        by_section_chunks[c.section_id].append(c)
        by_sector_chunks[c.gics_sector or "Unknown"].append(c)

        has_emb, dim_ok, finite_ok = _is_valid_finite_embedding(
            c.embedding, expected_embedding_dim
        )
        if not has_emb:
            anomalies.missing_embedding += 1
        elif not dim_ok:
            anomalies.dimension_mismatch += 1
        elif not finite_ok:
            anomalies.non_finite_embedding += 1
        else:
            valid_embedding_count += 1

        if c.char_count < MIN_HEALTHY_CHARS or not c.chunk_text.strip():
            anomalies.undersized_chunk += 1
        if c.char_count > MAX_HEALTHY_CHARS or c.token_estimate > MAX_HEALTHY_TOKENS:
            anomalies.oversized_chunk += 1
        if not _has_complete_metadata(c):
            anomalies.missing_metadata += 1

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
        sec: _compute_group_breakdown(sec, sec_chunks, expected_embedding_dim)
        for sec, sec_chunks in sorted(by_section_chunks.items())
    }
    by_sector = {
        sector: _compute_group_breakdown(sector, sec_chunks, expected_embedding_dim)
        for sector, sec_chunks in sorted(by_sector_chunks.items())
    }

    # Composite 0-100 health score:
    # - 45 pts: Embedding validity & dimension coverage
    # - 30 pts: Filing & dual-section (Item 1A + Item 7) completeness
    # - 15 pts: Chunk size compliance (neither undersized nor oversized)
    # - 10 pts: Metadata completeness
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


def build_cloud_monitoring_timeseries(
    report: ChunkHealthReport,
    *,
    timestamp_rfc3339: str | None = None,
) -> list[dict[str, Any]]:
    """Construct Google Cloud Monitoring v3 `TimeSeries` payloads from a `ChunkHealthReport`.

    Aligns 1:1 with the custom metric descriptors and dashboard widgets provisioned
    in `terraform/monitoring.tf`.
    """
    end_time = timestamp_rfc3339 or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    resource = {
        "type": "global",
        "labels": {"project_id": report.project_id},
    }
    base_labels = {
        "dataset_id": report.dataset_id,
        "table_id": report.table_id,
    }

    series: list[dict[str, Any]] = []

    def _add_gauge(
        metric_name: str,
        value_type: str,
        value: int | float,
        extra_labels: dict[str, str] | None = None,
    ) -> None:
        labels = dict(base_labels)
        if extra_labels:
            labels.update(extra_labels)
        point_val = (
            {"int64Value": str(int(value))}
            if value_type == "INT64"
            else {"doubleValue": float(value)}
        )
        series.append(
            {
                "metric": {
                    "type": f"{METRIC_PREFIX}/{metric_name}",
                    "labels": labels,
                },
                "resource": resource,
                "metricKind": "GAUGE",
                "valueType": value_type,
                "points": [
                    {
                        "interval": {"endTime": end_time},
                        "value": point_val,
                    }
                ],
            }
        )

    # 1. Overall health score (0-100)
    _add_gauge(
        "health_score",
        "DOUBLE",
        report.health_score,
        {"health_status": report.health_status},
    )

    # 2. Embedding coverage ratio (0.0-1.0)
    _add_gauge(
        "embedding_coverage_ratio",
        "DOUBLE",
        report.embedding_coverage_ratio,
        {"embedding_model": report.embedding_model},
    )

    # 3. 10-K filing coverage ratio (0.0-1.0)
    _add_gauge(
        "filing_coverage_ratio",
        "DOUBLE",
        report.full_section_coverage_ratio,
    )

    # 4. Total chunks broken down by GICS sector & section
    sector_section_counts: dict[tuple[str, str], int] = defaultdict(int)
    for sec_id, sec_breakdown in report.by_section.items():
        _add_gauge(
            "token_count_mean",
            "DOUBLE",
            sec_breakdown.avg_token_estimate,
            {"section_id": sec_id},
        )
    for sector, sector_breakdown in report.by_sector.items():
        _add_gauge(
            "total_chunks",
            "INT64",
            sector_breakdown.total_chunks,
            {"gics_sector": sector, "section_id": "all"},
        )
    for sec_id, sec_breakdown in report.by_section.items():
        _add_gauge(
            "total_chunks",
            "INT64",
            sec_breakdown.total_chunks,
            {"gics_sector": "all", "section_id": sec_id},
        )

    # 5. Anomaly counts by anomaly_type
    anomaly_map = {
        "missing_embedding": report.anomalies.missing_embedding,
        "dimension_mismatch": report.anomalies.dimension_mismatch,
        "non_finite_embedding": report.anomalies.non_finite_embedding,
        "undersized_chunk": report.anomalies.undersized_chunk,
        "oversized_chunk": report.anomalies.oversized_chunk,
        "missing_metadata": report.anomalies.missing_metadata,
        "missing_section_filings": report.anomalies.missing_section_filings,
    }
    for anomaly_type, count in anomaly_map.items():
        _add_gauge(
            "anomaly_chunk_count",
            "INT64",
            count,
            {"anomaly_type": anomaly_type},
        )

    return series


def format_structured_health_log(report: ChunkHealthReport) -> dict[str, Any]:
    """Build a Cloud Logging structured JSON log record for the chunk health report."""
    severity_map = {
        "HEALTHY": "INFO",
        "DEGRADED": "WARNING",
        "UNHEALTHY": "ERROR",
    }
    return {
        "severity": severity_map.get(report.health_status, "INFO"),
        "event_type": "sec_edgar_chunk_health_evaluation",
        "message": (
            f"SEC EDGAR RAG Vector DB Health: {report.health_status} "
            f"(score={report.health_score:.1f}/100, chunks={report.total_chunks}, "
            f"embedding_coverage={report.embedding_coverage_ratio * 100:.1f}%, "
            f"filing_coverage={report.full_section_coverage_ratio * 100:.1f}%)"
        ),
        "project_id": report.project_id,
        "dataset_id": report.dataset_id,
        "table_id": report.table_id,
        "health_score": report.health_score,
        "health_status": report.health_status,
        "total_chunks": report.total_chunks,
        "valid_embedding_count": report.valid_embedding_count,
        "embedding_coverage_ratio": report.embedding_coverage_ratio,
        "total_profiles_expected": report.total_profiles_expected,
        "total_profiles_indexed": report.total_profiles_indexed,
        "full_section_coverage_ratio": report.full_section_coverage_ratio,
        "avg_token_estimate": report.avg_token_estimate,
        "p95_token_estimate": report.p95_token_estimate,
        "anomalies": report.anomalies.model_dump(mode="json"),
        "evaluated_at": report.evaluated_at,
    }


class CloudMonitoringHealthPublisher:
    """Publishes chunk health time-series metrics to Google Cloud Monitoring v3 API."""

    def __init__(
        self,
        *,
        project_id: str,
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
    ) -> None:
        self.project_id = project_id
        self._http_client = http_client
        self._credentials = credentials

    def _get_access_token(self) -> str:
        if self._credentials is None:
            self._credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        if not self._credentials.valid:
            self._credentials.refresh(GoogleAuthRequest())
        return str(self._credentials.token)

    def publish_report(self, report: ChunkHealthReport) -> int:
        """Write custom chunk health time series to Cloud Monitoring and emit structured log."""
        structured_log = format_structured_health_log(report)
        logger.info("Chunk Health Telemetry: %s", json.dumps(structured_log))

        all_series = build_cloud_monitoring_timeseries(report)
        if not all_series:
            return 0

        url = f"https://monitoring.googleapis.com/v3/projects/{self.project_id}/timeSeries"
        token = self._get_access_token()

        client = self._http_client or httpx.Client(timeout=30.0)
        close_client = self._http_client is None
        try:
            # Cloud Monitoring allows up to 200 timeSeries per createTimeSeries request
            for start in range(0, len(all_series), 200):
                batch = all_series[start : start + 200]
                resp = client.post(
                    url,
                    json={"timeSeries": batch},
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                )
                if resp.status_code not in (200, 201):
                    raise RuntimeError(
                        f"Cloud Monitoring createTimeSeries failed (HTTP {resp.status_code}): "
                        f"{resp.text[:400]}"
                    )
        finally:
            if close_client:
                client.close()

        report.published_to_cloud_monitoring = True
        report.published_timeseries_count = len(all_series)
        return len(all_series)
