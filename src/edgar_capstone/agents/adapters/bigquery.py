"""Live BigQuery REST v2 adapter for SEC 10-K metrics, BI views, and `VECTOR_SEARCH`."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import httpx

from edgar_capstone.agents.adapters.registry import (
    GcpAdapterErrorTranslationMixin,
    GcpToolConfig,
)
from edgar_capstone.agents.core.resilience import ToolExecutionError, UpstreamTimeoutError
from edgar_capstone.common.domain import CORE_METRICS, normalize_edgar_ticker
from edgar_capstone.common.formatting import ratio_pct
from edgar_capstone.dataset.models import FinancialDataProfile
from edgar_capstone.gcp.bigquery import (
    BigQueryRestClient,
    bq_array_param,
    bq_scalar_param,
    validate_bq_identifier,
)
from edgar_capstone.gcp.rest_client import GcpRestError


def _distance_to_similarity(raw_distance: Any) -> float:
    """Convert BigQuery cosine distance into a clamped ``[0.0, 1.0]`` similarity score."""
    return max(0.0, min(1.0, 1.0 - float(raw_distance or 0.0)))


def _round_opt(value: Any, ndigits: int = 2) -> float | None:
    """Round an optional numeric cell value, returning ``None`` if absent."""
    return round(float(value), ndigits) if value is not None else None


def _build_scope_where_clauses(
    *,
    start_year: int,
    end_year: int,
    tickers: set[str] | None = None,
    gics_sector: str | None = None,
    gics_sectors: set[str] | None = None,
    section_id: str | None = None,
    extra_clauses: Sequence[str] = (),
) -> tuple[list[str], list[dict[str, Any]]]:
    """Build parameterized WHERE clauses shared across BigQuery metrics & vector queries."""
    clauses = ["fiscal_year BETWEEN @start_year AND @end_year", *extra_clauses]
    params: list[dict[str, Any]] = [
        bq_scalar_param("start_year", "INT64", int(start_year)),
        bq_scalar_param("end_year", "INT64", int(end_year)),
    ]
    if tickers:
        norm_tickers = sorted({normalize_edgar_ticker(t) for t in tickers if t and t.strip()})
        if norm_tickers:
            clauses.append("UPPER(REPLACE(ticker, '.', '-')) IN UNNEST(@tickers)")
            params.append(bq_array_param("tickers", "STRING", norm_tickers))
    if gics_sectors:
        norm_sectors = sorted({s.strip().lower() for s in gics_sectors if s and s.strip()})
        if norm_sectors:
            clauses.append("LOWER(gics_sector) IN UNNEST(@sectors)")
            params.append(bq_array_param("sectors", "STRING", norm_sectors))
    elif gics_sector:
        clauses.append("LOWER(gics_sector) = @gics_sector")
        params.append(bq_scalar_param("gics_sector", "STRING", gics_sector.strip().lower()))
    if section_id:
        clauses.append("LOWER(section_id) = @section_id")
        params.append(bq_scalar_param("section_id", "STRING", section_id.strip().lower()))
    return clauses, params


def _chunk_health_summary_entry(group_key: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "group_key": group_key,
        "total_chunks": int(row.get("total_chunks") or 0),
        "distinct_profiles": int(row.get("distinct_profiles") or 0),
        "distinct_tickers": int(row.get("distinct_tickers") or 0),
        "valid_embedding_count": int(row.get("valid_embedding_count") or 0),
        "embedding_coverage_ratio": float(row.get("embedding_coverage_ratio") or 0.0),
        "avg_char_count": float(row.get("avg_char_count") or 0.0),
        "avg_token_estimate": float(row.get("avg_token_estimate") or 0.0),
        "min_token_estimate": int(row.get("min_token_estimate") or 0),
        "max_token_estimate": int(row.get("max_token_estimate") or 0),
        "p50_token_estimate": float(row.get("p50_token_estimate") or 0.0),
        "p95_token_estimate": float(row.get("p95_token_estimate") or 0.0),
        "undersized_chunk_count": int(row.get("undersized_chunk_count") or 0),
        "oversized_chunk_count": int(row.get("oversized_chunk_count") or 0),
        "missing_metadata_count": int(row.get("missing_metadata_count") or 0),
    }


def _rollup_section_cells(sec_id: str, cells: list[dict[str, Any]]) -> dict[str, Any]:
    tot_chunks = sum(int(c.get("total_chunks") or 0) for c in cells)
    valid_embs = sum(int(c.get("valid_embedding_count") or 0) for c in cells)
    weighted_chars = sum(
        float(c.get("avg_char_count") or 0.0) * int(c.get("total_chunks") or 0) for c in cells
    )
    weighted_tokens = sum(
        float(c.get("avg_token_estimate") or 0.0) * int(c.get("total_chunks") or 0) for c in cells
    )
    return {
        "group_key": sec_id,
        "total_chunks": tot_chunks,
        "distinct_profiles": sum(int(c.get("distinct_profiles") or 0) for c in cells),
        "distinct_tickers": sum(int(c.get("distinct_tickers") or 0) for c in cells),
        "valid_embedding_count": valid_embs,
        "embedding_coverage_ratio": (round(valid_embs / tot_chunks, 4) if tot_chunks > 0 else 0.0),
        "avg_char_count": (round(weighted_chars / tot_chunks, 2) if tot_chunks > 0 else 0.0),
        "avg_token_estimate": (round(weighted_tokens / tot_chunks, 2) if tot_chunks > 0 else 0.0),
        "min_token_estimate": min(
            (int(c.get("min_token_estimate") or 0) for c in cells), default=0
        ),
        "max_token_estimate": max(
            (int(c.get("max_token_estimate") or 0) for c in cells), default=0
        ),
        "undersized_chunk_count": sum(int(c.get("undersized_chunk_count") or 0) for c in cells),
        "oversized_chunk_count": sum(int(c.get("oversized_chunk_count") or 0) for c in cells),
        "missing_metadata_count": sum(int(c.get("missing_metadata_count") or 0) for c in cells),
    }


class BigQueryAdapter(GcpAdapterErrorTranslationMixin, BigQueryRestClient):
    """Live BigQuery REST v2 adapter for SEC 10-K metrics, BI views, and `VECTOR_SEARCH`."""

    def __init__(
        self,
        config: GcpToolConfig,
        *,
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
    ) -> None:
        super().__init__(
            project_id=config.project_id,
            dataset_id=config.bq_dataset_id,
            location=config.location,
            http_client=http_client,
            credentials=credentials,
            default_timeout=45.0,
        )
        self.financial_metrics_table = validate_bq_identifier(
            config.bq_financial_metrics_table, "financial_metrics_table"
        )
        self.filing_chunks_table = validate_bq_identifier(
            config.bq_filing_chunks_table, "filing_chunks_table"
        )
        self.yoy_view = validate_bq_identifier(config.bq_yoy_view, "yoy_view")
        self.chunk_health_view = validate_bq_identifier(
            config.bq_chunk_health_view, "chunk_health_view"
        )
        self.sec_corpus_bucket = config.sec_corpus_bucket
        self.embedding_dimensions = config.embedding_dimensions

    def _table_ref(self, table_or_view: str) -> str:
        return self.table_ref(table_or_view)

    def run_query(
        self,
        sql: str,
        query_parameters: list[dict[str, Any]] | None = None,
        *,
        timeout_ms: int = 30000,
        max_results: int = 10000,
        max_polls: int = 20,
        poll_interval_seconds: float = 0.1,
    ) -> list[dict[str, Any]]:
        """Execute a parameterized Standard SQL query on BigQuery and return decoded rows."""
        try:
            return super().run_query(
                sql,
                query_parameters=query_parameters,
                timeout_ms=timeout_ms,
                max_results=max_results,
                max_polls=max_polls,
                poll_interval_seconds=poll_interval_seconds,
            )
        except GcpRestError as exc:
            raise UpstreamTimeoutError(str(exc)) from exc

    def _row_to_profile(self, row: dict[str, Any]) -> FinancialDataProfile:
        """Convert a decoded `financial_metrics` BigQuery row into `FinancialDataProfile`."""
        ticker = str(row["ticker"])
        fy = int(row["fiscal_year"])
        unstructured_blob = str(
            row.get("unstructured_blob_path") or f"unstructured/{ticker}/{fy}_10k.md"
        )
        structured_blob = str(
            row.get("structured_blob_path") or f"structured/{ticker}/{fy}_financials.json"
        )
        prov = row.get("metric_provenance")
        if not isinstance(prov, dict):
            prov = {}
        else:
            prov = {
                **{metric: prov.get(metric) for metric in CORE_METRICS},
                "notes": list(prov.get("notes") or []),
            }

        payload: dict[str, Any] = {
            "profile_id": str(row.get("profile_id") or f"{ticker}_{fy}"),
            "ticker": ticker,
            "cik": str(row.get("cik") or "0000000000"),
            "company_name": str(row.get("company_name") or ticker),
            "gics_sector": str(row.get("gics_sector") or "Unknown"),
            "gics_sub_industry": str(row.get("gics_sub_industry") or "Unknown"),
            "form_type": str(row.get("form_type") or "10-K"),
            "fiscal_year": fy,
            "fiscal_period": str(row.get("fiscal_period") or "FY"),
            "period_of_report": str(row.get("period_of_report") or ""),
            "filing_date": str(row.get("filing_date") or ""),
            "accession_number": str(row.get("accession_number") or ""),
            "currency": str(row.get("currency") or "USD"),
            "revenue": row.get("revenue"),
            "operating_income": row.get("operating_income"),
            "net_income": row.get("net_income"),
            "research_and_development": row.get("research_and_development"),
            "metric_provenance": prov,
            "extracted_sections": list(row.get("extracted_sections") or []),
            "unstructured_blob_path": unstructured_blob,
            "structured_blob_path": structured_blob,
            "unstructured_gcs_uri": str(
                row.get("unstructured_gcs_uri")
                or f"gs://{self.sec_corpus_bucket}/{unstructured_blob}"
            ),
            "structured_gcs_uri": str(
                row.get("structured_gcs_uri") or f"gs://{self.sec_corpus_bucket}/{structured_blob}"
            ),
            "sec_filing_url": str(row.get("sec_filing_url") or ""),
        }
        if row.get("ingested_at"):
            payload["ingested_at"] = str(row["ingested_at"])
        return FinancialDataProfile.model_validate(payload)

    def fetch_profiles(
        self,
        *,
        tickers: set[str],
        start_year: int,
        end_year: int,
        gics_sector: str | None = None,
        gics_sectors: set[str] | None = None,
    ) -> list[FinancialDataProfile]:
        """Query `sec_edgar.financial_metrics` with partition (`fiscal_year`) & cluster pruning."""
        clauses, params = _build_scope_where_clauses(
            start_year=start_year,
            end_year=end_year,
            tickers=tickers,
            gics_sector=gics_sector,
            gics_sectors=gics_sectors,
        )
        sql = (
            f"SELECT * FROM {self._table_ref(self.financial_metrics_table)} "
            f"WHERE {' AND '.join(clauses)} "
            f"ORDER BY ticker ASC, fiscal_year ASC"
        )
        rows = self.run_query(sql, params)
        return [self._row_to_profile(r) for r in rows]

    def query_yoy_variances(
        self,
        *,
        tickers: set[str],
        gics_sector: str | None,
        start_year: int,
        end_year: int,
    ) -> list[dict[str, Any]]:
        """Query `sec_edgar.v_financial_variances_yoy` for Looker YoY & peer comparison explores."""
        clauses, params = _build_scope_where_clauses(
            start_year=start_year,
            end_year=end_year,
            tickers=tickers,
            gics_sector=gics_sector,
        )
        sql = (
            "SELECT "
            "profile_id, ticker, company_name, gics_sector, fiscal_year, prev_fiscal_year, "
            "period_of_report, filing_date, accession_number, "
            "revenue, revenue_yoy_delta, revenue_yoy_pct, "
            "operating_income, operating_income_yoy_delta, operating_income_yoy_pct, "
            "net_income, net_income_yoy_delta, net_income_yoy_pct, "
            "research_and_development, research_and_development_yoy_delta, "
            "research_and_development_yoy_pct, sec_filing_url "
            f"FROM {self._table_ref(self.yoy_view)} "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY ticker ASC, fiscal_year ASC"
        )
        raw_rows = self.run_query(sql, params)
        formatted: list[dict[str, Any]] = []
        for r in raw_rows:
            rev = r.get("revenue")
            abs_rev = abs(float(rev)) if rev else None
            entry: dict[str, Any] = {
                "profile_id": r.get("profile_id"),
                "ticker": r.get("ticker"),
                "company_name": r.get("company_name"),
                "gics_sector": r.get("gics_sector") or "Unknown",
                "fiscal_year": int(r["fiscal_year"]),
                "prev_fiscal_year": (
                    int(r["prev_fiscal_year"]) if r.get("prev_fiscal_year") is not None else None
                ),
                "period_of_report": str(r.get("period_of_report") or "") or None,
                "filing_date": str(r.get("filing_date") or "") or None,
                "accession_number": str(r.get("accession_number") or "") or None,
                "operating_margin_pct": ratio_pct(
                    float(r["operating_income"]) if r.get("operating_income") is not None else None,
                    abs_rev,
                ),
                "net_margin_pct": ratio_pct(
                    float(r["net_income"]) if r.get("net_income") is not None else None,
                    abs_rev,
                ),
                "rd_intensity_pct": ratio_pct(
                    float(r["research_and_development"])
                    if r.get("research_and_development") is not None
                    else None,
                    abs_rev,
                ),
                "sec_filing_url": r.get("sec_filing_url") or "",
            }
            for metric in CORE_METRICS:
                entry[metric] = r.get(metric)
                entry[f"{metric}_yoy_delta"] = _round_opt(r.get(f"{metric}_yoy_delta"))
                entry[f"{metric}_yoy_pct"] = _round_opt(r.get(f"{metric}_yoy_pct"))
            formatted.append(entry)
        return formatted

    def query_sector_benchmarks(
        self,
        *,
        tickers: set[str],
        gics_sector: str | None,
        start_year: int,
        end_year: int,
    ) -> list[dict[str, Any]]:
        """Aggregate sector-level revenue, operating income, net income, and R&D benchmarks in BigQuery."""
        clauses, params = _build_scope_where_clauses(
            start_year=start_year,
            end_year=end_year,
            tickers=tickers,
            gics_sector=gics_sector,
        )
        sql = (
            "SELECT "
            "COALESCE(gics_sector, 'Unknown') AS gics_sector, "
            "fiscal_year, "
            "COUNT(*) AS company_count, "
            "ARRAY_AGG(DISTINCT ticker ORDER BY ticker) AS tickers, "
            "ROUND(SUM(revenue), 2) AS total_sector_revenue, "
            "ROUND(AVG(revenue), 2) AS avg_company_revenue, "
            "ROUND(SUM(operating_income), 2) AS total_sector_operating_income, "
            "ROUND(SUM(net_income), 2) AS total_sector_net_income, "
            "ROUND(SUM(research_and_development), 2) AS total_sector_research_and_development "
            f"FROM {self._table_ref(self.financial_metrics_table)} "
            f"WHERE {' AND '.join(clauses)} "
            "GROUP BY 1, 2 "
            "ORDER BY gics_sector ASC, fiscal_year ASC"
        )
        raw_rows = self.run_query(sql, params)
        return [
            {
                "gics_sector": str(r.get("gics_sector") or "Unknown"),
                "fiscal_year": int(r["fiscal_year"]),
                "company_count": int(r.get("company_count") or 0),
                "tickers": list(r.get("tickers") or []),
                "total_sector_revenue": r.get("total_sector_revenue"),
                "avg_company_revenue": r.get("avg_company_revenue"),
                "total_sector_operating_income": r.get("total_sector_operating_income"),
                "total_sector_net_income": r.get("total_sector_net_income"),
                "total_sector_research_and_development": r.get(
                    "total_sector_research_and_development"
                ),
            }
            for r in raw_rows
        ]

    def query_chunk_health(self) -> dict[str, Any]:
        """Query `sec_edgar.v_chunk_health_metrics` and format Looker `chunk_health` response."""
        sql = (
            f"SELECT * FROM {self._table_ref(self.chunk_health_view)} "
            "ORDER BY gics_sector ASC, section_id ASC"
        )
        rows = self.run_query(sql)
        if not rows:
            raise ToolExecutionError(
                f"BigQuery view {self.dataset_id}.{self.chunk_health_view} returned 0 rows."
            )

        overall: dict[str, Any] = {}
        by_sector: dict[str, dict[str, Any]] = {}
        by_section: dict[str, dict[str, Any]] = {}
        section_cells: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for r in rows:
            sec_sector = str(r.get("gics_sector") or "")
            sec_id = str(r.get("section_id") or "")
            if sec_sector == "ALL_SECTORS" and sec_id == "ALL_SECTIONS":
                overall = r
            elif sec_sector != "ALL_SECTORS" and sec_id == "ALL_SECTIONS":
                by_sector[sec_sector] = _chunk_health_summary_entry(sec_sector, r)
            elif sec_sector == "ALL_SECTORS" and sec_id != "ALL_SECTIONS":
                by_section[sec_id] = _chunk_health_summary_entry(sec_id, r)
            elif sec_sector != "ALL_SECTORS" and sec_id != "ALL_SECTIONS":
                section_cells[sec_id].append(r)

        if not by_section and section_cells:
            for sec_id, cells in sorted(section_cells.items()):
                by_section[sec_id] = _rollup_section_cells(sec_id, cells)

        if not overall:
            overall = rows[0]

        return {
            "status": "success",
            "explore_name": "chunk_health",
            "health_status": overall.get("health_status", "HEALTHY"),
            "health_score": overall.get("health_score"),
            "total_chunks": overall.get("total_chunks"),
            "embedding_coverage_ratio": overall.get("embedding_coverage_ratio"),
            "full_section_coverage_ratio": overall.get(
                "full_section_coverage_ratio",
                overall.get("filing_coverage_ratio"),
            ),
            "by_section": by_section,
            "by_sector": by_sector,
        }

    def vector_search_chunks(
        self,
        query_embedding: list[float],
        *,
        tickers: set[str],
        gics_sector: str | None,
        section_id: str | None,
        start_year: int,
        end_year: int,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Execute metadata-prefiltered BigQuery `VECTOR_SEARCH` over `sec_edgar.filing_chunks`."""
        clauses, params = _build_scope_where_clauses(
            start_year=start_year,
            end_year=end_year,
            tickers=tickers,
            gics_sector=gics_sector,
            section_id=section_id,
            extra_clauses=("ARRAY_LENGTH(embedding) = @embedding_dim",),
        )
        params.extend(
            [
                bq_scalar_param("embedding_dim", "INT64", int(len(query_embedding))),
                bq_array_param("query_embedding", "FLOAT64", query_embedding),
                bq_scalar_param("top_k", "INT64", int(max(1, min(top_k, 500)))),
            ]
        )
        sql = (
            "SELECT "
            "base.chunk_id, base.profile_id, base.ticker, base.company_name, "
            "base.gics_sector, base.fiscal_year, base.period_of_report, base.filing_date, "
            "base.accession_number, base.section_id, base.section_title, "
            "base.chunk_index, base.chunk_text, base.sec_filing_url, "
            "base.unstructured_gcs_uri, distance "
            "FROM VECTOR_SEARCH("
            "  ("
            "    SELECT chunk_id, profile_id, ticker, company_name, gics_sector, "
            "           fiscal_year, period_of_report, filing_date, accession_number, "
            "           section_id, section_title, chunk_index, "
            "           chunk_text, embedding, sec_filing_url, unstructured_gcs_uri "
            f"   FROM {self._table_ref(self.filing_chunks_table)} "
            f"   WHERE {' AND '.join(clauses)}"
            "  ), "
            "  'embedding', "
            "  (SELECT @query_embedding AS embedding), "
            "  top_k => @top_k, "
            "  distance_type => 'COSINE'"
            ") "
            "ORDER BY distance ASC"
        )
        rows = self.run_query(sql, params)
        return [
            {
                **r,
                "similarity": _distance_to_similarity(r.get("distance")),
                "source_backend": "bigquery_vector_search",
            }
            for r in rows
        ]

    def vector_search_sector_top_per_ticker(
        self,
        query_embedding: list[float],
        *,
        gics_sector: str,
        start_year: int,
        end_year: int,
        section_id: str | None = None,
        per_company_year: int = 3,
    ) -> list[dict[str, Any]]:
        """Retrieve top-N cosine-ranked chunks for EVERY issuer and fiscal year in a GICS sector.

        Uses one BigQuery SQL query (`ML.DISTANCE` + `QUALIFY ROW_NUMBER()`) so sector-wide
        thematic queries cover all ~76 Information Technology (or other GICS sector) issuers
        in ~2-3 seconds instead of issuing hundreds of per-company RPCs.
        """
        eff_section = (
            section_id.strip().lower()
            if section_id and section_id.strip().lower() in ("item_1a", "item_7")
            else None
        )
        return self._top_per_company_year(
            query_embedding,
            start_year=start_year,
            end_year=end_year,
            gics_sector=gics_sector,
            section_id=eff_section,
            per_company_year=per_company_year,
            source_backend="bigquery_sector_vector_search",
        )

    def vector_search_top_per_company_year(
        self,
        query_embedding: list[float],
        *,
        tickers: set[str],
        start_year: int,
        end_year: int,
        section_id: str | None = None,
        per_company_year: int = 2,
    ) -> list[dict[str, Any]]:
        """Retrieve the top-N cosine-ranked chunks for each `(ticker, fiscal_year)` in scope.

        Ticker-scoped counterpart of `vector_search_sector_top_per_ticker`, used by hybrid RAG to
        guarantee every company-year in a multi-year / multi-company window has at least one
        candidate passage. The prefiltered row set is small (a few companies x a few years), so
        an exact `ML.DISTANCE` scan in one query costs about the same as `VECTOR_SEARCH`.
        """
        return self._top_per_company_year(
            query_embedding,
            start_year=start_year,
            end_year=end_year,
            tickers=tickers,
            section_id=section_id,
            per_company_year=per_company_year,
            source_backend="bigquery_coverage_vector_search",
        )

    def _top_per_company_year(
        self,
        query_embedding: list[float],
        *,
        start_year: int,
        end_year: int,
        tickers: set[str] | None = None,
        gics_sector: str | None = None,
        section_id: str | None = None,
        per_company_year: int,
        source_backend: str,
    ) -> list[dict[str, Any]]:
        clauses, params = _build_scope_where_clauses(
            start_year=start_year,
            end_year=end_year,
            tickers=tickers,
            gics_sector=gics_sector,
            section_id=section_id,
            extra_clauses=("ARRAY_LENGTH(embedding) = @embedding_dim",),
        )
        params.extend(
            [
                bq_scalar_param("embedding_dim", "INT64", int(len(query_embedding))),
                bq_array_param("query_embedding", "FLOAT64", query_embedding),
                bq_scalar_param(
                    "per_company_year", "INT64", int(max(1, min(per_company_year, 10)))
                ),
            ]
        )
        sql = (
            "SELECT "
            "chunk_id, profile_id, ticker, company_name, gics_sector, fiscal_year, "
            "period_of_report, filing_date, accession_number, section_id, section_title, "
            "chunk_index, chunk_text, sec_filing_url, unstructured_gcs_uri, "
            "ML.DISTANCE(embedding, @query_embedding, 'COSINE') AS distance "
            f"FROM {self._table_ref(self.filing_chunks_table)} "
            f"WHERE {' AND '.join(clauses)} "
            "QUALIFY ROW_NUMBER() OVER ("
            "  PARTITION BY ticker, fiscal_year "
            "  ORDER BY ML.DISTANCE(embedding, @query_embedding, 'COSINE') ASC"
            ") <= @per_company_year "
            "ORDER BY ticker ASC, fiscal_year ASC, distance ASC"
        )
        rows = self.run_query(sql, params)
        return [
            {
                **r,
                "similarity": _distance_to_similarity(r.get("distance")),
                "source_backend": source_backend,
            }
            for r in rows
        ]

    def fetch_section_chunks(
        self,
        *,
        ticker: str,
        fiscal_year: int,
        section_id: str,
        include_embedding: bool = True,
    ) -> list[dict[str, Any]]:
        """Fetch canonical chunks (and optionally pre-computed embeddings) for one filing section."""
        emb_col = ", embedding" if include_embedding else ""
        sql = (
            "SELECT chunk_id, profile_id, ticker, company_name, gics_sector, "
            "fiscal_year, period_of_report, filing_date, accession_number, "
            f"section_id, section_title, chunk_index, chunk_text{emb_col}, "
            "sec_filing_url, unstructured_gcs_uri "
            f"FROM {self._table_ref(self.filing_chunks_table)} "
            "WHERE UPPER(REPLACE(ticker, '.', '-')) = @ticker "
            "  AND fiscal_year = @fiscal_year "
            "  AND LOWER(section_id) = @section_id "
            "ORDER BY chunk_index ASC"
        )
        params = [
            bq_scalar_param("ticker", "STRING", normalize_edgar_ticker(ticker)),
            bq_scalar_param("fiscal_year", "INT64", int(fiscal_year)),
            bq_scalar_param("section_id", "STRING", section_id.lower()),
        ]
        return self.run_query(sql, params)

    def fetch_sections_chunks_batch(
        self,
        *,
        tickers: set[str],
        start_year: int,
        end_year: int,
        section_id: str | None = None,
        include_embedding: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch canonical chunks across multiple `(ticker, fiscal_year, section_id)` slices in one RPC."""
        if not tickers:
            return []
        eff_sec = (
            section_id.strip().lower()
            if section_id and section_id.strip().lower() in ("item_1a", "item_7")
            else None
        )
        clauses, params = _build_scope_where_clauses(
            start_year=start_year,
            end_year=end_year,
            tickers=tickers,
            section_id=eff_sec,
        )
        emb_col = ", embedding" if include_embedding else ""
        sql = (
            "SELECT chunk_id, profile_id, ticker, company_name, gics_sector, "
            "fiscal_year, period_of_report, filing_date, accession_number, "
            f"section_id, section_title, chunk_index, chunk_text{emb_col}, "
            "sec_filing_url, unstructured_gcs_uri "
            f"FROM {self._table_ref(self.filing_chunks_table)} "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY ticker ASC, fiscal_year ASC, section_id ASC, chunk_index ASC"
        )
        return self.run_query(sql, params)
