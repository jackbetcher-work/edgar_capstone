"""Parallel hybrid RAG retrieval combining BigQuery `VECTOR_SEARCH` and Vertex AI Search."""

from __future__ import annotations

import logging
import re
from typing import Any

from edgar_capstone.agents.adapters.registry import GcpToolAdapters
from edgar_capstone.agents.adapters.vertex_search import (
    _default_section_title,
    _normalize_section_id,
)
from edgar_capstone.agents.resilience import ToolExecutionError, _is_retryable_exception
from edgar_capstone.agents.schemas import GroundedCitation
from edgar_capstone.common.concurrency import run_parallel_with_context
from edgar_capstone.common.domain import normalize_edgar_ticker

logger = logging.getLogger(__name__)

_WORD_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def _run_parallel_branches(
    adapters: GcpToolAdapters,
    *,
    query: str,
    ticker_filter: set[str],
    sector_norm: str | None,
    sec_norm: str | None,
    start_year: int,
    end_year: int,
    candidate_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Fan out BigQuery `VECTOR_SEARCH` and Vertex AI Search concurrently with context propagation."""

    def _run_branch(
        name: str,
    ) -> tuple[str, list[dict[str, Any]], list[str], Exception | None]:
        if name == "bq":
            try:
                q_emb = adapters.embed_query(query)
                hits = adapters.bigquery.vector_search_chunks(
                    q_emb,
                    tickers=ticker_filter,
                    gics_sector=sector_norm,
                    section_id=sec_norm,
                    start_year=start_year,
                    end_year=end_year,
                    top_k=candidate_k,
                )
                return ("bq", hits, ["vertex_ai_embeddings", "bigquery_vector_search"], None)
            except Exception as exc:
                if _is_retryable_exception(exc)[0]:
                    raise
                logger.warning("BigQuery VECTOR_SEARCH branch failed: %s", exc)
                return ("bq", [], [], exc)
        try:
            hits = adapters.vertex_ai_search.search_documents(
                query,
                tickers=ticker_filter,
                gics_sector=sector_norm,
                section_id=sec_norm,
                start_year=start_year,
                end_year=end_year,
                page_size=candidate_k,
            )
            return ("vais", hits, ["vertex_ai_search"], None)
        except Exception as exc:
            if _is_retryable_exception(exc)[0]:
                raise
            logger.info("Vertex AI Search (Discovery Engine) unavailable: %s", exc)
            return ("vais", [], [], exc)

    results = run_parallel_with_context(["bq", "vais"], _run_branch, max_workers=2)

    bq_hits: list[dict[str, Any]] = []
    vais_hits: list[dict[str, Any]] = []
    services_hit: list[str] = []
    backend_errors: list[Exception] = []

    for branch_name, hits, branch_services, err in results:
        services_hit.extend(branch_services)
        if err is not None:
            backend_errors.append(err)
        elif branch_name == "bq":
            bq_hits = hits
        else:
            vais_hits = hits

    retrieval_backends = {
        s for s in services_hit if s in ("vertex_ai_search", "bigquery_vector_search")
    }
    if not retrieval_backends:
        if backend_errors:
            raise backend_errors[0]
        raise ToolExecutionError(
            "No live GCP RAG retrieval backends (BigQuery VECTOR_SEARCH or Vertex AI Search) succeeded."
        )
    return bq_hits, vais_hits, services_hit


def _upsert_canonical_hit(
    fused: dict[str, dict[str, Any]],
    hit: dict[str, Any],
    rank: int,
    *,
    ticker_filter: set[str],
    sector_norm: str | None,
    sec_norm: str | None,
    start_year: int,
    end_year: int,
) -> None:
    """Validate metadata scope filters and accumulate RRF score for a retrieval candidate."""
    cid = str(hit.get("chunk_id") or "").strip()
    text = str(hit.get("chunk_text") or "").strip()
    ticker = str(hit.get("ticker") or "").strip()
    fy_val = hit.get("fiscal_year")
    if not cid or not text or not ticker or fy_val is None:
        return

    fy = int(fy_val)
    if fy < start_year or fy > end_year:
        return
    if ticker_filter and normalize_edgar_ticker(ticker) not in ticker_filter:
        return
    sector_val = str(hit.get("gics_sector") or "Unknown")
    if sector_norm and sector_val.lower() != sector_norm:
        return
    resolved_sec = _normalize_section_id(
        str(hit.get("section_id") or ""),
        requested_section=sec_norm,
        passage_text=text,
    )
    if sec_norm and resolved_sec != sec_norm:
        return

    rrf_inc = 1.0 / (60.0 + rank)
    sim = float(hit.get("similarity") or 0.0)
    if cid in fused:
        fused[cid]["rrf_score"] += rrf_inc
        fused[cid]["max_similarity"] = max(fused[cid]["max_similarity"], sim)
        return

    fused[cid] = {
        "chunk_id": cid,
        "profile_id": str(hit.get("profile_id") or f"{ticker}_{fy}"),
        "ticker": ticker,
        "company_name": str(hit.get("company_name") or ticker),
        "gics_sector": sector_val,
        "fiscal_year": fy,
        "period_of_report": str(hit.get("period_of_report") or "") or None,
        "filing_date": str(hit.get("filing_date") or "") or None,
        "accession_number": str(hit.get("accession_number") or "") or None,
        "section_id": resolved_sec,
        "section_title": str(hit.get("section_title") or _default_section_title(resolved_sec)),
        "chunk_text": text,
        "sec_filing_url": str(hit.get("sec_filing_url") or ""),
        "unstructured_gcs_uri": str(hit.get("unstructured_gcs_uri") or ""),
        "rrf_score": rrf_inc,
        "max_similarity": sim,
    }


def _reconcile_vais_hits(
    fused: dict[str, dict[str, Any]],
    vais_hits: list[dict[str, Any]],
    *,
    ticker_filter: set[str],
    sector_norm: str | None,
    sec_norm: str | None,
    start_year: int,
    end_year: int,
) -> None:
    """Merge overlapping Discovery Engine extractive segments into matching BigQuery chunks."""
    for rank, vais_hit in enumerate(vais_hits, start=1):
        seg_text = str(vais_hit.get("chunk_text") or "").strip()
        if not seg_text:
            continue
        seg_lower = seg_text.lower()
        seg_probe = seg_lower[:64]
        vais_prof = str(vais_hit.get("profile_id") or "")
        matched_cid: str | None = None
        for cid, cand in fused.items():
            if cand["profile_id"] != vais_prof:
                continue
            cand_lower = cand["chunk_text"].lower()
            if seg_probe in cand_lower or cand_lower[:64] in seg_lower:
                matched_cid = cid
                break
        if matched_cid is not None:
            fused[matched_cid]["rrf_score"] += 1.0 / (60.0 + rank)
            fused[matched_cid]["max_similarity"] = max(
                fused[matched_cid]["max_similarity"],
                float(vais_hit.get("similarity") or 0.0),
            )
        else:
            _upsert_canonical_hit(
                fused,
                vais_hit,
                rank,
                ticker_filter=ticker_filter,
                sector_norm=sector_norm,
                sec_norm=sec_norm,
                start_year=start_year,
                end_year=end_year,
            )


def _rerank_and_build_citations(
    fused: dict[str, dict[str, Any]],
    *,
    query: str,
    effective_top_k: int,
) -> list[dict[str, Any]]:
    """Score candidates with dense + RRF + lexical overlap, clamping emitted scores to `[0, 1]`."""
    query_terms = set(_WORD_TOKEN_RE.findall(query.lower()))
    scored_candidates: list[tuple[float, dict[str, Any]]] = []
    for cand in fused.values():
        text_lower = cand["chunk_text"].lower()
        lexical_boost = 0.0
        if query_terms:
            matched_terms = sum(1 for term in query_terms if term in text_lower)
            lexical_boost = 0.35 * (matched_terms / len(query_terms))
        rrf_norm = min(1.0, cand["rrf_score"] * 30.0)
        hybrid_score = 0.55 * float(cand["max_similarity"]) + 0.30 * rrf_norm + lexical_boost
        scored_candidates.append((hybrid_score, cand))

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    return [
        GroundedCitation.from_row(cand, score=score, clamp_score=True).model_dump(mode="json")
        for score, cand in scored_candidates[:effective_top_k]
    ]


def _execute_live_gcp_hybrid_rag(
    adapters: GcpToolAdapters,
    *,
    query: str,
    ticker_filter: set[str],
    sector_norm: str | None,
    sec_norm: str | None,
    start_year: int,
    end_year: int,
    effective_top_k: int,
) -> tuple[list[dict[str, Any]], int, list[str]]:
    """Run parallel hybrid retrieval across BigQuery `VECTOR_SEARCH` and Vertex AI Search."""
    candidate_k = min(25, max(effective_top_k * 2, 10))
    bq_hits, vais_hits, services_hit = _run_parallel_branches(
        adapters,
        query=query,
        ticker_filter=ticker_filter,
        sector_norm=sector_norm,
        sec_norm=sec_norm,
        start_year=start_year,
        end_year=end_year,
        candidate_k=candidate_k,
    )

    fused: dict[str, dict[str, Any]] = {}
    for rank, bq_hit in enumerate(bq_hits, start=1):
        _upsert_canonical_hit(
            fused,
            bq_hit,
            rank,
            ticker_filter=ticker_filter,
            sector_norm=sector_norm,
            sec_norm=sec_norm,
            start_year=start_year,
            end_year=end_year,
        )

    _reconcile_vais_hits(
        fused,
        vais_hits,
        ticker_filter=ticker_filter,
        sector_norm=sector_norm,
        sec_norm=sec_norm,
        start_year=start_year,
        end_year=end_year,
    )

    citations = _rerank_and_build_citations(
        fused,
        query=query,
        effective_top_k=effective_top_k,
    )
    return citations, len(fused), services_hit
