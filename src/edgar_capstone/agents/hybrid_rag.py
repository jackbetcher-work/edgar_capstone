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

# Coverage-aware retrieval: when a ticker-scoped search spans several company-years, every
# `(ticker, fiscal_year)` slice gets at least one passage before the remaining slots are filled
# by score. Without this, a single global top-k lets one or two strong years crowd out the
# start year or a middle year (e.g. Merck's FY2023 acquired-IPR&D spike).
MAX_COVERAGE_SLICES = 24  # beyond this the scope is too broad to cover slice-by-slice
COVERAGE_CITATION_CAP = 12  # matches prompt_builders._MAX_EVIDENCE_LEADS
_COVERAGE_PER_SLICE = 2


def coverage_slices(
    ticker_filter: set[str],
    start_year: int,
    end_year: int,
) -> list[tuple[str, int]]:
    """Return the `(ticker, fiscal_year)` slices to cover, in priority order.

    Boundary years (start, then end) come first for every ticker, then middle years, so if the
    citation budget runs out the comparison endpoints are the last thing dropped. Returns `[]`
    when coverage does not apply (no tickers, a single company-year, or too many slices).
    """
    tickers = sorted(normalize_edgar_ticker(t) for t in ticker_filter if t)
    if not tickers or end_year < start_year:
        return []
    years = list(range(int(start_year), int(end_year) + 1))
    n_slices = len(tickers) * len(years)
    if n_slices < 2 or n_slices > MAX_COVERAGE_SLICES:
        return []
    boundary = list(dict.fromkeys([years[0], years[-1]]))
    middle = [y for y in years if y not in boundary]
    return [(tk, fy) for fy in [*boundary, *middle] for tk in tickers]


def _run_coverage_query(
    adapters: GcpToolAdapters,
    q_emb: list[float],
    *,
    ticker_filter: set[str],
    sec_norm: str | None,
    start_year: int,
    end_year: int,
) -> list[dict[str, Any]]:
    """Best-effort per-company-year candidates; failures degrade to plain top-k retrieval."""
    try:
        return adapters.bigquery.vector_search_top_per_company_year(
            q_emb,
            tickers=ticker_filter,
            section_id=sec_norm,
            start_year=start_year,
            end_year=end_year,
            per_company_year=_COVERAGE_PER_SLICE,
        )
    except Exception as exc:
        logger.warning("Per-company-year coverage query failed; using plain top-k: %s", exc)
        return []


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
    with_coverage: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Fan out BigQuery `VECTOR_SEARCH` and Vertex AI Search concurrently with context propagation.

    When `with_coverage` is set, the BigQuery branch reuses its query embedding to run the
    per-company-year coverage query concurrently with `VECTOR_SEARCH`, so coverage adds no
    extra embedding call and no extra round-trip on the critical path.
    """

    def _run_branch(
        name: str,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[str], Exception | None]:
        if name == "bq":
            try:
                q_emb = adapters.embed_query(query)

                def _vector_search() -> list[dict[str, Any]]:
                    return adapters.bigquery.vector_search_chunks(
                        q_emb,
                        tickers=ticker_filter,
                        gics_sector=sector_norm,
                        section_id=sec_norm,
                        start_year=start_year,
                        end_year=end_year,
                        top_k=candidate_k,
                    )

                if with_coverage:
                    hits, slice_hits = run_parallel_with_context(
                        [
                            _vector_search,
                            lambda: _run_coverage_query(
                                adapters,
                                q_emb,
                                ticker_filter=ticker_filter,
                                sec_norm=sec_norm,
                                start_year=start_year,
                                end_year=end_year,
                            ),
                        ],
                        lambda fn: fn(),
                        max_workers=2,
                    )
                else:
                    hits, slice_hits = _vector_search(), []
                return (
                    "bq",
                    hits,
                    slice_hits,
                    ["vertex_ai_embeddings", "bigquery_vector_search"],
                    None,
                )
            except Exception as exc:
                if _is_retryable_exception(exc)[0]:
                    raise
                logger.warning("BigQuery VECTOR_SEARCH branch failed: %s", exc)
                return ("bq", [], [], [], exc)
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
            return ("vais", hits, [], ["vertex_ai_search"], None)
        except Exception as exc:
            if _is_retryable_exception(exc)[0]:
                raise
            logger.info("Vertex AI Search (Discovery Engine) unavailable: %s", exc)
            return ("vais", [], [], [], exc)

    results = run_parallel_with_context(["bq", "vais"], _run_branch, max_workers=2)

    bq_hits: list[dict[str, Any]] = []
    vais_hits: list[dict[str, Any]] = []
    coverage_hits: list[dict[str, Any]] = []
    services_hit: list[str] = []
    backend_errors: list[Exception] = []

    for branch_name, hits, slice_hits, branch_services, err in results:
        services_hit.extend(branch_services)
        if err is not None:
            backend_errors.append(err)
        elif branch_name == "bq":
            bq_hits = hits
            coverage_hits = slice_hits
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
    return bq_hits, vais_hits, coverage_hits, services_hit


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
    rrf_weight: float = 1.0,
) -> None:
    """Validate metadata scope filters and accumulate RRF score for a retrieval candidate.

    Coverage candidates use `rrf_weight=0.0`: they make a company-year *eligible* without
    boosting its rank, so ordering still reflects the two primary retrieval backends.
    """
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

    rrf_inc = rrf_weight / (60.0 + rank)
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


def _select_with_coverage(
    scored_candidates: list[tuple[float, dict[str, Any]]],
    *,
    effective_top_k: int,
    slices: list[tuple[str, int]],
) -> list[tuple[float, dict[str, Any]]]:
    """Pick the best passage per company-year slice first, then fill remaining slots by score.

    `scored_candidates` must already be sorted by score (descending). The budget grows past
    `effective_top_k` only as far as needed to cover each slice that has candidates (capped at
    `COVERAGE_CITATION_CAP`); slices are taken in priority order, so boundary years win if the
    cap binds. The result keeps score order.
    """
    if not slices:
        return scored_candidates[:effective_top_k]
    best_by_slice: dict[tuple[str, int], tuple[float, dict[str, Any]]] = {}
    for item in scored_candidates:
        cand = item[1]
        key = (normalize_edgar_ticker(cand["ticker"]), int(cand["fiscal_year"]))
        best_by_slice.setdefault(key, item)
    covered = [s for s in slices if s in best_by_slice]
    budget = min(25, max(effective_top_k, min(len(covered), COVERAGE_CITATION_CAP)))

    chosen: set[str] = set()
    for s in covered:
        if len(chosen) >= budget:
            break
        chosen.add(best_by_slice[s][1]["chunk_id"])
    for _, cand in scored_candidates:
        if len(chosen) >= budget:
            break
        chosen.add(cand["chunk_id"])
    return [item for item in scored_candidates if item[1]["chunk_id"] in chosen]


def _rerank_and_build_citations(
    fused: dict[str, dict[str, Any]],
    *,
    query: str,
    effective_top_k: int,
    slices: list[tuple[str, int]] | None = None,
) -> list[dict[str, Any]]:
    """Score candidates with dense + RRF + lexical overlap, clamping emitted scores to `[0, 1]`.

    With `slices`, selection is coverage-aware (see `_select_with_coverage`); otherwise it is a
    plain top-k by score.
    """
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
    selected = _select_with_coverage(
        scored_candidates, effective_top_k=effective_top_k, slices=slices or []
    )
    return [
        GroundedCitation.from_row(cand, score=score, clamp_score=True).model_dump(mode="json")
        for score, cand in selected
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
    """Run parallel hybrid retrieval across BigQuery `VECTOR_SEARCH` and Vertex AI Search.

    Ticker-scoped searches spanning 2+ company-years are coverage-aware: a per-company-year
    candidate query runs alongside `VECTOR_SEARCH`, and selection guarantees each company-year
    with matching passages at least one citation.
    """
    candidate_k = min(25, max(effective_top_k * 2, 10))
    slices = coverage_slices(ticker_filter, start_year, end_year)
    bq_hits, vais_hits, coverage_hits, services_hit = _run_parallel_branches(
        adapters,
        query=query,
        ticker_filter=ticker_filter,
        sector_norm=sector_norm,
        sec_norm=sec_norm,
        start_year=start_year,
        end_year=end_year,
        candidate_k=candidate_k,
        with_coverage=bool(slices),
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

    for rank, cov_hit in enumerate(coverage_hits, start=1):
        _upsert_canonical_hit(
            fused,
            cov_hit,
            rank,
            ticker_filter=ticker_filter,
            sector_norm=sector_norm,
            sec_norm=sec_norm,
            start_year=start_year,
            end_year=end_year,
            rrf_weight=0.0,
        )

    citations = _rerank_and_build_citations(
        fused,
        query=query,
        effective_top_k=effective_top_k,
        slices=slices,
    )
    return citations, len(fused), services_hit
