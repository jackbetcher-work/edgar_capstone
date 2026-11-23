"""Full-sector batched map-reduce pipeline for GICS sector-wide thematic queries.

When a user asks a sector-wide question (e.g. "How has AI risk wording changed across all
Information Technology companies in the S&P 500 since 2024?"), this module executes a
principled, verifiable batched map-reduce:
1. **Enumerate sector issuers** from the live BigQuery `financial_metrics` catalog (reports exact
   counts of total issuers, issuers with filings per year, and missing fiscal years like FY2026).
2. **Single-query sector retrieval** via `BigQueryAdapter.vector_search_sector_top_per_ticker`
   (`ML.DISTANCE(embedding, @query_embedding, 'COSINE')` + `QUALIFY ROW_NUMBER()`).
3. **Deterministic per-company alignment & mention counts**: Uses the word-trigram algorithm from
   `disclosure_deltas.py` across every issuer's retrieved passages.
4. **Parallel batched map (`gemini-3.5-flash-lite`)**: Groups matching companies into batches of ~14
   issuers and summarizes each batch concurrently with constrained JSON (`SectorBatchSummary`),
   verifying every cited `chunk_id` and verbatim quote against the real batch chunks.
5. **Reduce-ready output**: Returns verified batch digests, deterministic sector coverage metadata,
   merged Disclosure Delta structures, and prioritized `citations` traceable to real 10-K filings.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.genai import types as genai_types
from pydantic import BaseModel, Field

from edgar_capstone.agents.runtime.live_runtime import (
    resolve_live_adk_runtime,
    run_adk_agent_stream,
)
from edgar_capstone.agents.runtime.model_config import (
    low_thinking_config_for_model,
    resolve_model_for_task,
)
from edgar_capstone.agents.tools.disclosure_deltas import (
    SectionMaterial,
    build_company_evolution,
    chunk_row_to_citation,
    count_theme_mentions,
    resolve_theme_terms,
)
from edgar_capstone.agents.tools.sec_tools import _gcp_adapters
from edgar_capstone.common.concurrency import run_parallel_with_context
from edgar_capstone.common.domain import normalize_edgar_ticker, sha256_text
from edgar_capstone.common.formatting import normalize_quote_text, parse_model_output
from edgar_capstone.dataset.models import FinancialDataProfile
from edgar_capstone.observability.tracing import set_span_attributes, traced

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 14
DEFAULT_PER_COMPANY_YEAR = 3
DEFAULT_MAX_MAP_WORKERS = 6


class BatchEvidenceItem(BaseModel):
    """One verifiable company-level citation surfaced by a map batch."""

    ticker: str
    fiscal_year: int
    chunk_id: str
    verbatim_quote: str = Field(
        ...,
        description="Verbatim quote (10-35 words) copied directly from the referenced chunk_id.",
    )
    takeaway: str = Field(
        ...,
        description="One concise sentence explaining the theme trend or wording change.",
    )


class SectorBatchSummary(BaseModel):
    """Structured map summary produced by Gemini Flash for one batch of sector issuers."""

    batch_index: int
    tickers_covered: list[str] = Field(default_factory=list)
    dominant_themes: list[str] = Field(
        default_factory=list,
        description="Up to 4 recurring themes observed across companies in this batch.",
    )
    wording_shifts: list[str] = Field(
        default_factory=list,
        description="Up to 4 concise observations on how disclosure wording changed YoY.",
    )
    evidence: list[BatchEvidenceItem] = Field(
        default_factory=list,
        description="Up to 6 representative company-level claims backed by verbatim quotes and real chunk_ids.",
    )


def _normalize_for_quote_match(text: str) -> str:
    """Normalize smart punctuation, dashes, markdown escapes/emphasis, and underscores for matching."""
    cleaned = normalize_quote_text((text or "").replace("\\", ""))
    return " ".join(cleaned.replace("*", "").replace("_", " ").split())


def _create_sector_map_agent(model: str | BaseLlm) -> Agent:
    return Agent(
        name="SectorMapSummarizerAgent",
        model=model,
        description=(
            "Summarizes a batch of SEC 10-K company disclosures and year-over-year wording deltas "
            "into a strictly grounded JSON digest preserving company-level chunk_id citations."
        ),
        instruction=(
            "You are a financial disclosure analyst executing the MAP phase of a sector-wide "
            "SEC Form 10-K thematic review. Summarize ONLY the passages provided for your batch "
            "of companies. Output valid JSON matching `SectorBatchSummary`. Every item in `evidence` "
            "MUST use an exact `chunk_id` from the input batch and copy `verbatim_quote` (10-35 words) "
            "verbatim from that chunk's `excerpt`. Never invent chunk IDs, companies, or statistics."
        ),
        output_schema=SectorBatchSummary,
        output_key="sector_batch_summary",
        generate_content_config=genai_types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=4096,
            thinking_config=low_thinking_config_for_model(model),
        ),
    )


def _verify_batch_evidence(
    evidence_items: list[BatchEvidenceItem],
    chunk_lookup: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Filter batch evidence down to items whose `chunk_id` and verbatim quote exist in the batch."""
    verified: list[dict[str, Any]] = []
    dropped = 0
    for item in evidence_items[:6]:
        cit = chunk_lookup.get(item.chunk_id)
        if cit is None:
            dropped += 1
            continue
        excerpt_norm = _normalize_for_quote_match(
            str(cit.get("excerpt") or cit.get("chunk_text") or "")
        )
        quote_norm = _normalize_for_quote_match(item.verbatim_quote)
        if len(quote_norm) < 16 or quote_norm not in excerpt_norm:
            dropped += 1
            continue
        verified.append(
            {
                "ticker": str(cit["ticker"]),
                "fiscal_year": int(cit["fiscal_year"]),
                "period_of_report": cit.get("period_of_report"),
                "accession_number": cit.get("accession_number"),
                "section_id": cit.get("section_id"),
                "chunk_id": item.chunk_id,
                "verbatim_quote": item.verbatim_quote.strip(),
                "takeaway": item.takeaway.strip(),
            }
        )
    return verified, dropped


def _build_batch_prompt(
    *,
    batch_index: int,
    total_batches: int,
    gics_sector: str,
    theme: str,
    start_year: int,
    end_year: int,
    tickers_in_batch: list[str],
    company_payloads: list[dict[str, Any]],
) -> str:
    return (
        f"## Sector Map Batch {batch_index + 1}/{total_batches}\n"
        f"- GICS Sector: `{gics_sector}`\n"
        f"- Theme / Question: {theme}\n"
        f"- Fiscal Window: FY{start_year}–FY{end_year}\n"
        f"- Companies in batch ({len(tickers_in_batch)}): {', '.join(tickers_in_batch)}\n\n"
        "Summarize the recurring themes and year-over-year wording changes across these companies, "
        "returning JSON matching `SectorBatchSummary` (`batch_index`, `tickers_covered`, "
        "`dominant_themes`, `wording_shifts`, `evidence`). Pick up to 6 strong representative "
        "`evidence` items spanning different companies and years, copying `verbatim_quote` exactly "
        "from the chunk's `excerpt`.\n\n"
        f"```json\n{json.dumps(company_payloads, default=str)}\n```"
    )


def _summarize_batch_via_flash(
    *,
    batch_index: int,
    total_batches: int,
    gics_sector: str,
    theme: str,
    start_year: int,
    end_year: int,
    company_payloads: list[dict[str, Any]],
    chunk_lookup: dict[str, dict[str, Any]],
    map_llm: BaseLlm | None = None,
) -> dict[str, Any]:
    """Run one map batch through Gemini 3.5 Flash-Lite and filter out any unverifiable citations."""
    tickers_in_batch = [str(c["ticker"]) for c in company_payloads]
    prompt = _build_batch_prompt(
        batch_index=batch_index,
        total_batches=total_batches,
        gics_sector=gics_sector,
        theme=theme,
        start_year=start_year,
        end_year=end_year,
        tickers_in_batch=tickers_in_batch,
        company_payloads=company_payloads,
    )

    resolved = resolve_live_adk_runtime(custom_llm=map_llm)
    if map_llm is not None:
        model_obj: str | BaseLlm = map_llm
        model_used = map_llm.model
    else:
        from edgar_capstone.agents.runtime.live_runtime import build_adk_llm_for_task

        model_obj = build_adk_llm_for_task("context_compaction", resolution=resolved)
        model_used = resolve_model_for_task("context_compaction")

    stream_res = run_adk_agent_stream(
        agent=_create_sector_map_agent(model_obj),
        prompt_text=prompt,
        user_id="sector_mapper",
        session_id=f"sector-map-{batch_index}-{uuid.uuid4().hex[:8]}",
        app_name="edgar_sector_map_app",
        state_delta={"_model_armor_verified_prompt_sha256": sha256_text(prompt)},
        use_adk_app=False,
        backend=resolved.backend,
        max_llm_calls=2,
        plugins=[],
    )

    raw_obj = stream_res.session_state.get("sector_batch_summary")
    parsed = parse_model_output(raw_obj, SectorBatchSummary, fallback_text=stream_res.final_text)
    if parsed is None:
        logger.warning(
            "Sector map batch %d/%d did not parse into SectorBatchSummary; dropping unparsed output.",
            batch_index + 1,
            total_batches,
        )
        return {
            "batch_index": batch_index,
            "tickers_covered": tickers_in_batch,
            "dominant_themes": [],
            "wording_shifts": [],
            "evidence": [],
            "verified_evidence_count": 0,
            "dropped_evidence_count": 0,
            "model_used": model_used,
        }

    verified_evidence, dropped = _verify_batch_evidence(parsed.evidence, chunk_lookup)
    return {
        "batch_index": batch_index,
        "tickers_covered": tickers_in_batch,
        "dominant_themes": [t.strip() for t in parsed.dominant_themes[:4] if t.strip()],
        "wording_shifts": [w.strip() for w in parsed.wording_shifts[:4] if w.strip()],
        "evidence": verified_evidence,
        "verified_evidence_count": len(verified_evidence),
        "dropped_evidence_count": dropped,
        "model_used": model_used,
    }


def _load_sector_issuer_catalog(
    gics_sector: str,
    start_year: int,
    end_year: int,
    sector_profiles: list[FinancialDataProfile] | None,
) -> tuple[dict[str, set[int]], dict[str, str]]:
    """Load sector profiles (or filter injected profiles) into `(issuer_years, issuer_names)`."""
    if sector_profiles is None:
        profiles = _gcp_adapters().bigquery.fetch_profiles(
            tickers=set(),
            start_year=start_year,
            end_year=end_year,
            gics_sector=gics_sector,
        )
    else:
        target = gics_sector.strip().lower()
        profiles = [
            p
            for p in sector_profiles
            if str(p.gics_sector or "").strip().lower() == target
            and start_year <= p.fiscal_year <= end_year
        ]

    issuer_years: dict[str, set[int]] = defaultdict(set)
    issuer_names: dict[str, str] = {}
    for p in profiles:
        tk = normalize_edgar_ticker(p.ticker)
        issuer_years[tk].add(int(p.fiscal_year))
        issuer_names[tk] = p.company_name or tk
    return issuer_years, issuer_names


def _compute_sector_evolutions(
    *,
    analyzed_tickers: list[str],
    rows_by_ticker_fy: dict[str, dict[int, list[dict[str, Any]]]],
    issuer_names: dict[str, str],
    gics_sector: str,
    window: list[int],
    theme_terms: list[str],
    issuers_in_sector: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int], dict[str, int], int]:
    """Compute per-company disclosure evolutions, mention totals, and compact Flash payloads."""
    evolutions: list[dict[str, Any]] = []
    company_map_inputs: list[dict[str, Any]] = []
    delta_summary = {
        "added": 0,
        "removed": 0,
        "modified": 0,
        "unchanged": 0,
        "year_pairs": 0,
        "companies_analyzed": len(analyzed_tickers),
        "issuers_in_sector": issuers_in_sector,
    }
    mentions_by_year: dict[str, int] = {str(fy): 0 for fy in window}
    issuers_with_theme_matches = 0

    for tk in analyzed_tickers:
        fy_dict = rows_by_ticker_fy[tk]
        avail_yrs = sorted(fy_dict.keys())
        sections: dict[tuple[int, str], SectionMaterial] = {}
        theme_passages: dict[int, list[dict[str, Any]]] = {}
        company_mentions = 0
        best_sim = 0.0
        for fy in avail_yrs:
            cits = fy_dict[fy]
            theme_passages[fy] = cits
            by_sec: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for c in cits:
                best_sim = max(best_sim, float(c.get("score") or 0.0))
                by_sec[str(c.get("section_id") or "item_1a")].append(c)
                company_mentions += count_theme_mentions(str(c.get("excerpt") or ""), theme_terms)
            for sec_id, sec_cits in by_sec.items():
                sections[(fy, sec_id)] = SectionMaterial(
                    ticker=tk,
                    company_name=issuer_names.get(tk, tk),
                    fiscal_year=fy,
                    section_id=sec_id,
                    chunks=sec_cits,
                )

        if company_mentions > 0 or best_sim >= 0.45:
            issuers_with_theme_matches += 1

        evo, _ = build_company_evolution(
            ticker=tk,
            company_name=issuer_names.get(tk, tk),
            gics_sector=gics_sector,
            window_years=window,
            available_years=avail_yrs,
            theme_terms=theme_terms,
            sections=sections,
            theme_passages=theme_passages,
        )
        evolutions.append(evo.model_dump(mode="json"))

        for yr_str, cnt in evo.theme_mentions.items():
            mentions_by_year[yr_str] = mentions_by_year.get(yr_str, 0) + int(cnt)
        for pair in evo.year_pairs:
            delta_summary["year_pairs"] += 1
            for kind in ("added", "removed", "modified", "unchanged"):
                delta_summary[kind] += pair.counts.get(kind, 0)

        compact_passages = [
            {
                "fiscal_year": fy,
                "period_of_report": c.get("period_of_report"),
                "section_id": c.get("section_id"),
                "chunk_id": c["chunk_id"],
                "score": c.get("score"),
                "excerpt": " ".join(str(c.get("excerpt") or "").split())[:520],
            }
            for fy in avail_yrs
            for c in fy_dict[fy][:2]
        ]
        company_map_inputs.append(
            {
                "ticker": tk,
                "company_name": issuer_names.get(tk, tk),
                "fiscal_years_available": avail_yrs,
                "missing_fiscal_years": evo.missing_fiscal_years,
                "theme_mentions": evo.theme_mentions,
                "delta_counts": {
                    f"FY{p.prior_fiscal_year}->FY{p.current_fiscal_year}": p.counts
                    for p in evo.year_pairs
                },
                "top_similarity": round(best_sim, 4),
                "passages": compact_passages,
            }
        )

    company_rank = {
        c["ticker"]: (sum(c["theme_mentions"].values()), c["top_similarity"])
        for c in company_map_inputs
    }
    evolutions.sort(key=lambda e: company_rank.get(e["ticker"], (0, 0.0)), reverse=True)
    company_map_inputs.sort(key=lambda c: company_rank.get(c["ticker"], (0, 0.0)), reverse=True)
    return (
        evolutions,
        company_map_inputs,
        delta_summary,
        mentions_by_year,
        issuers_with_theme_matches,
    )


def _run_parallel_map_batches(
    *,
    batches: list[list[dict[str, Any]]],
    analyzed_count: int,
    issuers_in_sector: int,
    gics_sector: str,
    theme: str,
    start_year: int,
    end_year: int,
    all_citations_by_id: dict[str, dict[str, Any]],
    max_workers: int,
    map_llm: BaseLlm | None,
    on_progress: Callable[[str, str, dict[str, Any] | None], None] | None,
) -> list[dict[str, Any]]:
    """Execute Flash map batches concurrently while streaming progress as each batch completes."""
    total_batches = len(batches)
    if total_batches == 0:
        return []

    progress_lock = threading.Lock()
    state = {"analyzed": 0, "done": 0}

    def _run_one(item: tuple[int, list[dict[str, Any]]]) -> dict[str, Any]:
        idx, batch = item
        summary = _summarize_batch_via_flash(
            batch_index=idx,
            total_batches=total_batches,
            gics_sector=gics_sector,
            theme=theme,
            start_year=start_year,
            end_year=end_year,
            company_payloads=batch,
            chunk_lookup=all_citations_by_id,
            map_llm=map_llm,
        )
        with progress_lock:
            state["analyzed"] += len(batch)
            state["done"] += 1
            snap_analyzed = state["analyzed"]
            snap_done = state["done"]
        if on_progress is not None:
            on_progress(
                "sector_map_progress",
                (
                    f"{snap_analyzed}/{analyzed_count} {gics_sector} companies analyzed "
                    f"(batch {snap_done}/{total_batches})"
                ),
                {
                    "analyzed": snap_analyzed,
                    "total": analyzed_count,
                    "issuers_in_sector": issuers_in_sector,
                    "batch_completed": snap_done,
                    "total_batches": total_batches,
                },
            )
        return summary

    return run_parallel_with_context(
        list(enumerate(batches)),
        _run_one,
        max_workers=min(max_workers, total_batches),
    )


def _collect_prioritized_citations(
    batch_summaries: list[dict[str, Any]],
    company_map_inputs: list[dict[str, Any]],
    all_citations_by_id: dict[str, dict[str, Any]],
    *,
    max_citations: int = 60,
) -> list[dict[str, Any]]:
    """Collect all verified batch citations first, then fill with top representative passages."""
    prioritized: list[dict[str, Any]] = []
    seen_cids: set[str] = set()
    for bs in batch_summaries:
        for ev in bs.get("evidence") or []:
            cid = str(ev.get("chunk_id") or "")
            if cid and cid in all_citations_by_id and cid not in seen_cids:
                prioritized.append(all_citations_by_id[cid])
                seen_cids.add(cid)

    for c_in in company_map_inputs[:20]:
        for p in c_in["passages"]:
            if len(prioritized) >= max_citations:
                return prioritized
            cid = str(p["chunk_id"])
            if cid in all_citations_by_id and cid not in seen_cids:
                prioritized.append(all_citations_by_id[cid])
                seen_cids.add(cid)
    return prioritized


@traced("sector.map_reduce", record_args=("gics_sector", "start_year", "end_year"))
def run_sector_thematic_map_reduce(
    *,
    theme: str,
    gics_sector: str,
    section_id: str,
    start_year: int,
    end_year: int,
    theme_terms: str | Sequence[str] | None = None,
    sector_profiles: list[FinancialDataProfile] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    per_company_year: int = DEFAULT_PER_COMPANY_YEAR,
    max_workers: int = DEFAULT_MAX_MAP_WORKERS,
    map_llm: BaseLlm | None = None,
    on_progress: Callable[[str, str, dict[str, Any] | None], None] | None = None,
) -> dict[str, Any]:
    """Run full-sector batched map-reduce thematic analysis across ALL issuers in `gics_sector`.

    Emits incremental `on_progress(stage, message, meta)` events so SSE callers stream progress
    to the user within the 3.0 s TTFT window (e.g. `"28/76 Information Technology companies analyzed"`).
    """
    start_year, end_year = sorted((int(start_year), int(end_year)))
    window = list(range(start_year, end_year + 1))
    sec_norm = (section_id or "ALL").strip().lower()
    sec_filter = sec_norm if sec_norm in ("item_1a", "item_7") else None

    adapters = _gcp_adapters()

    # 1. Enumerate every issuer in the target GICS sector and its available filing years.
    issuer_years, issuer_names = _load_sector_issuer_catalog(
        gics_sector, start_year, end_year, sector_profiles
    )
    all_sector_tickers = sorted(issuer_years.keys())
    issuers_in_sector = len(all_sector_tickers)
    filings_by_year = {
        str(fy): sum(1 for yrs in issuer_years.values() if fy in yrs) for fy in window
    }
    missing_fy_counts = {
        str(fy): sum(1 for yrs in issuer_years.values() if fy not in yrs) for fy in window
    }

    if on_progress is not None:
        on_progress(
            "sector_retrieval",
            f"Scanning {issuers_in_sector} {gics_sector} companies (FY{start_year}–FY{end_year})...",
            {
                "gics_sector": gics_sector,
                "issuers_in_sector": issuers_in_sector,
                "analyzed": 0,
                "total": issuers_in_sector,
            },
        )

    # 2. Single BigQuery ML.DISTANCE query covering EVERY company-year in the entire sector.
    q_emb = adapters.embed_query(theme)
    raw_rows = adapters.bigquery.vector_search_sector_top_per_ticker(
        q_emb,
        gics_sector=gics_sector,
        start_year=start_year,
        end_year=end_year,
        section_id=sec_filter,
        per_company_year=per_company_year,
    )

    resolved_theme_terms = resolve_theme_terms(theme, theme_terms)
    rows_by_ticker_fy: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    all_citations_by_id: dict[str, dict[str, Any]] = {}
    for row in raw_rows:
        tk = normalize_edgar_ticker(str(row.get("ticker") or ""))
        fy = int(row["fiscal_year"])
        cit = chunk_row_to_citation(row, score=float(row.get("similarity") or 0.0))
        rows_by_ticker_fy[tk][fy].append(cit)
        all_citations_by_id[str(cit["chunk_id"])] = cit
        issuer_names.setdefault(tk, str(row.get("company_name") or tk))
        issuer_years[tk].add(fy)

    analyzed_tickers = sorted(rows_by_ticker_fy.keys())
    if not all_sector_tickers:
        all_sector_tickers = list(analyzed_tickers)
        issuers_in_sector = len(all_sector_tickers)

    # 3. Deterministic per-company alignment & theme mentions across ALL analyzed issuers.
    (
        evolutions,
        company_map_inputs,
        delta_summary,
        theme_mentions_totals_by_year,
        issuers_with_theme_matches,
    ) = _compute_sector_evolutions(
        analyzed_tickers=analyzed_tickers,
        rows_by_ticker_fy=rows_by_ticker_fy,
        issuer_names=issuer_names,
        gics_sector=gics_sector,
        window=window,
        theme_terms=resolved_theme_terms,
        issuers_in_sector=issuers_in_sector,
    )

    # 4. Parallel batched Map phase across ALL analyzed companies.
    eff_batch_size = max(1, int(batch_size))
    batches = [
        company_map_inputs[i : i + eff_batch_size]
        for i in range(0, len(company_map_inputs), eff_batch_size)
    ]
    total_batches = len(batches)
    if on_progress is not None and total_batches > 0:
        on_progress(
            "sector_map_start",
            (
                f"Retrieved {len(raw_rows)} passages across {len(analyzed_tickers)}/{issuers_in_sector} "
                f"{gics_sector} companies. Summarizing in {total_batches} parallel batch(es)..."
            ),
            {
                "gics_sector": gics_sector,
                "analyzed": 0,
                "total": len(analyzed_tickers),
                "issuers_in_sector": issuers_in_sector,
                "batches": total_batches,
            },
        )

    batch_summaries = _run_parallel_map_batches(
        batches=batches,
        analyzed_count=len(analyzed_tickers),
        issuers_in_sector=issuers_in_sector,
        gics_sector=gics_sector,
        theme=theme,
        start_year=start_year,
        end_year=end_year,
        all_citations_by_id=all_citations_by_id,
        max_workers=max_workers,
        map_llm=map_llm,
        on_progress=on_progress,
    )

    # 5. Collect ALL verified cited chunk_ids across every batch + top representative citations.
    prioritized_citations = _collect_prioritized_citations(
        batch_summaries, company_map_inputs, all_citations_by_id
    )

    sector_coverage = {
        "gics_sector": gics_sector,
        "start_year": start_year,
        "end_year": end_year,
        "issuers_in_sector": issuers_in_sector,
        "issuers_with_filings": len(all_sector_tickers),
        "issuers_analyzed": len(analyzed_tickers),
        "issuers_with_theme_matches": issuers_with_theme_matches,
        "filings_by_year": filings_by_year,
        "missing_fy_counts": missing_fy_counts,
        "theme_mentions_by_year": theme_mentions_totals_by_year,
        "total_chunks_evaluated": len(raw_rows),
        "batches_processed": total_batches,
        "batch_size": eff_batch_size,
        "mode": "batched_map_reduce",
    }

    set_span_attributes(
        **{
            "edgar.sector.issuers_in_sector": issuers_in_sector,
            "edgar.sector.issuers_analyzed": len(analyzed_tickers),
            "edgar.sector.batches_processed": total_batches,
            "edgar.sector.total_chunks_evaluated": len(raw_rows),
        }
    )

    disclosure_deltas_payload = {
        "status": "success",
        "theme": theme.strip(),
        "theme_terms": resolved_theme_terms,
        "section_id": sec_filter or "ALL",
        "start_year": start_year,
        "end_year": end_year,
        "companies_selected_by": "full_sector_map_reduce",
        "missing_tickers": [],
        "summary": delta_summary,
        "sector_coverage": sector_coverage,
        "companies": evolutions[:12],
        "total_companies_evaluated": len(evolutions),
        "citations": prioritized_citations,
    }

    return {
        "status": "success",
        "sector_coverage": sector_coverage,
        "sector_map_summaries": {
            "sector_coverage": sector_coverage,
            "batch_summaries": batch_summaries,
        },
        "disclosure_deltas": disclosure_deltas_payload,
        "citations": prioritized_citations,
    }
