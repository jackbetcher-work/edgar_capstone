"""Live Vertex AI Search (Discovery Engine) hybrid search adapter."""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

import httpx

from edgar_capstone.agents.adapters.registry import (
    GcpAdapterErrorTranslationMixin,
    GcpAdapterHttpError,
    GcpEndpointNotReadyError,
    GcpToolConfig,
)
from edgar_capstone.common.domain import (
    SECTION_TITLES,
    SectionId,
    canonicalize_gics_sector,
    normalize_edgar_ticker,
)
from edgar_capstone.gcp.discovery_engine import (
    DEFAULT_VERTEX_AI_SEARCH_COLLECTION,
    DEFAULT_VERTEX_AI_SEARCH_LOCATION,
    discovery_engine_root,
)
from edgar_capstone.gcp.rest_client import GcpRestClient

logger = logging.getLogger(__name__)

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _normalize_section_id(
    raw_section: str | None,
    *,
    requested_section: str | None = None,
    passage_text: str = "",
) -> SectionId:
    """Coerce section identifiers to ``SectionId`` (``'item_1a'`` or ``'item_7'``)."""
    if raw_section:
        low = raw_section.strip().lower()
        if "1a" in low or "risk" in low:
            return "item_1a"
        if "7" in low or "md&a" in low or "management" in low:
            return "item_7"

    probe = passage_text[:320].lower()
    if "item 1a" in probe or "risk factor" in probe:
        return "item_1a"
    if "item 7" in probe or "results of operations" in probe or "executive overview" in probe:
        return "item_7"

    if requested_section == "item_1a":
        return "item_1a"
    if requested_section == "item_7":
        return "item_7"

    lower_body = passage_text.lower()
    risk_cues = ("risk", "regulatory", "restriction", "litigation", "cybersecurity", "adversely")
    mda_cues = ("revenue", "operating income", "margin", "cash flow", "fiscal year", "segment")
    risk_hits = sum(1 for cue in risk_cues if cue in lower_body)
    mda_hits = sum(1 for cue in mda_cues if cue in lower_body)
    return "item_1a" if risk_hits > mda_hits else "item_7"


def _default_section_title(sec_id: SectionId) -> str:
    return SECTION_TITLES[sec_id]


def _collect_extracted_candidates(
    derived: dict[str, Any],
    struct_data: dict[str, Any],
    doc_rank: int,
) -> list[tuple[str, float, str | None, str | None]]:
    """Collect candidate `(text, score, raw_section_id, raw_section_title)` tuples from a doc."""
    candidates: list[tuple[str, float, str | None, str | None]] = []
    for seg in derived.get("extractive_segments") or []:
        if isinstance(seg, dict) and seg.get("content"):
            rel = float(seg.get("relevanceScore") or max(0.35, 0.92 - 0.05 * doc_rank))
            candidates.append(
                (str(seg["content"]).strip(), rel, seg.get("section_id"), seg.get("section_title"))
            )
    for ans in derived.get("extractive_answers") or []:
        if isinstance(ans, dict) and ans.get("content"):
            candidates.append(
                (
                    str(ans["content"]).strip(),
                    max(0.30, 0.85 - 0.05 * doc_rank),
                    ans.get("section_id"),
                    ans.get("section_title"),
                )
            )
    for snip in derived.get("snippets") or []:
        if isinstance(snip, dict) and snip.get("snippet"):
            clean_snip = _HTML_TAG_RE.sub("", str(snip["snippet"])).strip()
            if clean_snip:
                candidates.append((clean_snip, max(0.25, 0.75 - 0.05 * doc_rank), None, None))
    if struct_data.get("chunk_text"):
        candidates.append(
            (
                str(struct_data["chunk_text"]).strip(),
                max(0.35, 0.88 - 0.05 * doc_rank),
                struct_data.get("section_id"),
                struct_data.get("section_title"),
            )
        )
    return candidates


class VertexAISearchAdapter(GcpAdapterErrorTranslationMixin, GcpRestClient):
    """Live Discovery Engine (`servingConfigs/default_search:search`) hybrid search adapter."""

    def __init__(
        self,
        config: GcpToolConfig,
        *,
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
    ) -> None:
        super().__init__(http_client=http_client, credentials=credentials, default_timeout=30.0)
        self.project_id = config.project_id
        self.location = config.vais_location or DEFAULT_VERTEX_AI_SEARCH_LOCATION
        self.collection_id = config.vais_collection_id or DEFAULT_VERTEX_AI_SEARCH_COLLECTION
        self.engine_id = config.vais_engine_id
        self.datastore_id = config.vais_datastore_id
        self._resolved_url: str | None = None
        self._url_lock = threading.Lock()

    def _serving_config_urls(self) -> list[str]:
        root = discovery_engine_root(self.location)
        prefix = (
            f"{root}/projects/{self.project_id}/locations/{self.location}/"
            f"collections/{self.collection_id}"
        )
        urls: list[str] = []
        if self.engine_id:
            urls.append(f"{prefix}/engines/{self.engine_id}/servingConfigs/default_search:search")
        if self.datastore_id:
            urls.append(
                f"{prefix}/dataStores/{self.datastore_id}/servingConfigs/default_search:search"
            )
        return urls

    @staticmethod
    def build_filter_expression(
        *,
        tickers: set[str],
        gics_sector: str | None,
        section_id: str | None,
        start_year: int,
        end_year: int,
    ) -> str:
        """Build a Discovery Engine filter expression over `VertexAISearchManifestRecord.structData`."""
        clauses: list[str] = [
            f"fiscal_year >= {int(start_year)}",
            f"fiscal_year <= {int(end_year)}",
        ]
        if tickers:
            quoted_tickers = ", ".join(f'"{t.replace(chr(34), "")}"' for t in sorted(tickers))
            clauses.append(f"ticker: ANY({quoted_tickers})")
        canon_sector = canonicalize_gics_sector(gics_sector)
        if canon_sector:
            safe_sec = canon_sector.replace('"', '\\"')
            clauses.append(f'gics_sector: ANY("{safe_sec}")')
        if section_id in ("item_1a", "item_7"):
            clauses.append(f'extracted_sections: ANY("{section_id}")')
        return " AND ".join(clauses)

    def _post_with_url_fallback(
        self, urls: list[str], payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        """POST search payload across engine/datastore URLs, caching the first working URL."""
        with self._url_lock:
            cached_url = self._resolved_url
        candidate_urls = [cached_url, *(u for u in urls if u != cached_url)] if cached_url else urls
        last_err: Exception | None = None
        for idx, url in enumerate(candidate_urls):
            try:
                data = self._request_json("POST", url, json_body=payload)
                if url != cached_url:
                    with self._url_lock:
                        self._resolved_url = url
                return data if isinstance(data, dict) else {"items": data}
            except GcpAdapterHttpError as exc:
                last_err = exc
                if exc.status_code == 404 and idx + 1 < len(candidate_urls):
                    logger.info(
                        "Vertex AI Search endpoint returned 404; caching next servingConfig endpoint."
                    )
                    continue
                raise
        if last_err is not None:
            raise last_err
        return None

    def _extract_document_passages(
        self,
        doc_rank: int,
        item: dict[str, Any],
        *,
        tickers: set[str],
        gics_sector: str | None,
        section_id: str | None,
        start_year: int,
        end_year: int,
    ) -> list[dict[str, Any]]:
        """Normalize a single Discovery Engine result item into passage dicts."""
        doc = item.get("document") or {}
        doc_id = str(doc.get("id") or f"vais_doc_{doc_rank}")
        struct_data = doc.get("structData") or {}
        derived = doc.get("derivedStructData") or {}
        content_obj = doc.get("content") or {}

        ticker = str(struct_data.get("ticker") or "").strip().upper()
        fy_raw = struct_data.get("fiscal_year")
        if not ticker or fy_raw is None:
            id_parts = doc_id.split("_")
            if len(id_parts) >= 2 and id_parts[1].isdigit():
                ticker = ticker or id_parts[0].upper()
                fy_raw = fy_raw if fy_raw is not None else int(id_parts[1])
        if not ticker or fy_raw is None:
            return []

        fy = int(fy_raw)
        if fy < start_year or fy > end_year:
            return []
        if tickers and normalize_edgar_ticker(ticker) not in tickers:
            return []
        doc_sector = str(struct_data.get("gics_sector") or "Unknown")
        if gics_sector and doc_sector.lower() != gics_sector.lower():
            return []

        profile_id = str(struct_data.get("profile_id") or f"{ticker}_{fy}")
        company_name = str(struct_data.get("company_name") or ticker)
        sec_url = str(struct_data.get("sec_filing_url") or "")
        gcs_uri = str(
            content_obj.get("uri")
            or derived.get("link")
            or struct_data.get("unstructured_gcs_uri")
            or ""
        )

        passages: list[dict[str, Any]] = []
        for seg_idx, (seg_text, rel_score, raw_sec, raw_title) in enumerate(
            _collect_extracted_candidates(derived, struct_data, doc_rank)
        ):
            if not seg_text:
                continue
            resolved_sec = _normalize_section_id(
                raw_sec,
                requested_section=section_id,
                passage_text=seg_text,
            )
            if section_id in ("item_1a", "item_7") and resolved_sec != section_id:
                continue
            chunk_id = str(
                struct_data.get("chunk_id") or f"{profile_id}_{resolved_sec}_vais_{seg_idx:04d}"
            )
            passages.append(
                {
                    "chunk_id": chunk_id,
                    "profile_id": profile_id,
                    "ticker": ticker,
                    "company_name": company_name,
                    "gics_sector": doc_sector,
                    "fiscal_year": fy,
                    "period_of_report": str(struct_data.get("period_of_report") or "") or None,
                    "filing_date": str(struct_data.get("filing_date") or "") or None,
                    "accession_number": str(struct_data.get("accession_number") or "") or None,
                    "section_id": resolved_sec,
                    "section_title": str(raw_title or _default_section_title(resolved_sec)),
                    "chunk_text": seg_text,
                    "sec_filing_url": sec_url,
                    "unstructured_gcs_uri": gcs_uri,
                    "similarity": max(0.0, min(1.0, rel_score)),
                    "source_backend": "vertex_ai_search",
                }
            )
        return passages

    def search_documents(
        self,
        query: str,
        *,
        tickers: set[str],
        gics_sector: str | None,
        section_id: str | None,
        start_year: int,
        end_year: int,
        page_size: int = 10,
    ) -> list[dict[str, Any]]:
        """Execute a metadata-filtered hybrid search against Vertex AI Search (Discovery Engine)."""
        urls = self._serving_config_urls()
        if not urls:
            raise GcpEndpointNotReadyError(
                "Neither VERTEX_AI_SEARCH_ENGINE_ID nor VERTEX_AI_SEARCH_DATASTORE_ID is configured."
            )

        filter_expr = self.build_filter_expression(
            tickers=tickers,
            gics_sector=gics_sector,
            section_id=section_id,
            start_year=start_year,
            end_year=end_year,
        )
        payload: dict[str, Any] = {
            "query": query,
            "pageSize": max(1, min(int(page_size), 25)),
            "queryExpansionSpec": {"condition": "AUTO"},
            "spellCorrectionSpec": {"mode": "AUTO"},
            "contentSearchSpec": {
                "snippetSpec": {"returnSnippet": True, "maxSnippetCount": 3},
                "extractiveContentSpec": {
                    "maxExtractiveSegmentCount": 5,
                    "maxExtractiveAnswerCount": 3,
                    "returnExtractiveSegmentScore": True,
                },
            },
        }
        if filter_expr:
            payload["filter"] = filter_expr

        data = self._post_with_url_fallback(urls, payload)
        if not data:
            return []

        passages: list[dict[str, Any]] = []
        for doc_rank, item in enumerate(data.get("results") or []):
            passages.extend(
                self._extract_document_passages(
                    doc_rank,
                    item,
                    tickers=tickers,
                    gics_sector=gics_sector,
                    section_id=section_id,
                    start_year=start_year,
                    end_year=end_year,
                )
            )
        return passages
