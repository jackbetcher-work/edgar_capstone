"""Live Vertex AI Search (Discovery Engine) hybrid search SDK adapter."""

from __future__ import annotations

import logging
import re
import threading
from types import TracebackType
from typing import Any

from edgar_capstone.agents.adapters.registry import (
    GcpAdapterHttpError,
    GcpEndpointNotReadyError,
    GcpToolConfig,
    translate_gcp_adapter_error,
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
    discovery_engine_host,
    vertex_search_datastore_path,
    vertex_search_engine_path,
)
from edgar_capstone.gcp.errors import build_client_options, resolve_quota_project_id

logger = logging.getLogger(__name__)

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _escape_filter_literal(value: str) -> str:
    """Escape backslashes and double quotes for Discovery Engine filter string literals."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


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


def _proto_struct_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a protobuf ``Struct`` / ``MapComposite`` / ``dict`` into a plain Python ``dict``."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    try:
        from google.protobuf.json_format import MessageToDict

        pb = getattr(obj, "_pb", obj)
        if hasattr(pb, "DESCRIPTOR"):
            return dict(MessageToDict(pb))
    except Exception:
        pass
    try:
        return {str(k): _proto_value_to_python(v) for k, v in obj.items()}
    except Exception:
        return {}


def _proto_value_to_python(val: Any) -> Any:
    if isinstance(val, dict):
        return {str(k): _proto_value_to_python(v) for k, v in val.items()}
    if hasattr(val, "items"):
        return {str(k): _proto_value_to_python(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)) or (
        hasattr(val, "__iter__") and not isinstance(val, (str, bytes))
    ):
        return [_proto_value_to_python(item) for item in val]
    return val


def _collect_extracted_candidates(
    derived: dict[str, Any],
    struct_data: dict[str, Any],
    doc_rank: int,
) -> list[tuple[str, float, str | None, str | None]]:
    """Collect candidate `(text, score, raw_section_id, raw_section_title)` tuples from a doc."""
    doc_sec_id = struct_data.get("section_id")
    doc_sec_title = struct_data.get("section_title")
    candidates: list[tuple[str, float, str | None, str | None]] = []
    for seg in derived.get("extractive_segments") or []:
        if isinstance(seg, dict) and seg.get("content"):
            rel = float(
                seg.get("relevanceScore")
                or seg.get("relevance_score")
                or max(0.35, 0.92 - 0.05 * doc_rank)
            )
            candidates.append(
                (
                    str(seg["content"]).strip(),
                    rel,
                    seg.get("section_id") or doc_sec_id,
                    seg.get("section_title") or doc_sec_title,
                )
            )
    for ans in derived.get("extractive_answers") or []:
        if isinstance(ans, dict) and ans.get("content"):
            candidates.append(
                (
                    str(ans["content"]).strip(),
                    max(0.30, 0.85 - 0.05 * doc_rank),
                    ans.get("section_id") or doc_sec_id,
                    ans.get("section_title") or doc_sec_title,
                )
            )
    for snip in derived.get("snippets") or []:
        if isinstance(snip, dict) and snip.get("snippet"):
            clean_snip = _HTML_TAG_RE.sub("", str(snip["snippet"])).strip()
            if clean_snip:
                candidates.append(
                    (clean_snip, max(0.25, 0.75 - 0.05 * doc_rank), doc_sec_id, doc_sec_title)
                )
    if struct_data.get("chunk_text"):
        candidates.append(
            (
                str(struct_data["chunk_text"]).strip(),
                max(0.35, 0.88 - 0.05 * doc_rank),
                doc_sec_id,
                doc_sec_title,
            )
        )
    return candidates


class VertexAISearchAdapter:
    """Live ``google-cloud-discoveryengine`` (``SearchServiceClient``) hybrid search adapter."""

    def __init__(
        self,
        config: GcpToolConfig,
        *,
        search_client: Any | None = None,
        credentials: Any | None = None,
    ) -> None:
        self.project_id = config.project_id
        self.location = config.vais_location or DEFAULT_VERTEX_AI_SEARCH_LOCATION
        self.collection_id = config.vais_collection_id or DEFAULT_VERTEX_AI_SEARCH_COLLECTION
        self.engine_id = config.vais_engine_id
        self.datastore_id = config.vais_datastore_id
        self._injected_client = search_client
        self._owned_client: Any | None = None
        self._credentials = credentials
        self._client_lock = threading.Lock()
        self._resolved_serving_config: str | None = None
        self._url_lock = threading.Lock()

    def _get_client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        with self._client_lock:
            if self._owned_client is None:
                from google.cloud import discoveryengine_v1

                opts = build_client_options(
                    api_endpoint=discovery_engine_host(self.location),
                    quota_project_id=resolve_quota_project_id(
                        self.project_id, credentials=self._credentials
                    ),
                )
                kwargs: dict[str, Any] = {"client_options": opts}
                if self._credentials is not None:
                    kwargs["credentials"] = self._credentials
                self._owned_client = discoveryengine_v1.SearchServiceClient(**kwargs)
            return self._owned_client

    def close(self) -> None:
        """Close the internally managed SDK transport (leaves injected clients open)."""
        with self._client_lock:
            if self._owned_client is not None:
                transport = getattr(self._owned_client, "transport", None)
                close_fn = getattr(transport, "close", None)
                if callable(close_fn):
                    close_fn()
                self._owned_client = None

    def __enter__(self) -> VertexAISearchAdapter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _serving_config_paths(self) -> list[str]:
        paths: list[str] = []
        if self.engine_id:
            engine = vertex_search_engine_path(
                self.project_id,
                engine_id=self.engine_id,
                location=self.location,
                collection=self.collection_id,
            )
            paths.append(f"{engine}/servingConfigs/default_search")
        if self.datastore_id:
            ds = vertex_search_datastore_path(
                self.project_id,
                data_store_id=self.datastore_id,
                location=self.location,
                collection=self.collection_id,
            )
            paths.append(f"{ds}/servingConfigs/default_search")
        return paths

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
            quoted_tickers = ", ".join(f'"{_escape_filter_literal(t)}"' for t in sorted(tickers))
            clauses.append(f"ticker: ANY({quoted_tickers})")
        canon_sector = canonicalize_gics_sector(gics_sector)
        if canon_sector:
            safe_sec = _escape_filter_literal(canon_sector)
            clauses.append(f'gics_sector: ANY("{safe_sec}")')
        if section_id in ("item_1a", "item_7"):
            clauses.append(f'section_id: ANY("{section_id}")')
        return " AND ".join(clauses)

    def _search_with_fallback(
        self,
        serving_configs: list[str],
        *,
        query: str,
        filter_expr: str,
        page_size: int,
    ) -> list[Any]:
        """Execute ``SearchServiceClient.search`` across engine/datastore paths, caching the winner."""
        from google.cloud import discoveryengine_v1 as de

        with self._url_lock:
            cached = self._resolved_serving_config
        candidates = (
            [cached, *(p for p in serving_configs if p != cached)] if cached else serving_configs
        )
        client = self._get_client()
        last_err: Exception | None = None

        for idx, serving_config in enumerate(candidates):
            req = de.SearchRequest(
                serving_config=serving_config,
                query=query,
                filter=filter_expr or "",
                page_size=max(1, min(int(page_size), 25)),
                query_expansion_spec=de.SearchRequest.QueryExpansionSpec(
                    condition=de.SearchRequest.QueryExpansionSpec.Condition.AUTO
                ),
                spell_correction_spec=de.SearchRequest.SpellCorrectionSpec(
                    mode=de.SearchRequest.SpellCorrectionSpec.Mode.AUTO
                ),
                content_search_spec=de.SearchRequest.ContentSearchSpec(
                    snippet_spec=de.SearchRequest.ContentSearchSpec.SnippetSpec(
                        return_snippet=True,
                        max_snippet_count=3,
                    ),
                    extractive_content_spec=de.SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
                        max_extractive_segment_count=5,
                        max_extractive_answer_count=3,
                        return_extractive_segment_score=True,
                    ),
                ),
            )
            try:
                response = client.search(request=req, timeout=30.0)
                if serving_config != cached:
                    with self._url_lock:
                        self._resolved_serving_config = serving_config
                if hasattr(response, "pages"):
                    first_page = next(iter(response.pages), None)
                    return list(getattr(first_page, "results", None) or ()) if first_page else []
                return list(getattr(response, "results", None) or ())
            except Exception as raw_exc:
                err = translate_gcp_adapter_error(
                    raw_exc, context=f"Vertex AI Search ({serving_config})"
                )
                last_err = err
                if (
                    isinstance(err, GcpAdapterHttpError)
                    and err.status_code == 404
                    and idx + 1 < len(candidates)
                ):
                    logger.info(
                        "Vertex AI Search servingConfig returned 404; trying next servingConfig."
                    )
                    continue
                raise err from raw_exc

        if last_err is not None:
            raise last_err
        return []

    def _extract_document_passages(
        self,
        doc_rank: int,
        item: Any,
        *,
        tickers: set[str],
        gics_sector: str | None,
        section_id: str | None,
        start_year: int,
        end_year: int,
    ) -> list[dict[str, Any]]:
        """Normalize a single Discovery Engine ``SearchResult`` into passage dicts."""
        doc = getattr(item, "document", None)
        if doc is None:
            return []
        doc_id = str(getattr(doc, "id", "") or f"vais_doc_{doc_rank}")
        struct_data = _proto_struct_to_dict(getattr(doc, "struct_data", None))
        derived = _proto_struct_to_dict(getattr(doc, "derived_struct_data", None))
        content_obj = getattr(doc, "content", None)
        content_uri = str(getattr(content_obj, "uri", "") or "")

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
            content_uri or derived.get("link") or struct_data.get("unstructured_gcs_uri") or ""
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
                    "section_title": str(raw_title or SECTION_TITLES[resolved_sec]),
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
        serving_configs = self._serving_config_paths()
        if not serving_configs:
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
        results = self._search_with_fallback(
            serving_configs,
            query=query,
            filter_expr=filter_expr,
            page_size=page_size,
        )
        passages: list[dict[str, Any]] = []
        for doc_rank, item in enumerate(results):
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
