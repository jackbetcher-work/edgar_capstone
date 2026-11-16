"""Helpers for test LLMs that answer FinancialAnalystAgent requests with a structured AnalystReport."""

from __future__ import annotations

import json
import re
from typing import Any

from google.adk.models.llm_request import LlmRequest

from edgar_capstone.agents.schemas import AnalystReport

_LEADS_BLOCK_RE = re.compile(r"## 10-K Evidence Leads[^\n]*\n```json\n(.*?)\n```", re.S)


def request_text(llm_request: LlmRequest) -> str:
    return "\n".join(
        p.text for c in (llm_request.contents or []) for p in (c.parts or []) if p.text
    )


def evidence_leads(llm_request: LlmRequest) -> list[dict[str, Any]]:
    """Evidence leads (chunk_id + excerpt) embedded in the synthesis prompt."""
    m = _LEADS_BLOCK_RE.search(request_text(llm_request))
    if not m:
        return []
    return [lead for lead in json.loads(m.group(1)) if lead.get("chunk_id")]


def verbatim_quote(text: str, words: int = 14) -> str:
    return " ".join(str(text or "").split()[:words])


def finding_from_citation(
    citation: dict[str, Any], statement: str, *, words: int = 14
) -> dict[str, Any]:
    return {
        "statement": statement,
        "ticker": str(citation.get("ticker") or "NVDA"),
        "fiscal_year": int(citation.get("fiscal_year") or 2026),
        "chunk_ids": [str(citation["chunk_id"])],
        "evidence_quote": verbatim_quote(str(citation.get("excerpt") or ""), words),
    }


def report_json(
    *,
    headline: str,
    findings: list[dict[str, Any]] | None = None,
    key_metrics: list[tuple[str, str]] | None = None,
    notes: str = "",
) -> str:
    """Serialize a schema-valid AnalystReport, as constrained decoding would return it."""
    report = AnalystReport(
        headline=headline,
        key_metrics=[{"ticker": t, "metric": m} for t, m in (key_metrics or [])],
        findings=findings or [],
        notes=notes,
    )
    return report.model_dump_json()
