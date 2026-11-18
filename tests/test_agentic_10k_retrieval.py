"""Agent-driven 10-K retrieval: FinancialAnalystAgent queries the evidence each question needs.

Tools read live GCP (BigQuery `filing_chunks`, the GCS SEC corpus, Vertex AI embeddings) and
chat turns call live Gemini. A scripted ADK `BaseLlm` is injected only where a test targets
pipeline mechanics the live model can't be made to exercise on demand (a specific tool call
sequence, a tool call written as prose); its answers cite chunks the live tools returned.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types

from edgar_capstone.agents import (
    get_context_latency_tracker,
    get_conversation_memory_manager,
    query_cached_10k_context,
)
from edgar_capstone.agents.chat_engine import run_conversational_rag_turn
from edgar_capstone.agents.live_runtime import _build_financial_reasoning_prompt
from edgar_capstone.agents.tools import get_full_10k_section, load_cached_10k_section
from tests.structured_report_helpers import finding_from_citation, report_json

_RETRIEVAL_TOOLS = {
    "search_10k_chunks_rag",
    "query_cached_10k_context",
    "compare_disclosures_across_years",
}


def test_synthesis_prompt_carries_facts_and_retrieval_brief_not_full_sections() -> None:
    long_excerpt = "Data center revenue grew on Blackwell demand. " * 40
    prompt = _build_financial_reasoning_prompt(
        query="Why did NVIDIA's operating income grow faster than revenue?",
        intent="single_company_longitudinal",
        scope={
            "tickers_csv": "NVDA",
            "gics_sector": "ALL",
            "section_id": "ALL",
            "start_year": 2024,
            "end_year": 2026,
        },
        variance_reports=[{"ticker": "NVDA", "start_year": 2024, "end_year": 2026}],
        variance_evaluations=[
            {
                "ticker": "NVDA",
                "primary_metric_mover": "operating_income",
                "targeted_10k_queries": [
                    {
                        "ticker": "NVDA",
                        "fiscal_year": 2026,
                        "section_id": "item_7",
                        "variance_query": "NVDA FY2026 revenue operating income ...",
                        "driver_focus": "FY2024–FY2026 operating income & margin variance drivers",
                    }
                ],
            }
        ],
        cached_explanations=[{"cache_key": "NVDA_2026_item_7", "extracted_explanations": ["x"]}],
        citations=[
            {
                "chunk_id": f"NVDA_2026_item_7_{i:04d}",
                "ticker": "NVDA",
                "fiscal_year": 2026,
                "section_id": "item_7",
                "section_title": "Item 7",
                "sec_filing_url": "https://www.sec.gov/x",
                "excerpt": long_excerpt,
            }
            for i in range(16)
        ],
        sector_benchmarks=[],
        chunk_health=None,
    )
    assert "<10k_section" not in prompt
    assert "Full 10-K sections are NOT in this request" in prompt
    assert "query_cached_10k_context(" in prompt and "search_10k_chunks_rag(" in prompt
    assert "without redundant tool calls" not in prompt
    # Leads are capped in count (12) and length (850 chars) and point at citable chunk ids
    assert prompt.count('"chunk_id": "NVDA_2026_item_7_') == 12
    assert long_excerpt[:900] not in prompt
    assert '"pre_queried_sections"' in prompt
    # Suggested retrieval targets are surfaced once, not duplicated inside the evaluations JSON
    assert prompt.count("operating income & margin variance drivers") == 1
    assert "NVDA FY2026 revenue operating income ..." not in prompt


def test_query_cached_10k_context_ranks_full_live_section_and_returns_citable_passages() -> None:
    """Ranking covers the whole GCS section, not just the 50k-char excerpt returned to callers."""
    excerpt_res = load_cached_10k_section(
        ticker="NVDA", fiscal_year=2026, section_id="item_1a", max_chars=50_000
    )
    assert excerpt_res["status"] == "success", excerpt_res
    assert excerpt_res["content_source"] == "gcs"
    assert excerpt_res["truncated"] is True

    full_text = str(get_full_10k_section(excerpt_res["cache_key"])["full_text"])
    assert len(full_text) > 50_000
    # A distinctive paragraph that only exists beyond the 50k-char excerpt
    target = next(
        blk
        for blk in (b.strip() for b in re.split(r"\n\s*\n+", full_text))
        if 300 <= len(blk) <= 1500
        and not blk.startswith("#")
        and full_text.index(blk) > 50_000
        and full_text.count(blk[:120]) == 1
    )
    assert target not in excerpt_res["content"]

    res = query_cached_10k_context(
        ticker="NVDA",
        fiscal_year=2026,
        section_id="item_1a",
        variance_query=target[:600],
        max_passages=2,
    )
    assert res["status"] == "success"
    assert res["content_source"] == "bigquery_filing_chunks"
    assert "warning" not in res
    norm_passage = " ".join(res["passages"][0]["text"].replace("\\", "").split())
    norm_target = " ".join(target.replace("\\", "").split())
    assert norm_target[:120] in norm_passage, (norm_target[:120], norm_passage[:300])
    assert all({"text", "chunk_id", "chunk_match"} <= set(p) for p in res["passages"])

    # Every passage points at a live BigQuery chunk returned as a matched citation
    cited = {c["chunk_id"]: c for c in res["matched_citations"]}
    assert res["matched_chunk_ids"] == list(cited)
    for passage in res["passages"]:
        assert passage["chunk_id"] in cited
        assert passage["chunk_match"] in {"exact", "section"}
    for cit in cited.values():
        assert (cit["ticker"], cit["fiscal_year"], cit["section_id"]) == ("NVDA", 2026, "item_1a")
        assert cit["chunk_id"].startswith("NVDA_2026_item_1a")
        assert cit["excerpt"]
        assert cit["sec_filing_url"].startswith("https://www.sec.gov/")


class _RetrievingAnalystLlm(BaseLlm):
    """Calls search_10k_chunks_rag first, then answers citing the live chunk the tool returned."""

    model: str = "gemini-3.8-flash"
    requests: list[dict[str, Any]] = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        texts = [p.text for c in (llm_request.contents or []) for p in (c.parts or []) if p.text]
        tool_responses = [
            p.function_response
            for c in (llm_request.contents or [])
            for p in (c.parts or [])
            if p.function_response is not None
        ]
        self.requests.append(
            {
                "joined_text": "\n".join(texts),
                "cached_content": llm_request.config.cached_content if llm_request.config else None,
                "tool_responses": len(tool_responses),
            }
        )
        await asyncio.sleep(0.005)
        if not tool_responses:
            yield LlmResponse(
                model_version=self.model,
                partial=False,
                content=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part.from_function_call(
                            name="search_10k_chunks_rag",
                            args={
                                "query": "data center revenue growth drivers Blackwell",
                                "tickers": "NVDA",
                                "gics_sector": "ALL",
                                "section_id": "item_7",
                                "start_year": 2026,
                                "end_year": 2026,
                                "top_k": 3,
                            },
                        )
                    ],
                ),
                usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=3000, candidates_token_count=40
                ),
            )
            return

        payload = tool_responses[-1].response or {}
        cit = payload["citations"][-1]
        final_md = report_json(
            headline="NVIDIA Corp. (NVDA) — Agent-Retrieved 10-K Evidence",
            key_metrics=[("NVDA", "revenue"), ("NVDA", "operating_income")],
            findings=[finding_from_citation(cit, "The retrieved MD&A passage explains this.")],
        )
        yield LlmResponse(
            model_version=self.model,
            partial=False,
            content=genai_types.Content(role="model", parts=[genai_types.Part(text=final_md)]),
            usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
                prompt_token_count=4200, candidates_token_count=110
            ),
        )


def test_chat_turn_merges_agent_retrieved_evidence_and_citations_resolve() -> None:
    """Scripted tool sequence: a chunk the agent retrieves itself becomes a resolvable citation."""
    get_context_latency_tracker().reset()
    get_conversation_memory_manager().reset()
    llm = _RetrievingAnalystLlm(requests=[])

    turn = run_conversational_rag_turn(
        query="What drove NVIDIA's FY2024 to FY2026 data center revenue growth?",
        tickers="NVDA",
        start_year=2024,
        end_year=2026,
        session_id="agentic-retrieval-1",
        user_id="analyst@fde-workspace.corp",
        reasoning_llm=llm,
    )

    assert turn["execution_mode"] == "live_adk"
    assert turn["answer_verified"] is True, turn["adk_runtime"]["answer_withheld_reason"]
    assert turn["adk_runtime"]["answer_withheld_reason"] is None
    assert turn["fact_check"]["passed"] is True

    # No full-section bundle or remote cache is forced into the first request
    first = llm.requests[0]
    assert "<10k_section" not in first["joined_text"]
    assert "Full 10-K sections are NOT in this request" in first["joined_text"]
    assert first["cached_content"] is None
    assert llm.requests[-1]["tool_responses"] == 1

    retrieval = turn["adk_runtime"]["agent_retrieval"]
    assert retrieval["retrieval_tool_calls_count"] == 1
    assert retrieval["retrieval_tool_calls"][0]["name"] == "search_10k_chunks_rag"
    assert retrieval["cited_chunk_ids"], "answer must cite the retrieved chunk"
    assert retrieval["unresolved_cited_chunk_ids"] == []
    turn_chunk_ids = {c["chunk_id"] for c in turn["citations"]}
    assert set(retrieval["cited_chunk_ids"]) <= turn_chunk_ids
    assert set(retrieval["agent_retrieved_chunk_ids"]) <= turn_chunk_ids
    assert all(cid.startswith("NVDA_2026_item_7") for cid in retrieval["cited_chunk_ids"])

    # Structured answer: verified claims are returned as JSON alongside the rendered Markdown
    assert turn["structured_report"]["headline"].startswith("NVIDIA Corp. (NVDA)")
    assert turn["report_verification"]["findings_verified"] == 1
    assert turn["report_verification"]["usable"] is True
    assert "#### Findings" in turn["answer_markdown"]
    assert "*Evidence:*" in turn["answer_markdown"]


def test_live_gemini_turn_retrieval_and_citations_resolve() -> None:
    """Live Gemini decides what to retrieve; whatever it cites must resolve to turn citations."""
    get_context_latency_tracker().reset()
    get_conversation_memory_manager().reset()

    turn = run_conversational_rag_turn(
        query="How did NVIDIA's export-control risk disclosures change from FY2024 to FY2026?",
        tickers="NVDA",
        start_year=2024,
        end_year=2026,
        session_id="agentic-live-1",
        user_id="analyst@fde-workspace.corp",
    )

    assert turn["status"] == "success"
    assert turn["execution_mode"] == "live_adk"
    assert turn["adk_runtime"]["backend"] == "vertex_ai"
    assert turn["adk_runtime"]["executed_via_adk_runner"] is True

    turn_chunk_ids = {c["chunk_id"] for c in turn["citations"]}
    assert turn_chunk_ids
    retrieval = turn["adk_runtime"]["agent_retrieval"]
    assert retrieval["retrieval_tool_calls_count"] == len(retrieval["retrieval_tool_calls"])
    assert retrieval["retrieval_tool_calls_count"] <= 6  # retrieval budget
    assert {c["name"] for c in retrieval["retrieval_tool_calls"]} <= _RETRIEVAL_TOOLS
    assert set(retrieval["agent_retrieved_chunk_ids"]) <= turn_chunk_ids
    assert retrieval["unresolved_cited_chunk_ids"] == []
    assert set(retrieval["cited_chunk_ids"]) <= turn_chunk_ids

    md = turn["answer_markdown"]
    assert "default_api." not in md and "/*ACTION*/" not in md
    # Acceptance matrix (qualitative wording-shift question): the live agent selects no
    # variance or anomaly engine, so no variance table, flags, or Key Metrics are shown.
    invoked = set(turn["adk_runtime"]["agent_invoked_tools"])
    assert not invoked & {"calculate_period_variances", "flag_material_variances"}, invoked
    assert turn["variance_reports"] == []
    assert turn["material_variance_flags"] == []
    assert "#### Key Metrics" not in md
    assert turn["intent"] == "thematic_trend_tracking", turn["intent_source"]
    if turn["answer_verified"]:
        assert turn["adk_runtime"]["answer_withheld_reason"] is None
        assert turn["report_verification"]["usable"] is True
        assert turn["report_verification"]["findings_verified"] >= 1
        assert retrieval["cited_chunk_ids"], "a verified answer cites retrieved chunks"
        assert turn["fact_check"]["passed"] is True
    else:
        assert md.startswith("### Answer withheld")
        assert turn["adk_runtime"]["answer_withheld_reason"]


def test_judge_sees_the_passages_the_report_cites() -> None:
    from edgar_capstone.agents.live_runtime import select_citations_for_judge

    citations = [
        {"chunk_id": f"NVDA_2026_item_1a_{i:04d}", "excerpt": "x" * 5000} for i in range(40)
    ]
    report = (
        "Export rules tightened [NVDA FY2026 | Item 1A. Risk Factors | NVDA_2026_item_1a_0025]"
        "(https://sec.gov/a) and China sales fell "
        "[NVDA FY2026 | Item 1A. Risk Factors | NVDA_2026_item_1a_0012](https://sec.gov/b).\n\n"
        "| chunk_id |\n|---|\n| NVDA_2026_item_1a_0033 |\n"  # table-only reference
    )
    selected = select_citations_for_judge(report, citations)
    assert [c["chunk_id"] for c in selected[:3]] == [
        "NVDA_2026_item_1a_0025",
        "NVDA_2026_item_1a_0012",
        "NVDA_2026_item_1a_0033",
    ]
    assert len(selected) == 16
    assert len({c["chunk_id"] for c in selected}) == 16
    assert all(len(c["excerpt"]) == 4000 for c in selected)
    assert len(citations[0]["excerpt"]) == 5000  # inputs are not mutated

    # Reports referencing more than 16 chunks get all of them (up to 30)
    many = " ".join(f"NVDA_2026_item_1a_{i:04d}" for i in range(22))
    assert len(select_citations_for_judge(many, citations)) == 22


class _TextActionLlm(BaseLlm):
    """Reproduces the live failure: a tool call written as prose instead of a function call."""

    model: str = "gemini-3.8-flash"

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        yield LlmResponse(
            model_version=self.model,
            partial=False,
            content=genai_types.Content(
                role="model",
                parts=[
                    genai_types.Part(
                        text=(
                            "/*PLANNING*/ 1. Query FY2024 risk factors. 2. Compare with FY2026.\n"
                            "/*ACTION*/ default_api.query_cached_10k_context(ticker='NVDA', "
                            "fiscal_year=2024, section_id='item_1a', variance_query='export', "
                            "max_passages=4)"
                        )
                    )
                ],
            ),
        )


def test_unexecuted_text_tool_call_is_withheld_instead_of_shipping_a_plan() -> None:
    get_context_latency_tracker().reset()
    get_conversation_memory_manager().reset()
    turn = run_conversational_rag_turn(
        query="How did NVIDIA's export-control risk disclosures change from FY2024 to FY2026?",
        tickers="NVDA",
        start_year=2024,
        end_year=2026,
        session_id="agentic-text-action-1",
        user_id="analyst@fde-workspace.corp",
        reasoning_llm=_TextActionLlm(),
    )
    # The prose tool call fails AnalystReport validation (or the incomplete-output guard), so
    # the narrative is withheld; no template answer is substituted.
    assert turn["status"] == "success"
    assert turn["execution_mode"] == "live_adk"
    assert turn["answer_verified"] is False
    assert turn["adk_runtime"]["answer_verified"] is False
    assert turn["adk_runtime"]["answer_withheld_reason"]
    assert turn["answer_markdown"].startswith("### Answer withheld")
    assert "default_api." not in turn["answer_markdown"]
    assert "/*ACTION*/" not in turn["answer_markdown"]
    # Nothing is pre-computed: the agent invoked no deterministic engine, so no variance
    # table or flags are attached, while the retrieved evidence is still returned.
    assert turn["variance_reports"] == []
    assert turn["material_variance_flags"] == []
    assert turn["citations"]


def test_retrieval_budget_short_circuits_tools_after_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from edgar_capstone.agents.callbacks import retrieval_budget_before_tool_callback

    monkeypatch.setenv("EDGAR_ANALYST_RETRIEVAL_BUDGET", "2")
    ctx = SimpleNamespace(state={})
    search = SimpleNamespace(name="search_10k_chunks_rag")
    calc = SimpleNamespace(name="calculate_period_variances")
    assert retrieval_budget_before_tool_callback(search, {}, ctx) is None
    assert retrieval_budget_before_tool_callback(search, {}, ctx) is None
    blocked = retrieval_budget_before_tool_callback(search, {}, ctx)
    assert blocked is not None and blocked["status"] == "retrieval_budget_exhausted"
    assert "Write the AnalystReport now" in blocked["message"]
    # Non-retrieval tools are never budgeted
    assert retrieval_budget_before_tool_callback(calc, {}, ctx) is None
