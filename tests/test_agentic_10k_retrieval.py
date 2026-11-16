"""Agent-driven 10-K retrieval: FinancialAnalystAgent queries the evidence each question needs."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
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
from edgar_capstone.agents.tools import (
    _SESSION_10K_CONTEXT_CACHE,
    _load_all_chunks_cached,
    _load_all_profiles_cached,
    load_cached_10k_section,
)
from tests.structured_report_helpers import finding_from_citation, report_json


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
            for i in range(12)
        ],
        sector_benchmarks=[],
        chunk_health=None,
    )
    assert "<10k_section" not in prompt
    assert "Full 10-K sections are NOT in this request" in prompt
    assert "query_cached_10k_context(" in prompt and "search_10k_chunks_rag(" in prompt
    assert "without redundant tool calls" not in prompt
    # Leads are capped in count and length and point at citable chunk ids
    assert prompt.count('"chunk_id": "NVDA_2026_item_7_') == 8
    assert long_excerpt[:300] not in prompt
    assert '"pre_queried_sections"' in prompt
    # Suggested retrieval targets are surfaced once, not duplicated inside the evaluations JSON
    assert prompt.count("operating income & margin variance drivers") == 1
    assert "NVDA FY2026 revenue operating income ..." not in prompt


def test_query_cached_10k_context_ranks_full_section_and_returns_citable_passages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EDGAR_DATA_BACKEND", "local")
    filler = "\n\n".join(
        f"Paragraph {i}: routine discussion of general administrative matters and facilities."
        for i in range(900)
    )
    needle = (
        "Quantum interconnect licensing revenue from the Zephyr program drove the operating "
        "margin expansion during the fiscal year."
    )
    full_text = f"{filler}\n\n{needle}"
    assert full_text.index(needle) > 50_000  # beyond the 50k-char excerpt limit
    _SESSION_10K_CONTEXT_CACHE["ZZZQ_2026_item_7"] = {
        "profile_id": "ZZZQ_2026",
        "ticker": "ZZZQ",
        "company_name": "Zephyr Quantum Corp",
        "fiscal_year": 2026,
        "section_id": "item_7",
        "sec_filing_url": "https://www.sec.gov/zzzq",
        "unstructured_gcs_uri": "",
        "token_estimate": len(full_text) // 4,
        "full_text": full_text,
        "content_source": "local_markdown",
    }
    try:
        res = query_cached_10k_context(
            ticker="ZZZQ",
            fiscal_year=2026,
            section_id="item_7",
            variance_query="Zephyr quantum interconnect licensing revenue operating margin",
            max_passages=2,
        )
    finally:
        _SESSION_10K_CONTEXT_CACHE.pop("ZZZQ_2026_item_7", None)
    assert res["status"] == "success"
    assert res["passages"][0]["text"] == needle
    assert res["content_source"] == "local_markdown"
    assert "warning" not in res
    assert {"text", "chunk_id", "chunk_match"} <= set(res["passages"][0])


def test_section_tools_flag_synthesized_placeholder_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EDGAR_DATASET_DIR", str(tmp_path))
    monkeypatch.setenv("EDGAR_DATA_BACKEND", "local")
    _load_all_profiles_cached.cache_clear()
    _load_all_chunks_cached.cache_clear()
    _SESSION_10K_CONTEXT_CACHE.clear()
    try:
        sec = load_cached_10k_section(
            ticker="NVDA", fiscal_year=2026, section_id="item_7", max_chars=1000
        )
        assert sec["content_source"] == "synthesized_fallback"
        res = query_cached_10k_context(
            ticker="NVDA",
            fiscal_year=2026,
            section_id="item_7",
            variance_query="data center revenue growth",
            max_passages=2,
        )
        assert res["content_source"] == "synthesized_fallback"
        assert "do not cite" in res["warning"]
    finally:
        _SESSION_10K_CONTEXT_CACHE.clear()
        _load_all_profiles_cached.cache_clear()
        _load_all_chunks_cached.cache_clear()


class _RetrievingAnalystLlm(BaseLlm):
    """Calls search_10k_chunks_rag first, then answers citing the chunk the tool returned."""

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
            findings=[finding_from_citation(cit, "Data center demand drove the growth.")],
        )
        yield LlmResponse(
            model_version=self.model,
            partial=False,
            content=genai_types.Content(role="model", parts=[genai_types.Part(text=final_md)]),
            usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
                prompt_token_count=4200, candidates_token_count=110
            ),
        )


class _JudgeLlm(BaseLlm):
    model: str = "gemini-3.5-flash"

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
                            '{"evaluator_model": "gemini-3.5-flash", "task_category": "evaluation", '
                            '"grade": "pass", "overall_score": 0.95, "numeric_fidelity_score": 1.0, '
                            '"citation_grounding_score": 0.95, "explanation_quality_score": 0.9, '
                            '"thematic_consistency_score": 0.9, "rationale": "Grounded."}'
                        )
                    )
                ],
            ),
        )


def test_chat_turn_agent_retrieves_its_own_evidence_and_citations_resolve() -> None:
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
        judge_llm=_JudgeLlm(),
    )

    assert turn["execution_mode"] == "live_adk"
    assert turn["adk_runtime"]["grounding_fallback_applied"] is False
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

    # Structured answer: verified claims are returned as JSON alongside the rendered Markdown
    assert turn["structured_report"]["headline"].startswith("NVIDIA Corp. (NVDA)")
    assert turn["report_verification"]["findings_verified"] == 1
    assert turn["report_verification"]["usable"] is True
    assert "#### Findings" in turn["answer_markdown"]
    assert "*Evidence:*" in turn["answer_markdown"]


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


def test_unexecuted_text_tool_call_falls_back_instead_of_shipping_a_plan() -> None:
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
        judge_llm=_JudgeLlm(),
    )
    # With structured output, a prose tool call fails AnalystReport validation (or the
    # incomplete-output guard) and the turn falls back instead of shipping the plan.
    assert turn["adk_runtime"]["grounding_fallback_applied"] is True or (
        turn["execution_mode"] == "deterministic_fallback"
    )
    assert "default_api." not in turn["answer_markdown"]
    assert "/*ACTION*/" not in turn["answer_markdown"]


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
