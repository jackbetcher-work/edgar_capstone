"""Agentic tool use & selective display: the acceptance matrix for FinancialAnalystAgent.

`FinancialAnalystAgent` decides which deterministic engines a question needs; nothing is
pre-computed. Tool outputs are hydrated from ADK `session_state` before claim verification,
Markdown rendering, and the numeric fact-check, and only the payloads the agent produced are
returned with the turn (and rendered as cards by the UI).

Tools read live GCP (BigQuery `financial_metrics` / `filing_chunks`, Vertex AI Search, GCS).
Scripted ADK `BaseLlm`s pin the agent's tool choices so each matrix row is exercised
deterministically; their answers cite only chunks the live tools returned.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types

from edgar_capstone.agents import get_context_latency_tracker, get_conversation_memory_manager
from edgar_capstone.agents.chat_engine import _get_entity_catalog, run_conversational_rag_turn
from edgar_capstone.agents.prompt_builders import build_financial_reasoning_prompt
from edgar_capstone.agents.schemas import AnalystReport
from edgar_capstone.agents.scope_validation import build_coverage_notices
from edgar_capstone.agents.sector_tool import sector_thematic_map_reduce
from edgar_capstone.agents.structured_report import (
    render_analyst_report_markdown,
    verify_analyst_report,
)
from edgar_capstone.agents.tool_runtime import (
    STATE_CALCULATED_VARIANCES,
    STATE_DISCLOSURE_DELTAS,
    STATE_MATERIAL_FLAG_SUMMARY,
    STATE_MATERIAL_FLAGS,
    ToolOutputs,
    hydrate_tool_outputs,
    merge_disclosure_deltas,
    tool_turn_context,
    turn_scoped_state_seed,
)
from edgar_capstone.agents.tools import calculate_period_variances
from tests.structured_report_helpers import finding_from_citation

_USER = "analyst@fde-workspace.corp"


_LEADS_BLOCK_RE = re.compile(r"## 10-K Evidence Leads[^\n]*\n```json\n(.*?)\n```", re.S)


def _latest_evidence_leads(request_text: str) -> list[dict[str, Any]]:
    """Leads from the CURRENT turn's prompt (persistent sessions replay earlier turns)."""
    blocks = _LEADS_BLOCK_RE.findall(request_text)
    if not blocks:
        return []
    return [lead for lead in json.loads(blocks[-1]) if lead.get("chunk_id")]


def _report_json(
    *,
    intent: str | None,
    headline: str,
    leads: list[dict[str, Any]],
    key_metrics: list[tuple[str, str]] | None = None,
    max_findings: int = 2,
) -> str:
    findings = [
        finding_from_citation(lead, "The cited 10-K passage addresses the question.")
        for lead in leads[:max_findings]
    ]
    return AnalystReport(
        intent=intent,  # type: ignore[arg-type]
        headline=headline,
        key_metrics=[{"ticker": t, "metric": m} for t, m in (key_metrics or [])],
        findings=findings,
    ).model_dump_json()


class _ScriptedAnalystLlm(BaseLlm):
    """Round 0: issue `tool_calls` (if any) in parallel; final round: answer with an AnalystReport."""

    model: str = "gemini-3.8-flash"
    tool_calls: list[dict[str, Any]] = []
    intent: str | None = None
    key_metrics: list[tuple[str, str]] = []
    requests: list[str] = []
    _called_tools: bool = False

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        text = "\n".join(
            p.text for c in (llm_request.contents or []) for p in (c.parts or []) if p.text
        )
        self.requests.append(text)
        await asyncio.sleep(0.002)
        current_turn = llm_request.contents[-1:] if llm_request.contents else []
        pending_tools = not any(
            p.function_response is not None for c in current_turn for p in (c.parts or [])
        )
        if self.tool_calls and pending_tools and not self._called_tools:
            self._called_tools = True
            yield LlmResponse(
                model_version=self.model,
                partial=False,
                content=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part.from_function_call(name=c["name"], args=c["args"])
                        for c in self.tool_calls
                    ],
                ),
                usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=2500, candidates_token_count=30
                ),
            )
            return
        leads = _latest_evidence_leads(text)
        yield LlmResponse(
            model_version=self.model,
            partial=False,
            content=genai_types.Content(
                role="model",
                parts=[
                    genai_types.Part(
                        text=_report_json(
                            intent=self.intent,
                            headline="Grounded answer from the cited 10-K passages.",
                            leads=leads,
                            key_metrics=self.key_metrics,
                        )
                    )
                ],
            ),
            usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
                prompt_token_count=4000, candidates_token_count=120
            ),
        )


def _reset() -> None:
    get_context_latency_tracker().reset()
    get_conversation_memory_manager().reset()


def _trace_tools(turn: dict[str, Any]) -> list[str]:
    return [str(step["tool"]) for step in turn["agent_trace"]]


# ---------------------------------------------------------------------------
# Unit: session-state seeding, hydration, merge
# ---------------------------------------------------------------------------


def test_turn_scoped_seed_resets_every_tool_output_key_and_hydration_round_trips() -> None:
    seed = turn_scoped_state_seed(citations=[{"chunk_id": "A_2026_item_1a_0001"}])
    assert seed[STATE_CALCULATED_VARIANCES] == {}
    assert seed[STATE_MATERIAL_FLAGS] == []
    assert seed[STATE_MATERIAL_FLAG_SUMMARY] is None
    assert seed[STATE_DISCLOSURE_DELTAS] is None
    empty = hydrate_tool_outputs(seed)
    assert empty.variance_reports == [] and empty.material_variance_flags == []
    assert empty.disclosure_deltas is None and empty.sector_coverage is None
    assert [c["chunk_id"] for c in empty.citations] == ["A_2026_item_1a_0001"]

    prior = ToolOutputs(
        variance_reports=[{"ticker": "NVDA", "start_year": 2024, "end_year": 2026}],
        material_variance_flags=[{"flag_id": "f1", "ticker": "NVDA"}],
        sector_benchmarks=[{"gics_sector": "Information Technology"}],
    )
    hydrated = hydrate_tool_outputs(turn_scoped_state_seed(citations=[], prior=prior))
    assert hydrated.calculated_variances_map == {"NVDA_2024_2026": prior.variance_reports[0]}
    assert hydrated.material_variance_flags == prior.material_variance_flags
    assert hydrated.sector_benchmarks == prior.sector_benchmarks

    merged = hydrated.merge(
        ToolOutputs(
            variance_reports=[{"ticker": "AAPL", "start_year": 2024, "end_year": 2026}],
            material_variance_flags=[{"flag_id": "f1", "ticker": "NVDA", "severity": "high"}],
        )
    )
    assert sorted(merged.calculated_variances_map) == ["AAPL_2024_2026", "NVDA_2024_2026"]
    assert merged.material_variance_flags == [
        {"flag_id": "f1", "ticker": "NVDA", "severity": "high"}
    ]


def test_disclosure_delta_payloads_merge_by_ticker_with_summed_counts() -> None:
    a = {
        "status": "success",
        "start_year": 2024,
        "end_year": 2025,
        "summary": {"added": 1, "removed": 0, "modified": 2, "unchanged": 0, "year_pairs": 1},
        "companies": [{"ticker": "NVDA"}],
        "citations": [{"chunk_id": "x"}],
    }
    b = {
        "status": "success",
        "start_year": 2025,
        "end_year": 2026,
        "summary": {"added": 2, "removed": 1, "modified": 0, "unchanged": 3, "year_pairs": 1},
        "companies": [{"ticker": "AMD"}],
        "citations": [{"chunk_id": "x"}, {"chunk_id": "y"}],
    }
    merged = merge_disclosure_deltas(a, b)
    assert merged is not None
    assert [c["ticker"] for c in merged["companies"]] == ["NVDA", "AMD"]
    assert merged["summary"]["added"] == 3 and merged["summary"]["year_pairs"] == 2
    assert merged["summary"]["companies_analyzed"] == 2
    assert (merged["start_year"], merged["end_year"]) == (2024, 2026)
    assert [c["chunk_id"] for c in merged["citations"]] == ["x", "y"]
    assert merge_disclosure_deltas(None, {"status": "error"}) == {"status": "error"}
    assert merge_disclosure_deltas(a, {"status": "error"}) is a


def test_variance_tool_writes_reports_and_missing_tickers_to_session_state() -> None:
    ctx = SimpleNamespace(state={})
    res = calculate_period_variances(
        tickers="NVDA,ZZZZQ", start_year=2024, end_year=2026, tool_context=ctx
    )
    assert res["status"] == "success"
    out = hydrate_tool_outputs(ctx.state)
    assert [r["ticker"] for r in out.variance_reports] == ["NVDA"]
    assert out.missing_tickers == ["ZZZZQ"]


def test_variance_tool_reuses_cached_catalog_inside_a_turn() -> None:
    _, profiles = _get_entity_catalog(allow_fetch=True)
    direct = calculate_period_variances(tickers="AAPL", start_year=2024, end_year=2026)
    with (
        patch(
            "edgar_capstone.agents.tools._gcp_adapters",
            side_effect=AssertionError("cached turn catalog must avoid BigQuery RPCs"),
        ),
        tool_turn_context(cached_profiles=profiles),
    ):
        cached = calculate_period_variances(tickers="AAPL", start_year=2024, end_year=2026)
    assert cached["variance_reports"] == direct["variance_reports"]


def test_coverage_notices_come_from_profile_catalog_without_variance_reports() -> None:
    _, profiles = _get_entity_catalog(allow_fetch=True)
    notices = build_coverage_notices(
        variance_reports=[],
        profiles=profiles,
        tickers=["NVDA", "AAPL"],
        start_year=2024,
        end_year=2026,
    )
    codes = {n["code"] for n in notices}
    # NVDA's fiscal year ends in January, Apple's in September
    assert "fiscal_calendar_misalignment" in codes, notices
    cal = next(n for n in notices if n["code"] == "fiscal_calendar_misalignment")
    assert set(cal["tickers"]) == {"NVDA", "AAPL"}

    future = build_coverage_notices(
        variance_reports=[],
        profiles=profiles,
        tickers=["NVDA"],
        start_year=2026,
        end_year=2027,
    )
    missing = [n for n in future if n["code"] == "missing_fiscal_year"]
    assert missing and missing[0]["missing_years"] == [2027]


def test_intent_round_trips_through_verification() -> None:
    report = AnalystReport(intent="thematic_trend_tracking", headline="Wording shifted.")
    verified = verify_analyst_report(report, variance_reports=[], citations=[])
    assert verified.report.intent == "thematic_trend_tracking"
    md = render_analyst_report_markdown(verified, variance_reports=[], citations=[])
    assert "#### Key Metrics" not in md


def test_prompt_carries_tool_selection_policy_and_no_precomputed_payloads() -> None:
    scope = {
        "tickers": ["NVDA"],
        "tickers_csv": "NVDA",
        "gics_sector": "ALL",
        "section_id": "ALL",
        "start_year": 2024,
        "end_year": 2026,
    }
    prompt = build_financial_reasoning_prompt(
        query="How has NVIDIA's export control risk wording changed?",
        intent="longitudinal_single_company",
        scope=scope,
        variance_reports=[],
        variance_evaluations=[],
        cached_explanations=[],
        citations=[],
        sector_benchmarks=[],
        chunk_health=None,
    )
    assert "## Deterministic Tool Selection" in prompt
    assert "No variance tables, anomaly flags, or disclosure deltas were computed" in prompt
    assert "Authoritative Deterministic Period-over-Period Variance Calculations" not in prompt
    assert "Authoritative Material Variance Flags" not in prompt
    assert "sector_thematic_map_reduce(" in prompt
    assert "leave `key_metrics` EMPTY for" in prompt
    assert "`intent`" in prompt


# ---------------------------------------------------------------------------
# Acceptance matrix (scripted agent tool choices, live GCP tools)
# ---------------------------------------------------------------------------


def test_qualitative_risk_wording_turn_shows_no_variance_flags_or_key_metrics() -> None:
    _reset()
    llm = _ScriptedAnalystLlm(tool_calls=[], intent="thematic_trend_tracking", requests=[])
    progress: list[tuple[str, dict[str, Any] | None]] = []
    turn = run_conversational_rag_turn(
        query="How has NVIDIA's export control risk wording changed from FY2024 to FY2026?",
        tickers="NVDA",
        start_year=2024,
        end_year=2026,
        session_id="agentic-qual-1",
        user_id=_USER,
        reasoning_llm=llm,
        on_progress=lambda stage, _msg, meta: progress.append((stage, meta)),
    )
    assert turn["status"] == "success"
    assert turn["answer_verified"] is True, turn["adk_runtime"]["answer_withheld_reason"]
    assert turn["variance_reports"] == []
    assert turn["material_variance_flags"] == []
    assert turn["material_variance_summary"] is None
    assert turn["disclosure_deltas"] is None
    assert "#### Key Metrics" not in turn["answer_markdown"]
    assert "XBRL" not in turn["answer_markdown"]
    assert turn["intent"] == "thematic_trend_tracking"
    assert turn["intent_source"] == "agent"
    assert turn["fact_check"]["passed"] is True
    tools = _trace_tools(turn)
    assert "resolve_query_scope" in tools and "classify_query_intent" not in tools
    assert not {"calculate_period_variances", "flag_material_variances"} & set(tools)
    assert turn["adk_runtime"]["agent_invoked_tools"] == []
    # No pre-computed payload reached the agent
    assert "Authoritative Deterministic Period-over-Period" not in llm.requests[0]
    scope_evt = next(meta for stage, meta in progress if stage == "scope_resolved")
    assert scope_evt is not None and scope_evt["intent"] is None
    assert any(stage == "intent_resolved" for stage, _ in progress)


def test_quantitative_peer_turn_computes_variances_and_only_requested_key_metrics() -> None:
    _reset()
    llm = _ScriptedAnalystLlm(
        tool_calls=[
            {
                "name": "calculate_period_variances",
                "args": {"tickers": "AAPL,MSFT", "start_year": 2024, "end_year": 2026},
            }
        ],
        intent="multi_company_peer_comparison",
        key_metrics=[("AAPL", "research_and_development"), ("MSFT", "research_and_development")],
        requests=[],
    )
    progress: list[str] = []
    turn = run_conversational_rag_turn(
        query="Compare AAPL and MSFT R&D spend FY24-FY26",
        tickers="AAPL,MSFT",
        start_year=2024,
        end_year=2026,
        session_id="agentic-quant-1",
        user_id=_USER,
        reasoning_llm=llm,
        on_progress=lambda stage, _msg, _meta: progress.append(stage),
    )
    assert turn["answer_verified"] is True, turn["adk_runtime"]["answer_withheld_reason"]
    assert sorted(r["ticker"] for r in turn["variance_reports"]) == ["AAPL", "MSFT"]
    assert turn["material_variance_flags"] == []
    assert turn["material_variance_summary"] is None
    md = turn["answer_markdown"]
    assert "#### Key Metrics (deterministic engine)" in md
    assert md.count("R&D Expense:**") == 2
    assert "Revenue:**" not in md and "Net Income:**" not in md
    assert "| XBRL " in md and "acc:" in md
    assert turn["fact_check"]["passed"] is True
    assert turn["fact_check"]["checked_metrics_count"] >= 2
    assert turn["intent"] == "multi_company_peer_comparison"
    assert turn["adk_runtime"]["agent_invoked_tools"] == ["calculate_period_variances"]
    assert turn["adk_runtime"]["key_metric_backfill_tickers"] == []
    step = next(s for s in turn["agent_trace"] if s["tool"] == "calculate_period_variances")
    assert step["s2s_verified"] is True and "Agent-invoked" in step["summary"]
    assert "agent_tool" in progress
    # Fiscal calendar notice is preserved (AAPL Sep vs MSFT Jun year-ends)
    assert any(n["code"] == "fiscal_calendar_misalignment" for n in turn["coverage_notices"])


def test_anomaly_turn_invokes_variances_and_flags_with_mdna_explanations() -> None:
    _reset()
    llm = _ScriptedAnalystLlm(
        tool_calls=[
            {
                "name": "calculate_period_variances",
                "args": {"tickers": "INTC", "start_year": 2023, "end_year": 2025},
            },
            {
                "name": "flag_material_variances",
                "args": {
                    "tickers": "INTC",
                    "start_year": 2023,
                    "end_year": 2025,
                    "explain_top_flags": True,
                },
            },
        ],
        intent="longitudinal_single_company",
        key_metrics=[("INTC", "net_income")],
        requests=[],
    )
    turn = run_conversational_rag_turn(
        query="Why did Intel's net income turn negative between 2023 and 2025?",
        tickers="INTC",
        start_year=2023,
        end_year=2025,
        session_id="agentic-anomaly-1",
        user_id=_USER,
        reasoning_llm=llm,
    )
    assert turn["answer_verified"] is True, turn["adk_runtime"]["answer_withheld_reason"]
    assert [r["ticker"] for r in turn["variance_reports"]] == ["INTC"]
    flags = turn["material_variance_flags"]
    assert flags, "Intel's FY2024 net loss must be flagged"
    assert any(f["metric"] == "net_income" and f["rule"] == "sign_reversal" for f in flags), [
        (f["metric"], f["rule"]) for f in flags
    ]
    summary = turn["material_variance_summary"]
    assert summary is not None and summary["total_flags"] >= 1
    explained = [f for f in flags if f.get("explanation")]
    assert summary["explained_flags"] == len(explained) >= 1
    turn_chunk_ids = {c["chunk_id"] for c in turn["citations"]}
    assert all(f["explanation"]["chunk_id"] in turn_chunk_ids for f in explained)
    assert set(turn["adk_runtime"]["agent_invoked_tools"]) == {
        "calculate_period_variances",
        "flag_material_variances",
    }
    assert "flag_material_variances" in _trace_tools(turn)


def test_key_metrics_without_variance_tool_are_backfilled_deterministically() -> None:
    """XBRL chips stay backed by `variance_reports` even if the agent skipped the tool."""
    _reset()
    llm = _ScriptedAnalystLlm(
        tool_calls=[],
        intent="longitudinal_single_company",
        key_metrics=[("NVDA", "revenue"), ("ZZZZ", "revenue")],
        requests=[],
    )
    turn = run_conversational_rag_turn(
        query="What was NVIDIA's revenue from FY2024 to FY2026?",
        tickers="NVDA",
        start_year=2024,
        end_year=2026,
        session_id="agentic-backfill-1",
        user_id=_USER,
        reasoning_llm=llm,
    )
    assert turn["answer_verified"] is True, turn["adk_runtime"]["answer_withheld_reason"]
    assert turn["adk_runtime"]["key_metric_backfill_tickers"] == ["NVDA"]
    assert [r["ticker"] for r in turn["variance_reports"]] == ["NVDA"]
    assert "Revenue:**" in turn["answer_markdown"] and "acc:" in turn["answer_markdown"]
    assert turn["report_verification"]["metrics_dropped"] == ["ZZZZ:revenue"]
    assert turn["fact_check"]["passed"] is True


def test_tool_outputs_do_not_leak_into_the_next_turn_of_a_persistent_session() -> None:
    _reset()
    quant = _ScriptedAnalystLlm(
        tool_calls=[
            {
                "name": "calculate_period_variances",
                "args": {"tickers": "NVDA", "start_year": 2024, "end_year": 2026},
            }
        ],
        intent="longitudinal_single_company",
        key_metrics=[("NVDA", "revenue")],
        requests=[],
    )
    first = run_conversational_rag_turn(
        query="How did NVIDIA's revenue change from FY2024 to FY2026?",
        tickers="NVDA",
        session_id="agentic-session-iso",
        user_id=_USER,
        reasoning_llm=quant,
    )
    assert first["variance_reports"]
    qual = _ScriptedAnalystLlm(tool_calls=[], intent="thematic_trend_tracking", requests=[])
    second = run_conversational_rag_turn(
        query="How did NVIDIA describe supply chain risks in its FY2026 risk factors?",
        tickers="NVDA",
        start_year=2026,
        end_year=2026,
        session_id="agentic-session-iso",
        user_id=_USER,
        reasoning_llm=qual,
    )
    assert second["answer_verified"] is True, second["adk_runtime"]["answer_withheld_reason"]
    assert second["variance_reports"] == []
    assert second["material_variance_flags"] == []
    assert "#### Key Metrics" not in second["answer_markdown"]


def _fake_sector_payload(sector: str, start_year: int, end_year: int) -> dict[str, Any]:
    cov = {
        "gics_sector": sector,
        "start_year": start_year,
        "end_year": end_year,
        "issuers_in_sector": 31,
        "issuers_with_filings": 31,
        "issuers_analyzed": 29,
        "issuers_with_theme_matches": 17,
        "filings_by_year": {str(start_year): 31, str(end_year): 30},
        "missing_fy_counts": {str(start_year): 0, str(end_year): 1},
        "theme_mentions_by_year": {str(start_year): 12, str(end_year): 40},
        "total_chunks_evaluated": 174,
        "batches_processed": 3,
        "batch_size": 14,
        "mode": "batched_map_reduce",
    }
    citation = {
        "chunk_id": "PLD_2025_item_1a_0003",
        "ticker": "PLD",
        "fiscal_year": end_year,
        "section_id": "item_1a",
        "section_title": "Item 1A. Risk Factors",
        "excerpt": "Climate change and extreme weather could adversely affect our properties.",
        "sec_filing_url": "https://www.sec.gov/pld",
    }
    return {
        "status": "success",
        "sector_coverage": cov,
        "sector_map_summaries": {"sector_coverage": cov, "batch_summaries": []},
        "disclosure_deltas": {
            "status": "success",
            "theme": "climate risk",
            "section_id": "item_1a",
            "start_year": start_year,
            "end_year": end_year,
            "companies_selected_by": "full_sector_map_reduce",
            "missing_tickers": [],
            "summary": {
                "added": 4,
                "removed": 1,
                "modified": 6,
                "unchanged": 9,
                "year_pairs": 29,
                "companies_analyzed": 29,
            },
            "companies": [{"ticker": "PLD", "year_pairs": []}],
            "citations": [citation],
        },
        "citations": [citation],
    }


def test_sector_wide_thematic_turn_runs_map_reduce_tool_with_sse_progress() -> None:
    _reset()
    captured: dict[str, Any] = {}

    def _fake_map_reduce(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        on_progress = kwargs["on_progress"]
        assert on_progress is not None, "SSE progress callback must reach the tool"
        for done in (14, 28, 29):
            on_progress(
                "sector_map_progress",
                f"{done}/29 {kwargs['gics_sector']} companies analyzed",
                {"analyzed": done, "total": 29},
            )
        return _fake_sector_payload(
            kwargs["gics_sector"], int(kwargs["start_year"]), int(kwargs["end_year"])
        )

    llm = _ScriptedAnalystLlm(
        tool_calls=[
            {
                "name": "sector_thematic_map_reduce",
                "args": {
                    "theme": "climate and extreme weather risk",
                    "gics_sector": "Real Estate",
                    "section_id": "item_1a",
                    "start_year": 2020,
                    "end_year": 2025,
                },
            }
        ],
        intent=None,  # exercise the deterministic tool-trace intent fallback
        requests=[],
    )
    stages: list[str] = []
    sentinel_map_llm = object()
    with patch(
        "edgar_capstone.agents.sector_map_reduce.run_sector_thematic_map_reduce",
        side_effect=_fake_map_reduce,
    ):
        turn = run_conversational_rag_turn(
            query="How have climate risk disclosures changed across Real Estate companies from 2020 to 2025?",
            gics_sector="Real Estate",
            start_year=2020,
            end_year=2025,
            session_id="agentic-sector-1",
            user_id=_USER,
            reasoning_llm=llm,
            map_llm=sentinel_map_llm,  # type: ignore[arg-type]
            on_progress=lambda stage, _msg, _meta: stages.append(stage),
        )
    assert turn["answer_verified"] is True, turn["adk_runtime"]["answer_withheld_reason"]
    assert captured["gics_sector"] == "Real Estate"
    assert captured["map_llm"] is sentinel_map_llm
    assert captured["sector_profiles"], "cached catalog profiles are reused by the tool"
    assert stages.count("sector_map_progress") == 3 and "agent_tool" in stages
    assert turn["sector_coverage"]["issuers_analyzed"] == 29
    assert turn["disclosure_deltas"]["companies_selected_by"] == "full_sector_map_reduce"
    assert turn["variance_reports"] == [] and turn["material_variance_flags"] == []
    assert turn["intent"] == "thematic_trend_tracking"
    assert turn["intent_source"] == "tool_inference"
    assert turn["candidates_searched"] == 174
    assert any(n["code"] == "sector_coverage_summary" for n in turn["coverage_notices"])
    assert "PLD_2025_item_1a_0003" in {c["chunk_id"] for c in turn["citations"]}
    assert "sector_thematic_map_reduce" in _trace_tools(turn)


def test_sector_tool_requires_a_sector_and_writes_full_payload_to_state() -> None:
    assert (
        sector_thematic_map_reduce(
            theme="AI risk", gics_sector="ALL", section_id="ALL", start_year=2024, end_year=2026
        )["status"]
        == "error"
    )
    ctx = SimpleNamespace(state={})
    with patch(
        "edgar_capstone.agents.sector_map_reduce.run_sector_thematic_map_reduce",
        side_effect=lambda **kw: _fake_sector_payload(
            kw["gics_sector"], kw["start_year"], kw["end_year"]
        ),
    ):
        digest = sector_thematic_map_reduce(
            theme="climate risk",
            gics_sector="real estate",
            section_id="item_1a",
            start_year=2020,
            end_year=2025,
            tool_context=ctx,
        )
    assert digest["status"] == "success"
    assert digest["sector_coverage"]["gics_sector"] == "Real Estate"
    assert digest["citations"][0]["chunk_id"] == "PLD_2025_item_1a_0003"
    hydrated = hydrate_tool_outputs(ctx.state)
    assert hydrated.sector_coverage and hydrated.sector_coverage["issuers_analyzed"] == 29
    assert hydrated.disclosure_deltas and hydrated.disclosure_deltas["summary"]["added"] == 4
    assert [c["chunk_id"] for c in hydrated.citations] == ["PLD_2025_item_1a_0003"]


@pytest.mark.parametrize("explain", [True, False])
def test_flag_tool_attaches_optional_explanations_and_summary_to_state(explain: bool) -> None:
    from edgar_capstone.agents.material_variances import flag_material_variances

    ctx = SimpleNamespace(state={})
    res = flag_material_variances(
        tickers="INTC",
        start_year=2023,
        end_year=2025,
        explain_top_flags=explain,
        tool_context=ctx,
    )
    assert res["status"] == "success" and res["flags"]
    out = hydrate_tool_outputs(ctx.state)
    assert out.material_variance_flags == res["flags"]
    assert out.material_variance_summary == res["summary"]
    explained = [f for f in res["flags"] if f.get("explanation")]
    assert res["summary"]["explained_flags"] == len(res["explanation_citations"])
    if explain:
        assert explained and out.citations
    else:
        assert not explained and res["explanation_citations"] == []
