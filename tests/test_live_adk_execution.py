"""Live Google ADK `Runner` SSE execution & Gemini runtime wiring.

Chat turns, the narrative judge, and the coding assistant call live Gemini on Vertex AI and read
live GCP data. Scripted ADK `BaseLlm`s are injected only for mechanics the live model can't be
made to exercise on demand: exact SSE timing (TTFT attribution), a hallucinated finding that
verification must reject, and a fixed RootOrchestrator tool-call / PlanReAct tag sequence.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from google.adk.models import Gemini
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types

from edgar_capstone.agents import (
    build_adk_llm_for_task,
    calculate_period_variances,
    create_financial_analyst_agent,
    evaluate_variance_explanation,
    generate_financial_coding_assistance,
    get_context_latency_tracker,
    get_conversation_memory_manager,
    get_live_runtime_status_metadata,
    resolve_live_adk_runtime,
    run_adk_agent_stream,
    run_live_root_orchestrator_turn,
)
from edgar_capstone.agents.chat_engine import run_conversational_rag_turn
from edgar_capstone.agents.tools import search_10k_chunks_rag
from tests.structured_report_helpers import evidence_leads, finding_from_citation, report_json

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_PROJECT_ID = "iron-radio-514217-q7"

_ADK_RUNTIME_KEYS = {
    "backend",
    "vertex_project_id",
    "vertex_location",
    "gemini_api_key_configured",
    "streaming_mode",
    "executed_via_adk_runner",
    "observed_ttft_ms",
    "streamed_chunks_count",
    "adk_events_count",
    "adk_tool_calls",
    "answer_verified",
    "answer_withheld_reason",
    "structured_report",
    "report_verification",
    "agent_retrieval",
}


def _is_model(name: str, family: str) -> bool:
    """Live responses may report a versioned model name (e.g. `gemini-3.5-flash-001`)."""
    return str(name).split("/")[-1].startswith(family)


class StreamingProAnalystLlm(BaseLlm):
    """Scripted SSE stream with a controlled prefill (TTFT) delay and decode delay."""

    model: str = "gemini-2.5-pro"
    ttft_delay_s: float = 0.018
    decode_delay_s: float = 0.030
    override_statement: str | None = None

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        assert stream is True, "ADK Runner must invoke Gemini with stream=True (StreamingMode.SSE)"

        # Evidence leads are live chunks the ReAct pre-pass retrieved for this turn.
        leads = evidence_leads(llm_request)
        if self.override_statement is not None:
            # Hallucinated statement: its figure appears in neither the quote nor the payload
            final_md = report_json(
                headline="Hallucinated Report",
                findings=[finding_from_citation(leads[0], self.override_statement)],
            )
        else:
            final_md = report_json(
                headline="Streaming timing probe",
                findings=[],
                notes="No claims; this response only exercises SSE timing.",
            )

        await asyncio.sleep(self.ttft_delay_s)
        yield LlmResponse(
            model_version=self.model,
            partial=True,
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part(text="Reviewing the variance payload...", thought=True)],
            ),
        )

        midpoint = len(final_md) // 2
        await asyncio.sleep(0.010)
        yield LlmResponse(
            model_version=self.model,
            partial=True,
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part(text=final_md[:midpoint])],
            ),
        )

        await asyncio.sleep(self.decode_delay_s)
        yield LlmResponse(
            model_version=self.model,
            partial=False,
            content=genai_types.Content(
                role="model",
                parts=[
                    genai_types.Part(text="Reviewing the variance payload...", thought=True),
                    genai_types.Part(text=final_md),
                ],
            ),
            usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
                prompt_token_count=3400,
                cached_content_token_count=2100,
                candidates_token_count=165,
                thoughts_token_count=38,
            ),
        )


class RootOrchestratorToolCallingLlm(BaseLlm):
    """Scripted `RootOrchestrator` turn: one tool call, then PlanReAct-tagged final text."""

    model: str = "gemini-2.5-pro"

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        assert stream is True
        has_tool_response = any(
            getattr(p, "function_response", None) is not None
            for c in (llm_request.contents or [])
            for p in (c.parts or [])
        )
        await asyncio.sleep(0.008)
        if not has_tool_response:
            yield LlmResponse(
                model_version=self.model,
                partial=False,
                content=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part(
                            text="/*PLANNING*/ 1. Call calculate_period_variances for NVDA.",
                            thought=True,
                        ),
                        genai_types.Part.from_function_call(
                            name="calculate_period_variances",
                            args={"tickers": "NVDA", "start_year": 2024, "end_year": 2026},
                        ),
                    ],
                ),
                usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=650,
                    candidates_token_count=35,
                    thoughts_token_count=18,
                ),
            )
            return

        yield LlmResponse(
            model_version=self.model,
            partial=True,
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part(text="/*FINAL_ANSWER*/ ### NVDA FY2024–FY2026 ")],
            ),
        )
        await asyncio.sleep(0.008)
        yield LlmResponse(
            model_version=self.model,
            partial=False,
            content=genai_types.Content(
                role="model",
                parts=[
                    genai_types.Part(
                        text=(
                            "/*REASONING*/ Deterministic variance calculation succeeded. "
                            "/*FINAL_ANSWER*/ ### NVDA FY2024–FY2026 Supervisor Summary\n"
                            "Verified deterministic variance metrics via `RootOrchestrator`."
                        )
                    )
                ],
            ),
            usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
                prompt_token_count=1450,
                candidates_token_count=54,
                thoughts_token_count=14,
            ),
        )


def test_live_runtime_resolution_and_gemini_client_wiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vertex AI by default (Gemini on `global`), Gemini API key mode, and custom LLM injection."""
    for env_key in (
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "EDGAR_GEMINI_LOCATION",
    ):
        monkeypatch.delenv(env_key, raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", TEST_PROJECT_ID)

    # 1. Default: Vertex AI with ADC, Gemini served from the `global` endpoint, HTTP 429 retries
    vertex_res = resolve_live_adk_runtime()
    assert vertex_res.backend == "vertex_ai"
    assert vertex_res.project_id == TEST_PROJECT_ID
    assert vertex_res.location == "global"
    assert vertex_res.api_key_configured is False

    pro_gemini = build_adk_llm_for_task("financial_reasoning", resolution=vertex_res)
    flash_gemini = build_adk_llm_for_task("evaluation", resolution=vertex_res)
    assert isinstance(pro_gemini, Gemini)
    assert isinstance(flash_gemini, Gemini)
    assert pro_gemini.model == "gemini-3.8-flash"
    assert flash_gemini.model == "gemini-3.5-flash"
    assert pro_gemini.client_kwargs == {
        "vertexai": True,
        "project": TEST_PROJECT_ID,
        "location": "global",
    }
    assert pro_gemini.retry_options is not None
    assert pro_gemini.retry_options.attempts == 3
    assert 429 in (pro_gemini.retry_options.http_status_codes or [])

    # Gemini can be pinned to a regional endpoint independently of BigQuery / GCS regions
    monkeypatch.setenv("EDGAR_GEMINI_LOCATION", "us-central1")
    regional_res = resolve_live_adk_runtime()
    assert regional_res.location == "us-central1"
    regional_gemini = build_adk_llm_for_task("financial_reasoning", resolution=regional_res)
    assert regional_gemini.client_kwargs["location"] == "us-central1"
    monkeypatch.delenv("EDGAR_GEMINI_LOCATION")

    # 2. Gemini API key mode -> secret redacted from repr & public status metadata
    synthetic_key = "test-ephemeral-" + "credential-value"
    monkeypatch.setenv("GEMINI_API_KEY", synthetic_key)
    api_res = resolve_live_adk_runtime()
    assert api_res.backend == "gemini_api"
    assert api_res.api_key_configured is True
    assert synthetic_key not in repr(api_res)
    api_gemini = build_adk_llm_for_task("financial_reasoning", resolution=api_res)
    assert isinstance(api_gemini, Gemini)
    assert api_gemini.client_kwargs == {"api_key": synthetic_key}
    pub_meta = get_live_runtime_status_metadata()
    assert pub_meta["backend"] == "gemini_api"
    assert pub_meta["gemini_api_key_configured"] is True
    assert synthetic_key not in json.dumps(pub_meta)

    # 3. GOOGLE_GENAI_USE_VERTEXAI wins over a configured key (and the key is not retained)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    both_res = resolve_live_adk_runtime()
    assert both_res.backend == "vertex_ai"
    assert both_res.api_key_configured is True
    assert both_res._api_key is None
    assert get_live_runtime_status_metadata()["backend"] == "vertex_ai"

    # 4. An injected ADK BaseLlm is used as-is
    custom = StreamingProAnalystLlm()
    custom_res = resolve_live_adk_runtime(custom_llm=custom)
    assert custom_res.backend == "custom_llm"
    assert build_adk_llm_for_task("financial_reasoning", custom_llm=custom) is custom


def test_adk_runner_sse_streaming_captures_real_ttft_without_duplicate_partial_records() -> None:
    """StreamingMode.SSE captures first-chunk TTFT and records only 1 telemetry entry per call."""
    tracker = get_context_latency_tracker()
    tracker.reset()
    get_conversation_memory_manager().reset()

    pro_llm = StreamingProAnalystLlm(ttft_delay_s=0.020, decode_delay_s=0.045)
    agent = create_financial_analyst_agent(model=pro_llm)

    stream_res = run_adk_agent_stream(
        agent=agent,
        prompt_text="Summarize the streaming timing probe.",
        session_id="sse-ttft-test-1",
        use_adk_app=True,
        summarizer_llm=pro_llm,
        backend="custom_llm",
    )

    assert stream_res.partial_chunks_count == 2
    assert stream_res.total_events_count >= 3
    assert stream_res.observed_ttft_ms is not None
    # TTFT must reflect the ~20ms prefill delay and be strictly less than full ~75ms stream duration
    assert 12.0 <= stream_res.observed_ttft_ms < (stream_res.total_stream_latency_ms - 25.0)
    assert len(stream_res.thoughts) >= 1

    # Exactly 1 latency record must be recorded (partial SSE chunks must NOT double-record!)
    all_records = tracker.recent_records(limit=10)
    assert len(all_records) == 1
    rec = all_records[0]
    assert rec.model_name == "gemini-2.5-pro"
    assert rec.context_breakdown.total_context_tokens == 3400
    assert rec.context_breakdown.cached_context_tokens == 2100
    assert rec.output_tokens == 165
    assert rec.thoughts_tokens == 38
    assert rec.context_prefill_latency_ms == pytest.approx(stream_res.observed_ttft_ms, abs=1.0)
    assert rec.generation_latency_ms >= 25.0


def test_conversational_rag_turn_executes_live_gemini_end_to_end() -> None:
    """Full chat engine on live Gemini (analyst + Flash judge) over live BigQuery / GCS data."""
    tracker = get_context_latency_tracker()
    tracker.reset()
    get_conversation_memory_manager().reset()

    turn = run_conversational_rag_turn(
        query="Analyze NVIDIA FY2024 to FY2026 revenue and operating income drivers",
        tickers="NVDA",
        start_year=2024,
        end_year=2026,
        session_id="live-chat-sess-01",
        user_id="analyst@fde-workspace.corp",
    )

    assert turn["status"] == "success"
    assert turn["execution_mode"] == "live_adk"
    runtime = turn["adk_runtime"]
    assert set(runtime) == _ADK_RUNTIME_KEYS
    assert runtime["backend"] == "vertex_ai"
    assert runtime["vertex_project_id"] == TEST_PROJECT_ID
    assert runtime["executed_via_adk_runner"] is True
    assert runtime["streaming_mode"] == "sse"
    assert runtime["observed_ttft_ms"] is not None and runtime["observed_ttft_ms"] > 0.0
    assert runtime["adk_events_count"] >= 1

    # The live answer passed claim-level verification and the numeric fact-check
    assert turn["answer_verified"] is True, runtime["answer_withheld_reason"]
    assert runtime["answer_verified"] is True
    assert runtime["answer_withheld_reason"] is None
    assert turn["fact_check"]["passed"] is True
    assert turn["structured_report"] is not None
    assert turn["report_verification"]["usable"] is True
    assert not turn["answer_markdown"].startswith("### Answer withheld")

    # Key metrics are the deterministic engine's figures for the same live BigQuery rows
    direct = calculate_period_variances(tickers="NVDA", start_year=2024, end_year=2026)
    nvda = direct["variance_reports"][0]
    turn_nvda = next(r for r in turn["variance_reports"] if r["ticker"] == "NVDA")
    assert turn_nvda["revenue_variance"] == nvda["revenue_variance"]
    assert turn_nvda["operating_income_variance"] == nvda["operating_income_variance"]
    assert "#### Key Metrics (deterministic engine)" in turn["answer_markdown"]

    evaluation = turn["evaluation"]
    assert _is_model(evaluation["evaluator_model"], "gemini-3.5-flash")
    assert evaluation["grade"] in {"pass", "needs_revision", "fail"}
    assert 0.0 <= evaluation["overall_score"] <= 1.0

    # Both the analyst (gemini-3.8-flash) and the judge (gemini-3.5-flash) calls were recorded
    models_called = [r["model_name"] for r in turn["context_and_memory"]["model_call_records"]]
    assert any(_is_model(m, "gemini-3.8-flash") for m in models_called), models_called
    assert any(_is_model(m, "gemini-3.5-flash") for m in models_called), models_called

    # All S2S cryptographic agent attestations remain strictly enforced
    assert len(turn["agent_trace"]) >= 4
    for step in turn["agent_trace"]:
        assert step["s2s_verified"] is True
        assert step["agent_service_account"].endswith(".iam.gserviceaccount.com")
        assert step["spiffe_id"].startswith("spiffe://")


def test_unverifiable_live_answer_is_withheld_not_replaced() -> None:
    """A finding whose figure isn't in its quote is rejected and the narrative is withheld."""
    get_context_latency_tracker().reset()
    get_conversation_memory_manager().reset()

    hallucinating_llm = StreamingProAnalystLlm(
        ttft_delay_s=0.005,
        decode_delay_s=0.005,
        override_statement="NVIDIA revenue declined -99.99% in FY2026.",
    )

    turn = run_conversational_rag_turn(
        query="Analyze NVIDIA FY2024 to FY2026 revenue variance",
        tickers="NVDA",
        start_year=2024,
        end_year=2026,
        reasoning_llm=hallucinating_llm,
    )

    assert turn["status"] == "success"
    assert turn["execution_mode"] == "live_adk"
    assert turn["answer_verified"] is False
    assert turn["adk_runtime"]["answer_verified"] is False
    assert "could be verified" in str(turn["adk_runtime"]["answer_withheld_reason"])
    assert turn["answer_markdown"].startswith("### Answer withheld")
    assert "Hallucinated Report" not in turn["answer_markdown"]
    assert "-99.99%" not in turn["answer_markdown"]

    verification = turn["report_verification"]
    assert verification["usable"] is False
    assert (verification["findings_total"], verification["findings_verified"]) == (1, 0)
    reasons = " ".join(verification["findings"][0]["reasons"])
    assert "figures not in the quote" in reasons and "99.99%" in reasons
    assert "too short" not in reasons and "not found" not in reasons  # only the figure fails
    # Deterministic outputs for the turn are still returned
    assert turn["variance_reports"] and turn["citations"]


def test_live_narrative_evaluator_and_coding_assistant() -> None:
    """Live `gemini-3.5-flash` judge (capped by the deterministic fact-check) and coding agent."""
    get_context_latency_tracker().reset()

    nvda_rep = calculate_period_variances(tickers="NVDA", start_year=2024, end_year=2026)[
        "variance_reports"
    ][0]
    rev_pct = nvda_rep["revenue_variance"]["percentage_delta"]
    search = search_10k_chunks_rag(
        query="data center revenue growth drivers",
        tickers="NVDA",
        gics_sector="ALL",
        section_id="item_7",
        start_year=2026,
        end_year=2026,
        top_k=1,
    )
    cit = search["citations"][0]
    tag = (
        f"[{cit['ticker']} FY{cit['fiscal_year']} | {cit['section_title']} | {cit['chunk_id']}]"
        f"({cit['sec_filing_url']})"
    )

    def _evaluate(pct: float):
        return evaluate_variance_explanation(
            query="Evaluate NVDA FY2024-FY2026 revenue variance summary",
            response_markdown=f"NVIDIA revenue changed {pct:+.2f}% from FY2024 to FY2026 {tag}.",
            variance_reports=[nvda_rep],
            citations=[cit],
        )

    grounded = _evaluate(rev_pct)
    assert _is_model(grounded.evaluator_model, "gemini-3.5-flash")
    assert grounded.grade in {"pass", "needs_revision", "fail"}
    assert grounded.context_tokens > 0
    assert grounded.context_latency_ms > 0.0
    for score in (
        grounded.overall_score,
        grounded.numeric_fidelity_score,
        grounded.citation_grounding_score,
    ):
        assert 0.0 <= score <= 1.0

    # A figure that contradicts the deterministic engine caps numeric fidelity, whatever the judge says
    wrong = _evaluate(rev_pct + 37.0)
    assert wrong.numeric_fidelity_score <= 0.45
    assert wrong.grade != "pass"

    code_report = generate_financial_coding_assistance(
        prompt="Generate BigQuery YoY revenue variance SQL for NVDA and AAPL",
        language="sql",
        tickers="NVDA,AAPL",
        start_year=2024,
        end_year=2026,
    )
    assert _is_model(code_report.assistant_model, "gemini-3.5-flash")
    assert code_report.language == "sql"
    assert "SELECT" in code_report.code.upper()
    assert "NVDA" in code_report.code
    assert code_report.context_tokens > 0
    assert code_report.context_latency_ms > 0.0


def test_root_orchestrator_app_tool_call_and_planner_tag_splitting() -> None:
    """Scripted RootOrchestrator turn calls the live variance tool and strips PlanReAct tags."""
    get_context_latency_tracker().reset()
    root_stream = run_live_root_orchestrator_turn(
        query="Compute NVIDIA period variances from 2024 to 2026",
        orchestrator_llm=RootOrchestratorToolCallingLlm(),
    )
    assert any(fc["name"] == "calculate_period_variances" for fc in root_stream.function_calls)
    assert "NVDA FY2024–FY2026 Supervisor Summary" in root_stream.final_text
    assert "/*FINAL_ANSWER*/" not in root_stream.final_text
    assert root_stream.observed_ttft_ms is not None


def test_deploy_configs_wire_vertex_ai_env() -> None:
    """Cloud Run Terraform & CI/CD pipelines wire the Vertex AI / ADK env vars."""
    cicd_tf = (REPO_ROOT / "terraform" / "cicd_and_secrets.tf").read_text(encoding="utf-8")
    cloudbuild_yml = (REPO_ROOT / "cloudbuild.yaml").read_text(encoding="utf-8")
    gha_yml = (REPO_ROOT / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")
    for cfg_text in (cicd_tf, cloudbuild_yml, gha_yml):
        assert "GOOGLE_GENAI_USE_VERTEXAI" in cfg_text
        assert "GOOGLE_CLOUD_PROJECT" in cfg_text
        assert "GOOGLE_CLOUD_LOCATION" in cfg_text
