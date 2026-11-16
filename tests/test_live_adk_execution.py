"""Automated tests for live Google ADK `Runner` SSE streaming execution & Gemini runtime wiring."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
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
from edgar_capstone.security import issue_local_dev_workspace_sso_token
from edgar_capstone.server import app

REPO_ROOT = Path(__file__).resolve().parents[1]


class StreamingProAnalystLlm(BaseLlm):
    """Simulated live `gemini-2.5-pro` streaming SSE chunks with measurable TTFT vs decode delay."""

    model: str = "gemini-2.5-pro"
    ttft_delay_s: float = 0.018
    decode_delay_s: float = 0.030
    override_markdown: str | None = None

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        assert stream is True, "ADK Runner must invoke Gemini with stream=True (StreamingMode.SSE)"

        # Extract deterministic variance % from prompt so synthesized Markdown satisfies 100% numeric lock
        prompt_blob = "\n".join(
            p.text
            for c in (llm_request.contents or [])
            for p in (c.parts or [])
            if getattr(p, "text", None)
        )
        if self.override_markdown is not None:
            final_md = self.override_markdown
        else:
            var_res = calculate_period_variances(tickers="NVDA", start_year=2024, end_year=2026)
            nvda_rep = var_res["variance_reports"][0]
            rev_pct = nvda_rep["revenue_variance"]["percentage_delta"]
            op_pct = nvda_rep["operating_income_variance"]["percentage_delta"]
            final_md = (
                f"### NVIDIA Corp. (NVDA) — Live ADK Grounded Variance Synthesis\n\n"
                f"- **Deterministic Variance Alignment:** Across FY2024–FY2026, NVIDIA revenue expanded "
                f"**+{rev_pct:.2f}%** while operating income increased **+{op_pct:.2f}%**, driven by "
                f"Blackwell and Hopper Data Center compute demand "
                f"[NVDA FY2026 | Item 7 MD&A | NVDA_2026_item_7_0000](https://www.sec.gov/Archives/edgar/data/1045810).\n"
                f"- **Risk Factor Context:** Export licensing controls and supply chain capacity commitments "
                f"remain key operational variables cited in Item 1A (`prompt_chars={len(prompt_blob)}`)."
            )

        await asyncio.sleep(self.ttft_delay_s)
        yield LlmResponse(
            model_version=self.model,
            partial=True,
            content=genai_types.Content(
                role="model",
                parts=[
                    genai_types.Part(
                        text="Evaluating deterministic NVDA variance payload and cached 10-K sections...",
                        thought=True,
                    )
                ],
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
                    genai_types.Part(
                        text="Evaluating deterministic NVDA variance payload and cached 10-K sections...",
                        thought=True,
                    ),
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


class StreamingFlashStructuredLlm(BaseLlm):
    """Simulated live `gemini-3.5-flash` streaming structured JSON for evaluation & coding agents."""

    model: str = "gemini-3.5-flash"
    mode: str = "evaluation"

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        assert stream is True
        if self.mode == "coding":
            payload = json.dumps(
                {
                    "assistant_model": self.model,
                    "task_category": "coding_assistance",
                    "language": "sql",
                    "artifact_title": "Live ADK BigQuery YoY Variance Audit (NVDA, AAPL)",
                    "code": (
                        "SELECT ticker, fiscal_year, revenue, ROUND(revenue_yoy_pct, 2) AS revenue_yoy_pct\n"
                        "FROM `sec_edgar.v_financial_variances_yoy`\n"
                        "WHERE ticker IN ('AAPL', 'NVDA') AND fiscal_year BETWEEN 2024 AND 2026\n"
                        "ORDER BY ticker, fiscal_year;"
                    ),
                    "explanation": "Queries authoritative BigQuery YoY variance view via live CodingAssistantAgent.",
                    "tickers_referenced": ["AAPL", "NVDA"],
                }
            )
        else:
            payload = json.dumps(
                {
                    "evaluator_model": self.model,
                    "task_category": "evaluation",
                    "grade": "pass",
                    "overall_score": 0.97,
                    "numeric_fidelity_score": 1.0,
                    "citation_grounding_score": 0.96,
                    "explanation_quality_score": 0.95,
                    "thematic_consistency_score": 0.95,
                    "rationale": "Live Gemini 3.5 Flash judge verified 100% numeric & citation fidelity.",
                }
            )

        await asyncio.sleep(0.010)
        yield LlmResponse(
            model_version=self.model,
            partial=True,
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part(text=payload[:32])],
            ),
        )
        await asyncio.sleep(0.012)
        yield LlmResponse(
            model_version=self.model,
            partial=False,
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part(text=payload)],
            ),
            usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
                prompt_token_count=920,
                cached_content_token_count=0,
                candidates_token_count=88,
                thoughts_token_count=0,
            ),
        )


class RootOrchestratorToolCallingLlm(BaseLlm):
    """Multi-turn streaming LLM driving `RootOrchestrator` tool call -> final grounded synthesis."""

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
    """Verify Vertex AI, Gemini API key, deterministic, and strict live mode resolution."""
    for env_key in (
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "EDGAR_EXECUTION_MODE",
        "EDGAR_GEMINI_LOCATION",
        "GOOGLE_CLOUD_LOCATION",
    ):
        monkeypatch.delenv(env_key, raising=False)

    # 1. Default offline auto mode -> deterministic_fallback
    offline_res = resolve_live_adk_runtime()
    assert offline_res.requested_mode == "auto"
    assert offline_res.live_adk_enabled is False
    assert offline_res.backend == "deterministic_fallback"

    # 2. Vertex AI mode -> configures ADK Gemini with vertexai=True, project, location, and HTTP 429 retries
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GCP_PROJECT_ID", "iron-radio-514217-q7")
    monkeypatch.setenv("GCP_REGION", "us-central1")
    vertex_res = resolve_live_adk_runtime()
    assert vertex_res.live_adk_enabled is True
    assert vertex_res.backend == "vertex_ai"
    assert vertex_res.project_id == "iron-radio-514217-q7"
    assert vertex_res.location == "us-central1"

    pro_gemini = build_adk_llm_for_task("financial_reasoning", resolution=vertex_res)
    flash_gemini = build_adk_llm_for_task("evaluation", resolution=vertex_res)
    assert isinstance(pro_gemini, Gemini)
    assert isinstance(flash_gemini, Gemini)
    assert pro_gemini.model == "gemini-3.8-flash"
    assert flash_gemini.model == "gemini-3.5-flash"
    assert pro_gemini.client_kwargs == {
        "vertexai": True,
        "project": "iron-radio-514217-q7",
        "location": "us-central1",
    }
    assert pro_gemini.retry_options is not None
    assert pro_gemini.retry_options.attempts == 3
    assert 429 in (pro_gemini.retry_options.http_status_codes or [])

    # Gemini can be pinned to the `global` endpoint while GCP_REGION stays regional
    monkeypatch.setenv("EDGAR_GEMINI_LOCATION", "global")
    global_res = resolve_live_adk_runtime()
    assert global_res.location == "global"
    global_gemini = build_adk_llm_for_task("financial_reasoning", resolution=global_res)
    assert global_gemini.client_kwargs["location"] == "global"
    monkeypatch.delenv("EDGAR_GEMINI_LOCATION")

    # 3. Gemini API key mode -> redacts secret from repr & public status metadata
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    synthetic_key = "test-ephemeral-" + "credential-value"
    monkeypatch.setenv("GEMINI_API_KEY", synthetic_key)
    api_res = resolve_live_adk_runtime()
    assert api_res.live_adk_enabled is True
    assert api_res.backend == "gemini_api"
    assert api_res.api_key_configured is True
    assert synthetic_key not in repr(api_res)
    pub_meta = get_live_runtime_status_metadata()
    assert pub_meta["gemini_api_key_configured"] is True
    assert synthetic_key not in json.dumps(pub_meta)

    # 4. Explicit deterministic override ignores active credentials
    det_override = resolve_live_adk_runtime(execution_mode="deterministic")
    assert det_override.live_adk_enabled is False
    assert det_override.backend == "deterministic_fallback"

    # 5. Strict live mode without credentials raises RuntimeError / HTTP 503
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="EDGAR_EXECUTION_MODE='live'"):
        resolve_live_adk_runtime(execution_mode="live")

    client = TestClient(app)
    token, _ = issue_local_dev_workspace_sso_token()
    resp = client.post(
        "/api/v1/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "Analyze NVDA revenue", "execution_mode": "live"},
    )
    assert resp.status_code == 503
    assert resp.json()["detail"]["error"] == "live_adk_unavailable"


def test_adk_runner_sse_streaming_captures_real_ttft_without_duplicate_partial_records() -> None:
    """Verify StreamingMode.SSE captures real first-chunk TTFT and records only 1 telemetry entry per call."""
    tracker = get_context_latency_tracker()
    tracker.reset()
    get_conversation_memory_manager().reset()

    pro_llm = StreamingProAnalystLlm(ttft_delay_s=0.020, decode_delay_s=0.045)
    agent = create_financial_analyst_agent(model=pro_llm)

    stream_res = run_adk_agent_stream(
        agent=agent,
        prompt_text="Synthesize NVDA FY2024-FY2026 revenue variance.",
        session_id="sse-ttft-test-1",
        use_adk_app=True,
        summarizer_llm=StreamingFlashStructuredLlm(),
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


def test_conversational_rag_turn_executes_live_adk_runner_pro_and_flash_end_to_end() -> None:
    """Verify full `/api/v1/chat` engine executes live ADK `gemini-2.5-pro` + `gemini-3.5-flash`."""
    tracker = get_context_latency_tracker()
    tracker.reset()
    get_conversation_memory_manager().reset()

    pro_llm = StreamingProAnalystLlm(ttft_delay_s=0.015, decode_delay_s=0.020)
    flash_llm = StreamingFlashStructuredLlm(mode="evaluation")

    turn = run_conversational_rag_turn(
        query="Analyze NVIDIA FY2024 to FY2026 revenue and operating income drivers",
        tickers="NVDA",
        start_year=2024,
        end_year=2026,
        session_id="live-chat-sess-01",
        user_id="analyst@fde-workspace.corp",
        reasoning_llm=pro_llm,
        judge_llm=flash_llm,
    )

    assert turn["status"] == "success"
    assert turn["execution_mode"] == "live_adk"
    assert turn["adk_runtime"]["executed_via_adk_runner"] is True
    assert turn["adk_runtime"]["streaming_mode"] == "sse"
    assert turn["adk_runtime"]["observed_ttft_ms"] is not None
    assert turn["adk_runtime"]["observed_ttft_ms"] > 0.0
    assert turn["adk_runtime"]["streamed_chunks_count"] == 2
    assert turn["adk_runtime"]["grounding_fallback_applied"] is False

    assert "Live ADK Grounded Variance Synthesis" in turn["answer_markdown"]
    assert turn["fact_check"]["passed"] is True
    assert turn["evaluation"]["evaluator_model"] == "gemini-3.5-flash"
    assert turn["evaluation"]["grade"] == "pass"
    assert turn["evaluation"]["overall_score"] >= 0.90

    # Both gemini-2.5-pro and gemini-3.5-flash calls recorded via ADK callbacks
    models_called = {r["model_name"] for r in turn["context_and_memory"]["model_call_records"]}
    assert models_called == {"gemini-2.5-pro", "gemini-3.5-flash"}

    # All S2S cryptographic agent attestations remain strictly enforced
    assert len(turn["agent_trace"]) >= 4
    for step in turn["agent_trace"]:
        assert step["s2s_verified"] is True
        assert step["agent_service_account"].endswith(".iam.gserviceaccount.com")
        assert step["spiffe_id"].startswith("spiffe://")


def test_live_adk_post_generation_grounding_lock_prevents_numeric_hallucination() -> None:
    """Verify post-generation fact-check rejects hallucinated live LLM numbers and falls back safely."""
    get_context_latency_tracker().reset()
    get_conversation_memory_manager().reset()

    hallucinating_llm = StreamingProAnalystLlm(
        ttft_delay_s=0.005,
        decode_delay_s=0.005,
        override_markdown=(
            "### Hallucinated Report\n"
            "NVIDIA revenue declined -99.99% in FY2026 without any SEC 10-K citation tags."
        ),
    )
    flash_llm = StreamingFlashStructuredLlm(mode="evaluation")

    turn = run_conversational_rag_turn(
        query="Analyze NVIDIA FY2024 to FY2026 revenue variance",
        tickers="NVDA",
        start_year=2024,
        end_year=2026,
        reasoning_llm=hallucinating_llm,
        judge_llm=flash_llm,
    )

    assert turn["status"] == "success"
    assert turn["execution_mode"] == "live_adk"
    assert turn["adk_runtime"]["grounding_fallback_applied"] is True
    assert "Hallucinated Report" not in turn["answer_markdown"]
    assert "-99.99%" not in turn["answer_markdown"]
    assert turn["fact_check"]["passed"] is True


def test_live_adk_evaluate_coding_assist_and_root_orchestrator_app() -> None:
    """Verify live ADK Runner execution for NarrativeEvaluatorAgent, CodingAssistantAgent, & RootOrchestrator."""
    tracker = get_context_latency_tracker()
    tracker.reset()
    get_conversation_memory_manager().reset()

    var_res = calculate_period_variances(tickers="NVDA", start_year=2024, end_year=2026)
    nvda_rep = var_res["variance_reports"][0]
    rev_pct = nvda_rep["revenue_variance"]["percentage_delta"]

    eval_report = evaluate_variance_explanation(
        query="Evaluate NVDA FY2024-FY2026 variance summary",
        response_markdown=(
            f"NVIDIA revenue rose +{rev_pct:.2f}% from FY2024 to FY2026 "
            f"[NVDA FY2026 | Item 7 MD&A | NVDA_2026_item_7_0000](https://www.sec.gov)."
        ),
        variance_reports=[nvda_rep],
        citations=[
            {
                "chunk_id": "NVDA_2026_item_7_0000",
                "ticker": "NVDA",
                "fiscal_year": 2026,
                "section_id": "item_7",
            }
        ],
        llm=StreamingFlashStructuredLlm(mode="evaluation"),
    )
    assert eval_report.evaluator_model == "gemini-3.5-flash"
    assert eval_report.grade == "pass"
    assert eval_report.context_tokens == 920
    assert eval_report.context_latency_ms > 0.0

    code_report = generate_financial_coding_assistance(
        prompt="Generate BigQuery YoY variance SQL for NVDA and AAPL",
        language="sql",
        tickers="NVDA,AAPL",
        start_year=2024,
        end_year=2026,
        llm=StreamingFlashStructuredLlm(mode="coding"),
    )
    assert code_report.assistant_model == "gemini-3.5-flash"
    assert "v_financial_variances_yoy" in code_report.code
    assert code_report.context_tokens == 920
    assert code_report.context_latency_ms > 0.0

    # Full RootOrchestrator ADK App multi-turn tool call + PlanReActPlanner tag splitting
    root_stream = run_live_root_orchestrator_turn(
        query="Compute NVIDIA period variances from 2024 to 2026",
        orchestrator_llm=RootOrchestratorToolCallingLlm(),
        flash_llm=StreamingFlashStructuredLlm(mode="evaluation"),
    )
    assert any(fc["name"] == "calculate_period_variances" for fc in root_stream.function_calls)
    assert "NVDA FY2024–FY2026 Supervisor Summary" in root_stream.final_text
    assert "/*FINAL_ANSWER*/" not in root_stream.final_text
    assert root_stream.observed_ttft_ms is not None

    # Verify Cloud Run Terraform & CI/CD pipelines wire Vertex AI / ADK env vars
    cicd_tf = (REPO_ROOT / "terraform" / "cicd_and_secrets.tf").read_text(encoding="utf-8")
    cloudbuild_yml = (REPO_ROOT / "cloudbuild.yaml").read_text(encoding="utf-8")
    gha_yml = (REPO_ROOT / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")
    for cfg_text in (cicd_tf, cloudbuild_yml, gha_yml):
        assert "GOOGLE_GENAI_USE_VERTEXAI" in cfg_text
        assert "GOOGLE_CLOUD_PROJECT" in cfg_text
        assert "GOOGLE_CLOUD_LOCATION" in cfg_text
        assert "EDGAR_EXECUTION_MODE" in cfg_text
