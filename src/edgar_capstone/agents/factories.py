"""Cycle-free ADK `Agent` factory helpers for the SEC EDGAR Natural Language Analyst."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.planners import BasePlanner, BuiltInPlanner
from google.genai import types as genai_types
from pydantic import BaseModel

from edgar_capstone.agents.callbacks import (
    after_model_fact_check_callback,
    after_model_telemetry_callback,
    before_agent_telemetry_callback,
    before_model_safety_and_latency_callback,
    before_model_telemetry_callback,
    before_tool_agent_identity_callback,
    retrieval_budget_before_tool_callback,
)
from edgar_capstone.agents.context_memory import (
    after_agent_memory_callback,
    low_thinking_config_for_model,
    resolve_model_for_task,
)
from edgar_capstone.agents.disclosure_deltas import DisclosureDeltaTool
from edgar_capstone.agents.material_variances import AnomalyFlagTool
from edgar_capstone.agents.prompt_store import make_adk_instruction_provider
from edgar_capstone.agents.schemas import (
    AnalystReport,
    ModelTaskCategory,
    NarrativeEvaluationReport,
)
from edgar_capstone.agents.tools import (
    HybridSearchTool,
    LookerQueryTool,
    MetricFetchTool,
    QueryCached10KTool,
    VarianceCalculationTool,
)

# The analyst follows the registry's `latest` prompt (v5: structured AnalystReport without `notes`,
# anomaly flags + disclosure deltas) rather than the store-wide default `v1`.
DEFAULT_ANALYST_PROMPT_VERSION = "latest"


def build_edgar_agent(
    *,
    name: str,
    description: str,
    prompt_id: str,
    task_category: ModelTaskCategory,
    model: str | BaseLlm | None = None,
    prompt_version: str | None = None,
    output_schema: type[BaseModel] | None = None,
    output_key: str | None = None,
    tools: Sequence[Any] = (),
    planner: BasePlanner | None = None,
    generate_content_config: genai_types.GenerateContentConfig | None = None,
    before_tool_callback: Any | None = None,
    include_fact_check_callback: bool = True,
) -> Agent:
    """Build an ADK `Agent` with dynamic model resolution, versioned prompt, and standard callbacks.

    When `include_fact_check_callback=False` (used by `NarrativeEvaluatorAgent`), the agent uses
    the lightweight telemetry-only `before_model_telemetry_callback` and
    `after_model_telemetry_callback`, recording exact TTFT/tokens/cost into
    `ContextLatencyTracker` without firing Model Armor HTTP checks, `parse_analyst_report()`,
    or `verify_narrative_against_calculations()` on judge JSON.
    """
    resolved_model: str | BaseLlm = (
        model if model is not None else resolve_model_for_task(task_category)
    )
    before_model_cb = (
        before_model_safety_and_latency_callback
        if include_fact_check_callback
        else before_model_telemetry_callback
    )
    after_model_cb = (
        after_model_fact_check_callback
        if include_fact_check_callback
        else after_model_telemetry_callback
    )
    kwargs: dict[str, Any] = {
        "name": name,
        "model": resolved_model,
        "description": description,
        "instruction": make_adk_instruction_provider(prompt_id, version=prompt_version),
        "before_agent_callback": before_agent_telemetry_callback,
        "after_agent_callback": after_agent_memory_callback,
        "before_model_callback": before_model_cb,
        "after_model_callback": after_model_cb,
    }
    if tools:
        kwargs["tools"] = list(tools)
    if planner is not None:
        kwargs["planner"] = planner
    if output_schema is not None:
        kwargs["output_schema"] = output_schema
    if output_key is not None:
        kwargs["output_key"] = output_key
    if generate_content_config is not None:
        kwargs["generate_content_config"] = generate_content_config
    if before_tool_callback is not None:
        kwargs["before_tool_callback"] = before_tool_callback
    return Agent(**kwargs)


def create_financial_analyst_agent(
    *,
    name: str = "FinancialAnalystAgent",
    model: str | BaseLlm | None = None,
    prompt_version: str | None = None,
) -> Agent:
    """Create the specialized `FinancialAnalystAgent` ADK `LlmAgent`.

    Uses default native thinking (`BuiltInPlanner(ThinkingConfig(include_thoughts=True))`) rather
    than forced low thinking so complex SEC 10-K reasoning quality is preserved, and resolves the
    model dynamically at call time via `resolve_model_for_task("financial_reasoning")` when
    `model` is omitted.
    """
    thinking_cfg = genai_types.ThinkingConfig(include_thoughts=True)
    return build_edgar_agent(
        name=name,
        model=model,
        task_category="financial_reasoning",
        prompt_id="financial_analyst",
        prompt_version=prompt_version or DEFAULT_ANALYST_PROMPT_VERSION,
        planner=BuiltInPlanner(thinking_config=thinking_cfg),
        description=(
            "Gemini SEC 10-K financial reasoning agent employing a structured ReAct "
            "loop to evaluate deterministic period-over-period variances, retrieve only the 10-K "
            "evidence each question needs (Item 1A Risk Factors & Item 7 MD&A passages), and "
            "format strictly cited reports."
        ),
        tools=[
            VarianceCalculationTool,
            AnomalyFlagTool,
            MetricFetchTool,
            LookerQueryTool,
            # Targeted retrieval only: ranked passages and hybrid search, each returning
            # chunk_ids. Whole-section loading (`load_cached_10k_section`) stays with the
            # source-viewer API, not the synthesis agent.
            HybridSearchTool,
            QueryCached10KTool,
            # Deterministic year-over-year wording comparison with both-year chunk_ids.
            DisclosureDeltaTool,
        ],
        generate_content_config=genai_types.GenerateContentConfig(
            temperature=0.1,
            # Includes thinking tokens on Gemini 3.x; a 4096 cap truncated full reports.
            max_output_tokens=16384,
        ),
        # Constrained JSON answer (`headline`, `key_metrics`, `findings`),
        # verified claim by claim in `structured_report.py` before rendering.
        output_schema=AnalystReport,
        output_key="financial_analyst_report",
        before_tool_callback=[
            before_tool_agent_identity_callback,
            retrieval_budget_before_tool_callback,
        ],
        include_fact_check_callback=True,
    )


def create_narrative_evaluator_agent(
    *,
    name: str = "NarrativeEvaluatorAgent",
    model: str | BaseLlm | None = None,
    prompt_version: str | None = None,
) -> Agent:
    """Create the `NarrativeEvaluatorAgent` ADK `LlmAgent` pinned to `gemini-3.5-flash`.

    Wired with telemetry-only model callbacks (`include_fact_check_callback=False`) so judge
    calls record exact TTFT, tokens, and USD cost in `ContextLatencyTracker` without firing
    redundant Model Armor HTTP checks or `AnalystReport` parsing on the judge's JSON output.
    """
    resolved_model: str | BaseLlm = (
        model if model is not None else resolve_model_for_task("evaluation")
    )
    return build_edgar_agent(
        name=name,
        model=resolved_model,
        task_category="evaluation",
        prompt_id="narrative_evaluator",
        prompt_version=prompt_version,
        description=(
            "Gemini 3.5 Flash LLM-as-a-Judge evaluation specialist scoring numeric fidelity, "
            "10-K citation grounding, variance explanation quality, and thematic coherence."
        ),
        output_schema=NarrativeEvaluationReport,
        output_key="narrative_evaluation_report",
        # Thinking tokens count against max_output_tokens; keep headroom for the JSON report.
        generate_content_config=genai_types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=8192,
            thinking_config=low_thinking_config_for_model(resolved_model),
        ),
        include_fact_check_callback=False,
    )


__all__ = [
    "DEFAULT_ANALYST_PROMPT_VERSION",
    "build_edgar_agent",
    "create_financial_analyst_agent",
    "create_narrative_evaluator_agent",
]
