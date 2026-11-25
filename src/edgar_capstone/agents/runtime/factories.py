"""Cycle-free ADK `Agent` factory helpers for the SEC EDGAR Natural Language Analyst."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.planners import BasePlanner, BuiltInPlanner
from google.genai import types as genai_types
from pydantic import BaseModel

from edgar_capstone.agents.core.schemas import (
    AnalystReport,
    ModelTaskCategory,
    NarrativeEvaluationReport,
)
from edgar_capstone.agents.prompts.store import make_adk_instruction_provider
from edgar_capstone.agents.runtime.callbacks import (
    after_model_telemetry_callback,
    agent_tool_progress_before_tool_callback,
    agent_tool_timing_after_tool_callback,
    before_agent_telemetry_callback,
    before_model_telemetry_callback,
    retrieval_budget_before_tool_callback,
)
from edgar_capstone.agents.runtime.context_memory import after_agent_memory_callback
from edgar_capstone.agents.runtime.model_config import (
    low_thinking_config_for_model,
    resolve_model_for_task,
)
from edgar_capstone.agents.tools.disclosure_deltas import DisclosureDeltaTool
from edgar_capstone.agents.tools.material_variances import AnomalyFlagTool
from edgar_capstone.agents.tools.sec_tools import (
    AnalyticsViewQueryTool,
    HybridSearchTool,
    QueryCached10KTool,
    VarianceCalculationTool,
)
from edgar_capstone.agents.tools.sector_tool import SectorThematicMapReduceTool

# The analyst follows the registry's `latest` prompt (`v7`: agent-driven deterministic tool
# selection, selective key metrics, citation-coverage rule, structured intent).
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
    after_tool_callback: Any | None = None,
) -> Agent:
    """Build an ADK `Agent` with dynamic model resolution, versioned prompt, and standard callbacks."""
    resolved_model: str | BaseLlm = (
        model if model is not None else resolve_model_for_task(task_category)
    )
    kwargs: dict[str, Any] = {
        "name": name,
        "model": resolved_model,
        "description": description,
        "instruction": make_adk_instruction_provider(prompt_id, version=prompt_version),
        "before_agent_callback": before_agent_telemetry_callback,
        "after_agent_callback": after_agent_memory_callback,
        "before_model_callback": before_model_telemetry_callback,
        "after_model_callback": after_model_telemetry_callback,
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
    if after_tool_callback is not None:
        kwargs["after_tool_callback"] = after_tool_callback
    return Agent(**kwargs)


def create_financial_analyst_agent(
    *,
    name: str = "FinancialAnalystAgent",
    model: str | BaseLlm | None = None,
    prompt_version: str | None = None,
    enable_tools: bool = True,
    low_thinking: bool = False,
) -> Agent:
    """Create the specialized `FinancialAnalystAgent` ADK `LlmAgent`.

    Uses default native thinking (`BuiltInPlanner(ThinkingConfig(include_thoughts=True))`) rather
    than forced low thinking so complex SEC 10-K reasoning quality is preserved, and resolves the
    model dynamically at call time via `resolve_model_for_task("financial_reasoning")` when
    `model` is omitted. Verification-guided repair retries pass `enable_tools=False,
    low_thinking=True` to execute a single fast JSON repair call over already-retrieved evidence.
    """
    resolved_model: str | BaseLlm = (
        model if model is not None else resolve_model_for_task("financial_reasoning")
    )
    thinking_cfg = (
        low_thinking_config_for_model(resolved_model, include_thoughts=True)
        if low_thinking
        else None
    ) or genai_types.ThinkingConfig(include_thoughts=True)
    tools: Sequence[Any] = (
        [
            VarianceCalculationTool,
            AnomalyFlagTool,
            AnalyticsViewQueryTool,
            # Targeted retrieval only: ranked passages and hybrid search, each returning
            # chunk_ids. Whole-section loading (`load_cached_10k_section`) stays with the
            # source-viewer API, not the synthesis agent.
            HybridSearchTool,
            QueryCached10KTool,
            # Deterministic year-over-year wording comparison with both-year chunk_ids.
            DisclosureDeltaTool,
            # Full-sector batched map-reduce for sector-wide thematic questions (SSE progress).
            SectorThematicMapReduceTool,
        ]
        if enable_tools
        else ()
    )
    return build_edgar_agent(
        name=name,
        model=resolved_model,
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
        # Deterministic engines are invoked by the agent ONLY when the question needs them
        # (nothing is pre-computed); each writes its full output to session_state, which the
        # chat pipeline hydrates before claim verification and rendering.
        tools=tools,
        generate_content_config=genai_types.GenerateContentConfig(
            temperature=0.1,
            # Includes thinking tokens on Gemini 3.x; a 4096 cap truncated full reports.
            max_output_tokens=16384,
        ),
        # Constrained JSON answer (`headline`, `key_metrics`, `findings`),
        # verified claim by claim in `structured_report.py` before rendering.
        output_schema=AnalystReport,
        output_key="financial_analyst_report",
        before_tool_callback=(
            [
                retrieval_budget_before_tool_callback,
                agent_tool_progress_before_tool_callback,
            ]
            if enable_tools
            else None
        ),
        after_tool_callback=agent_tool_timing_after_tool_callback if enable_tools else None,
    )


def create_narrative_evaluator_agent(
    *,
    name: str = "NarrativeEvaluatorAgent",
    model: str | BaseLlm | None = None,
    prompt_version: str | None = None,
) -> Agent:
    """Create the `NarrativeEvaluatorAgent` ADK `LlmAgent` (`gemini-3.8-flash`, `ThinkingLevel.LOW`)."""
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
            "Gemini 3.8 Flash (low-thinking) LLM-as-a-Judge evaluation specialist scoring numeric "
            "fidelity, 10-K citation grounding, variance explanation quality, and thematic coherence."
        ),
        output_schema=NarrativeEvaluationReport,
        output_key="narrative_evaluation_report",
        # Thinking tokens count against max_output_tokens; keep headroom for the JSON report.
        generate_content_config=genai_types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=8192,
            thinking_config=low_thinking_config_for_model(resolved_model),
        ),
    )


__all__ = [
    "DEFAULT_ANALYST_PROMPT_VERSION",
    "build_edgar_agent",
    "create_financial_analyst_agent",
    "create_narrative_evaluator_agent",
]
