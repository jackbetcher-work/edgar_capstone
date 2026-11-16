"""ContextCacheAgent: Least-privilege ADK sub-agent for 10-K section loading & context caching."""

from __future__ import annotations

from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.genai import types as genai_types

from edgar_capstone.agents.callbacks import (
    after_model_fact_check_callback,
    before_agent_telemetry_callback,
    before_model_safety_and_latency_callback,
    before_tool_agent_identity_callback,
)
from edgar_capstone.agents.context_memory import (
    FLASH_CODING_AND_EVAL_MODEL,
    after_agent_memory_callback,
)
from edgar_capstone.agents.prompt_store import (
    load_agent_prompt,
    make_adk_instruction_provider,
)
from edgar_capstone.agents.tools import ContextCacheTool

CONTEXT_CACHE_AGENT_INSTRUCTION: str = load_agent_prompt("context_cache_agent")


def create_context_cache_agent(
    *,
    name: str = "ContextCacheAgent",
    model: str | BaseLlm = FLASH_CODING_AND_EVAL_MODEL,
    prompt_version: str | None = None,
) -> Agent:
    """Create the least-privilege `ContextCacheAgent` bound strictly to `ContextCacheTool`."""
    return Agent(
        name=name,
        model=model,
        description=(
            "Least-privilege specialist agent (`context-cache-agent-sa`) scoped strictly to "
            "fetching curated SEC 10-K Markdown sections from GCS and warming session context cache."
        ),
        instruction=make_adk_instruction_provider("context_cache_agent", version=prompt_version),
        tools=[ContextCacheTool],
        generate_content_config=genai_types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=2048,
        ),
        output_key="cached_10k_section",
        before_agent_callback=before_agent_telemetry_callback,
        after_agent_callback=after_agent_memory_callback,
        before_model_callback=before_model_safety_and_latency_callback,
        after_model_callback=after_model_fact_check_callback,
        before_tool_callback=before_tool_agent_identity_callback,
    )


context_cache_agent: Agent = create_context_cache_agent()
ContextCacheAgent: Agent = context_cache_agent
