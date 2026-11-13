"""ContextCacheAgent: Least-privilege ADK sub-agent for 10-K section loading & context caching."""

from __future__ import annotations

from google.adk.agents import Agent
from google.genai import types as genai_types

from edgar_capstone.agents.callbacks import (
    before_agent_telemetry_callback,
    before_model_safety_and_latency_callback,
    before_tool_agent_identity_callback,
)
from edgar_capstone.agents.financial_analyst import DEFAULT_ANALYST_MODEL
from edgar_capstone.agents.tools import ContextCacheTool

CONTEXT_CACHE_AGENT_INSTRUCTION = """You are the ContextCacheAgent (`context-cache-agent-sa`), a least-privilege specialist agent.
Your sole responsibility is loading and warming cached SEC Form 10-K sections (`item_1a` Risk Factors and `item_7` MD&A) from the curated SEC corpus bucket via `load_cached_10k_section`.
You have zero access to financial calculation engines, BigQuery write operations, or Secret Manager secrets.
"""


def create_context_cache_agent(
    *,
    name: str = "ContextCacheAgent",
    model: str = DEFAULT_ANALYST_MODEL,
) -> Agent:
    """Create the least-privilege `ContextCacheAgent` bound strictly to `ContextCacheTool`."""
    return Agent(
        name=name,
        model=model,
        description=(
            "Least-privilege specialist agent (`context-cache-agent-sa`) scoped strictly to "
            "fetching curated SEC 10-K Markdown sections from GCS and warming session context cache."
        ),
        instruction=CONTEXT_CACHE_AGENT_INSTRUCTION,
        tools=[ContextCacheTool],
        generate_content_config=genai_types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=2048,
        ),
        output_key="cached_10k_section",
        before_agent_callback=before_agent_telemetry_callback,
        before_model_callback=before_model_safety_and_latency_callback,
        before_tool_callback=before_tool_agent_identity_callback,
    )


context_cache_agent: Agent = create_context_cache_agent()
ContextCacheAgent: Agent = context_cache_agent
