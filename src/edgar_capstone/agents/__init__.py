"""Core Agent Development Kit (ADK) orchestration, specialist agents, ReAct loop, and context/memory tools.

Organized into cohesive subpackages:
- ``core``: Pydantic schemas, retry/telemetry resilience, and canonical 10-K citation helpers.
- ``adapters``: Live GCP service adapters (BigQuery, Vertex AI Search, GCS, registry).
- ``prompts``: Versioned prompt store (local/GCS/Vertex) and prompt compaction/builders.
- ``tools``: ADK tools, turn-scoped tool state, and deterministic financial/disclosure/sector engines.
- ``runtime``: ADK execution runtime, agent factories, lifecycle callbacks, and context/memory strategy.
- ``evaluation``: Claim-by-claim ``AnalystReport`` verification/rendering and Flash LLM-as-a-Judge.
- ``orchestration``: Conversational RAG turn pipeline, intent resolution, scope validation, and ReAct loop.

Intentionally empty of eager imports so ``import edgar_capstone.agents`` never loads heavy SDKs
(``google.adk``, ``google.genai``, ``edgar``, ``pandas``, ``google.cloud.storage``) on cold start.
Import symbols directly from their canonical subpackage modules.
"""
