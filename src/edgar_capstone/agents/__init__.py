"""Core Agent Development Kit (ADK) orchestration, specialist agents, ReAct loop, and context/memory tools.

Uses PEP 562 lazy module attribute loading so ``import edgar_capstone.agents`` does not eagerly
import ``google.adk`` / ``google.genai`` / ``edgar`` / ``pandas`` / ``google.cloud.storage``,
read prompt YAML files from disk, or instantiate ADK ``Agent`` objects at import time.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from edgar_capstone.agents.callbacks import (
        ModelTelemetryPlugin,
        after_model_fact_check_callback,
        after_model_telemetry_callback,
        before_agent_telemetry_callback,
        before_model_safety_and_latency_callback,
        before_model_telemetry_callback,
        before_tool_agent_identity_callback,
        retrieval_budget_before_tool_callback,
        verify_narrative_against_calculations,
    )
    from edgar_capstone.agents.chat_engine import (
        ACTIVE_AGENT_NAMES,
        PRIMARY_ORCHESTRATOR_NAME,
        EntityCatalogCache,
        list_curated_companies,
        resolve_query_scope,
        run_conversational_rag_turn,
    )
    from edgar_capstone.agents.context_memory import (
        FINANCIAL_REASONING_MODEL,
        FLASH_CODING_AND_EVAL_MODEL,
        TTFT_SLO_TARGET_MS,
        BoundedInMemoryMemoryService,
        ContextLatencyTracker,
        ConversationMemoryManager,
        SessionStoreProtocol,
        after_agent_memory_callback,
        attribute_context_latency,
        create_edgar_adk_app,
        extract_llm_request_context_breakdown,
        get_context_latency_tracker,
        get_conversation_memory_manager,
        get_model_strategy_metadata,
        get_ttft_slo_ms,
        is_ephemeral_session_id,
        low_thinking_config_for_model,
        reconcile_breakdown_with_usage_metadata,
        resolve_model_for_task,
        resolve_task_and_model_for_agent,
    )
    from edgar_capstone.agents.disclosure_deltas import (
        DisclosureDeltaTool,
        compare_disclosures_across_years,
    )
    from edgar_capstone.agents.evaluation import (
        DEFAULT_FLASH_MODEL,
        evaluate_variance_explanation,
        evaluate_variance_explanation_with_latency,
    )
    from edgar_capstone.agents.factories import (
        DEFAULT_ANALYST_PROMPT_VERSION,
        build_edgar_agent,
        create_financial_analyst_agent,
        create_narrative_evaluator_agent,
    )
    from edgar_capstone.agents.financial_analyst import DEFAULT_ANALYST_MODEL
    from edgar_capstone.agents.intent import (
        QueryIntentType,
        TurnIntentType,
        infer_intent_from_tool_calls,
        resolve_turn_intent,
    )
    from edgar_capstone.agents.live_runtime import (
        AdkStreamResult,
        LiveBackendType,
        LiveRuntimeResolution,
        build_adk_llm_for_task,
        get_live_runtime_status_metadata,
        resolve_live_adk_runtime,
        run_adk_agent_stream,
        run_adk_agent_stream_async,
        run_live_adk_financial_reasoning,
        run_live_adk_financial_reasoning_async,
        run_live_adk_narrative_evaluation,
        run_live_adk_narrative_evaluation_async,
    )
    from edgar_capstone.agents.material_variances import (
        AnomalyFlagTool,
        MaterialityThresholds,
        compute_material_variance_flags,
        flag_material_variances,
    )
    from edgar_capstone.agents.prompt_builders import (
        build_financial_reasoning_prompt,
        build_narrative_evaluation_prompt,
        select_citations_for_judge,
    )
    from edgar_capstone.agents.prompt_store import (
        DEFAULT_PROMPT_VERSION,
        GcsPromptStoreBackend,
        LocalPromptStoreBackend,
        PromptArtifact,
        PromptBackendMode,
        VersionedPromptStore,
        VertexPromptStoreBackend,
        configure_prompt_store,
        get_prompt_store,
        get_prompt_store_status_metadata,
        load_agent_prompt,
        load_agent_prompt_artifact,
        make_adk_instruction_provider,
        reset_prompt_store,
    )
    from edgar_capstone.agents.react_loop import (
        StructuredReActExecutor,
        evaluate_calculated_variances,
        flush_pending_evaluations,
        query_cached_10k_for_explanations,
    )
    from edgar_capstone.agents.resilience import (
        RateLimitExceededError,
        ToolExecutionError,
        UpstreamTimeoutError,
        retry_transient_errors,
        with_tool_resilience,
    )
    from edgar_capstone.agents.schemas import (
        AnalystFinding,
        AnalystMetricReference,
        AnalystReport,
        AnnualStepVariance,
        Cached10KExplanation,
        CompanyVarianceReport,
        ContextBreakdown,
        ContextLatencySummary,
        CoverageNotice,
        FactCheckVerification,
        GroundedCitation,
        MetricRecord,
        MetricVariance,
        ModelContextLatencyRecord,
        ModelTaskCategory,
        NarrativeEvaluationReport,
        NumericCitation,
        ReActPhaseType,
        ReActStep,
        VarianceEvaluationItem,
    )
    from edgar_capstone.agents.sector_map_reduce import run_sector_thematic_map_reduce
    from edgar_capstone.agents.sector_tool import (
        SectorThematicMapReduceTool,
        sector_thematic_map_reduce,
    )
    from edgar_capstone.agents.tool_runtime import (
        ToolOutputs,
        hydrate_tool_outputs,
        tool_turn_context,
    )
    from edgar_capstone.agents.tools import (
        BigQueryAdapter,
        ContextCacheTool,
        GcpAdapterHttpError,
        GcpEndpointNotReadyError,
        GcpToolAdapters,
        GcpToolConfig,
        GcsCorpusAdapter,
        HybridSearchTool,
        LookerQueryTool,
        MetricFetchTool,
        QueryCached10KTool,
        VarianceCalculationTool,
        VertexAISearchAdapter,
        calculate_period_variances,
        configure_gcp_tool_adapters,
        fetch_financial_metrics,
        get_active_gcp_tool_adapters,
        get_context_cache_runtime_stats,
        get_gcp_adapters_status_metadata,
        load_cached_10k_section,
        query_cached_10k_context,
        query_looker_financial_explore,
        reset_gcp_tool_adapters,
        search_10k_chunks_rag,
    )

_EXPORTS: dict[str, tuple[str, str]] = {
    # callbacks
    "ModelTelemetryPlugin": ("edgar_capstone.agents.callbacks", "ModelTelemetryPlugin"),
    "after_model_fact_check_callback": (
        "edgar_capstone.agents.callbacks",
        "after_model_fact_check_callback",
    ),
    "after_model_telemetry_callback": (
        "edgar_capstone.agents.callbacks",
        "after_model_telemetry_callback",
    ),
    "before_agent_telemetry_callback": (
        "edgar_capstone.agents.callbacks",
        "before_agent_telemetry_callback",
    ),
    "before_model_safety_and_latency_callback": (
        "edgar_capstone.agents.callbacks",
        "before_model_safety_and_latency_callback",
    ),
    "before_model_telemetry_callback": (
        "edgar_capstone.agents.callbacks",
        "before_model_telemetry_callback",
    ),
    "before_tool_agent_identity_callback": (
        "edgar_capstone.agents.callbacks",
        "before_tool_agent_identity_callback",
    ),
    "retrieval_budget_before_tool_callback": (
        "edgar_capstone.agents.callbacks",
        "retrieval_budget_before_tool_callback",
    ),
    "verify_narrative_against_calculations": (
        "edgar_capstone.agents.callbacks",
        "verify_narrative_against_calculations",
    ),
    # chat_engine
    "ACTIVE_AGENT_NAMES": ("edgar_capstone.agents.chat_engine", "ACTIVE_AGENT_NAMES"),
    "PRIMARY_ORCHESTRATOR_NAME": (
        "edgar_capstone.agents.chat_engine",
        "PRIMARY_ORCHESTRATOR_NAME",
    ),
    "EntityCatalogCache": ("edgar_capstone.agents.chat_engine", "EntityCatalogCache"),
    "list_curated_companies": ("edgar_capstone.agents.chat_engine", "list_curated_companies"),
    "resolve_query_scope": ("edgar_capstone.agents.chat_engine", "resolve_query_scope"),
    "run_conversational_rag_turn": (
        "edgar_capstone.agents.chat_engine",
        "run_conversational_rag_turn",
    ),
    # context_memory
    "FINANCIAL_REASONING_MODEL": (
        "edgar_capstone.agents.context_memory",
        "FINANCIAL_REASONING_MODEL",
    ),
    "FLASH_CODING_AND_EVAL_MODEL": (
        "edgar_capstone.agents.context_memory",
        "FLASH_CODING_AND_EVAL_MODEL",
    ),
    "TTFT_SLO_TARGET_MS": ("edgar_capstone.agents.context_memory", "TTFT_SLO_TARGET_MS"),
    "BoundedInMemoryMemoryService": (
        "edgar_capstone.agents.context_memory",
        "BoundedInMemoryMemoryService",
    ),
    "ContextLatencyTracker": ("edgar_capstone.agents.context_memory", "ContextLatencyTracker"),
    "ConversationMemoryManager": (
        "edgar_capstone.agents.context_memory",
        "ConversationMemoryManager",
    ),
    "SessionStoreProtocol": ("edgar_capstone.agents.context_memory", "SessionStoreProtocol"),
    "after_agent_memory_callback": (
        "edgar_capstone.agents.context_memory",
        "after_agent_memory_callback",
    ),
    "attribute_context_latency": (
        "edgar_capstone.agents.context_memory",
        "attribute_context_latency",
    ),
    "create_edgar_adk_app": ("edgar_capstone.agents.context_memory", "create_edgar_adk_app"),
    "extract_llm_request_context_breakdown": (
        "edgar_capstone.agents.context_memory",
        "extract_llm_request_context_breakdown",
    ),
    "get_context_latency_tracker": (
        "edgar_capstone.agents.context_memory",
        "get_context_latency_tracker",
    ),
    "get_conversation_memory_manager": (
        "edgar_capstone.agents.context_memory",
        "get_conversation_memory_manager",
    ),
    "get_model_strategy_metadata": (
        "edgar_capstone.agents.context_memory",
        "get_model_strategy_metadata",
    ),
    "get_ttft_slo_ms": ("edgar_capstone.agents.context_memory", "get_ttft_slo_ms"),
    "is_ephemeral_session_id": (
        "edgar_capstone.agents.context_memory",
        "is_ephemeral_session_id",
    ),
    "low_thinking_config_for_model": (
        "edgar_capstone.agents.context_memory",
        "low_thinking_config_for_model",
    ),
    "reconcile_breakdown_with_usage_metadata": (
        "edgar_capstone.agents.context_memory",
        "reconcile_breakdown_with_usage_metadata",
    ),
    "resolve_model_for_task": ("edgar_capstone.agents.context_memory", "resolve_model_for_task"),
    "resolve_task_and_model_for_agent": (
        "edgar_capstone.agents.context_memory",
        "resolve_task_and_model_for_agent",
    ),
    # disclosure_deltas
    "DisclosureDeltaTool": ("edgar_capstone.agents.disclosure_deltas", "DisclosureDeltaTool"),
    "compare_disclosures_across_years": (
        "edgar_capstone.agents.disclosure_deltas",
        "compare_disclosures_across_years",
    ),
    # evaluation
    "DEFAULT_FLASH_MODEL": ("edgar_capstone.agents.evaluation", "DEFAULT_FLASH_MODEL"),
    "evaluate_variance_explanation": (
        "edgar_capstone.agents.evaluation",
        "evaluate_variance_explanation",
    ),
    "evaluate_variance_explanation_with_latency": (
        "edgar_capstone.agents.evaluation",
        "evaluate_variance_explanation_with_latency",
    ),
    # factories
    "DEFAULT_ANALYST_PROMPT_VERSION": (
        "edgar_capstone.agents.factories",
        "DEFAULT_ANALYST_PROMPT_VERSION",
    ),
    "build_edgar_agent": ("edgar_capstone.agents.factories", "build_edgar_agent"),
    "create_financial_analyst_agent": (
        "edgar_capstone.agents.factories",
        "create_financial_analyst_agent",
    ),
    "create_narrative_evaluator_agent": (
        "edgar_capstone.agents.factories",
        "create_narrative_evaluator_agent",
    ),
    # financial_analyst
    "DEFAULT_ANALYST_MODEL": ("edgar_capstone.agents.financial_analyst", "DEFAULT_ANALYST_MODEL"),
    # intent
    "QueryIntentType": ("edgar_capstone.agents.intent", "QueryIntentType"),
    "TurnIntentType": ("edgar_capstone.agents.intent", "TurnIntentType"),
    "infer_intent_from_tool_calls": (
        "edgar_capstone.agents.intent",
        "infer_intent_from_tool_calls",
    ),
    "resolve_turn_intent": ("edgar_capstone.agents.intent", "resolve_turn_intent"),
    # live_runtime
    "AdkStreamResult": ("edgar_capstone.agents.live_runtime", "AdkStreamResult"),
    "LiveBackendType": ("edgar_capstone.agents.live_runtime", "LiveBackendType"),
    "LiveRuntimeResolution": ("edgar_capstone.agents.live_runtime", "LiveRuntimeResolution"),
    "build_adk_llm_for_task": ("edgar_capstone.agents.live_runtime", "build_adk_llm_for_task"),
    "get_live_runtime_status_metadata": (
        "edgar_capstone.agents.live_runtime",
        "get_live_runtime_status_metadata",
    ),
    "resolve_live_adk_runtime": (
        "edgar_capstone.agents.live_runtime",
        "resolve_live_adk_runtime",
    ),
    "run_adk_agent_stream": ("edgar_capstone.agents.live_runtime", "run_adk_agent_stream"),
    "run_adk_agent_stream_async": (
        "edgar_capstone.agents.live_runtime",
        "run_adk_agent_stream_async",
    ),
    "run_live_adk_financial_reasoning": (
        "edgar_capstone.agents.live_runtime",
        "run_live_adk_financial_reasoning",
    ),
    "run_live_adk_financial_reasoning_async": (
        "edgar_capstone.agents.live_runtime",
        "run_live_adk_financial_reasoning_async",
    ),
    "run_live_adk_narrative_evaluation": (
        "edgar_capstone.agents.live_runtime",
        "run_live_adk_narrative_evaluation",
    ),
    "run_live_adk_narrative_evaluation_async": (
        "edgar_capstone.agents.live_runtime",
        "run_live_adk_narrative_evaluation_async",
    ),
    # material_variances
    "AnomalyFlagTool": ("edgar_capstone.agents.material_variances", "AnomalyFlagTool"),
    "MaterialityThresholds": (
        "edgar_capstone.agents.material_variances",
        "MaterialityThresholds",
    ),
    "compute_material_variance_flags": (
        "edgar_capstone.agents.material_variances",
        "compute_material_variance_flags",
    ),
    "flag_material_variances": (
        "edgar_capstone.agents.material_variances",
        "flag_material_variances",
    ),
    # prompt_builders
    "build_financial_reasoning_prompt": (
        "edgar_capstone.agents.prompt_builders",
        "build_financial_reasoning_prompt",
    ),
    "build_narrative_evaluation_prompt": (
        "edgar_capstone.agents.prompt_builders",
        "build_narrative_evaluation_prompt",
    ),
    "select_citations_for_judge": (
        "edgar_capstone.agents.prompt_builders",
        "select_citations_for_judge",
    ),
    # prompt_store
    "DEFAULT_PROMPT_VERSION": ("edgar_capstone.agents.prompt_store", "DEFAULT_PROMPT_VERSION"),
    "GcsPromptStoreBackend": ("edgar_capstone.agents.prompt_store", "GcsPromptStoreBackend"),
    "LocalPromptStoreBackend": ("edgar_capstone.agents.prompt_store", "LocalPromptStoreBackend"),
    "PromptArtifact": ("edgar_capstone.agents.prompt_store", "PromptArtifact"),
    "PromptBackendMode": ("edgar_capstone.agents.prompt_store", "PromptBackendMode"),
    "VersionedPromptStore": ("edgar_capstone.agents.prompt_store", "VersionedPromptStore"),
    "VertexPromptStoreBackend": (
        "edgar_capstone.agents.prompt_store",
        "VertexPromptStoreBackend",
    ),
    "configure_prompt_store": ("edgar_capstone.agents.prompt_store", "configure_prompt_store"),
    "get_prompt_store": ("edgar_capstone.agents.prompt_store", "get_prompt_store"),
    "get_prompt_store_status_metadata": (
        "edgar_capstone.agents.prompt_store",
        "get_prompt_store_status_metadata",
    ),
    "load_agent_prompt": ("edgar_capstone.agents.prompt_store", "load_agent_prompt"),
    "load_agent_prompt_artifact": (
        "edgar_capstone.agents.prompt_store",
        "load_agent_prompt_artifact",
    ),
    "make_adk_instruction_provider": (
        "edgar_capstone.agents.prompt_store",
        "make_adk_instruction_provider",
    ),
    "reset_prompt_store": ("edgar_capstone.agents.prompt_store", "reset_prompt_store"),
    # react_loop
    "StructuredReActExecutor": ("edgar_capstone.agents.react_loop", "StructuredReActExecutor"),
    "evaluate_calculated_variances": (
        "edgar_capstone.agents.react_loop",
        "evaluate_calculated_variances",
    ),
    "flush_pending_evaluations": (
        "edgar_capstone.agents.react_loop",
        "flush_pending_evaluations",
    ),
    "query_cached_10k_for_explanations": (
        "edgar_capstone.agents.react_loop",
        "query_cached_10k_for_explanations",
    ),
    # resilience
    "RateLimitExceededError": ("edgar_capstone.agents.resilience", "RateLimitExceededError"),
    "ToolExecutionError": ("edgar_capstone.agents.resilience", "ToolExecutionError"),
    "UpstreamTimeoutError": ("edgar_capstone.agents.resilience", "UpstreamTimeoutError"),
    "retry_transient_errors": ("edgar_capstone.agents.resilience", "retry_transient_errors"),
    "with_tool_resilience": ("edgar_capstone.agents.resilience", "with_tool_resilience"),
    # schemas
    "AnalystFinding": ("edgar_capstone.agents.schemas", "AnalystFinding"),
    "AnalystMetricReference": ("edgar_capstone.agents.schemas", "AnalystMetricReference"),
    "AnalystReport": ("edgar_capstone.agents.schemas", "AnalystReport"),
    "AnnualStepVariance": ("edgar_capstone.agents.schemas", "AnnualStepVariance"),
    "Cached10KExplanation": ("edgar_capstone.agents.schemas", "Cached10KExplanation"),
    "CompanyVarianceReport": ("edgar_capstone.agents.schemas", "CompanyVarianceReport"),
    "ContextBreakdown": ("edgar_capstone.agents.schemas", "ContextBreakdown"),
    "ContextLatencySummary": ("edgar_capstone.agents.schemas", "ContextLatencySummary"),
    "CoverageNotice": ("edgar_capstone.agents.schemas", "CoverageNotice"),
    "FactCheckVerification": ("edgar_capstone.agents.schemas", "FactCheckVerification"),
    "GroundedCitation": ("edgar_capstone.agents.schemas", "GroundedCitation"),
    "MetricRecord": ("edgar_capstone.agents.schemas", "MetricRecord"),
    "MetricVariance": ("edgar_capstone.agents.schemas", "MetricVariance"),
    "ModelContextLatencyRecord": ("edgar_capstone.agents.schemas", "ModelContextLatencyRecord"),
    "ModelTaskCategory": ("edgar_capstone.agents.schemas", "ModelTaskCategory"),
    "NarrativeEvaluationReport": ("edgar_capstone.agents.schemas", "NarrativeEvaluationReport"),
    "NumericCitation": ("edgar_capstone.agents.schemas", "NumericCitation"),
    "ReActPhaseType": ("edgar_capstone.agents.schemas", "ReActPhaseType"),
    "ReActStep": ("edgar_capstone.agents.schemas", "ReActStep"),
    "VarianceEvaluationItem": ("edgar_capstone.agents.schemas", "VarianceEvaluationItem"),
    # sector_tool & tool_runtime
    "SectorThematicMapReduceTool": (
        "edgar_capstone.agents.sector_tool",
        "SectorThematicMapReduceTool",
    ),
    "sector_thematic_map_reduce": (
        "edgar_capstone.agents.sector_tool",
        "sector_thematic_map_reduce",
    ),
    "ToolOutputs": ("edgar_capstone.agents.tool_runtime", "ToolOutputs"),
    "hydrate_tool_outputs": ("edgar_capstone.agents.tool_runtime", "hydrate_tool_outputs"),
    "tool_turn_context": ("edgar_capstone.agents.tool_runtime", "tool_turn_context"),
    # sector_map_reduce
    "run_sector_thematic_map_reduce": (
        "edgar_capstone.agents.sector_map_reduce",
        "run_sector_thematic_map_reduce",
    ),
    # tools
    "BigQueryAdapter": ("edgar_capstone.agents.tools", "BigQueryAdapter"),
    "ContextCacheTool": ("edgar_capstone.agents.tools", "ContextCacheTool"),
    "GcpAdapterHttpError": ("edgar_capstone.agents.tools", "GcpAdapterHttpError"),
    "GcpEndpointNotReadyError": ("edgar_capstone.agents.tools", "GcpEndpointNotReadyError"),
    "GcpToolAdapters": ("edgar_capstone.agents.tools", "GcpToolAdapters"),
    "GcpToolConfig": ("edgar_capstone.agents.tools", "GcpToolConfig"),
    "GcsCorpusAdapter": ("edgar_capstone.agents.tools", "GcsCorpusAdapter"),
    "HybridSearchTool": ("edgar_capstone.agents.tools", "HybridSearchTool"),
    "LookerQueryTool": ("edgar_capstone.agents.tools", "LookerQueryTool"),
    "MetricFetchTool": ("edgar_capstone.agents.tools", "MetricFetchTool"),
    "QueryCached10KTool": ("edgar_capstone.agents.tools", "QueryCached10KTool"),
    "VarianceCalculationTool": ("edgar_capstone.agents.tools", "VarianceCalculationTool"),
    "VertexAISearchAdapter": ("edgar_capstone.agents.tools", "VertexAISearchAdapter"),
    "calculate_period_variances": (
        "edgar_capstone.agents.tools",
        "calculate_period_variances",
    ),
    "configure_gcp_tool_adapters": (
        "edgar_capstone.agents.tools",
        "configure_gcp_tool_adapters",
    ),
    "fetch_financial_metrics": ("edgar_capstone.agents.tools", "fetch_financial_metrics"),
    "get_active_gcp_tool_adapters": (
        "edgar_capstone.agents.tools",
        "get_active_gcp_tool_adapters",
    ),
    "get_context_cache_runtime_stats": (
        "edgar_capstone.agents.tools",
        "get_context_cache_runtime_stats",
    ),
    "get_gcp_adapters_status_metadata": (
        "edgar_capstone.agents.tools",
        "get_gcp_adapters_status_metadata",
    ),
    "load_cached_10k_section": ("edgar_capstone.agents.tools", "load_cached_10k_section"),
    "query_cached_10k_context": ("edgar_capstone.agents.tools", "query_cached_10k_context"),
    "query_looker_financial_explore": (
        "edgar_capstone.agents.tools",
        "query_looker_financial_explore",
    ),
    "reset_gcp_tool_adapters": ("edgar_capstone.agents.tools", "reset_gcp_tool_adapters"),
    "search_10k_chunks_rag": ("edgar_capstone.agents.tools", "search_10k_chunks_rag"),
}

__all__ = [
    "ACTIVE_AGENT_NAMES",
    "DEFAULT_ANALYST_MODEL",
    "DEFAULT_ANALYST_PROMPT_VERSION",
    "DEFAULT_FLASH_MODEL",
    "DEFAULT_PROMPT_VERSION",
    "FINANCIAL_REASONING_MODEL",
    "FLASH_CODING_AND_EVAL_MODEL",
    "PRIMARY_ORCHESTRATOR_NAME",
    "TTFT_SLO_TARGET_MS",
    "AdkStreamResult",
    "AnalystFinding",
    "AnalystMetricReference",
    "AnalystReport",
    "AnnualStepVariance",
    "AnomalyFlagTool",
    "BigQueryAdapter",
    "BoundedInMemoryMemoryService",
    "Cached10KExplanation",
    "CompanyVarianceReport",
    "ContextBreakdown",
    "ContextCacheTool",
    "ContextLatencySummary",
    "ContextLatencyTracker",
    "ConversationMemoryManager",
    "CoverageNotice",
    "DisclosureDeltaTool",
    "EntityCatalogCache",
    "FactCheckVerification",
    "GcpAdapterHttpError",
    "GcpEndpointNotReadyError",
    "GcpToolAdapters",
    "GcpToolConfig",
    "GcsCorpusAdapter",
    "GcsPromptStoreBackend",
    "GroundedCitation",
    "HybridSearchTool",
    "LiveBackendType",
    "LiveRuntimeResolution",
    "LocalPromptStoreBackend",
    "LookerQueryTool",
    "MaterialityThresholds",
    "MetricFetchTool",
    "MetricRecord",
    "MetricVariance",
    "ModelContextLatencyRecord",
    "ModelTaskCategory",
    "ModelTelemetryPlugin",
    "NarrativeEvaluationReport",
    "NumericCitation",
    "PromptArtifact",
    "PromptBackendMode",
    "QueryCached10KTool",
    "QueryIntentType",
    "RateLimitExceededError",
    "ReActPhaseType",
    "ReActStep",
    "SessionStoreProtocol",
    "StructuredReActExecutor",
    "ToolExecutionError",
    "TurnIntentType",
    "UpstreamTimeoutError",
    "VarianceCalculationTool",
    "VarianceEvaluationItem",
    "VersionedPromptStore",
    "VertexAISearchAdapter",
    "VertexPromptStoreBackend",
    "after_agent_memory_callback",
    "after_model_fact_check_callback",
    "after_model_telemetry_callback",
    "attribute_context_latency",
    "before_agent_telemetry_callback",
    "before_model_safety_and_latency_callback",
    "before_model_telemetry_callback",
    "before_tool_agent_identity_callback",
    "build_adk_llm_for_task",
    "build_edgar_agent",
    "build_financial_reasoning_prompt",
    "build_narrative_evaluation_prompt",
    "calculate_period_variances",
    "infer_intent_from_tool_calls",
    "compare_disclosures_across_years",
    "compute_material_variance_flags",
    "configure_gcp_tool_adapters",
    "configure_prompt_store",
    "create_edgar_adk_app",
    "create_financial_analyst_agent",
    "create_narrative_evaluator_agent",
    "evaluate_calculated_variances",
    "evaluate_variance_explanation",
    "evaluate_variance_explanation_with_latency",
    "extract_llm_request_context_breakdown",
    "fetch_financial_metrics",
    "flag_material_variances",
    "flush_pending_evaluations",
    "get_active_gcp_tool_adapters",
    "get_context_cache_runtime_stats",
    "get_context_latency_tracker",
    "get_conversation_memory_manager",
    "get_gcp_adapters_status_metadata",
    "get_live_runtime_status_metadata",
    "get_model_strategy_metadata",
    "get_prompt_store",
    "get_prompt_store_status_metadata",
    "get_ttft_slo_ms",
    "is_ephemeral_session_id",
    "list_curated_companies",
    "load_agent_prompt",
    "load_agent_prompt_artifact",
    "load_cached_10k_section",
    "low_thinking_config_for_model",
    "make_adk_instruction_provider",
    "query_cached_10k_context",
    "query_cached_10k_for_explanations",
    "query_looker_financial_explore",
    "reconcile_breakdown_with_usage_metadata",
    "reset_gcp_tool_adapters",
    "reset_prompt_store",
    "resolve_live_adk_runtime",
    "resolve_model_for_task",
    "resolve_query_scope",
    "resolve_task_and_model_for_agent",
    "retrieval_budget_before_tool_callback",
    "retry_transient_errors",
    "run_adk_agent_stream",
    "run_adk_agent_stream_async",
    "run_conversational_rag_turn",
    "run_live_adk_financial_reasoning",
    "run_live_adk_financial_reasoning_async",
    "run_live_adk_narrative_evaluation",
    "run_live_adk_narrative_evaluation_async",
    "resolve_turn_intent",
    "run_sector_thematic_map_reduce",
    "sector_thematic_map_reduce",
    "SectorThematicMapReduceTool",
    "ToolOutputs",
    "hydrate_tool_outputs",
    "tool_turn_context",
    "search_10k_chunks_rag",
    "select_citations_for_judge",
    "verify_narrative_against_calculations",
    "with_tool_resilience",
]


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
