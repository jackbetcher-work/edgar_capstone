"""Health probe router (`GET /health`)."""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Depends

from edgar_capstone.agents.orchestration.chat_engine import (
    ACTIVE_AGENT_NAMES,
    PRIMARY_ORCHESTRATOR_NAME,
)
from edgar_capstone.agents.prompts.store import get_prompt_store_status_metadata
from edgar_capstone.agents.runtime.context_memory import get_conversation_memory_manager
from edgar_capstone.agents.runtime.live_runtime import get_live_runtime_status_metadata
from edgar_capstone.agents.runtime.model_config import get_model_strategy_metadata
from edgar_capstone.agents.tools import get_gcp_adapters_status_metadata
from edgar_capstone.api.dependencies import get_runtime_sa_config
from edgar_capstone.gcp.defaults import resolve_gcp_project_id
from edgar_capstone.observability import (
    logging_status_metadata,
    tracing_status_metadata,
)
from edgar_capstone.security.agent_identity import PER_AGENT_IDENTITIES
from edgar_capstone.security.model_armor import get_model_armor_status_metadata
from edgar_capstone.security.rbac import RuntimeServiceAccountConfig
from edgar_capstone.telemetry import get_feedback_store, get_telemetry_sink

router = APIRouter(tags=["Health"])


@router.get("/health")
def health(
    runtime_sa: RuntimeServiceAccountConfig = Depends(get_runtime_sa_config),
) -> dict[str, Any]:
    """Cloud Run external health check and readiness probe endpoint."""
    model_meta = get_model_strategy_metadata()
    active_agents = list(ACTIVE_AGENT_NAMES)
    return {
        "status": "ok",
        "service": os.environ.get("K_SERVICE", "sec-edgar-analyst"),
        "project_id": resolve_gcp_project_id(),
        "runtime_service_account": runtime_sa.email,
        "runtime_service_account_project_name": runtime_sa.project_name_alias_email,
        "root_agent": PRIMARY_ORCHESTRATOR_NAME,
        "active_agents": active_agents,
        "sub_agents": active_agents,
        "reasoning_strategy": "structured_react_loop",
        "models": {
            "financial_reasoning": model_meta["financial_reasoning_model"],
            "coding_and_evaluation": model_meta["coding_and_evaluation_model"],
            "compaction_and_mapping": model_meta["compaction_and_mapping_model"],
        },
        "adk_runtime": get_live_runtime_status_metadata(),
        "model_armor": get_model_armor_status_metadata(),
        "prompt_store": get_prompt_store_status_metadata(),
        "telemetry_sink": get_telemetry_sink().status_metadata(),
        "feedback_store": get_feedback_store().status_metadata(),
        "session_memory": get_conversation_memory_manager().persistence_status(),
        "agent_identities": sorted(PER_AGENT_IDENTITIES.keys()),
        "iap_jwt_auth_enabled": True,
        "data_backend": get_gcp_adapters_status_metadata(),
        "observability": {
            "tracing": tracing_status_metadata(),
            "logging": logging_status_metadata(),
        },
    }
