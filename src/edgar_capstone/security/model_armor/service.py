"""Unified Model Armor + Cloud SDP/DLP + SEC Financial Projection guardrail service."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Literal

import httpx

from edgar_capstone.observability.tracing import traced
from edgar_capstone.security.agent_identity import (
    GcpAgentIamCredentialsClient,
    authorize_agent_s2s_hop,
)
from edgar_capstone.security.auth import record_security_audit_event
from edgar_capstone.security.model_armor.client import ModelArmorClient
from edgar_capstone.security.model_armor.config import (
    GuardrailDecision,
    ModelArmorConfig,
    ModelArmorSanitizationSummary,
    _load_guardrail_policy_dict,
)
from edgar_capstone.security.model_armor.projection_filter import FinancialProjectionFilter
from edgar_capstone.security.rbac import CUSTOM_RUNTIME_SA_ACCOUNT_ID

logger = logging.getLogger(__name__)

_HARD_BLOCK_SDP_TYPES: frozenset[str] = frozenset({"GCP_API_KEY", "AUTH_TOKEN"})


def schema_safe_block_text(original_text: str, block_message: str) -> str:
    """Replacement text for a blocked model response that keeps structured outputs parseable.

    FinancialAnalystAgent answers with constrained `AnalystReport` JSON. Replacing a blocked
    answer with plain text would fail ADK's output-schema validation, crash the run, and lose
    the block decision, so structured answers get a minimal AnalystReport carrying the
    block message instead.
    """
    stripped = (original_text or "").strip()
    if stripped.startswith("{") and '"headline"' in stripped:
        return json.dumps({"headline": block_message, "key_metrics": [], "findings": []})
    return block_message


class ModelArmorGuardrailService:
    """Orchestrates Model Armor v1 API, Cloud SDP/DLP, and SEC Financial Projection Filters."""

    def __init__(
        self,
        client: ModelArmorClient | None = None,
        *,
        projection_filter: FinancialProjectionFilter | None = None,
    ) -> None:
        self.client = client or ModelArmorClient()
        self.projection_filter = projection_filter or FinancialProjectionFilter()

    @property
    def config(self) -> ModelArmorConfig:
        return self.client.config

    def _attest_guardrail_s2s(self, caller_agent: str) -> tuple[str, str]:
        eff_caller = (
            caller_agent
            if caller_agent
            in ("RootOrchestrator", "FinancialAnalystAgent", CUSTOM_RUNTIME_SA_ACCOUNT_ID)
            else CUSTOM_RUNTIME_SA_ACCOUNT_ID
        )
        claims = authorize_agent_s2s_hop(
            caller_agent=eff_caller,
            callee_agent="ModelArmorGuardrail",
        )
        return claims.service_account_email, claims.spiffe_id

    def _inspect(
        self,
        *,
        stage: Literal["prompt", "response"],
        text: str,
        caller_agent: str,
    ) -> GuardrailDecision:
        """Shared inspection pipeline across both prompt and model-response stages."""
        start = time.perf_counter()
        sa_email, spiffe_id = self._attest_guardrail_s2s(caller_agent)
        policy = _load_guardrail_policy_dict()

        is_prompt = stage == "prompt"
        block_msg_key = "blocked_prompt_message" if is_prompt else "blocked_response_message"
        fail_closed_cat = (
            "model_armor_unavailable_fail_closed"
            if is_prompt
            else "model_armor_response_unavailable_fail_closed"
        )
        fail_closed_event = (
            "model_armor_fail_closed" if is_prompt else "model_armor_response_fail_closed"
        )
        sdp_fail_cat = (
            "sdp_unavailable_fail_closed" if is_prompt else "sdp_response_unavailable_fail_closed"
        )
        sdp_fail_event = "sdp_fail_closed" if is_prompt else "sdp_response_fail_closed"
        rpc_label = "sanitizeUserPrompt" if is_prompt else "sanitizeModelResponse"

        def _elapsed_ms() -> float:
            return round((time.perf_counter() - start) * 1000.0, 2)

        def _blocked(
            *,
            categories: tuple[str, ...],
            block_message: str,
            backend_used: str,
            projection_blocked: bool = False,
            sdp_info_types: tuple[str, ...] = (),
        ) -> GuardrailDecision:
            return GuardrailDecision(
                action="blocked",
                blocked=True,
                sdp_redacted=False,
                projection_blocked=projection_blocked,
                original_text=text,
                sanitized_text="",
                matched_categories=categories,
                sdp_info_types=sdp_info_types,
                block_message=block_message,
                backend_used=backend_used,
                latency_ms=_elapsed_ms(),
                s2s_service_account=sa_email,
                s2s_spiffe_id=spiffe_id,
            )

        if self.config.mode == "disabled" or not text or not text.strip():
            return GuardrailDecision(
                action="allowed",
                blocked=False,
                sdp_redacted=False,
                projection_blocked=False,
                original_text=text,
                sanitized_text=text,
                backend_used="disabled" if self.config.mode == "disabled" else "noop_empty",
                latency_ms=_elapsed_ms(),
                s2s_service_account=sa_email,
                s2s_spiffe_id=spiffe_id,
            )

        # 1. Invoke Google Cloud Model Armor (`sanitizeUserPrompt` or `sanitizeModelResponse`)
        try:
            ma_summary = (
                self.client.sanitize_user_prompt(text)
                if is_prompt
                else self.client.sanitize_model_response(text)
            )
        except Exception as exc:
            logger.error("Model Armor %s failed: %s", rpc_label, exc)
            if self.config.fail_closed:
                record_security_audit_event(
                    event_type=fail_closed_event,
                    user=None,
                    inherit_request_principal=True,
                    resource=self.config.template_resource_name,
                    status_outcome="blocked_fail_closed",
                    detail=str(exc)[:200],
                )
                return _blocked(
                    categories=(fail_closed_cat,),
                    block_message=str(policy[block_msg_key]),
                    backend_used="gcp_model_armor_fail_closed",
                )
            ma_summary = ModelArmorSanitizationSummary(
                filter_match_state="NO_MATCH_FOUND",
                blocking_categories=(),
                sdp_matched=False,
                sdp_info_types=(),
                sdp_deidentified_text=None,
                backend_used="degraded_fail_open",
            )

        if ma_summary.blocking_categories:
            record_security_audit_event(
                event_type=f"model_armor_{stage}_blocked",
                user=None,
                inherit_request_principal=True,
                resource=self.config.template_resource_name,
                status_outcome="blocked",
                detail=f"categories={list(ma_summary.blocking_categories)}",
            )
            return _blocked(
                categories=ma_summary.blocking_categories,
                block_message=str(policy[block_msg_key]),
                backend_used=ma_summary.backend_used,
                sdp_info_types=ma_summary.sdp_info_types,
            )

        # 2. SEC Financial Projection & Confidential MNPI Compliance Filter
        if self.config.projection_filter_enabled:
            proj_res = self.projection_filter.evaluate(text, stage=stage)
            if proj_res.matched:
                cat = proj_res.category or "ungrounded_financial_projection"
                record_security_audit_event(
                    event_type=f"projection_filter_{stage}_blocked",
                    user=None,
                    inherit_request_principal=True,
                    resource=self.config.template_resource_name,
                    status_outcome="blocked",
                    detail=proj_res.reason,
                )
                return _blocked(
                    categories=(cat, *proj_res.matched_rules),
                    block_message=str(policy["blocked_projection_message"]),
                    backend_used=ma_summary.backend_used,
                    projection_blocked=True,
                )

        # 3. Sensitive Data Protection (Model Armor SDP + Cloud DLP v2 fallback)
        working_text = text
        sdp_types: set[str] = set(ma_summary.sdp_info_types)
        sdp_triggered = ma_summary.sdp_matched
        if ma_summary.sdp_deidentified_text is not None and ma_summary.sdp_matched:
            working_text = ma_summary.sdp_deidentified_text

        if self.config.enable_cloud_dlp_api:
            try:
                dlp_matched, dlp_types, dlp_text = self.client.inspect_and_deidentify_sdp(
                    working_text,
                    model_armor_sdp_executed=ma_summary.sdp_executed,
                )
                if dlp_matched:
                    sdp_triggered = True
                    sdp_types.update(dlp_types)
                    working_text = dlp_text
            except Exception as exc:
                logger.warning("Cloud SDP deidentify check failed on %s: %s", stage, exc)
                # Fail closed whenever live GCP was active (both strict gcp/live AND auto mode).
                if self.config.fail_closed and self.client.is_live_gcp_active:
                    record_security_audit_event(
                        event_type=sdp_fail_event,
                        user=None,
                        inherit_request_principal=True,
                        resource=self.config.template_resource_name,
                        status_outcome="blocked_fail_closed",
                        detail=str(exc)[:200],
                    )
                    return _blocked(
                        categories=(sdp_fail_cat,),
                        block_message=str(policy[block_msg_key]),
                        backend_used=ma_summary.backend_used,
                    )

        if sdp_triggered:
            sorted_types = tuple(sorted(sdp_types))
            # Block leaked credentials (`GCP_API_KEY`, `AUTH_TOKEN`) on BOTH prompt AND response.
            if self.config.sdp_action == "block" or bool(sdp_types & _HARD_BLOCK_SDP_TYPES):
                record_security_audit_event(
                    event_type=f"sdp_sensitive_{stage}_blocked",
                    user=None,
                    inherit_request_principal=True,
                    resource=self.config.template_resource_name,
                    status_outcome="blocked",
                    detail=f"info_types={list(sorted_types)}",
                )
                return _blocked(
                    categories=tuple(f"sdp:{t}" for t in sorted_types) or ("sdp",),
                    block_message=str(policy[block_msg_key]),
                    backend_used=ma_summary.backend_used,
                    sdp_info_types=sorted_types,
                )

            return GuardrailDecision(
                action="sanitized",
                blocked=False,
                sdp_redacted=True,
                projection_blocked=False,
                original_text=text,
                sanitized_text=working_text,
                matched_categories=tuple(f"sdp_deidentified:{t}" for t in sorted_types),
                sdp_info_types=sorted_types,
                backend_used=ma_summary.backend_used,
                latency_ms=_elapsed_ms(),
                s2s_service_account=sa_email,
                s2s_spiffe_id=spiffe_id,
            )

        return GuardrailDecision(
            action="allowed",
            blocked=False,
            sdp_redacted=False,
            projection_blocked=False,
            original_text=text,
            sanitized_text=text,
            backend_used=ma_summary.backend_used,
            latency_ms=_elapsed_ms(),
            s2s_service_account=sa_email,
            s2s_spiffe_id=spiffe_id,
        )

    @traced("guardrail.inspect_user_prompt")
    def inspect_user_prompt(
        self,
        text: str,
        *,
        caller_agent: str = CUSTOM_RUNTIME_SA_ACCOUNT_ID,
    ) -> GuardrailDecision:
        """Inspect and sanitize an incoming user prompt via Model Armor + SDP + Projection filters."""
        return self._inspect(stage="prompt", text=text, caller_agent=caller_agent)

    @traced("guardrail.inspect_model_response")
    def inspect_model_response(
        self,
        text: str,
        *,
        caller_agent: str = "FinancialAnalystAgent",
    ) -> GuardrailDecision:
        """Inspect and sanitize a synthesized model response via Model Armor + SDP + Projection filters."""
        return self._inspect(stage="response", text=text, caller_agent=caller_agent)

    def to_status_metadata(self) -> dict[str, Any]:
        return {
            "mode": self.config.mode,
            "live_gcp_active": self.client.is_live_gcp_active,
            "project_id": self.config.project_id,
            "location": self.config.location,
            "template_id": self.config.template_id,
            "template_resource_name": self.config.template_resource_name,
            "regional_api_endpoint": self.config.regional_api_endpoint,
            "sdp_action": self.config.sdp_action,
            "sdp_inspect_template_configured": bool(self.config.sdp_inspect_template_id),
            "sdp_deidentify_template_configured": bool(self.config.sdp_deidentify_template_id),
            "projection_filter_enabled": self.config.projection_filter_enabled,
            "fail_closed": self.config.fail_closed,
        }


# ---------------------------------------------------------------------------
# Process-Wide Guardrail Singleton & Test/Runtime Injection Helpers
# ---------------------------------------------------------------------------

_GUARDRAIL_LOCK = threading.Lock()
_GUARDRAIL_HOLDER: dict[str, ModelArmorGuardrailService | None] = {"instance": None}


def get_model_armor_guardrail() -> ModelArmorGuardrailService:
    """Return the process-wide `ModelArmorGuardrailService` instance."""
    with _GUARDRAIL_LOCK:
        svc = _GUARDRAIL_HOLDER["instance"]
        if svc is None:
            svc = ModelArmorGuardrailService()
            _GUARDRAIL_HOLDER["instance"] = svc
        return svc


def configure_model_armor_guardrail(
    service: ModelArmorGuardrailService | None = None,
    *,
    config: ModelArmorConfig | None = None,
    http_client: httpx.Client | None = None,
    credentials: Any | None = None,
    iam_credentials_client: GcpAgentIamCredentialsClient | None = None,
) -> ModelArmorGuardrailService:
    """Configure or inject a custom `ModelArmorGuardrailService` (e.g. for Cloud Run startup or tests)."""
    with _GUARDRAIL_LOCK:
        previous = _GUARDRAIL_HOLDER["instance"]
        if service is not None:
            new_svc = service
        else:
            client = ModelArmorClient(
                config=config,
                http_client=http_client,
                credentials=credentials,
                iam_credentials_client=iam_credentials_client,
            )
            new_svc = ModelArmorGuardrailService(client=client)
        _GUARDRAIL_HOLDER["instance"] = new_svc
    if previous is not None and previous is not new_svc:
        previous.client.close()
    return new_svc


def reset_model_armor_guardrail() -> None:
    """Reset the process-wide `ModelArmorGuardrailService` singleton and release its pooled clients."""
    with _GUARDRAIL_LOCK:
        previous = _GUARDRAIL_HOLDER["instance"]
        _GUARDRAIL_HOLDER["instance"] = None
    if previous is not None:
        previous.client.close()


def get_model_armor_status_metadata() -> dict[str, Any]:
    """Return non-secret status metadata describing the active Model Armor & SDP configuration."""
    return get_model_armor_guardrail().to_status_metadata()
