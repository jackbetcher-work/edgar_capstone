"""Google Cloud Managed Model Armor API, Sensitive Data Protection (SDP / Cloud DLP),
and SEC Financial Projection Compliance Guardrails.

Modular package replacing the monolithic ``model_armor.py`` while preserving all
public and internal exports for backward compatibility (zero regex imports).
"""

from __future__ import annotations

from edgar_capstone.security.model_armor.adk_plugin import ModelArmorSafetyPlugin
from edgar_capstone.security.model_armor.client import ModelArmorClient
from edgar_capstone.security.model_armor.config import (
    DEFAULT_MODEL_ARMOR_LOCATION,
    DEFAULT_MODEL_ARMOR_TEMPLATE_ID,
    DEFAULT_SDP_INFO_TYPES,
    GuardrailAction,
    GuardrailDecision,
    ModelArmorConfig,
    ModelArmorMode,
    ModelArmorSanitizationSummary,
    SdpActionMode,
    parse_model_armor_response,
)
from edgar_capstone.security.model_armor.config import (
    _load_guardrail_policy_dict as _load_guardrail_policy_dict,
)
from edgar_capstone.security.model_armor.config import (
    _normalize_location as _normalize_location,
)
from edgar_capstone.security.model_armor.config import (
    _normalize_words as _normalize_words,
)
from edgar_capstone.security.model_armor.emulator import LocalModelArmorEmulatorTransport
from edgar_capstone.security.model_armor.projection_filter import (
    FinancialProjectionFilter,
    ProjectionFilterResult,
    format_redaction_tag,
)
from edgar_capstone.security.model_armor.projection_filter import (
    _detect_sensitive_spans as _detect_sensitive_spans,
)
from edgar_capstone.security.model_armor.projection_filter import (
    _redact_spans as _redact_spans,
)
from edgar_capstone.security.model_armor.service import (
    ModelArmorGuardrailService,
    configure_model_armor_guardrail,
    get_model_armor_guardrail,
    get_model_armor_status_metadata,
    reset_model_armor_guardrail,
    schema_safe_block_text,
)

__all__ = [
    "DEFAULT_MODEL_ARMOR_LOCATION",
    "DEFAULT_MODEL_ARMOR_TEMPLATE_ID",
    "DEFAULT_SDP_INFO_TYPES",
    "FinancialProjectionFilter",
    "GuardrailAction",
    "GuardrailDecision",
    "LocalModelArmorEmulatorTransport",
    "ModelArmorClient",
    "ModelArmorConfig",
    "ModelArmorGuardrailService",
    "ModelArmorMode",
    "ModelArmorSafetyPlugin",
    "ModelArmorSanitizationSummary",
    "ProjectionFilterResult",
    "SdpActionMode",
    "configure_model_armor_guardrail",
    "format_redaction_tag",
    "get_model_armor_guardrail",
    "get_model_armor_status_metadata",
    "parse_model_armor_response",
    "reset_model_armor_guardrail",
    "schema_safe_block_text",
]
