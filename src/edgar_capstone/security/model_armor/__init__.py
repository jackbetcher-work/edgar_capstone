"""Google Cloud Managed Model Armor API, Sensitive Data Protection (SDP / Cloud DLP),
and SEC Financial Projection Compliance Guardrails (zero regex imports).
"""

from __future__ import annotations

from edgar_capstone.security.model_armor.client import ModelArmorClient
from edgar_capstone.security.model_armor.config import (
    DEFAULT_MODEL_ARMOR_LOCATION,
    DEFAULT_MODEL_ARMOR_TEMPLATE_ID,
    GuardrailAction,
    GuardrailDecision,
    ModelArmorConfig,
    ModelArmorMode,
    ModelArmorSanitizationSummary,
    SdpActionMode,
    parse_model_armor_proto,
)
from edgar_capstone.security.model_armor.config import (
    _normalize_location as _normalize_location,
)
from edgar_capstone.security.model_armor.emulator import LocalModelArmorEmulator
from edgar_capstone.security.model_armor.projection_filter import (
    FinancialProjectionFilter,
    ProjectionFilterResult,
    format_redaction_tag,
)
from edgar_capstone.security.model_armor.service import (
    ModelArmorGuardrailService,
    configure_model_armor_guardrail,
    get_model_armor_guardrail,
    get_model_armor_status_metadata,
    reset_model_armor_guardrail,
)

__all__ = [
    "DEFAULT_MODEL_ARMOR_LOCATION",
    "DEFAULT_MODEL_ARMOR_TEMPLATE_ID",
    "FinancialProjectionFilter",
    "GuardrailAction",
    "GuardrailDecision",
    "LocalModelArmorEmulator",
    "ModelArmorClient",
    "ModelArmorConfig",
    "ModelArmorGuardrailService",
    "ModelArmorMode",
    "ModelArmorSanitizationSummary",
    "ProjectionFilterResult",
    "SdpActionMode",
    "configure_model_armor_guardrail",
    "format_redaction_tag",
    "get_model_armor_guardrail",
    "get_model_armor_status_metadata",
    "parse_model_armor_proto",
    "reset_model_armor_guardrail",
]
