"""Configuration, policy loading, response parsing, and decision models for Model Armor."""

from __future__ import annotations

import logging
import os
import string
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from edgar_capstone.common.env import env_bool
from edgar_capstone.gcp.defaults import (
    DEFAULT_PROJECT_ID,
    resolve_gcp_location,
    resolve_gcp_project_id,
)
from edgar_capstone.gcp.paths import resolve_asset_dir

logger = logging.getLogger(__name__)

ModelArmorMode = Literal["auto", "gcp", "live", "emulator", "disabled"]
SdpActionMode = Literal["deidentify", "block"]
GuardrailAction = Literal["allowed", "sanitized", "blocked"]

DEFAULT_MODEL_ARMOR_TEMPLATE_ID = "sec-edgar-analyst-armor"
DEFAULT_MODEL_ARMOR_LOCATION = "us-central1"

DEFAULT_SDP_INFO_TYPES: tuple[str, ...] = (
    "US_SOCIAL_SECURITY_NUMBER",
    "CREDIT_CARD_NUMBER",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "GCP_API_KEY",
    "AUTH_TOKEN",
)

_SAFE_EMAIL_DOMAIN_SUFFIXES: tuple[str, ...] = (
    "iam.gserviceaccount.com",
    "sec.gov",
    "googleapis.com",
)

_PUNCT_TRANSLATION_TABLE = str.maketrans({ch: " " for ch in string.punctuation})

# Minimal emergency fallback used ONLY when neither VersionedPromptStore nor
# `prompts/v1/model_armor_policy.yaml` can be read.
_MINIMAL_FALLBACK_POLICY: dict[str, Any] = {
    "blocked_prompt_message": (
        "Request blocked by SEC Analyst Model Armor & compliance guardrails: I can only "
        "answer financial and SEC 10-K analysis queries grounded in verified historical "
        "EDGAR filings and deterministic variance calculations."
    ),
    "blocked_projection_message": (
        "Request blocked by SEC Financial Projection & Grounding Compliance Filter: "
        "This workspace is restricted to verified historical SEC Form 10-K disclosures "
        "(through FY2026) and deterministic variance metrics, and cannot generate "
        "forward-looking financial projections, future earnings forecasts, stock price "
        "targets, or investment recommendations."
    ),
    "blocked_response_message": (
        "Response withheld by Model Armor & SEC Compliance Filter: The generated output "
        "triggered prompt-safety, sensitive-data (SDP), or ungrounded financial projection guardrails."
    ),
    "emulator_pi_token_clauses": [],
    "projection_phrases": [],
    "mnpi_markers": [],
    "future_fiscal_years": [],
    "projection_verbs_and_nouns": [],
}

_YAML_CACHE_LOCK = threading.Lock()
_YAML_POLICY_CACHE: dict[str, tuple[Path, int, dict[str, Any]] | None] = {"entry": None}


def _normalize_location(raw_location: str | None) -> str:
    loc = (raw_location or "").strip()
    if not loc or loc.lower() == "global":
        return DEFAULT_MODEL_ARMOR_LOCATION
    return loc


def _normalize_words(text: str) -> list[str]:
    """Tokenize text into normalized lowercase alphanumeric words without regexes."""
    return text.lower().translate(_PUNCT_TRANSLATION_TABLE).split()


def _load_policy_from_yaml_file() -> dict[str, Any] | None:
    policy_path = resolve_asset_dir("prompts") / "v1" / "model_armor_policy.yaml"
    try:
        mtime_ns = policy_path.stat().st_mtime_ns
    except OSError:
        return None

    with _YAML_CACHE_LOCK:
        cached = _YAML_POLICY_CACHE["entry"]
        if cached is not None and cached[0] == policy_path and cached[1] == mtime_ns:
            return cached[2]

    try:
        parsed = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Unable to parse %s: %s", policy_path, exc)
        return None

    policy = parsed.get("policy") if isinstance(parsed, dict) else None
    if not isinstance(policy, dict):
        return None

    with _YAML_CACHE_LOCK:
        _YAML_POLICY_CACHE["entry"] = (policy_path, mtime_ns, policy)
    return policy


def _load_guardrail_policy_dict() -> dict[str, Any]:
    """Load versioned Model Armor & projection filter policy from prompt store or YAML asset."""
    try:
        from edgar_capstone.agents.prompt_store import get_prompt_store

        artifact = get_prompt_store().load_prompt_artifact("model_armor_policy")
        policy = artifact.raw_metadata.get("policy")
        if isinstance(policy, dict):
            return policy
    except Exception as exc:
        logger.debug(
            "Prompt store unavailable for model_armor_policy (%s); trying YAML asset.", exc
        )

    yaml_policy = _load_policy_from_yaml_file()
    if yaml_policy is not None:
        return yaml_policy

    return _MINIMAL_FALLBACK_POLICY


@dataclass(frozen=True)
class ModelArmorConfig:
    """Configuration for Google Cloud Model Armor, Cloud SDP/DLP, and Projection Filters."""

    mode: ModelArmorMode = "auto"
    project_id: str = DEFAULT_PROJECT_ID
    location: str = DEFAULT_MODEL_ARMOR_LOCATION
    template_id: str = DEFAULT_MODEL_ARMOR_TEMPLATE_ID
    sdp_inspect_template_id: str | None = None
    sdp_deidentify_template_id: str | None = None
    sdp_action: SdpActionMode = "deidentify"
    enable_cloud_dlp_api: bool = True
    projection_filter_enabled: bool = True
    inspect_tool_outputs: bool = False
    fail_closed: bool = True
    timeout_seconds: float = 10.0

    @classmethod
    def from_env(cls) -> ModelArmorConfig:
        raw_mode = os.environ.get("EDGAR_MODEL_ARMOR_MODE", "auto").strip().lower()
        if raw_mode not in ("auto", "gcp", "live", "emulator", "disabled"):
            raise ValueError(
                f"Unsupported EDGAR_MODEL_ARMOR_MODE={raw_mode!r}; expected "
                "'auto', 'gcp', 'live', 'emulator', or 'disabled'."
            )
        project_id = resolve_gcp_project_id()
        explicit_loc = os.environ.get("MODEL_ARMOR_LOCATION", "").strip() or None
        location = _normalize_location(
            explicit_loc
            or resolve_gcp_location(allow_global=False, default=DEFAULT_MODEL_ARMOR_LOCATION)
        )
        raw_template = (
            os.environ.get("MODEL_ARMOR_TEMPLATE_ID", "").strip() or DEFAULT_MODEL_ARMOR_TEMPLATE_ID
        )
        # Allow passing full resource path `projects/{p}/locations/{l}/templates/{t}`
        if raw_template.startswith("projects/"):
            segs = raw_template.split("/")
            if (
                len(segs) >= 6
                and segs[0] == "projects"
                and segs[2] == "locations"
                and segs[4] == "templates"
            ):
                project_id = segs[1]
                location = _normalize_location(segs[3])
                raw_template = segs[5]

        raw_sdp_action = os.environ.get("MODEL_ARMOR_SDP_ACTION", "deidentify").strip().lower()
        sdp_action: SdpActionMode = "block" if raw_sdp_action == "block" else "deidentify"

        return cls(
            mode=raw_mode,  # type: ignore[arg-type]
            project_id=project_id,
            location=location,
            template_id=raw_template,
            sdp_inspect_template_id=os.environ.get("SDP_INSPECT_TEMPLATE_ID", "").strip() or None,
            sdp_deidentify_template_id=os.environ.get("SDP_DEIDENTIFY_TEMPLATE_ID", "").strip()
            or None,
            sdp_action=sdp_action,
            enable_cloud_dlp_api=env_bool("MODEL_ARMOR_ENABLE_DLP_API", True),
            projection_filter_enabled=env_bool("MODEL_ARMOR_PROJECTION_FILTER_ENABLED", True),
            inspect_tool_outputs=env_bool("MODEL_ARMOR_INSPECT_TOOL_OUTPUTS", False),
            fail_closed=env_bool("MODEL_ARMOR_FAIL_CLOSED", True),
        )

    @property
    def template_resource_name(self) -> str:
        return f"projects/{self.project_id}/locations/{self.location}/templates/{self.template_id}"

    @property
    def regional_api_endpoint(self) -> str:
        return f"https://modelarmor.{self.location}.rep.googleapis.com"


@dataclass(frozen=True)
class ModelArmorSanitizationSummary:
    """Parsed result from Model Armor `sanitizeUserPrompt` / `sanitizeModelResponse`."""

    filter_match_state: str
    blocking_categories: tuple[str, ...]
    sdp_matched: bool
    sdp_info_types: tuple[str, ...]
    sdp_deidentified_text: str | None
    invocation_result: str = "SUCCESS"
    backend_used: str = "gcp_model_armor_v1"
    raw_response: dict[str, Any] = field(default_factory=dict)
    sdp_executed: bool = False


@dataclass(frozen=True)
class GuardrailDecision:
    """Unified verdict from Model Armor, Cloud SDP/DLP, and SEC Projection Filters."""

    action: GuardrailAction
    blocked: bool
    sdp_redacted: bool
    projection_blocked: bool
    original_text: str
    sanitized_text: str
    matched_categories: tuple[str, ...] = ()
    sdp_info_types: tuple[str, ...] = ()
    block_message: str = ""
    backend_used: str = "gcp_model_armor_v1"
    latency_ms: float = 0.0
    s2s_service_account: str = ""
    s2s_spiffe_id: str = ""

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "blocked": self.blocked,
            "sdp_redacted": self.sdp_redacted,
            "projection_blocked": self.projection_blocked,
            "matched_categories": list(self.matched_categories),
            "sdp_info_types": list(self.sdp_info_types),
            "backend_used": self.backend_used,
            "latency_ms": round(self.latency_ms, 2),
            "s2s_service_account": self.s2s_service_account,
            "s2s_spiffe_id": self.s2s_spiffe_id,
        }

    @classmethod
    def from_audit_dict(
        cls,
        data: dict[str, Any],
        *,
        text: str = "",
        block_message: str | None = None,
    ) -> GuardrailDecision:
        """Reconstruct a `GuardrailDecision` from a serialized `to_audit_dict()` payload."""
        blocked = bool(data.get("blocked", False))
        sdp_redacted = bool(data.get("sdp_redacted", False))
        raw_action = data.get("action")
        if raw_action in ("allowed", "sanitized", "blocked"):
            action: GuardrailAction = raw_action
        else:
            action = "blocked" if blocked else ("sanitized" if sdp_redacted else "allowed")

        raw_cats = data.get("matched_categories") or ()
        matched_categories = (
            tuple(str(c) for c in raw_cats)
            if raw_cats
            else (("policy_violation",) if blocked else ())
        )
        resolved_block_msg = (
            str(block_message)
            if block_message is not None
            else str(data.get("block_message") or "")
        )
        return cls(
            action=action,
            blocked=blocked,
            sdp_redacted=sdp_redacted,
            projection_blocked=bool(data.get("projection_blocked", False)),
            original_text=text,
            sanitized_text="" if blocked else text,
            matched_categories=matched_categories,
            sdp_info_types=tuple(str(t) for t in (data.get("sdp_info_types") or ())),
            block_message=resolved_block_msg,
            backend_used=str(data.get("backend_used") or "gcp_model_armor_v1"),
            latency_ms=float(data.get("latency_ms") or 0.0),
            s2s_service_account=str(data.get("s2s_service_account") or ""),
            s2s_spiffe_id=str(data.get("s2s_spiffe_id") or ""),
        )


def _is_matched(val: Any) -> bool:
    if isinstance(val, int) and not isinstance(val, bool):
        return val == 2  # FilterMatchState.MATCH_FOUND proto enum int
    text = str(val or "").upper()
    return text.endswith("MATCH_FOUND") and not text.endswith("NO_MATCH_FOUND")


def _sdp_subresult_executed(sub: Any) -> bool:
    if not isinstance(sub, dict) or not sub:
        return False
    exec_state = sub.get("executionState") or sub.get("execution_state")
    if exec_state is not None:
        if isinstance(exec_state, int) and not isinstance(exec_state, bool):
            return exec_state == 1  # EXECUTION_SUCCESS proto enum int
        return str(exec_state).upper().endswith("EXECUTION_SUCCESS")
    return bool(sub.get("matchState") or sub.get("match_state"))


def _extract_filter_result(
    filter_results: dict[str, Any],
    key: str,
    camel_name: str,
    snake_name: str,
) -> dict[str, Any]:
    entry = filter_results.get(key)
    if not isinstance(entry, dict):
        return {}
    nested = entry.get(camel_name) or entry.get(snake_name) or entry
    return nested if isinstance(nested, dict) else {}


def _get_match_state(res: dict[str, Any]) -> Any:
    return res.get("matchState") if "matchState" in res else res.get("match_state")


def _parse_pi_filter(filter_results: dict[str, Any]) -> list[str]:
    pi_res = _extract_filter_result(
        filter_results,
        "pi_and_jailbreak",
        "piAndJailbreakFilterResult",
        "pi_and_jailbreak_filter_result",
    )
    if _is_matched(_get_match_state(pi_res)):
        conf = pi_res.get("confidenceLevel") or pi_res.get("confidence_level") or "HIGH"
        return [f"pi_and_jailbreak:{conf}"]
    return []


def _parse_rai_filter(filter_results: dict[str, Any]) -> list[str]:
    rai_res = _extract_filter_result(filter_results, "rai", "raiFilterResult", "rai_filter_result")
    if not _is_matched(_get_match_state(rai_res)):
        return []
    type_map = rai_res.get("raiFilterTypeResults") or rai_res.get("rai_filter_type_results") or {}
    if not isinstance(type_map, dict):
        return ["rai"]
    subtypes = sorted(
        str(name)
        for name, detail in type_map.items()
        if isinstance(detail, dict) and _is_matched(_get_match_state(detail))
    )
    return [f"rai:{sub}" for sub in subtypes] if subtypes else ["rai"]


_SIMPLE_FILTER_SPECS: tuple[tuple[str, str, str], ...] = (
    ("malicious_uris", "maliciousUriFilterResult", "malicious_uri_filter_result"),
    ("csam", "csamFilterFilterResult", "csam_filter_filter_result"),
)


def _parse_sdp_filter(
    filter_results: dict[str, Any],
) -> tuple[bool, bool, list[str], str | None]:
    sdp_res = _extract_filter_result(filter_results, "sdp", "sdpFilterResult", "sdp_filter_result")
    if not sdp_res:
        return False, False, [], None

    insp = sdp_res.get("inspectResult") or sdp_res.get("inspect_result") or {}
    deid = sdp_res.get("deidentifyResult") or sdp_res.get("deidentify_result") or {}
    sdp_executed = _sdp_subresult_executed(insp) or _sdp_subresult_executed(deid)

    sdp_matched = False
    info_types: list[str] = []
    deid_text: str | None = None

    if isinstance(insp, dict) and _is_matched(_get_match_state(insp)):
        sdp_matched = True
        for f_item in insp.get("findings") or []:
            if isinstance(f_item, dict):
                it_raw = f_item.get("infoType") or f_item.get("info_type")
                it_name = it_raw.get("name") if isinstance(it_raw, dict) else it_raw
                if it_name:
                    info_types.append(str(it_name))

    if isinstance(deid, dict) and _is_matched(_get_match_state(deid)):
        sdp_matched = True
        data_obj = deid.get("data") or {}
        if isinstance(data_obj, dict) and isinstance(data_obj.get("text"), str):
            deid_text = data_obj["text"]
        for it_name in deid.get("infoTypes") or deid.get("info_types") or []:
            if it_name:
                info_types.append(str(it_name))

    return sdp_executed, sdp_matched, info_types, deid_text


def parse_model_armor_response(
    payload: dict[str, Any],
    *,
    backend_used: str = "gcp_model_armor_v1",
) -> ModelArmorSanitizationSummary:
    """Parse Google Cloud Model Armor v1 JSON (`camelCase` or `snake_case`) into structured findings."""
    sr = payload.get("sanitizationResult") or payload.get("sanitization_result") or payload
    if not isinstance(sr, dict):
        sr = {}

    match_state = str(
        sr.get("filterMatchState") or sr.get("filter_match_state") or "NO_MATCH_FOUND"
    ).upper()
    inv_result = str(sr.get("invocationResult") or sr.get("invocation_result") or "SUCCESS").upper()
    filter_results = sr.get("filterResults") or sr.get("filter_results") or {}
    if not isinstance(filter_results, dict):
        filter_results = {}

    blocking: list[str] = []
    blocking.extend(_parse_pi_filter(filter_results))
    blocking.extend(_parse_rai_filter(filter_results))
    for key, camel_name, snake_name in _SIMPLE_FILTER_SPECS:
        res = _extract_filter_result(filter_results, key, camel_name, snake_name)
        if _is_matched(_get_match_state(res)):
            blocking.append(key)

    sdp_executed, sdp_matched, sdp_info_types, sdp_deid_text = _parse_sdp_filter(filter_results)

    if _is_matched(match_state) and not blocking and not sdp_matched:
        blocking.append("policy_violation")

    return ModelArmorSanitizationSummary(
        filter_match_state=match_state,
        blocking_categories=tuple(dict.fromkeys(blocking)),
        sdp_matched=sdp_matched,
        sdp_info_types=tuple(sorted(set(sdp_info_types))),
        sdp_deidentified_text=sdp_deid_text,
        invocation_result=inv_result,
        backend_used=backend_used,
        raw_response=payload,
        sdp_executed=sdp_executed,
    )
