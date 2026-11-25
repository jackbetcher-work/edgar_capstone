"""Configuration, policy loading, response parsing, and decision models for Model Armor."""

from __future__ import annotations

import logging
import os
import string
import threading
from dataclasses import dataclass
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

ModelArmorMode = Literal["auto", "gcp", "emulator", "disabled"]
SdpActionMode = Literal["deidentify", "block"]
GuardrailAction = Literal["allowed", "sanitized", "blocked"]

DEFAULT_MODEL_ARMOR_TEMPLATE_ID = "sec-edgar-analyst-armor"
DEFAULT_MODEL_ARMOR_LOCATION = "us-central1"

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
    "future_prompt_cues": [],
    "relative_future_phrases": [],
    "financial_metric_terms": [],
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
        from edgar_capstone.agents.prompts.store import get_prompt_store

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
    sdp_action: SdpActionMode = "deidentify"
    projection_filter_enabled: bool = True
    fail_closed: bool = True
    timeout_seconds: float = 10.0

    @classmethod
    def from_env(cls) -> ModelArmorConfig:
        raw_mode = os.environ.get("EDGAR_MODEL_ARMOR_MODE", "auto").strip().lower()
        if raw_mode not in ("auto", "gcp", "emulator", "disabled"):
            raise ValueError(
                f"Unsupported EDGAR_MODEL_ARMOR_MODE={raw_mode!r}; expected "
                "'auto', 'gcp', 'emulator', or 'disabled'."
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
            sdp_action=sdp_action,
            projection_filter_enabled=env_bool("MODEL_ARMOR_PROJECTION_FILTER_ENABLED", True),
            fail_closed=env_bool("MODEL_ARMOR_FAIL_CLOSED", True),
        )

    @property
    def template_resource_name(self) -> str:
        return f"projects/{self.project_id}/locations/{self.location}/templates/{self.template_id}"

    @property
    def regional_api_host(self) -> str:
        return f"modelarmor.{self.location}.rep.googleapis.com"

    @property
    def regional_api_endpoint(self) -> str:
        return f"https://{self.regional_api_host}"


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
        }


def _enum_name(val: Any, default: str) -> str:
    if val is None:
        return default
    name = getattr(val, "name", None)
    if isinstance(name, str) and name:
        return name.upper()
    return str(val).upper() or default


def _is_match_found(state: Any) -> bool:
    if isinstance(state, int) and not isinstance(state, bool):
        return state == 2  # FilterMatchState.MATCH_FOUND
    name = _enum_name(state, "")
    return name.endswith("MATCH_FOUND") and not name.endswith("NO_MATCH_FOUND")


def parse_model_armor_proto(
    response: Any,
    *,
    backend_used: str = "gcp_model_armor_v1",
) -> ModelArmorSanitizationSummary:
    """Parse a typed ``google.cloud.modelarmor_v1`` sanitize response into structured findings."""
    sr = getattr(response, "sanitization_result", None)
    if sr is None:
        return ModelArmorSanitizationSummary(
            filter_match_state="FILTER_MATCH_STATE_UNSPECIFIED",
            blocking_categories=(),
            sdp_matched=False,
            sdp_info_types=(),
            sdp_deidentified_text=None,
            invocation_result="FAILURE",
            backend_used=backend_used,
        )

    match_state = _enum_name(
        getattr(sr, "filter_match_state", None), "FILTER_MATCH_STATE_UNSPECIFIED"
    )
    inv_result = _enum_name(getattr(sr, "invocation_result", None), "INVOCATION_RESULT_UNSPECIFIED")
    filter_results = getattr(sr, "filter_results", None) or {}

    blocking: list[str] = []

    pi_entry = filter_results.get("pi_and_jailbreak")
    if pi_entry is not None and "pi_and_jailbreak_filter_result" in pi_entry:
        pi_res = pi_entry.pi_and_jailbreak_filter_result
        if _is_match_found(pi_res.match_state):
            conf = _enum_name(pi_res.confidence_level, "HIGH")
            if conf == "DETECTION_CONFIDENCE_LEVEL_UNSPECIFIED":
                conf = "HIGH"
            blocking.append(f"pi_and_jailbreak:{conf}")

    rai_entry = filter_results.get("rai")
    if rai_entry is not None and "rai_filter_result" in rai_entry:
        rai_res = rai_entry.rai_filter_result
        if _is_match_found(rai_res.match_state):
            type_map = getattr(rai_res, "rai_filter_type_results", None) or {}
            subtypes = sorted(
                str(name)
                for name, detail in type_map.items()
                if _is_match_found(getattr(detail, "match_state", None))
            )
            blocking.extend(f"rai:{sub}" for sub in subtypes) if subtypes else blocking.append(
                "rai"
            )

    for key, attr_name in (
        ("malicious_uris", "malicious_uri_filter_result"),
        ("csam", "csam_filter_filter_result"),
        ("virus_scan", "virus_scan_filter_result"),
    ):
        entry = filter_results.get(key)
        if entry is not None and attr_name in entry:
            sub = getattr(entry, attr_name)
            if _is_match_found(getattr(sub, "match_state", None)):
                blocking.append(key)

    sdp_matched = False
    sdp_info_types: list[str] = []
    sdp_deid_text: str | None = None

    sdp_entry = filter_results.get("sdp")
    if sdp_entry is not None and "sdp_filter_result" in sdp_entry:
        sdp_res = sdp_entry.sdp_filter_result
        if "inspect_result" in sdp_res:
            insp = sdp_res.inspect_result
            if _is_match_found(insp.match_state):
                sdp_matched = True
                for finding in getattr(insp, "findings", None) or ():
                    it_name = getattr(finding, "info_type", "")
                    if it_name:
                        sdp_info_types.append(str(it_name))
        if "deidentify_result" in sdp_res:
            deid = sdp_res.deidentify_result
            if _is_match_found(deid.match_state):
                sdp_matched = True
                data_obj = getattr(deid, "data", None)
                if data_obj is not None and isinstance(getattr(data_obj, "text", None), str):
                    sdp_deid_text = data_obj.text
                for it_name in getattr(deid, "info_types", None) or ():
                    if it_name:
                        sdp_info_types.append(str(it_name))

    if _is_match_found(match_state) and not blocking and not sdp_matched:
        blocking.append("policy_violation")

    return ModelArmorSanitizationSummary(
        filter_match_state=match_state,
        blocking_categories=tuple(dict.fromkeys(blocking)),
        sdp_matched=sdp_matched,
        sdp_info_types=tuple(sorted(set(sdp_info_types))),
        sdp_deidentified_text=sdp_deid_text,
        invocation_result=inv_result,
        backend_used=backend_used,
    )
