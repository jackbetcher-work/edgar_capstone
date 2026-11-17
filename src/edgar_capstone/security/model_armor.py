"""Google Cloud Managed Model Armor API, Sensitive Data Protection (SDP / Cloud DLP),
and SEC Financial Projection Compliance Guardrails.

Replaces legacy regex prompt-injection filters with:
1. **Google Cloud Managed Model Armor v1 API** (`sanitizeUserPrompt` and `sanitizeModelResponse`)
   enforcing `pi_and_jailbreak`, `rai`, `malicious_uris`, `csam`, and integrated `sdp`
   inspection/de-identification via regional endpoints
   (`https://modelarmor.{location}.rep.googleapis.com/v1/...`).
2. **Google Cloud Sensitive Data Protection (SDP / Cloud DLP v2 API)** (`content:inspect`
   and `content:deidentify`) redacting or blocking PII and credentials before prompts reach
   Gemini or before responses enter session memory / UI responses.
3. **SEC Financial Projection & MNPI Compliance Filter** preventing ungrounded forward-looking
   financial forecasts, stock price targets, buy/sell investment advice, and confidential
   MNPI leaks across both user inputs and synthesized outputs.
4. **ADK `ModelArmorSafetyPlugin` (`BasePlugin`)** hooking `on_user_message_callback`,
   `before_run_callback`, `before_model_callback`, `after_tool_callback`, and
   `after_model_callback` with multi-turn session-poisoning prevention.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import string
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

import google.auth
import httpx
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.genai import types as genai_types

from edgar_capstone.observability.tracing import traced
from edgar_capstone.security.agent_identity import (
    GcpAgentIamCredentialsClient,
    authorize_agent_s2s_hop,
)
from edgar_capstone.security.auth import record_security_audit_event
from edgar_capstone.security.rbac import CUSTOM_RUNTIME_SA_ACCOUNT_ID
from edgar_capstone.security.secrets import DEFAULT_PROJECT_ID

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


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _normalize_location(raw_location: str | None) -> str:
    loc = (raw_location or "").strip()
    if not loc or loc.lower() == "global":
        return DEFAULT_MODEL_ARMOR_LOCATION
    return loc


def _normalize_words(text: str) -> list[str]:
    """Tokenize text into normalized lowercase alphanumeric words without regexes."""
    table = str.maketrans({ch: " " for ch in string.punctuation})
    return text.lower().translate(table).split()


def _load_guardrail_policy_dict() -> dict[str, Any]:
    """Load versioned Model Armor & projection filter policy from `VersionedPromptStore`."""
    try:
        from edgar_capstone.agents.prompt_store import get_prompt_store

        artifact = get_prompt_store().load_prompt_artifact("model_armor_policy")
        policy = artifact.raw_metadata.get("policy")
        if isinstance(policy, dict):
            return policy
    except Exception as exc:
        logger.debug("Falling back to default guardrail policy metadata: %s", exc)
    return {
        "blocked_prompt_message": (
            "Request blocked by SEC Analyst Model Armor & compliance guardrails: I can only "
            "answer financial and SEC 10-K analysis queries grounded in verified historical "
            "EDGAR filings and deterministic variance calculations."
        ),
        "blocked_projection_message": (
            "Request blocked by SEC Financial Projection & Grounding Compliance Filter: "
            "This workspace is restricted to verified historical SEC Form 10-K disclosures "
            "(through FY2026) and deterministic variance metrics, and cannot generate "
            "forward-looking financial projections, stock price targets, or investment advice."
        ),
        "blocked_response_message": (
            "Response withheld by Model Armor & SEC Compliance Filter: The generated output "
            "triggered prompt-safety, sensitive-data (SDP), or financial projection guardrails."
        ),
        "emulator_pi_token_clauses": [
            ["ignore", "previous", "instructions"],
            ["ignore", "prior", "instructions"],
            ["ignore", "system", "instructions"],
            ["disregard", "previous", "instructions"],
            ["override", "system", "instructions"],
            ["bypass", "grounding"],
            ["disable", "grounding"],
            ["bypass", "model", "armor"],
            ["reveal", "system", "prompt"],
            ["print", "system", "prompt"],
            ["fabricate", "financial", "numbers"],
            ["invent", "financial", "numbers"],
            ["developer", "mode", "jailbreak"],
        ],
        "projection_phrases": [
            "price target",
            "target price",
            "12-month target",
            "12 month target",
            "stock price forecast",
            "share price prediction",
            "predict stock price",
            "forecast stock price",
            "strong buy",
            "strong sell",
            "buy rating",
            "sell rating",
            "should i buy",
            "should i sell",
            "guaranteed return",
        ],
        "mnpi_markers": [
            "confidential mnpi",
            "material non-public information",
            "material nonpublic information",
            "non-public financial projection",
            "internal embargoed draft",
            "unreleased earnings preview",
            "pre-release earnings guidance",
        ],
        "future_fiscal_years": ["2027", "2028", "2029", "2030", "2031", "2032"],
        "projection_verbs_and_nouns": [
            "forecast",
            "forecasts",
            "forecasted",
            "predict",
            "predicts",
            "prediction",
            "predictions",
            "projection",
            "projections",
            "projected",
        ],
    }


# ---------------------------------------------------------------------------
# Structural PII / Credential Inspection Helpers (for SDP & Emulator Parity)
# ---------------------------------------------------------------------------


def _luhn_checksum_valid(digits: str) -> bool:
    total = 0
    reverse_digits = digits[::-1]
    for idx, ch in enumerate(reverse_digits):
        n = ord(ch) - 48
        if idx % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _detect_sensitive_spans(text: str) -> list[tuple[int, int, str]]:
    """Locate PII and credential spans `(start, end, info_type)` for SDP inspection/redaction."""
    spans: list[tuple[int, int, str]] = []
    n = len(text)

    # 1. US Social Security Numbers: XXX-XX-XXXX (strict boundary check avoids SEC accession numbers)
    for i in range(0, max(0, n - 10)):
        if i > 0 and (text[i - 1].isalnum() or text[i - 1] == "-"):
            continue
        if i + 11 < n and (text[i + 11].isalnum() or text[i + 11] == "-"):
            continue
        seg = text[i : i + 11]
        if (
            len(seg) == 11
            and seg[3] == "-"
            and seg[6] == "-"
            and seg[:3].isdigit()
            and seg[4:6].isdigit()
            and seg[7:].isdigit()
        ):
            area, grp, ser = seg[:3], seg[4:6], seg[7:]
            if (
                area not in ("000", "666")
                and not area.startswith("9")
                and grp != "00"
                and ser != "0000"
            ):
                spans.append((i, i + 11, "US_SOCIAL_SECURITY_NUMBER"))

    # 2. Credit Card Numbers (13-19 digits with spaces/hyphens passing Luhn & major IIN prefix)
    idx = 0
    while idx < n:
        if text[idx].isdigit() and (idx == 0 or not text[idx - 1].isalnum()):
            end = idx
            digits_buf: list[str] = []
            while end < n:
                ch = text[end]
                if ch.isdigit():
                    digits_buf.append(ch)
                    end += 1
                elif ch in (" ", "-") and end + 1 < n and text[end + 1].isdigit():
                    end += 1
                else:
                    break
            digits_str = "".join(digits_buf)
            if (
                13 <= len(digits_str) <= 19
                and (end == n or not text[end].isalnum())
                and len(set(digits_str)) >= 3
                and not digits_str.endswith("000000")
                and digits_str[0] in ("3", "4", "5", "6")
                and _luhn_checksum_valid(digits_str)
            ):
                spans.append((idx, end, "CREDIT_CARD_NUMBER"))
            idx = max(idx + 1, end)
        else:
            idx += 1

    # 3. US Phone Numbers: (XXX) XXX-XXXX or +1-XXX-XXX-XXXX / +1 XXX XXX XXXX
    for i in range(0, n):
        if text[i] == "(" and i + 13 <= n:
            # (415) 555-0199 or (415)555-0199
            if text[i + 1 : i + 4].isdigit() and text[i + 4] == ")":
                rest_start = i + 6 if (i + 5 < n and text[i + 5] == " ") else i + 5
                if (
                    rest_start + 8 <= n
                    and text[rest_start : rest_start + 3].isdigit()
                    and text[rest_start + 3] in ("-", " ")
                    and text[rest_start + 4 : rest_start + 8].isdigit()
                    and (rest_start + 8 == n or not text[rest_start + 8].isdigit())
                ):
                    spans.append((i, rest_start + 8, "PHONE_NUMBER"))
        elif text.startswith("+1", i) and i + 14 <= n and text[i + 2] in ("-", " "):
            sep = text[i + 2]
            seg = text[i + 3 : i + 15]
            if (
                len(seg) == 12
                and seg[0:3].isdigit()
                and seg[3] == sep
                and seg[4:7].isdigit()
                and seg[7] == sep
                and seg[8:12].isdigit()
                and (i + 15 == n or not text[i + 15].isdigit())
            ):
                spans.append((i, i + 15, "PHONE_NUMBER"))

    # 4. Personal/External Email Addresses (excluding GCP service accounts & sec.gov)
    email_local_chars = set(string.ascii_letters + string.digits + "._%+-")
    email_domain_chars = set(string.ascii_letters + string.digits + ".-")
    for pos, ch in enumerate(text):
        if ch != "@":
            continue
        left = pos
        while left > 0 and text[left - 1] in email_local_chars:
            left -= 1
        right = pos + 1
        while right < n and text[right] in email_domain_chars:
            right += 1
        while right > pos + 1 and text[right - 1] in ".-":
            right -= 1
        local_part = text[left:pos]
        domain_part = text[pos + 1 : right]
        if (
            len(local_part) >= 1
            and "." in domain_part
            and len(domain_part.rsplit(".", 1)[-1]) >= 2
            and not domain_part.lower().endswith(_SAFE_EMAIL_DOMAIN_SUFFIXES)
        ):
            spans.append((left, right, "EMAIL_ADDRESS"))

    # 5. Leaked Cloud / API Credentials
    token_chars = set(string.ascii_letters + string.digits + "-_.")
    t_idx = 0
    while t_idx < n:
        if text[t_idx] in token_chars:
            t_end = t_idx
            while t_end < n and text[t_end] in token_chars:
                t_end += 1
            tok = text[t_idx:t_end]
            if tok.startswith("AIza") and len(tok) >= 33:
                spans.append((t_idx, t_end, "GCP_API_KEY"))
            elif tok.startswith("ya29.") and len(tok) >= 24:
                spans.append((t_idx, t_end, "AUTH_TOKEN"))
            elif tok.startswith(("ghp_", "gho_", "ghs_", "github_pat_")) and len(tok) >= 28:
                spans.append((t_idx, t_end, "AUTH_TOKEN"))
            elif (
                (tok.startswith("AKIA") or tok.startswith("ASIA"))
                and len(tok) == 20
                and tok.isalnum()
            ):
                spans.append((t_idx, t_end, "AUTH_TOKEN"))
            t_idx = t_end
        else:
            t_idx += 1

    if not spans:
        return []

    # Sort and deduplicate overlapping spans (preferring earlier / longer span)
    spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    merged: list[tuple[int, int, str]] = []
    for start, end, info_type in spans:
        if merged and start < merged[-1][1]:
            continue
        merged.append((start, end, info_type))
    return merged


def _redact_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    if not spans:
        return text
    out: list[str] = []
    cursor = 0
    for start, end, info_type in spans:
        out.append(text[cursor:start])
        out.append(f"[REDACTED_{info_type}]")
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


# ---------------------------------------------------------------------------
# Financial Projection & MNPI Leak Filter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectionFilterResult:
    """Evaluation result from the SEC Financial Projection & MNPI Compliance Filter."""

    matched: bool
    category: str | None = None
    matched_rules: tuple[str, ...] = ()
    reason: str = ""


def _contains_token_sequence(words: list[str], phrase_tokens: list[str]) -> bool:
    """Return True if `phrase_tokens` appears as a contiguous slice inside `words`."""
    k = len(phrase_tokens)
    if k == 0 or k > len(words):
        return False
    first = phrase_tokens[0]
    for idx in range(len(words) - k + 1):
        if words[idx] == first and words[idx : idx + k] == phrase_tokens:
            return True
    return False


class FinancialProjectionFilter:
    """Detects ungrounded forward-looking financial projections, target prices, and MNPI leaks."""

    @staticmethod
    def _extract_query_segment_if_structured_prompt(text: str) -> str:
        """If `text` is an internal ADK synthesis/eval prompt, scope projection checks to the user query."""
        for header in (
            "## Analyst Query\n",
            "### Analyst Query\n",
            "## User Query\n",
            "### User Query\n",
            "## Query\n",
            "### Query\n",
        ):
            pos = text.find(header)
            if pos != -1:
                after = text[pos + len(header) :]
                for next_hdr in ("\n## ", "\n### "):
                    end_pos = after.find(next_hdr)
                    if end_pos != -1:
                        return after[:end_pos].strip()
                return after.strip()
        return text

    def evaluate(
        self, text: str, *, stage: Literal["prompt", "response"] = "prompt"
    ) -> ProjectionFilterResult:
        if not text or not text.strip():
            return ProjectionFilterResult(matched=False)

        policy = _load_guardrail_policy_dict()
        target_text = (
            self._extract_query_segment_if_structured_prompt(text) if stage == "prompt" else text
        )
        lowered = " ".join(target_text.lower().split())
        words = _normalize_words(target_text)
        matched_rules: list[str] = []

        # 1. Confidential / Non-Public MNPI markers (scoped to target_text; allow 10-K policy disclosures in responses)
        is_10k_policy_disclosure = stage == "response" and any(
            term in lowered
            for term in ("insider trading", "item 10", "item 408", "10b5-1", "code of ethics")
        )
        for marker in policy.get("mnpi_markers", []):
            marker_norm = str(marker).strip().lower()
            if not marker_norm:
                continue
            if is_10k_policy_disclosure and marker_norm in (
                "material non-public information",
                "material nonpublic information",
            ):
                continue
            if marker_norm in lowered:
                matched_rules.append(f"mnpi:{marker_norm}")
        if matched_rules:
            return ProjectionFilterResult(
                matched=True,
                category="mnpi_confidential_leak",
                matched_rules=tuple(matched_rules),
                reason=(
                    f"Detected confidential MNPI or unreleased guidance marker in {stage}: "
                    f"{', '.join(matched_rules)}"
                ),
            )

        # 2. Explicit stock price targets, buy/sell ratings, or ungrounded prediction phrases (word-boundary)
        for phrase in policy.get("projection_phrases", []):
            phrase_norm = str(phrase).strip().lower()
            phrase_tokens = _normalize_words(phrase_norm)
            if phrase_tokens and _contains_token_sequence(words, phrase_tokens):
                matched_rules.append(f"projection_phrase:{phrase_norm}")

        # 3. Future fiscal-year projections beyond the verified 10-K filing window (FY2027+)
        # Require local token proximity (<= 8 words) between a forecast term and future year,
        # and avoid blocking explicit refusal statements in model responses.
        is_refusal_response = stage == "response" and any(
            ref in lowered
            for ref in (
                "cannot forecast",
                "cannot predict",
                "cannot project",
                "unable to forecast",
                "unable to predict",
                "unable to project",
                "restricted to verified historical",
            )
        )
        if not is_refusal_response:
            future_years = {str(y).strip() for y in policy.get("future_fiscal_years", [])}
            future_year_tokens = future_years.union({f"fy{y}" for y in future_years})
            proj_terms = {
                str(t).strip().lower() for t in policy.get("projection_verbs_and_nouns", [])
            }

            year_positions = [(i, tok) for i, tok in enumerate(words) if tok in future_year_tokens]
            proj_positions = [(i, tok) for i, tok in enumerate(words) if tok in proj_terms]
            for y_idx, y_tok in year_positions:
                for p_idx, p_tok in proj_positions:
                    if abs(y_idx - p_idx) <= 8:
                        clean_yr = y_tok[2:] if y_tok.startswith("fy") else y_tok
                        matched_rules.append(f"future_period_projection:FY{clean_yr}+{p_tok}")
                        break
                if any(r.startswith("future_period_projection:") for r in matched_rules):
                    break

        if matched_rules:
            return ProjectionFilterResult(
                matched=True,
                category="ungrounded_financial_projection",
                matched_rules=tuple(matched_rules),
                reason=(
                    f"Blocked ungrounded forward-looking financial projection or investment advice "
                    f"in {stage} ({', '.join(matched_rules)})."
                ),
            )

        return ProjectionFilterResult(matched=False)


# ---------------------------------------------------------------------------
# Offline Model Armor v1 & Cloud DLP v2 Wire Emulator (Zero Prompt-Injection Regexes)
# ---------------------------------------------------------------------------


class LocalModelArmorEmulatorTransport(httpx.BaseTransport):
    """Offline `httpx.BaseTransport` speaking Google Cloud Model Armor v1 & Cloud DLP v2 JSON wire schemas.

    Used strictly in `EDGAR_MODEL_ARMOR_MODE='auto'` (or `'emulator'`) when GCP Application
    Default Credentials are absent in local dev / CI environments. Uses lexical token-clause
    policy evaluation from `prompts/v1/model_armor_policy.yaml` (zero prompt-injection regexes).
    """

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        raw_body = request.content.decode("utf-8") if request.content else "{}"
        payload = json.loads(raw_body) if raw_body.strip() else {}

        if "dlp.googleapis.com" in url_str:
            return self._handle_dlp_request(url_str, payload)

        if ":sanitizeUserPrompt" in url_str:
            text = str((payload.get("userPromptData") or {}).get("text") or "")
            return httpx.Response(
                200, json=self._build_sanitization_response(text, is_user_prompt=True)
            )

        if ":sanitizeModelResponse" in url_str:
            text = str((payload.get("modelResponseData") or {}).get("text") or "")
            return httpx.Response(
                200, json=self._build_sanitization_response(text, is_user_prompt=False)
            )

        return httpx.Response(
            404, json={"error": {"code": 404, "message": f"Unknown endpoint: {url_str}"}}
        )

    def _handle_dlp_request(self, url_str: str, payload: dict[str, Any]) -> httpx.Response:
        text = str((payload.get("item") or {}).get("value") or "")
        spans = _detect_sensitive_spans(text)
        if url_str.endswith(":inspect"):
            findings = [
                {
                    "infoType": {"name": info_type},
                    "likelihood": "VERY_LIKELY",
                    "location": {"codepointRange": {"start": str(start), "end": str(end)}},
                }
                for start, end, info_type in spans
            ]
            return httpx.Response(
                200, json={"result": {"findings": findings, "findingsTruncated": False}}
            )

        if url_str.endswith(":deidentify"):
            redacted = _redact_spans(text, spans)
            summaries = [
                {"infoType": {"name": info_type}, "results": [{"count": "1", "code": "SUCCESS"}]}
                for _, _, info_type in spans
            ]
            return httpx.Response(
                200,
                json={
                    "item": {"value": redacted},
                    "overview": {"transformationSummaries": summaries},
                },
            )
        return httpx.Response(
            404, json={"error": {"code": 404, "message": "Unsupported DLP operation"}}
        )

    def _build_sanitization_response(self, text: str, *, is_user_prompt: bool) -> dict[str, Any]:
        policy = _load_guardrail_policy_dict()
        words = _normalize_words(text)
        word_set = set(words)

        pi_matched = False
        if is_user_prompt and word_set:
            for clause in policy.get("emulator_pi_token_clauses", []):
                clause_tokens = [str(tok).strip().lower() for tok in clause if str(tok).strip()]
                if clause_tokens and all(tok in word_set for tok in clause_tokens):
                    # Verify ordered proximity window so unrelated distant words do not collide
                    positions = [words.index(tok) for tok in clause_tokens]
                    if positions == sorted(positions) and (positions[-1] - positions[0]) <= 8:
                        pi_matched = True
                        break

        malicious_uri_matched = any(
            scheme in text.lower()
            for scheme in ("javascript:", "data:text/html", "vbscript:", "http://169.254.169.254")
        )

        spans = _detect_sensitive_spans(text)
        sdp_matched = len(spans) > 0
        redacted_text = _redact_spans(text, spans) if sdp_matched else text
        sdp_findings = [
            {"infoType": info_type, "likelihood": "VERY_LIKELY"} for _, _, info_type in spans
        ]

        overall_match = pi_matched or malicious_uri_matched or sdp_matched
        return {
            "sanitizationResult": {
                "filterMatchState": "MATCH_FOUND" if overall_match else "NO_MATCH_FOUND",
                "invocationResult": "SUCCESS",
                "filterResults": {
                    "pi_and_jailbreak": {
                        "piAndJailbreakFilterResult": {
                            "executionState": "EXECUTION_SUCCESS",
                            "matchState": "MATCH_FOUND" if pi_matched else "NO_MATCH_FOUND",
                            "confidenceLevel": "HIGH" if pi_matched else "LOW",
                        }
                    },
                    "malicious_uris": {
                        "maliciousUriFilterResult": {
                            "executionState": "EXECUTION_SUCCESS",
                            "matchState": "MATCH_FOUND"
                            if malicious_uri_matched
                            else "NO_MATCH_FOUND",
                        }
                    },
                    "rai": {
                        "raiFilterResult": {
                            "executionState": "EXECUTION_SUCCESS",
                            "matchState": "NO_MATCH_FOUND",
                            "raiFilterTypeResults": {
                                "dangerous": {"matchState": "NO_MATCH_FOUND"},
                                "harassment": {"matchState": "NO_MATCH_FOUND"},
                                "hate_speech": {"matchState": "NO_MATCH_FOUND"},
                                "sexually_explicit": {"matchState": "NO_MATCH_FOUND"},
                            },
                        }
                    },
                    "csam": {
                        "csamFilterFilterResult": {
                            "executionState": "EXECUTION_SUCCESS",
                            "matchState": "NO_MATCH_FOUND",
                        }
                    },
                    "sdp": {
                        "sdpFilterResult": {
                            "inspectResult": {
                                "executionState": "EXECUTION_SUCCESS",
                                "matchState": "MATCH_FOUND" if sdp_matched else "NO_MATCH_FOUND",
                                "findings": sdp_findings,
                            },
                            "deidentifyResult": {
                                "executionState": "EXECUTION_SUCCESS",
                                "matchState": "MATCH_FOUND" if sdp_matched else "NO_MATCH_FOUND",
                                "data": {"text": redacted_text},
                                "infoTypes": sorted({it for _, _, it in spans}),
                            },
                        }
                    },
                },
            }
        }


# ---------------------------------------------------------------------------
# Managed Google Cloud Model Armor v1 & Cloud SDP/DLP Clients
# ---------------------------------------------------------------------------


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
        project_id = (
            os.environ.get("GCP_PROJECT_ID", "").strip()
            or os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
            or DEFAULT_PROJECT_ID
        )
        raw_loc = (
            os.environ.get("MODEL_ARMOR_LOCATION", "").strip()
            or os.environ.get("GCP_REGION", "").strip()
            or os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip()
            or DEFAULT_MODEL_ARMOR_LOCATION
        )
        location = _normalize_location(raw_loc)
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
            enable_cloud_dlp_api=_truthy_env("MODEL_ARMOR_ENABLE_DLP_API", True),
            projection_filter_enabled=_truthy_env("MODEL_ARMOR_PROJECTION_FILTER_ENABLED", True),
            inspect_tool_outputs=_truthy_env("MODEL_ARMOR_INSPECT_TOOL_OUTPUTS", False),
            fail_closed=_truthy_env("MODEL_ARMOR_FAIL_CLOSED", True),
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
    sdp_matched = False
    sdp_info_types: list[str] = []
    sdp_deid_text: str | None = None

    def _is_matched(val: Any) -> bool:
        if isinstance(val, int):
            return val == 2  # FilterMatchState.MATCH_FOUND enum int in proto
        s = str(val or "").upper()
        return s.endswith("MATCH_FOUND") and not s.endswith("NO_MATCH_FOUND")

    # 1. Prompt Injection & Jailbreak
    pi_entry = filter_results.get("pi_and_jailbreak") or {}
    pi_res = (
        pi_entry.get("piAndJailbreakFilterResult")
        or pi_entry.get("pi_and_jailbreak_filter_result")
        or pi_entry
    )
    if isinstance(pi_res, dict) and _is_matched(
        pi_res.get("matchState") or pi_res.get("match_state")
    ):
        conf = pi_res.get("confidenceLevel") or pi_res.get("confidence_level") or "HIGH"
        blocking.append(f"pi_and_jailbreak:{conf}")

    # 2. Responsible AI (RAI)
    rai_entry = filter_results.get("rai") or {}
    rai_res = rai_entry.get("raiFilterResult") or rai_entry.get("rai_filter_result") or rai_entry
    if isinstance(rai_res, dict) and _is_matched(
        rai_res.get("matchState") or rai_res.get("match_state")
    ):
        type_map = (
            rai_res.get("raiFilterTypeResults") or rai_res.get("rai_filter_type_results") or {}
        )
        matched_rai_subtypes: list[str] = []
        if isinstance(type_map, dict):
            for r_name, r_detail in type_map.items():
                if isinstance(r_detail, dict) and _is_matched(
                    r_detail.get("matchState") or r_detail.get("match_state")
                ):
                    matched_rai_subtypes.append(str(r_name))
        if matched_rai_subtypes:
            for sub in sorted(matched_rai_subtypes):
                blocking.append(f"rai:{sub}")
        else:
            blocking.append("rai")

    # 3. Malicious URIs
    uri_entry = filter_results.get("malicious_uris") or {}
    uri_res = (
        uri_entry.get("maliciousUriFilterResult")
        or uri_entry.get("malicious_uri_filter_result")
        or uri_entry
    )
    if isinstance(uri_res, dict) and _is_matched(
        uri_res.get("matchState") or uri_res.get("match_state")
    ):
        blocking.append("malicious_uris")

    # 4. CSAM
    csam_entry = filter_results.get("csam") or {}
    csam_res = (
        csam_entry.get("csamFilterFilterResult")
        or csam_entry.get("csam_filter_filter_result")
        or csam_entry
    )
    if isinstance(csam_res, dict) and _is_matched(
        csam_res.get("matchState") or csam_res.get("match_state")
    ):
        blocking.append("csam")

    # 5. Sensitive Data Protection (SDP) Inspect & De-identify
    sdp_entry = filter_results.get("sdp") or {}
    sdp_res = sdp_entry.get("sdpFilterResult") or sdp_entry.get("sdp_filter_result") or sdp_entry
    if isinstance(sdp_res, dict):
        insp = sdp_res.get("inspectResult") or sdp_res.get("inspect_result") or {}
        deid = sdp_res.get("deidentifyResult") or sdp_res.get("deidentify_result") or {}
        if isinstance(insp, dict) and _is_matched(
            insp.get("matchState") or insp.get("match_state")
        ):
            sdp_matched = True
            for f_item in insp.get("findings") or []:
                if isinstance(f_item, dict):
                    it_raw = f_item.get("infoType") or f_item.get("info_type")
                    it_name = it_raw.get("name") if isinstance(it_raw, dict) else it_raw
                    if it_name:
                        sdp_info_types.append(str(it_name))
        if isinstance(deid, dict) and _is_matched(
            deid.get("matchState") or deid.get("match_state")
        ):
            sdp_matched = True
            data_obj = deid.get("data") or {}
            if isinstance(data_obj, dict) and isinstance(data_obj.get("text"), str):
                sdp_deid_text = data_obj["text"]
            for it_name in deid.get("infoTypes") or deid.get("info_types") or []:
                if it_name:
                    sdp_info_types.append(str(it_name))

    # Fallback: if top-level filterMatchState is MATCH_FOUND with no specific sub-filter parsed
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
    )


_F = TypeVar("_F", bound=Callable[..., Any])


def _resilient_security_rpc(
    *,
    tool_name: str,
    max_retries: int = 3,
    base_delay_seconds: float = 0.05,
) -> Callable[[_F], _F]:
    """Lazily wrap security RPCs with `with_tool_resilience` to avoid package import cycles."""

    def decorator(func: _F) -> _F:
        wrapped_fn: Callable[..., Any] | None = None

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal wrapped_fn
            from edgar_capstone.agents.resilience import ToolExecutionError, with_tool_resilience

            if wrapped_fn is None:
                wrapped_fn = with_tool_resilience(
                    tool_name=tool_name,
                    max_retries=max_retries,
                    base_delay_seconds=base_delay_seconds,
                )(func)
            res = wrapped_fn(*args, **kwargs)
            if isinstance(res, dict) and res.get("status") == "error":
                raise ToolExecutionError(str(res.get("message") or f"{tool_name} failed"))
            return res

        return wrapper  # type: ignore[return-value]

    return decorator


def _raise_for_security_http_status(resp: httpx.Response, service_label: str) -> None:
    if resp.status_code < 400:
        return
    from edgar_capstone.agents.resilience import (
        RateLimitExceededError,
        ToolExecutionError,
        UpstreamTimeoutError,
    )

    snippet = resp.text[:300]
    if resp.status_code == 429:
        retry_hdr = resp.headers.get("Retry-After")
        retry_s = (
            float(retry_hdr) if retry_hdr and retry_hdr.replace(".", "", 1).isdigit() else None
        )
        raise RateLimitExceededError(
            f"{service_label} HTTP 429 rate limited: {snippet}",
            retry_after_seconds=retry_s,
        )
    if resp.status_code in (408, 500, 502, 503, 504):
        raise UpstreamTimeoutError(f"{service_label} transient HTTP {resp.status_code}: {snippet}")
    raise ToolExecutionError(f"{service_label} HTTP {resp.status_code}: {snippet}")


class ModelArmorClient:
    """Regional Google Cloud Model Armor v1 & Cloud SDP/DLP REST client with S2S attestation & retries."""

    def __init__(
        self,
        config: ModelArmorConfig | None = None,
        *,
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
        iam_credentials_client: GcpAgentIamCredentialsClient | None = None,
        sdk_client: Any | None = None,
    ) -> None:
        self.config = config or ModelArmorConfig.from_env()
        self._http_client = http_client
        self._credentials = credentials
        self._iam_credentials_client = iam_credentials_client
        self._sdk_client = sdk_client
        self._emulator_client = httpx.Client(transport=LocalModelArmorEmulatorTransport())

    def _should_use_live_gcp(self) -> bool:
        if self.config.mode == "disabled":
            return False
        if self.config.mode == "emulator":
            return False
        if self.config.mode in ("gcp", "live"):
            return True
        # In "auto" mode: use live GCP when explicitly injected with a transport/SDK/credentials,
        # or when running on Vertex AI / Cloud Run GCP backend.
        if (
            self._http_client is not None
            or self._sdk_client is not None
            or self._credentials is not None
        ):
            return True
        if _truthy_env("GOOGLE_GENAI_USE_VERTEXAI", False):
            return True
        if os.environ.get("EDGAR_DATA_BACKEND", "").strip().lower() == "gcp":
            return True
        return False

    def _get_auth_token(self, *, target_service: str = "model-armor-service") -> str:
        if self._iam_credentials_client is not None:
            secret_tok = self._iam_credentials_client.mint_downscoped_agent_access_token(
                agent_name="ModelArmorGuardrail",
                target_service=target_service,
            )
            return secret_tok.get_secret_value()

        if self._credentials is None:
            self._credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        if not self._credentials.valid or not getattr(self._credentials, "token", None):
            self._credentials.refresh(GoogleAuthRequest())
        return str(self._credentials.token)

    def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        service_label: str,
        target_service: str,
    ) -> tuple[dict[str, Any], str]:
        if not self._should_use_live_gcp():
            resp = self._emulator_client.post(url, json=payload)
            _raise_for_security_http_status(resp, f"{service_label} (Emulator)")
            return resp.json(), "emulator_model_armor_v1"

        from edgar_capstone.agents.resilience import (
            RateLimitExceededError,
            ToolExecutionError,
            UpstreamTimeoutError,
        )

        try:
            token = self._get_auth_token(target_service=target_service)
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            if self._http_client is not None:
                resp = self._http_client.post(url, headers=headers, json=payload)
            else:
                with httpx.Client(timeout=self.config.timeout_seconds) as client:
                    resp = client.post(url, headers=headers, json=payload)
            _raise_for_security_http_status(resp, service_label)
            return resp.json(), "gcp_model_armor_v1"
        except (RateLimitExceededError, UpstreamTimeoutError):
            raise
        except Exception as exc:
            if (
                self.config.mode == "auto"
                and self._http_client is None
                and self._credentials is None
            ):
                logger.warning(
                    "%s live GCP call unavailable in auto mode (%s); routing via Model Armor wire emulator.",
                    service_label,
                    type(exc).__name__,
                )
                resp = self._emulator_client.post(url, json=payload)
                _raise_for_security_http_status(resp, f"{service_label} (Emulator)")
                return resp.json(), "emulator_model_armor_v1"
            raise ToolExecutionError(f"{service_label} request failed: {exc}") from exc

    @_resilient_security_rpc(
        tool_name="model_armor_sanitize_user_prompt",
        max_retries=3,
        base_delay_seconds=0.05,
    )
    def sanitize_user_prompt(self, text: str) -> ModelArmorSanitizationSummary:
        """Invoke Google Cloud Model Armor `templates.sanitizeUserPrompt`."""
        if self._sdk_client is not None:
            sdk_resp = self._sdk_client.sanitize_user_prompt(
                request={
                    "name": self.config.template_resource_name,
                    "user_prompt_data": {"text": text},
                }
            )
            as_dict = (
                sdk_resp
                if isinstance(sdk_resp, dict)
                else type(sdk_resp).to_dict(sdk_resp)
                if hasattr(type(sdk_resp), "to_dict")
                else {}
            )
            return parse_model_armor_response(as_dict, backend_used="gcp_model_armor_sdk")

        url = (
            f"{self.config.regional_api_endpoint}/v1/"
            f"{self.config.template_resource_name}:sanitizeUserPrompt"
        )
        raw, backend = self._post_json(
            url,
            {"userPromptData": {"text": text}},
            service_label="Model Armor sanitizeUserPrompt",
            target_service="model-armor-service",
        )
        return parse_model_armor_response(raw, backend_used=backend)

    @_resilient_security_rpc(
        tool_name="model_armor_sanitize_model_response",
        max_retries=3,
        base_delay_seconds=0.05,
    )
    def sanitize_model_response(self, text: str) -> ModelArmorSanitizationSummary:
        """Invoke Google Cloud Model Armor `templates.sanitizeModelResponse`."""
        if self._sdk_client is not None:
            sdk_resp = self._sdk_client.sanitize_model_response(
                request={
                    "name": self.config.template_resource_name,
                    "model_response_data": {"text": text},
                }
            )
            as_dict = (
                sdk_resp
                if isinstance(sdk_resp, dict)
                else type(sdk_resp).to_dict(sdk_resp)
                if hasattr(type(sdk_resp), "to_dict")
                else {}
            )
            return parse_model_armor_response(as_dict, backend_used="gcp_model_armor_sdk")

        url = (
            f"{self.config.regional_api_endpoint}/v1/"
            f"{self.config.template_resource_name}:sanitizeModelResponse"
        )
        raw, backend = self._post_json(
            url,
            {"modelResponseData": {"text": text}},
            service_label="Model Armor sanitizeModelResponse",
            target_service="model-armor-service",
        )
        return parse_model_armor_response(raw, backend_used=backend)

    @_resilient_security_rpc(
        tool_name="cloud_sdp_deidentify_content",
        max_retries=3,
        base_delay_seconds=0.05,
    )
    def inspect_and_deidentify_sdp(self, text: str) -> tuple[bool, tuple[str, ...], str]:
        """Inspect & de-identify PII/credentials via Google Cloud Sensitive Data Protection (DLP v2)."""
        # Fast pre-check: only invoke remote DLP deidentify when candidate PII/credential spans exist
        # or when running strictly against live GCP with an explicit SDP inspect/deidentify template.
        local_spans = _detect_sensitive_spans(text)
        has_explicit_templates = bool(
            self.config.sdp_inspect_template_id or self.config.sdp_deidentify_template_id
        )
        if not local_spans and not (self._should_use_live_gcp() and has_explicit_templates):
            return False, (), text

        parent = f"projects/{self.config.project_id}/locations/{self.config.location}"
        info_type_objs = [{"name": name} for name in DEFAULT_SDP_INFO_TYPES]
        deid_payload: dict[str, Any] = {
            "item": {"value": text},
            "inspectConfig": {
                "infoTypes": info_type_objs,
                "minLikelihood": "POSSIBLE",
            },
            "deidentifyConfig": {
                "infoTypeTransformations": {
                    "transformations": [{"primitiveTransformation": {"replaceWithInfoTypeConfig": {}}}]
                }
            },
        }
        if self.config.sdp_inspect_template_id:
            insp_t = self.config.sdp_inspect_template_id
            deid_payload["inspectTemplateName"] = (
                insp_t if insp_t.startswith("projects/") else f"{parent}/inspectTemplates/{insp_t}"
            )
        if self.config.sdp_deidentify_template_id:
            deid_t = self.config.sdp_deidentify_template_id
            deid_payload["deidentifyTemplateName"] = (
                deid_t
                if deid_t.startswith("projects/")
                else f"{parent}/deidentifyTemplates/{deid_t}"
            )

        url = f"https://dlp.googleapis.com/v2/{parent}/content:deidentify"
        raw, _ = self._post_json(
            url,
            deid_payload,
            service_label="Cloud SDP content:deidentify",
            target_service="sdp-dlp-service",
        )
        deid_text = str((raw.get("item") or {}).get("value") or text)
        info_types: list[str] = [it for _, _, it in local_spans]
        summaries = (raw.get("overview") or {}).get("transformationSummaries") or []
        for s_item in summaries:
            if isinstance(s_item, dict):
                it_name = (s_item.get("infoType") or {}).get("name")
                if it_name:
                    info_types.append(str(it_name))
        matched = bool(info_types) or (deid_text != text)
        return matched, tuple(sorted(set(info_types))), deid_text


# ---------------------------------------------------------------------------
# Unified Guardrail Service Orchestrating Model Armor + SDP + Projection Filters
# ---------------------------------------------------------------------------


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

    @traced("guardrail.inspect_user_prompt")
    def inspect_user_prompt(
        self,
        text: str,
        *,
        caller_agent: str = CUSTOM_RUNTIME_SA_ACCOUNT_ID,
    ) -> GuardrailDecision:
        """Inspect and sanitize an incoming user prompt via Model Armor + SDP + Projection filters."""
        start = time.perf_counter()
        sa_email, spiffe_id = self._attest_guardrail_s2s(caller_agent)
        policy = _load_guardrail_policy_dict()

        if self.config.mode == "disabled" or not text or not text.strip():
            return GuardrailDecision(
                action="allowed",
                blocked=False,
                sdp_redacted=False,
                projection_blocked=False,
                original_text=text,
                sanitized_text=text,
                backend_used="disabled" if self.config.mode == "disabled" else "noop_empty",
                latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
                s2s_service_account=sa_email,
                s2s_spiffe_id=spiffe_id,
            )

        # 1. Invoke Google Cloud Model Armor `sanitizeUserPrompt`
        try:
            ma_summary = self.client.sanitize_user_prompt(text)
        except Exception as exc:
            logger.error("Model Armor sanitizeUserPrompt failed: %s", exc)
            if self.config.fail_closed:
                record_security_audit_event(
                    event_type="model_armor_fail_closed",
                    user=None,
                    resource=self.config.template_resource_name,
                    status_outcome="blocked_fail_closed",
                    detail=str(exc)[:200],
                )
                return GuardrailDecision(
                    action="blocked",
                    blocked=True,
                    sdp_redacted=False,
                    projection_blocked=False,
                    original_text=text,
                    sanitized_text="",
                    matched_categories=("model_armor_unavailable_fail_closed",),
                    block_message=str(policy["blocked_prompt_message"]),
                    backend_used="gcp_model_armor_fail_closed",
                    latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
                    s2s_service_account=sa_email,
                    s2s_spiffe_id=spiffe_id,
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
                event_type="model_armor_prompt_blocked",
                user=None,
                resource=self.config.template_resource_name,
                status_outcome="blocked",
                detail=f"categories={list(ma_summary.blocking_categories)}",
            )
            return GuardrailDecision(
                action="blocked",
                blocked=True,
                sdp_redacted=False,
                projection_blocked=False,
                original_text=text,
                sanitized_text="",
                matched_categories=ma_summary.blocking_categories,
                sdp_info_types=ma_summary.sdp_info_types,
                block_message=str(policy["blocked_prompt_message"]),
                backend_used=ma_summary.backend_used,
                latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
                s2s_service_account=sa_email,
                s2s_spiffe_id=spiffe_id,
            )

        # 2. SEC Financial Projection & Confidential MNPI Compliance Filter
        if self.config.projection_filter_enabled:
            proj_res = self.projection_filter.evaluate(text, stage="prompt")
            if proj_res.matched:
                cat = proj_res.category or "ungrounded_financial_projection"
                record_security_audit_event(
                    event_type="projection_filter_prompt_blocked",
                    user=None,
                    resource=self.config.template_resource_name,
                    status_outcome="blocked",
                    detail=proj_res.reason,
                )
                return GuardrailDecision(
                    action="blocked",
                    blocked=True,
                    sdp_redacted=False,
                    projection_blocked=True,
                    original_text=text,
                    sanitized_text="",
                    matched_categories=(cat, *proj_res.matched_rules),
                    block_message=str(policy["blocked_projection_message"]),
                    backend_used=ma_summary.backend_used,
                    latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
                    s2s_service_account=sa_email,
                    s2s_spiffe_id=spiffe_id,
                )

        # 3. Sensitive Data Protection (Model Armor SDP + Cloud DLP v2)
        working_text = text
        sdp_types: set[str] = set(ma_summary.sdp_info_types)
        sdp_triggered = ma_summary.sdp_matched
        if ma_summary.sdp_deidentified_text is not None and ma_summary.sdp_matched:
            working_text = ma_summary.sdp_deidentified_text

        if self.config.enable_cloud_dlp_api and not sdp_triggered:
            try:
                dlp_matched, dlp_types, dlp_text = self.client.inspect_and_deidentify_sdp(
                    working_text
                )
                if dlp_matched:
                    sdp_triggered = True
                    sdp_types.update(dlp_types)
                    working_text = dlp_text
            except Exception as exc:
                logger.warning("Cloud SDP deidentify check failed: %s", exc)
                if self.config.fail_closed and self.config.mode in ("gcp", "live"):
                    return GuardrailDecision(
                        action="blocked",
                        blocked=True,
                        sdp_redacted=False,
                        projection_blocked=False,
                        original_text=text,
                        sanitized_text="",
                        matched_categories=("sdp_unavailable_fail_closed",),
                        block_message=str(policy["blocked_prompt_message"]),
                        backend_used=ma_summary.backend_used,
                        latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
                        s2s_service_account=sa_email,
                        s2s_spiffe_id=spiffe_id,
                    )

        if sdp_triggered:
            sorted_types = tuple(sorted(sdp_types))
            if (
                self.config.sdp_action == "block"
                or "GCP_API_KEY" in sdp_types
                or "AUTH_TOKEN" in sdp_types
            ):
                record_security_audit_event(
                    event_type="sdp_sensitive_prompt_blocked",
                    user=None,
                    resource=self.config.template_resource_name,
                    status_outcome="blocked",
                    detail=f"info_types={list(sorted_types)}",
                )
                return GuardrailDecision(
                    action="blocked",
                    blocked=True,
                    sdp_redacted=False,
                    projection_blocked=False,
                    original_text=text,
                    sanitized_text="",
                    matched_categories=tuple(f"sdp:{t}" for t in sorted_types) or ("sdp",),
                    sdp_info_types=sorted_types,
                    block_message=str(policy["blocked_prompt_message"]),
                    backend_used=ma_summary.backend_used,
                    latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
                    s2s_service_account=sa_email,
                    s2s_spiffe_id=spiffe_id,
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
                latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
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
            latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
            s2s_service_account=sa_email,
            s2s_spiffe_id=spiffe_id,
        )

    @traced("guardrail.inspect_model_response")
    def inspect_model_response(
        self,
        text: str,
        *,
        caller_agent: str = "FinancialAnalystAgent",
    ) -> GuardrailDecision:
        """Inspect and sanitize a synthesized model response via Model Armor + SDP + Projection filters."""
        start = time.perf_counter()
        sa_email, spiffe_id = self._attest_guardrail_s2s(caller_agent)
        policy = _load_guardrail_policy_dict()

        if self.config.mode == "disabled" or not text or not text.strip():
            return GuardrailDecision(
                action="allowed",
                blocked=False,
                sdp_redacted=False,
                projection_blocked=False,
                original_text=text,
                sanitized_text=text,
                backend_used="disabled" if self.config.mode == "disabled" else "noop_empty",
                latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
                s2s_service_account=sa_email,
                s2s_spiffe_id=spiffe_id,
            )

        # 1. Invoke Google Cloud Model Armor `sanitizeModelResponse`
        try:
            ma_summary = self.client.sanitize_model_response(text)
        except Exception as exc:
            logger.error("Model Armor sanitizeModelResponse failed: %s", exc)
            if self.config.fail_closed:
                return GuardrailDecision(
                    action="blocked",
                    blocked=True,
                    sdp_redacted=False,
                    projection_blocked=False,
                    original_text=text,
                    sanitized_text="",
                    matched_categories=("model_armor_response_unavailable_fail_closed",),
                    block_message=str(policy["blocked_response_message"]),
                    backend_used="gcp_model_armor_fail_closed",
                    latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
                    s2s_service_account=sa_email,
                    s2s_spiffe_id=spiffe_id,
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
                event_type="model_armor_response_blocked",
                user=None,
                resource=self.config.template_resource_name,
                status_outcome="blocked",
                detail=f"categories={list(ma_summary.blocking_categories)}",
            )
            return GuardrailDecision(
                action="blocked",
                blocked=True,
                sdp_redacted=False,
                projection_blocked=False,
                original_text=text,
                sanitized_text="",
                matched_categories=ma_summary.blocking_categories,
                sdp_info_types=ma_summary.sdp_info_types,
                block_message=str(policy["blocked_response_message"]),
                backend_used=ma_summary.backend_used,
                latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
                s2s_service_account=sa_email,
                s2s_spiffe_id=spiffe_id,
            )

        # 2. SEC Financial Projection & Confidential MNPI Leak Filter on output
        if self.config.projection_filter_enabled:
            proj_res = self.projection_filter.evaluate(text, stage="response")
            if proj_res.matched:
                cat = proj_res.category or "ungrounded_financial_projection"
                record_security_audit_event(
                    event_type="projection_filter_response_blocked",
                    user=None,
                    resource=self.config.template_resource_name,
                    status_outcome="blocked",
                    detail=proj_res.reason,
                )
                return GuardrailDecision(
                    action="blocked",
                    blocked=True,
                    sdp_redacted=False,
                    projection_blocked=True,
                    original_text=text,
                    sanitized_text="",
                    matched_categories=(cat, *proj_res.matched_rules),
                    block_message=str(policy["blocked_projection_message"]),
                    backend_used=ma_summary.backend_used,
                    latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
                    s2s_service_account=sa_email,
                    s2s_spiffe_id=spiffe_id,
                )

        # 3. Sensitive Data Protection (SDP) De-identification / Blocking on output
        working_text = text
        sdp_types: set[str] = set(ma_summary.sdp_info_types)
        sdp_triggered = ma_summary.sdp_matched
        if ma_summary.sdp_deidentified_text is not None and ma_summary.sdp_matched:
            working_text = ma_summary.sdp_deidentified_text

        if self.config.enable_cloud_dlp_api and not sdp_triggered:
            try:
                dlp_matched, dlp_types, dlp_text = self.client.inspect_and_deidentify_sdp(
                    working_text
                )
                if dlp_matched:
                    sdp_triggered = True
                    sdp_types.update(dlp_types)
                    working_text = dlp_text
            except Exception as exc:
                logger.warning("Cloud SDP deidentify check failed on response: %s", exc)
                if self.config.fail_closed and self.config.mode in ("gcp", "live"):
                    return GuardrailDecision(
                        action="blocked",
                        blocked=True,
                        sdp_redacted=False,
                        projection_blocked=False,
                        original_text=text,
                        sanitized_text="",
                        matched_categories=("sdp_response_unavailable_fail_closed",),
                        block_message=str(policy["blocked_response_message"]),
                        backend_used=ma_summary.backend_used,
                        latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
                        s2s_service_account=sa_email,
                        s2s_spiffe_id=spiffe_id,
                    )

        if sdp_triggered:
            sorted_types = tuple(sorted(sdp_types))
            if self.config.sdp_action == "block":
                return GuardrailDecision(
                    action="blocked",
                    blocked=True,
                    sdp_redacted=False,
                    projection_blocked=False,
                    original_text=text,
                    sanitized_text="",
                    matched_categories=tuple(f"sdp:{t}" for t in sorted_types) or ("sdp",),
                    sdp_info_types=sorted_types,
                    block_message=str(policy["blocked_response_message"]),
                    backend_used=ma_summary.backend_used,
                    latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
                    s2s_service_account=sa_email,
                    s2s_spiffe_id=spiffe_id,
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
                latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
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
            latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
            s2s_service_account=sa_email,
            s2s_spiffe_id=spiffe_id,
        )

    def to_status_metadata(self) -> dict[str, Any]:
        return {
            "mode": self.config.mode,
            "live_gcp_active": self.client._should_use_live_gcp(),
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
_GLOBAL_GUARDRAIL: ModelArmorGuardrailService | None = None


def get_model_armor_guardrail() -> ModelArmorGuardrailService:
    """Return the process-wide `ModelArmorGuardrailService` instance."""
    global _GLOBAL_GUARDRAIL
    with _GUARDRAIL_LOCK:
        if _GLOBAL_GUARDRAIL is None:
            _GLOBAL_GUARDRAIL = ModelArmorGuardrailService()
        return _GLOBAL_GUARDRAIL


def configure_model_armor_guardrail(
    service: ModelArmorGuardrailService | None = None,
    *,
    config: ModelArmorConfig | None = None,
    http_client: httpx.Client | None = None,
    credentials: Any | None = None,
    iam_credentials_client: GcpAgentIamCredentialsClient | None = None,
    sdk_client: Any | None = None,
) -> ModelArmorGuardrailService:
    """Configure or inject a custom `ModelArmorGuardrailService` (e.g. for Cloud Run startup or tests)."""
    global _GLOBAL_GUARDRAIL
    with _GUARDRAIL_LOCK:
        if service is not None:
            _GLOBAL_GUARDRAIL = service
        else:
            client = ModelArmorClient(
                config=config,
                http_client=http_client,
                credentials=credentials,
                iam_credentials_client=iam_credentials_client,
                sdk_client=sdk_client,
            )
            _GLOBAL_GUARDRAIL = ModelArmorGuardrailService(client=client)
        return _GLOBAL_GUARDRAIL


def reset_model_armor_guardrail() -> None:
    """Reset the process-wide `ModelArmorGuardrailService` singleton."""
    global _GLOBAL_GUARDRAIL
    with _GUARDRAIL_LOCK:
        _GLOBAL_GUARDRAIL = None


def get_model_armor_status_metadata() -> dict[str, Any]:
    """Return non-secret status metadata describing the active Model Armor & SDP configuration."""
    return get_model_armor_guardrail().to_status_metadata()


# ---------------------------------------------------------------------------
# Official ADK 2.14.0 `BasePlugin` Integration (`ModelArmorSafetyPlugin`)
# ---------------------------------------------------------------------------


def schema_safe_block_text(original_text: str, block_message: str) -> str:
    """Replacement text for a blocked model response that keeps structured outputs parseable.

    FinancialAnalystAgent answers with constrained `AnalystReport` JSON. Replacing a blocked
    answer with plain text would fail ADK's output-schema validation, crash the run, and lose
    the block decision, so structured answers get a minimal AnalystReport carrying the
    block message instead.
    """
    stripped = (original_text or "").strip()
    if stripped.startswith("{") and '"headline"' in stripped:
        return json.dumps(
            {"headline": block_message, "key_metrics": [], "findings": [], "notes": ""}
        )
    return block_message


_PLUGIN_BLOCK_STATE_KEY = "_model_armor_plugin_pending_block_message"
_PENDING_STATE_DELTA_KEY = "_model_armor_pending_state_delta"
_VERIFIED_PROMPT_HASH_KEY = "_model_armor_verified_prompt_sha256"
_VERIFIED_RESPONSE_HASH_KEY = "_model_armor_verified_response_sha256"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ModelArmorSafetyPlugin(BasePlugin):
    """ADK `BasePlugin` enforcing Google Cloud Model Armor, SDP de-identification, and Projection Filters.

    Follows the Google ADK `safety-plugins` architecture updated for ADK 2.14.0 keyword-only
    plugin callbacks, including session-poisoning prevention on `on_user_message_callback`
    before short-circuiting the runner in `before_run_callback`.
    """

    def __init__(
        self,
        *,
        name: str = "ModelArmorSafetyPlugin",
        guardrail: ModelArmorGuardrailService | None = None,
        inspect_tool_outputs: bool | None = None,
    ) -> None:
        super().__init__(name=name)
        self._guardrail = guardrail
        self._inspect_tool_outputs = inspect_tool_outputs

    @property
    def guardrail(self) -> ModelArmorGuardrailService:
        return self._guardrail if self._guardrail is not None else get_model_armor_guardrail()

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: genai_types.Content,
    ) -> genai_types.Content | None:
        parts = list(user_message.parts or [])
        text = "\n".join(p.text for p in parts if getattr(p, "text", None))
        if not text.strip():
            return None

        state = invocation_context.session.state
        decision = self.guardrail.inspect_user_prompt(
            text,
            caller_agent=CUSTOM_RUNTIME_SA_ACCOUNT_ID,
        )
        delta: dict[str, Any] = {
            "model_armor_last_decision": decision.to_audit_dict(),
            "model_armor_blocked": bool(decision.blocked),
            "model_armor_response_blocked": False,
            "model_armor_response_block_message": "",
        }

        if decision.blocked:
            delta["model_armor_findings"] = list(decision.matched_categories)
            state.update(delta)
            state[_PLUGIN_BLOCK_STATE_KEY] = decision.block_message
            state[_PENDING_STATE_DELTA_KEY] = delta
            # Replace unsafe prompt in session history so subsequent turns are never poisoned
            return genai_types.Content(
                role=user_message.role or "user",
                parts=[
                    genai_types.Part(
                        text=(
                            "[User message withheld by Model Armor Guardrail: "
                            f"{', '.join(decision.matched_categories)}]"
                        )
                    )
                ],
            )

        if decision.sdp_redacted:
            delta["sdp_redacted"] = True
            delta["sdp_info_types"] = list(decision.sdp_info_types)
            delta[_VERIFIED_PROMPT_HASH_KEY] = _sha256_text(decision.sanitized_text)
            state.update(delta)
            state[_PENDING_STATE_DELTA_KEY] = delta
            return genai_types.Content(
                role=user_message.role or "user",
                parts=[genai_types.Part(text=decision.sanitized_text)],
            )

        delta[_VERIFIED_PROMPT_HASH_KEY] = _sha256_text(text)
        state.update(delta)
        state[_PENDING_STATE_DELTA_KEY] = delta
        return None

    async def before_run_callback(
        self,
        *,
        invocation_context: InvocationContext,
    ) -> genai_types.Content | None:
        state = invocation_context.session.state
        pending_msg = state.get(_PLUGIN_BLOCK_STATE_KEY)
        if pending_msg:
            state[_PLUGIN_BLOCK_STATE_KEY] = ""
            return genai_types.Content(
                role="model",
                parts=[genai_types.Part(text=str(pending_msg))],
            )
        return None

    async def on_event_callback(
        self,
        *,
        invocation_context: InvocationContext,
        event: Event,
    ) -> Event | None:
        if getattr(event, "partial", False):
            return None
        pending = invocation_context.session.state.pop(_PENDING_STATE_DELTA_KEY, None)
        if isinstance(pending, dict) and pending:
            event.actions.state_delta.update(pending)
            return event
        return None

    async def before_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> LlmResponse | None:
        latest_user_text = ""
        latest_user_part: genai_types.Part | None = None
        for content in llm_request.contents or []:
            if content.role == "user":
                for part in content.parts or []:
                    if part.text:
                        latest_user_text = part.text
                        latest_user_part = part

        if not latest_user_text.strip():
            return None

        state = callback_context.state
        if state.get(_VERIFIED_PROMPT_HASH_KEY) == _sha256_text(latest_user_text):
            return None

        decision = self.guardrail.inspect_user_prompt(
            latest_user_text,
            caller_agent=str(
                getattr(callback_context, "agent_name", "") or CUSTOM_RUNTIME_SA_ACCOUNT_ID
            ),
        )
        state["model_armor_last_decision"] = decision.to_audit_dict()
        if decision.blocked:
            state["model_armor_blocked"] = True
            state["model_armor_findings"] = list(decision.matched_categories)
            return LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text=decision.block_message)],
                )
            )

        if decision.sdp_redacted and latest_user_part is not None:
            latest_user_part.text = decision.sanitized_text
            state["sdp_redacted"] = True
            state["sdp_info_types"] = list(decision.sdp_info_types)
        state[_VERIFIED_PROMPT_HASH_KEY] = _sha256_text(decision.sanitized_text)
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        _ = tool_args
        should_inspect = (
            self._inspect_tool_outputs
            if self._inspect_tool_outputs is not None
            else self.guardrail.config.inspect_tool_outputs
        )
        if not should_inspect or not isinstance(result, dict):
            return None

        serialized = json.dumps(result, default=str)
        decision = self.guardrail.inspect_user_prompt(
            serialized,
            caller_agent=str(
                getattr(tool_context, "agent_name", "") or CUSTOM_RUNTIME_SA_ACCOUNT_ID
            ),
        )
        if decision.blocked:
            tool_context.state["model_armor_blocked"] = True
            tool_context.state["model_armor_tool_blocked"] = getattr(tool, "name", "tool")
            return {
                "status": "blocked_by_model_armor",
                "error": (
                    f"Tool output from '{getattr(tool, 'name', 'tool')}' withheld by Model Armor: "
                    f"{', '.join(decision.matched_categories)}"
                ),
            }
        return None

    async def after_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> LlmResponse | None:
        # Never block or add network hops to partial SSE streaming chunks or intermediate tool calls
        if getattr(llm_response, "partial", False):
            return None
        if not llm_response.content or not llm_response.content.parts:
            return None

        text_parts: list[genai_types.Part] = []
        has_fn_call = False
        for part in llm_response.content.parts:
            if getattr(part, "function_call", None) is not None:
                has_fn_call = True
            if getattr(part, "text", None) and not getattr(part, "thought", False):
                text_parts.append(part)

        if has_fn_call or not text_parts:
            return None

        combined_text = "\n".join(p.text for p in text_parts if p.text)
        if not combined_text.strip():
            return None

        state = callback_context.state
        if state.get(_VERIFIED_RESPONSE_HASH_KEY) == _sha256_text(combined_text):
            return None

        decision = self.guardrail.inspect_model_response(
            combined_text,
            caller_agent=str(
                getattr(callback_context, "agent_name", "") or "FinancialAnalystAgent"
            ),
        )
        state["model_armor_last_response_decision"] = decision.to_audit_dict()

        if decision.blocked:
            state["model_armor_blocked"] = True
            state["model_armor_response_blocked"] = True
            state["model_armor_response_block_message"] = decision.block_message
            state["model_armor_findings"] = list(decision.matched_categories)
            return LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part(
                            text=schema_safe_block_text(combined_text, decision.block_message)
                        )
                    ],
                )
            )

        state["model_armor_response_blocked"] = False
        state["model_armor_response_block_message"] = ""

        if decision.sdp_redacted:
            state["sdp_redacted"] = True
            state["sdp_info_types"] = list(decision.sdp_info_types)
            state[_VERIFIED_RESPONSE_HASH_KEY] = _sha256_text(decision.sanitized_text)
            return LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text=decision.sanitized_text)],
                ),
                usage_metadata=llm_response.usage_metadata,
                model_version=llm_response.model_version,
            )

        state[_VERIFIED_RESPONSE_HASH_KEY] = _sha256_text(combined_text)
        return None
