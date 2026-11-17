"""Offline Google Cloud Model Armor v1 & Cloud DLP v2 wire-protocol emulator (zero regexes)."""

from __future__ import annotations

import bisect
import json
from collections import defaultdict
from typing import Any

import httpx

from edgar_capstone.security.model_armor.config import (
    _load_guardrail_policy_dict,
    _normalize_words,
)
from edgar_capstone.security.model_armor.projection_filter import (
    _detect_sensitive_spans,
    _redact_spans,
)

_MALICIOUS_URI_SCHEMES: tuple[str, ...] = (
    "javascript:",
    "data:text/html",
    "vbscript:",
    "http://169.254.169.254",
)


def _matches_ordered_proximity_clause(
    word_positions: dict[str, list[int]],
    clause_tokens: list[str],
    *,
    max_span: int = 8,
) -> bool:
    """Return True if there exists an ordered subsequence `(p_0 < p_1 < ... < p_k)` with `(p_k - p_0) <= max_span`.

    Checks EVERY occurrence of each token (via binary search) so repeated earlier/decoy words
    such as ``"Instructions for Q1: ignore previous instructions"`` cannot bypass detection.
    """
    if not clause_tokens:
        return False
    first_positions = word_positions.get(clause_tokens[0])
    if not first_positions:
        return False
    if len(clause_tokens) == 1:
        return True
    if any(tok not in word_positions for tok in clause_tokens[1:]):
        return False

    for start_pos in first_positions:
        prev_pos = start_pos
        matched = True
        for tok in clause_tokens[1:]:
            candidates = word_positions[tok]
            next_idx = bisect.bisect_right(candidates, prev_pos)
            if next_idx >= len(candidates):
                matched = False
                break
            prev_pos = candidates[next_idx]
            if prev_pos - start_pos > max_span:
                matched = False
                break
        if matched:
            return True
    return False


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
        if not url_str.endswith(":deidentify"):
            return httpx.Response(
                404, json={"error": {"code": 404, "message": "Unsupported DLP operation"}}
            )
        text = str((payload.get("item") or {}).get("value") or "")
        spans = _detect_sensitive_spans(text)
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

    def _build_sanitization_response(self, text: str, *, is_user_prompt: bool) -> dict[str, Any]:
        policy = _load_guardrail_policy_dict()
        words = _normalize_words(text)
        word_positions: dict[str, list[int]] = defaultdict(list)
        for idx, token in enumerate(words):
            word_positions[token].append(idx)

        pi_matched = False
        if is_user_prompt and word_positions:
            for clause in policy.get("emulator_pi_token_clauses", []):
                clause_tokens = [str(tok).strip().lower() for tok in clause if str(tok).strip()]
                if _matches_ordered_proximity_clause(word_positions, clause_tokens, max_span=8):
                    pi_matched = True
                    break

        lowered_text = text.lower()
        malicious_uri_matched = any(scheme in lowered_text for scheme in _MALICIOUS_URI_SCHEMES)

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
                            "matchState": (
                                "MATCH_FOUND" if malicious_uri_matched else "NO_MATCH_FOUND"
                            ),
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
