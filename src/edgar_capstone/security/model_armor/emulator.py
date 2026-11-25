"""Offline Google Cloud Model Armor v1 typed proto emulator (zero regexes)."""

from __future__ import annotations

import bisect
from collections import defaultdict
from typing import Any

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
    """Return True if an ordered subsequence `(p_0 < ... < p_k)` exists within `max_span`."""
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


class LocalModelArmorEmulator:
    """In-process ``modelarmor_v1.ModelArmorClient`` emulator returning real Model Armor protos.

    Used strictly in ``EDGAR_MODEL_ARMOR_MODE='emulator'`` (or ``'auto'`` fallback when ADC is
    absent in local dev / CI environments). Uses lexical token-clause policy evaluation from
    ``prompts/v1/model_armor_policy.yaml`` (zero prompt-injection regexes).
    """

    def _build_sanitization_result(self, text: str, *, is_user_prompt: bool) -> Any:
        from google.cloud import modelarmor_v1 as ma

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
        info_types = sorted({info_type for _, _, info_type in spans})

        any_match = pi_matched or malicious_uri_matched or sdp_matched
        match_state = (
            ma.FilterMatchState.MATCH_FOUND if any_match else ma.FilterMatchState.NO_MATCH_FOUND
        )
        pi_state = (
            ma.FilterMatchState.MATCH_FOUND if pi_matched else ma.FilterMatchState.NO_MATCH_FOUND
        )
        uri_state = (
            ma.FilterMatchState.MATCH_FOUND
            if malicious_uri_matched
            else ma.FilterMatchState.NO_MATCH_FOUND
        )
        sdp_state = (
            ma.FilterMatchState.MATCH_FOUND if sdp_matched else ma.FilterMatchState.NO_MATCH_FOUND
        )

        filter_results: dict[str, Any] = {
            "pi_and_jailbreak": ma.FilterResult(
                pi_and_jailbreak_filter_result=ma.PiAndJailbreakFilterResult(
                    execution_state=ma.FilterExecutionState.EXECUTION_SUCCESS,
                    match_state=pi_state,
                    confidence_level=(
                        ma.DetectionConfidenceLevel.HIGH
                        if pi_matched
                        else ma.DetectionConfidenceLevel.LOW_AND_ABOVE
                    ),
                )
            ),
            "malicious_uris": ma.FilterResult(
                malicious_uri_filter_result=ma.MaliciousUriFilterResult(
                    execution_state=ma.FilterExecutionState.EXECUTION_SUCCESS,
                    match_state=uri_state,
                )
            ),
            "rai": ma.FilterResult(
                rai_filter_result=ma.RaiFilterResult(
                    execution_state=ma.FilterExecutionState.EXECUTION_SUCCESS,
                    match_state=ma.FilterMatchState.NO_MATCH_FOUND,
                )
            ),
            "sdp": ma.FilterResult(
                sdp_filter_result=ma.SdpFilterResult(
                    deidentify_result=ma.SdpDeidentifyResult(
                        execution_state=ma.FilterExecutionState.EXECUTION_SUCCESS,
                        match_state=sdp_state,
                        data=ma.DataItem(text=redacted_text),
                        info_types=info_types,
                    )
                )
            ),
        }

        return ma.SanitizationResult(
            filter_match_state=match_state,
            invocation_result=ma.InvocationResult.SUCCESS,
            filter_results=filter_results,
        )

    def sanitize_user_prompt(self, request: Any, *, timeout: float | None = None) -> Any:
        from google.cloud import modelarmor_v1 as ma

        _ = timeout
        data = getattr(request, "user_prompt_data", None)
        text = str(getattr(data, "text", "") or "")
        return ma.SanitizeUserPromptResponse(
            sanitization_result=self._build_sanitization_result(text, is_user_prompt=True)
        )

    def sanitize_model_response(self, request: Any, *, timeout: float | None = None) -> Any:
        from google.cloud import modelarmor_v1 as ma

        _ = timeout
        data = getattr(request, "model_response_data", None)
        text = str(getattr(data, "text", "") or "")
        return ma.SanitizeModelResponseResponse(
            sanitization_result=self._build_sanitization_result(text, is_user_prompt=False)
        )
