"""Sensitive-span scanner (non-regex) and SEC Financial Projection & MNPI compliance filter."""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Any, Literal

from edgar_capstone.security.model_armor.config import (
    _SAFE_EMAIL_DOMAIN_SUFFIXES,
    _load_guardrail_policy_dict,
    _normalize_words,
)

_EMAIL_LOCAL_CHARS = frozenset(string.ascii_letters + string.digits + "._%+-")
_EMAIL_DOMAIN_CHARS = frozenset(string.ascii_letters + string.digits + ".-")
_TOKEN_CHARS = frozenset(string.ascii_letters + string.digits + "-_.")

_STRUCTURED_QUERY_HEADERS: tuple[str, ...] = (
    "## Analyst Query\n",
    "### Analyst Query\n",
    "## User Query\n",
    "### User Query\n",
    "## Query\n",
    "### Query\n",
)

_POLICY_DISCLOSURE_TERMS: tuple[str, ...] = (
    "insider trading",
    "item 10",
    "item 408",
    "10b5-1",
    "code of ethics",
)

_REFUSAL_RESPONSE_MARKERS: tuple[str, ...] = (
    "cannot forecast",
    "cannot predict",
    "cannot project",
    "unable to forecast",
    "unable to predict",
    "unable to project",
    "restricted to verified historical",
)


# ---------------------------------------------------------------------------
# Structural PII / Credential Inspection Helpers (Zero Regexes)
# ---------------------------------------------------------------------------


def format_redaction_tag(info_type: str) -> str:
    """Return the canonical Cloud DLP `replaceWithInfoTypeConfig` redaction placeholder."""
    return f"[{info_type}]"


def _luhn_checksum_valid(digits: str) -> bool:
    total = 0
    for idx, ch in enumerate(reversed(digits)):
        n = ord(ch) - 48
        if idx % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _scan_ssn_spans(text: str) -> list[tuple[int, int, str]]:
    """Detect US SSNs (`XXX-XX-XXXX`) with boundary guards avoiding SEC accession numbers."""
    spans: list[tuple[int, int, str]] = []
    n = len(text)
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
    return spans


def _scan_credit_card_spans(text: str) -> list[tuple[int, int, str]]:
    """Detect 13-19 digit credit card numbers passing major IIN prefix and Luhn checks."""
    spans: list[tuple[int, int, str]] = []
    n = len(text)
    idx = 0
    while idx < n:
        if not (text[idx].isdigit() and (idx == 0 or not text[idx - 1].isalnum())):
            idx += 1
            continue
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
    return spans


def _scan_phone_spans(text: str) -> list[tuple[int, int, str]]:
    """Detect US phone numbers in `(XXX) XXX-XXXX` and `+1-XXX-XXX-XXXX` formats."""
    spans: list[tuple[int, int, str]] = []
    n = len(text)
    for i in range(n):
        if text[i] == "(" and i + 13 <= n:
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
    return spans


def _scan_email_spans(text: str) -> list[tuple[int, int, str]]:
    """Detect personal/external email addresses, exempting GCP SAs and SEC domains."""
    spans: list[tuple[int, int, str]] = []
    n = len(text)
    for pos, ch in enumerate(text):
        if ch != "@":
            continue
        left = pos
        while left > 0 and text[left - 1] in _EMAIL_LOCAL_CHARS:
            left -= 1
        right = pos + 1
        while right < n and text[right] in _EMAIL_DOMAIN_CHARS:
            right += 1
        while right > pos + 1 and text[right - 1] in ".-":
            right -= 1
        local_part = text[left:pos]
        domain_part = text[pos + 1 : right]
        if (
            local_part
            and "." in domain_part
            and len(domain_part.rsplit(".", 1)[-1]) >= 2
            and not domain_part.lower().endswith(_SAFE_EMAIL_DOMAIN_SUFFIXES)
        ):
            spans.append((left, right, "EMAIL_ADDRESS"))
    return spans


def _scan_credential_spans(text: str) -> list[tuple[int, int, str]]:
    """Detect leaked GCP API keys, OAuth tokens, GitHub PATs, and AWS access key IDs."""
    spans: list[tuple[int, int, str]] = []
    n = len(text)
    idx = 0
    while idx < n:
        if text[idx] not in _TOKEN_CHARS:
            idx += 1
            continue
        end = idx
        while end < n and text[end] in _TOKEN_CHARS:
            end += 1
        tok = text[idx:end]
        if tok.startswith("AIza") and len(tok) >= 33:
            spans.append((idx, end, "GCP_API_KEY"))
        elif tok.startswith("ya29.") and len(tok) >= 24:
            spans.append((idx, end, "AUTH_TOKEN"))
        elif tok.startswith(("ghp_", "gho_", "ghs_", "github_pat_")) and len(tok) >= 28:
            spans.append((idx, end, "AUTH_TOKEN"))
        elif tok.startswith(("AKIA", "ASIA")) and len(tok) == 20 and tok.isalnum():
            spans.append((idx, end, "AUTH_TOKEN"))
        idx = end
    return spans


def _merge_non_overlapping_spans(
    spans: list[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    if not spans:
        return []
    spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    merged: list[tuple[int, int, str]] = []
    for start, end, info_type in spans:
        if merged and start < merged[-1][1]:
            continue
        merged.append((start, end, info_type))
    return merged


def _detect_sensitive_spans(text: str) -> list[tuple[int, int, str]]:
    """Locate PII and credential spans `(start, end, info_type)` for SDP inspection/redaction."""
    if not text:
        return []
    spans: list[tuple[int, int, str]] = []
    spans.extend(_scan_ssn_spans(text))
    spans.extend(_scan_credit_card_spans(text))
    spans.extend(_scan_phone_spans(text))
    spans.extend(_scan_email_spans(text))
    spans.extend(_scan_credential_spans(text))
    return _merge_non_overlapping_spans(spans)


def _redact_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    """Replace sensitive spans using canonical `[INFO_TYPE]` placeholders."""
    if not spans:
        return text
    out: list[str] = []
    cursor = 0
    for start, end, info_type in spans:
        out.append(text[cursor:start])
        out.append(format_redaction_tag(info_type))
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


def _find_token_sequence_positions(words: list[str], phrase_tokens: list[str]) -> list[int]:
    """Return start indices where `phrase_tokens` appears contiguously inside `words`."""
    k = len(phrase_tokens)
    if k == 0 or k > len(words):
        return []
    first = phrase_tokens[0]
    return [
        idx
        for idx in range(len(words) - k + 1)
        if words[idx] == first and words[idx : idx + k] == phrase_tokens
    ]


def _find_future_projection_rule(
    words: list[str],
    policy: dict[str, Any],
    *,
    stage: Literal["prompt", "response"] = "prompt",
    window: int = 8,
    prompt_window: int = 14,
) -> str | None:
    future_years = {str(y).strip() for y in policy.get("future_fiscal_years", [])}
    future_year_tokens = future_years.union({f"fy{y}" for y in future_years})
    proj_terms = {str(t).strip().lower() for t in policy.get("projection_verbs_and_nouns", [])}

    year_positions = [(i, tok) for i, tok in enumerate(words) if tok in future_year_tokens]
    if not year_positions:
        return None

    proj_positions = [(i, tok) for i, tok in enumerate(words) if tok in proj_terms]
    for y_idx, y_tok in year_positions:
        for p_idx, p_tok in proj_positions:
            if abs(y_idx - p_idx) <= window:
                clean_yr = y_tok[2:] if y_tok.startswith("fy") else y_tok
                return f"future_period_projection:FY{clean_yr}+{p_tok}"

    if stage == "prompt":
        prompt_cues = {str(t).strip().lower() for t in policy.get("future_prompt_cues", [])}
        cue_positions = [(i, tok) for i, tok in enumerate(words) if tok in prompt_cues]
        for y_idx, y_tok in year_positions:
            for c_idx, c_tok in cue_positions:
                if abs(y_idx - c_idx) <= prompt_window:
                    clean_yr = y_tok[2:] if y_tok.startswith("fy") else y_tok
                    return f"future_period_projection:FY{clean_yr}+{c_tok}"
    return None


def _find_relative_future_projection_rule(
    words: list[str],
    policy: dict[str, Any],
    *,
    window: int = 14,
) -> str | None:
    """Detect forward-looking financial calls over relative horizons (e.g. 'next fiscal year')."""
    phrases = [str(p).strip().lower() for p in policy.get("relative_future_phrases", []) if p]
    if not phrases:
        return None
    metric_terms = {str(t).strip().lower() for t in policy.get("financial_metric_terms", [])}
    spec_terms = {str(t).strip().lower() for t in policy.get("projection_verbs_and_nouns", [])} | {
        str(t).strip().lower() for t in policy.get("future_prompt_cues", [])
    }
    if not metric_terms or not spec_terms:
        return None

    metric_positions = [(i, tok) for i, tok in enumerate(words) if tok in metric_terms]
    spec_positions = [(i, tok) for i, tok in enumerate(words) if tok in spec_terms]
    if not metric_positions or not spec_positions:
        return None

    for phrase in phrases:
        p_toks = _normalize_words(phrase)
        for start_idx in _find_token_sequence_positions(words, p_toks):
            end_idx = start_idx + len(p_toks) - 1
            near_metric = next(
                (
                    tok
                    for m_idx, tok in metric_positions
                    if min(abs(m_idx - start_idx), abs(m_idx - end_idx)) <= window
                ),
                None,
            )
            near_spec = next(
                (
                    tok
                    for s_idx, tok in spec_positions
                    if min(abs(s_idx - start_idx), abs(s_idx - end_idx)) <= window
                ),
                None,
            )
            if near_metric and near_spec:
                return f"relative_future_projection:{phrase}+{near_metric}+{near_spec}"
    return None


class FinancialProjectionFilter:
    """Detects ungrounded forward-looking financial projections, target prices, and MNPI leaks."""

    @staticmethod
    def _extract_query_segment_if_structured_prompt(text: str) -> str:
        """If `text` is an internal ADK synthesis/eval prompt, scope projection checks to the user query."""
        for header in _STRUCTURED_QUERY_HEADERS:
            pos = text.find(header)
            if pos != -1:
                after = text[pos + len(header) :]
                for next_hdr in ("\n## ", "\n### "):
                    end_pos = after.find(next_hdr)
                    if end_pos != -1:
                        return after[:end_pos].strip()
                return after.strip()
        return text

    @staticmethod
    def _match_mnpi_markers(
        lowered: str,
        policy: dict[str, Any],
        *,
        stage: Literal["prompt", "response"],
    ) -> list[str]:
        is_10k_policy_disclosure = stage == "response" and any(
            term in lowered for term in _POLICY_DISCLOSURE_TERMS
        )
        matched: list[str] = []
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
                matched.append(f"mnpi:{marker_norm}")
        return matched

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

        # 1. Confidential / Non-Public MNPI markers
        mnpi_rules = self._match_mnpi_markers(lowered, policy, stage=stage)
        if mnpi_rules:
            return ProjectionFilterResult(
                matched=True,
                category="mnpi_confidential_leak",
                matched_rules=tuple(mnpi_rules),
                reason=(
                    f"Detected confidential MNPI or unreleased guidance marker in {stage}: "
                    f"{', '.join(mnpi_rules)}"
                ),
            )

        words = _normalize_words(target_text)
        matched_rules: list[str] = []

        # 2. Explicit stock price targets, buy/sell ratings, or ungrounded prediction phrases
        for phrase in policy.get("projection_phrases", []):
            phrase_norm = str(phrase).strip().lower()
            phrase_tokens = _normalize_words(phrase_norm)
            if phrase_tokens and _contains_token_sequence(words, phrase_tokens):
                matched_rules.append(f"projection_phrase:{phrase_norm}")

        # 3. Future fiscal-year projections beyond the verified 10-K window (FY2027+)
        #    and relative-horizon forward-looking calls ("next fiscal year", etc.) on prompts
        is_refusal_response = stage == "response" and any(
            ref in lowered for ref in _REFUSAL_RESPONSE_MARKERS
        )
        if not is_refusal_response:
            future_rule = _find_future_projection_rule(words, policy, stage=stage)
            if future_rule:
                matched_rules.append(future_rule)
            if stage == "prompt":
                relative_rule = _find_relative_future_projection_rule(words, policy)
                if relative_rule:
                    matched_rules.append(relative_rule)

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
