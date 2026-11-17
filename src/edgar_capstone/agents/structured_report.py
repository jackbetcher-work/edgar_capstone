"""Verification and rendering of FinancialAnalystAgent's structured `AnalystReport`.

The analyst answers with constrained JSON (`AnalystReport`) instead of free-form Markdown,
which makes the answer checkable claim by claim:

- **Key metrics** are references (ticker + metric). The figures are filled in from the
  deterministic variance engine, so the model never writes those numbers itself.
- **Findings** each carry `chunk_ids` and a verbatim `evidence_quote`. A finding is kept only
  when every chunk_id resolves to a passage retrieved this turn, the quote appears in one of
  the cited passages, and every figure in the statement appears in the quote or in the
  deterministic payload. Rejected findings are reported, not shown.
- **Headline and notes** are checked for figures that can't be traced to the payload or a
  verified quote. Unsupported figures are removed (headline replaced, note sentences dropped).

`render_analyst_report_markdown` turns the verified report into the compact Markdown that the
UI, fact-check, and judge consume. Inline citation tags stay in
`[TICKER FYxxxx | Section Title | chunk_id](sec_filing_url)` form.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from edgar_capstone.agents.material_variances import material_flag_payload_figures
from edgar_capstone.agents.schemas import AnalystFinding, AnalystMetricReference, AnalystReport
from edgar_capstone.agents.tools import get_full_10k_section

MAX_FINDINGS = 6
MAX_KEY_METRICS = 6
MAX_CHUNK_IDS_PER_FINDING = 3
MAX_HEADLINE_CHARS = 300
MAX_NOTES_CHARS = 900
MIN_QUOTE_CHARS = 20

_METRIC_LABELS = {
    "revenue": "Revenue",
    "operating_income": "Operating Income",
    "net_income": "Net Income",
    "research_and_development": "R&D Expense",
}
_SECTION_TITLES = {"item_1a": "Item 1A. Risk Factors", "item_7": "Item 7. MD&A"}
_CHUNK_SECTION_RE = re.compile(r"^(?P<key>.+_\d{4}_(?:item_1a|item_7))_\d+$")

# Figures worth verifying: currency amounts, magnitudes, percentages, and basis points.
# Bare integers (years, counts, list numbers) are ignored.
_FIGURE_RE = re.compile(
    r"(?P<dollar>\$)?\s?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s?"
    r"(?P<unit>%|percent\b|basis points\b|bps\b|trillion\b|billion\b|million\b|thousand\b"
    r"|[BMK]\b)?",
    re.IGNORECASE,
)
_MAGNITUDES = {
    "trillion": 1e12,
    "billion": 1e9,
    "b": 1e9,
    "million": 1e6,
    "m": 1e6,
    "thousand": 1e3,
    "k": 1e3,
}
FigureKind = Literal["amount", "pct", "bps"]


@dataclass(frozen=True)
class Figure:
    kind: FigureKind
    value: float
    text: str


def extract_figures(text: str) -> list[Figure]:
    """Extract verifiable figures ($ amounts, magnitudes, %, bps) from text."""
    # The 10-K corpus is Markdown-escaped ("$4\\.5 billion"); quotes copied from it may be too.
    text = (text or "").replace("\\", "")
    figures: list[Figure] = []
    for m in _FIGURE_RE.finditer(text):
        dollar, unit = m.group("dollar"), (m.group("unit") or "").lower()
        if not dollar and not unit:
            continue
        start = m.start("num") if not dollar else m.start()
        if start > 0 and text[start - 1].isalpha():
            continue  # identifiers such as FY2026, Q1, H20, A800
        value = float(m.group("num").replace(",", ""))
        if unit in ("%", "percent"):
            figures.append(Figure("pct", value, m.group(0).strip()))
        elif unit in ("bps", "basis points"):
            figures.append(Figure("bps", value, m.group(0).strip()))
        else:
            figures.append(Figure("amount", value * _MAGNITUDES.get(unit, 1.0), m.group(0).strip()))
    return figures


def _figures_match(a: Figure, b: Figure) -> bool:
    if a.kind != b.kind:
        return False
    if a.kind == "amount":
        return abs(a.value - b.value) <= max(0.005 * max(abs(a.value), abs(b.value)), 0.5)
    if a.kind == "pct":
        return abs(abs(a.value) - abs(b.value)) <= 0.05
    return abs(abs(a.value) - abs(b.value)) <= 0.5


def _payload_figures(
    variance_reports: list[dict[str, Any]],
    material_flags: list[dict[str, Any]] | None = None,
) -> list[Figure]:
    """Every figure the deterministic engine produced, for tolerance matching.

    Includes figures from deterministic material-variance flags (YoY margin bps shifts, sector
    medians, sign-reversal amounts) so an analyst finding may quote them.
    """
    out: list[Figure] = [
        Figure(kind, value, "material_flag")  # type: ignore[arg-type]
        for kind, value in material_flag_payload_figures(material_flags)
    ]
    for rep in variance_reports:
        for key in (
            "revenue_variance",
            "operating_income_variance",
            "net_income_variance",
            "research_and_development_variance",
        ):
            mv = rep.get(key) or {}
            for amount_key in ("start_value", "end_value", "absolute_delta"):
                if mv.get(amount_key) is not None:
                    out.append(Figure("amount", abs(float(mv[amount_key])), amount_key))
            for pct_key in ("percentage_delta", "cagr_pct"):
                if mv.get(pct_key) is not None:
                    out.append(Figure("pct", float(mv[pct_key]), pct_key))
        for bps_key in (
            "operating_margin_bps_delta",
            "net_margin_bps_delta",
            "rd_intensity_bps_delta",
        ):
            if rep.get(bps_key) is not None:
                out.append(Figure("bps", float(rep[bps_key]), bps_key))
        for step in rep.get("annual_series") or []:
            for metric in ("revenue", "operating_income", "net_income", "research_and_development"):
                if step.get(metric) is not None:
                    out.append(Figure("amount", abs(float(step[metric])), metric))
                if step.get(f"{metric}_yoy_delta") is not None:
                    out.append(Figure("amount", abs(float(step[f"{metric}_yoy_delta"])), metric))
                if step.get(f"{metric}_yoy_pct") is not None:
                    out.append(Figure("pct", float(step[f"{metric}_yoy_pct"]), metric))
    return out


def _payload_pct_matches(fig: Figure, payload: list[Figure]) -> bool:
    # Same 0.6-point tolerance as verify_narrative_against_calculations for percentages.
    if fig.kind == "pct":
        return any(p.kind == "pct" and abs(abs(p.value) - abs(fig.value)) <= 0.6 for p in payload)
    return any(_figures_match(fig, p) for p in payload)


def _normalize(text: str) -> str:
    text = (text or "").replace("\\", "")
    text = text.translate(
        str.maketrans(
            {
                "\u2019": "'",
                "\u2018": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\u2014": "-",
                "\u2013": "-",
                "*": "",
                "_": " ",
            }
        )
    )
    return " ".join(text.lower().split())


# ---------------------------------------------------------------------------
# Verification results
# ---------------------------------------------------------------------------


class FindingVerification(BaseModel):
    index: int
    verified: bool
    reasons: list[str] = Field(default_factory=list)
    quote_located_in: Literal["chunk", "section"] | None = None


class ReportVerification(BaseModel):
    findings_total: int = 0
    findings_verified: int = 0
    findings: list[FindingVerification] = Field(default_factory=list)
    headline_verified: bool = True
    headline_reasons: list[str] = Field(default_factory=list)
    metrics_dropped: list[str] = Field(default_factory=list)
    notes_removed_sentences: list[str] = Field(default_factory=list)
    usable: bool = True
    unusable_reason: str | None = None


@dataclass
class VerifiedAnalystReport:
    report: AnalystReport
    verification: ReportVerification
    verified_findings: list[AnalystFinding] = field(default_factory=list)


def parse_analyst_report(raw: Any) -> AnalystReport | None:
    """Parse the ADK output_key value (dict or JSON string) into an `AnalystReport`."""
    if raw is None or raw == "":
        return None
    try:
        if isinstance(raw, AnalystReport):
            return raw
        if isinstance(raw, str):
            return AnalystReport.model_validate_json(raw)
        return AnalystReport.model_validate(raw)
    except ValidationError:
        return None


def _trim_words(text: str, max_chars: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + "…"


def _locate_quote(
    quote_norm: str, chunk_ids: list[str], by_id: dict[str, dict[str, Any]]
) -> Literal["chunk", "section"] | None:
    for cid in chunk_ids:
        if quote_norm in _normalize(str(by_id[cid].get("excerpt") or "")):
            return "chunk"
    # Passages from query_cached_10k_context are paragraphs of the full section; a paragraph
    # can straddle chunk boundaries, so accept a quote found in the cited chunk's section.
    for cid in chunk_ids:
        m = _CHUNK_SECTION_RE.match(cid)
        entry = get_full_10k_section(m["key"]) if m else None
        if entry:
            if quote_norm in _normalize(str(entry.get("full_text") or "")):
                return "section"
    return None


def verify_analyst_report(
    report: AnalystReport,
    *,
    variance_reports: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    material_flags: list[dict[str, Any]] | None = None,
) -> VerifiedAnalystReport:
    """Check every claim in the report against retrieved passages and deterministic figures."""
    by_id = {str(c.get("chunk_id")): c for c in citations if c.get("chunk_id")}
    payload = _payload_figures(variance_reports, material_flags)
    payload_tickers = {str(r.get("ticker", "")).upper() for r in variance_reports}

    trimmed = AnalystReport(
        headline=_trim_words(report.headline, MAX_HEADLINE_CHARS),
        key_metrics=report.key_metrics[:MAX_KEY_METRICS],
        findings=[
            f.model_copy(
                update={"chunk_ids": list(dict.fromkeys(f.chunk_ids))[:MAX_CHUNK_IDS_PER_FINDING]}
            )
            for f in report.findings[:MAX_FINDINGS]
        ],
        notes=report.notes,
    )
    verification = ReportVerification(findings_total=len(trimmed.findings))
    verified: list[AnalystFinding] = []
    verified_quote_figures: list[Figure] = []

    for idx, finding in enumerate(trimmed.findings):
        reasons: list[str] = []
        if not finding.chunk_ids:
            reasons.append("no chunk_ids cited")
        unresolved = [cid for cid in finding.chunk_ids if cid not in by_id]
        if unresolved:
            reasons.append(f"chunk_ids not retrieved this turn: {unresolved}")
        quote_norm = _normalize(finding.evidence_quote)
        located: Literal["chunk", "section"] | None = None
        if len(quote_norm) < MIN_QUOTE_CHARS:
            reasons.append("evidence_quote missing or too short")
        elif not unresolved and finding.chunk_ids:
            located = _locate_quote(quote_norm, finding.chunk_ids, by_id)
            if located is None:
                reasons.append("evidence_quote not found in the cited passages")
        quote_figs = extract_figures(finding.evidence_quote)
        unsupported = [
            f.text
            for f in extract_figures(finding.statement)
            if not any(_figures_match(f, q) for q in quote_figs)
            and not _payload_pct_matches(f, payload)
        ]
        if unsupported:
            reasons.append(f"figures not in the quote or deterministic payload: {unsupported}")
        ok = not reasons
        verification.findings.append(
            FindingVerification(index=idx, verified=ok, reasons=reasons, quote_located_in=located)
        )
        if ok:
            verified.append(finding)
            verified_quote_figures.extend(quote_figs)

    verification.findings_verified = len(verified)

    def _traceable(fig: Figure) -> bool:
        return _payload_pct_matches(fig, payload) or any(
            _figures_match(fig, q) for q in verified_quote_figures
        )

    headline_bad = [f.text for f in extract_figures(trimmed.headline) if not _traceable(f)]
    if headline_bad or not trimmed.headline.strip():
        verification.headline_verified = False
        verification.headline_reasons = (
            [f"unsupported figures: {headline_bad}"] if headline_bad else ["empty headline"]
        )

    kept_notes: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", " ".join(trimmed.notes.split())):
        if not sentence:
            continue
        if any(not _traceable(f) for f in extract_figures(sentence)):
            verification.notes_removed_sentences.append(sentence)
        else:
            kept_notes.append(sentence)

    kept_metrics = []
    for ref in trimmed.key_metrics:
        if ref.ticker.upper() in payload_tickers:
            kept_metrics.append(ref)
        else:
            verification.metrics_dropped.append(f"{ref.ticker}:{ref.metric}")

    if verification.findings_total > 0 and not verified:
        verification.usable = False
        verification.unusable_reason = (
            "No finding in the structured report could be verified against the retrieved passages"
        )

    final = trimmed.model_copy(
        update={
            "key_metrics": kept_metrics,
            "findings": verified,
            "notes": _trim_words(" ".join(kept_notes), MAX_NOTES_CHARS),
        }
    )
    return VerifiedAnalystReport(
        report=final, verification=verification, verified_findings=verified
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt_usd(value: float | None) -> str:
    if value is None:
        return "N/A"
    sign = "-" if value < 0 else ""
    v = abs(value)
    if v >= 1e12:
        return f"{sign}${v / 1e12:.2f}T"
    if v >= 1e9:
        return f"{sign}${v / 1e9:.2f}B"
    if v >= 1e6:
        return f"{sign}${v / 1e6:.2f}M"
    return f"{sign}${v:,.0f}"


def _fmt_pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.2f}%"


def _citation_tag(cit: dict[str, Any], fallback_ticker: str, fallback_fy: int) -> str:
    ticker = cit.get("ticker") or fallback_ticker
    fy = cit.get("fiscal_year") or fallback_fy
    section = cit.get("section_title") or _SECTION_TITLES.get(
        str(cit.get("section_id") or ""), "10-K"
    )
    return (
        f"[{ticker} FY{fy} | {section} | {cit.get('chunk_id')}]({cit.get('sec_filing_url') or ''})"
    )


def render_analyst_report_markdown(
    verified: VerifiedAnalystReport,
    *,
    variance_reports: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    scope: dict[str, Any] | None = None,
) -> str:
    """Render the verified report as compact Markdown with inline citation tags."""
    report = verified.report
    scope = scope or {}
    by_id = {str(c.get("chunk_id")): c for c in citations if c.get("chunk_id")}
    reps = {str(r.get("ticker", "")).upper(): r for r in variance_reports}

    headline = report.headline
    if not verified.verification.headline_verified:
        headline = (
            f"Grounded analysis for {scope.get('tickers_csv') or ', '.join(reps) or 'the selected companies'}, "
            f"FY{scope.get('start_year', '')}–FY{scope.get('end_year', '')}"
        )
    lines = [f"### {headline}", ""]

    metric_refs = list(report.key_metrics)
    if not metric_refs:
        # Default: the three core metrics for up to two companies in the payload.
        metric_refs = [
            AnalystMetricReference(ticker=t, metric=m)
            for t in list(reps)[:2]
            for m in ("revenue", "operating_income", "net_income")
        ]
    metric_lines: list[str] = []
    for ref in metric_refs:
        rep = reps.get(ref.ticker.upper())
        mv = (rep or {}).get(f"{ref.metric}_variance") or {}
        if not rep or mv.get("start_value") is None or mv.get("end_value") is None:
            continue
        note = (
            f" ({_trim_words(ref.note, 80)})" if ref.note and not extract_figures(ref.note) else ""
        )
        metric_lines.append(
            f"- **{rep.get('company_name', ref.ticker)} ({ref.ticker.upper()}) · "
            f"{_METRIC_LABELS[ref.metric]}:** {_fmt_usd(mv.get('start_value'))} "
            f"(FY{mv.get('start_year', rep.get('start_year'))}) → {_fmt_usd(mv.get('end_value'))} "
            f"(FY{mv.get('end_year', rep.get('end_year'))}), **{_fmt_pct(mv.get('percentage_delta'))}** "
            f"(CAGR {_fmt_pct(mv.get('cagr_pct'))}){note}"
        )
    if metric_lines:
        lines += ["#### Key Metrics (deterministic engine)", *metric_lines, ""]

    if report.findings:
        lines.append("#### Findings")
        for f in report.findings:
            tags = " ".join(
                _citation_tag(by_id[cid], f.ticker, f.fiscal_year)
                for cid in f.chunk_ids
                if cid in by_id
            )
            lines.append(f"- {' '.join(f.statement.split())} {tags}".rstrip())
            lines.append(f'  - *Evidence:* "{" ".join(f.evidence_quote.split())}"')
        lines.append("")

    if report.notes:
        lines += ["#### Notes (not source-verified)", report.notes, ""]
    return "\n".join(lines).strip() + "\n"
