"""Golden eval runner: executes every golden item through the RAG pipeline, scores it, and
applies the regression gates from `evals/thresholds.yaml`.

Every run is live: answers come from the Gemini `FinancialAnalystAgent`, retrieval reads
BigQuery / Vertex AI Search / Vertex AI embeddings / GCS, and the Gemini 3.5 Flash judge scores
each answer against its reference explanation.

By default, items execute concurrently across a bounded worker pool (`DEFAULT_EVAL_CONCURRENCY`),
which runs sync GCP pre-pass I/O in parallel threads while multiplexing live ADK Gemini streams
asynchronously onto `_ADK_LOOP_MANAGER`. Concurrent requests (especially alongside sector
map-reduce batches) can inflate `p95_ttft_ms` due to client/loop contention and Vertex AI queueing,
so the `p95_ttft_ms` gate is automatically skipped whenever more than one item runs concurrently
(the observed `p95_ttft_ms` is still recorded for visibility). Pass `max_workers=1`
(`--concurrency 1`) to run sequentially and enforce the `p95_ttft_ms` gate.
"""

from __future__ import annotations

import contextvars
import inspect
import json
import logging
import os
import platform
import subprocess
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from edgar_capstone.evals.golden import (
    REPO_ROOT,
    SCOPE_CATEGORY,
    GoldenEvalItem,
    dataset_fingerprint,
)
from edgar_capstone.evals.metrics import (
    percentile,
    score_anomaly_flags,
    score_citation_coverage,
    score_grounding,
    score_metadata_filter_correctness,
    score_numeric_agreement,
    score_numeric_citation_coverage,
    score_retrieval_precision_at_10,
    score_routing,
    score_safety,
    score_scope_or_coverage,
    score_selective_display,
    score_unsupported_claims,
    score_wording_delta_accuracy,
)

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS_PATH = REPO_ROOT / "evals" / "thresholds.yaml"
DEFAULT_EVAL_CONCURRENCY = 4


@dataclass(frozen=True)
class EvalThresholds:
    numeric_agreement: float = 1.0
    citation_coverage: float = 1.0
    numeric_citation_coverage: float = 1.0
    grounding_recall: float = 0.92
    metadata_filter_correctness: float = 1.0
    retrieval_precision_at_10: float = 0.85
    wording_delta_accuracy: float = 0.90
    anomaly_flag_recall: float = 1.0
    selective_display_accuracy: float = 0.9
    scope_and_coverage_accuracy: float = 1.0
    routing_accuracy: float = 1.0
    judge_faithfulness: float = 0.85
    judge_citation_precision: float = 0.95
    judge_explanation_quality: float = 0.75
    judge_thematic_consistency: float = 0.75
    unsupported_claims_total: int = 0
    safety_compliance: float = 1.0
    p95_ttft_ms: float = 3000.0
    narrative_pct_tolerance: float = 0.1
    min_scored_items: int = 20

    @classmethod
    def load(cls, path: Path | str | None = None) -> EvalThresholds:
        """Load threshold overrides from YAML.

        Raises `FileNotFoundError` when an explicit `path` argument is supplied and does not
        exist; falls back to `cls()` defaults only when `path is None` and the default repo
        file is absent.
        """
        if path is not None:
            resolved = Path(path)
            if not resolved.exists():
                raise FileNotFoundError(f"Eval thresholds file not found: {resolved}")
        else:
            resolved = DEFAULT_THRESHOLDS_PATH
            if not resolved.exists():
                return cls()

        raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        gates = raw.get("gates", raw)
        defaults = cls()
        known = set(cls.__dataclass_fields__)
        unknown = set(gates) - known
        if unknown:
            raise ValueError(f"Unknown threshold keys in {resolved}: {sorted(unknown)}")
        return cls(**{k: type(getattr(defaults, k))(v) for k, v in gates.items()})


@dataclass
class ItemResult:
    item_id: str
    category: str
    query: str
    passed: bool = False
    status: str = ""
    intent: str | None = None
    intent_source: str | None = None
    execution_mode: str | None = None
    numeric_agreement: bool | None = None
    citation_coverage: bool | None = None
    numeric_citation_coverage: bool | None = None
    routing_ok: bool | None = None
    grounding_matched: int = 0
    grounding_expected: int = 0
    metadata_filter_ok: bool | None = None
    metadata_filter_ratio: float | None = None
    retrieval_precision_at_10: float | None = None
    delta_matched: int = 0
    delta_expected: int = 0
    flag_matched: int = 0
    flag_expected: int = 0
    selective_display_ok: bool | None = None
    scope_check_ok: bool | None = None
    is_scope_early_exit: bool = False
    unsupported_claims_count: int = 0
    faithfulness_score: float | None = None
    citation_precision_score: float | None = None
    explanation_quality_score: float | None = None
    thematic_consistency_score: float | None = None
    judge_grade: str | None = None
    judge_model: str | None = None
    safety_blocked: bool | None = None
    ttft_ms: float | None = None
    total_latency_ms: float | None = None
    citations_count: int = 0
    failure_reasons: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def grounding_recall(self) -> float | None:
        if self.grounding_expected == 0:
            return None
        return round(self.grounding_matched / self.grounding_expected, 4)

    @property
    def delta_accuracy(self) -> float | None:
        if self.delta_expected == 0:
            return None
        return round(self.delta_matched / self.delta_expected, 4)

    @property
    def flag_recall(self) -> float | None:
        if self.flag_expected == 0:
            return None
        return round(self.flag_matched / self.flag_expected, 4)


@dataclass
class GateResult:
    name: str
    value: float | None
    threshold: float
    comparator: Literal[">=", "<="]
    passed: bool
    skipped: bool = False
    detail: str = ""


@dataclass
class EvalRunReport:
    run_id: str
    started_at: str
    finished_at: str
    eval_mode: str
    dataset_source: str
    dataset_version: str
    corpus_kind: str
    commit_sha: str
    environment: str
    analyst_model: str | None
    judge_model: str | None
    item_results: list[ItemResult]
    metrics: dict[str, float | int | None]
    gates: list[GateResult]
    thresholds: EvalThresholds
    duration_seconds: float
    max_workers: int = 1

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.gates)

    @property
    def regressions(self) -> list[str]:
        return [g.name for g in self.gates if not g.passed]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["passed"] = self.passed
        payload["regressions"] = self.regressions
        for row, res in zip(payload["item_results"], self.item_results, strict=True):
            row["grounding_recall"] = res.grounding_recall
            row["delta_accuracy"] = res.delta_accuracy
            row["flag_recall"] = res.flag_recall
        return payload

    def to_bigquery_row(self) -> dict[str, Any]:
        """Flatten into the `sec_edgar.eval_results` schema (terraform/schemas/eval_results.json)."""
        return {
            "run_id": self.run_id,
            "recorded_at": self.finished_at,
            "started_at": self.started_at,
            "eval_mode": self.eval_mode,
            "passed": self.passed,
            "regressions": self.regressions,
            "commit_sha": self.commit_sha,
            "environment": self.environment,
            "dataset_source": self.dataset_source,
            "dataset_version": self.dataset_version,
            "corpus_kind": self.corpus_kind,
            "analyst_model": self.analyst_model,
            "judge_model": self.judge_model,
            "item_count": len(self.item_results),
            "items_passed": sum(1 for r in self.item_results if r.passed),
            **{k: v for k, v in self.metrics.items()},
            "gates": [
                {
                    "name": g.name,
                    "value": g.value,
                    "threshold": g.threshold,
                    "comparator": g.comparator,
                    "passed": g.passed,
                    "skipped": g.skipped,
                    "detail": g.detail,
                }
                for g in self.gates
            ],
            "item_results": [
                {
                    "item_id": r.item_id,
                    "category": r.category,
                    "passed": r.passed,
                    "status": r.status,
                    "intent": r.intent,
                    "numeric_agreement": r.numeric_agreement,
                    "citation_coverage": r.citation_coverage,
                    "numeric_citation_coverage": r.numeric_citation_coverage,
                    "routing_ok": r.routing_ok,
                    "grounding_recall": r.grounding_recall,
                    "metadata_filter_ratio": r.metadata_filter_ratio,
                    "retrieval_precision_at_10": r.retrieval_precision_at_10,
                    "delta_accuracy": r.delta_accuracy,
                    "flag_recall": r.flag_recall,
                    "selective_display_ok": r.selective_display_ok,
                    "scope_check_ok": r.scope_check_ok,
                    "unsupported_claims_count": r.unsupported_claims_count,
                    "faithfulness_score": r.faithfulness_score,
                    "citation_precision_score": r.citation_precision_score,
                    "explanation_quality_score": r.explanation_quality_score,
                    "thematic_consistency_score": r.thematic_consistency_score,
                    "judge_grade": r.judge_grade,
                    "safety_blocked": r.safety_blocked,
                    "ttft_ms": r.ttft_ms,
                    "total_latency_ms": r.total_latency_ms,
                    "failure_reasons": r.failure_reasons[:20],
                }
                for r in self.item_results
            ],
            "duration_seconds": round(self.duration_seconds, 3),
            "runner_host": platform.node()[:120],
        }


TurnFn = Callable[..., dict[str, Any]]
JudgeFn = Callable[..., Any]


def detect_environment() -> str:
    if os.environ.get("BUILD_ID") and os.environ.get("PROJECT_ID"):
        return "cloud_build"
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return "github_actions"
    return "local"


def detect_commit_sha() -> str:
    for var in ("COMMIT_SHA", "GITHUB_SHA", "SHORT_SHA"):
        if os.environ.get(var):
            return os.environ[var]
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
            ).strip()
            or "unknown"
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _default_turn_fn() -> TurnFn:
    from edgar_capstone.agents.chat_engine import run_conversational_rag_turn

    return run_conversational_rag_turn


def _default_judge_fn() -> JudgeFn:
    from edgar_capstone.agents.evaluation import evaluate_variance_explanation

    return evaluate_variance_explanation


def _supports_kwarg(fn: Callable[..., Any], name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    if name in sig.parameters:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def _mark_scored_failure(
    res: ItemResult,
    item: GoldenEvalItem,
    *,
    score_chunk_ids: bool,
) -> None:
    """Populate deterministic denominators on failed/errored items so aggregates aren't inflated."""
    if item.is_safety:
        res.safety_blocked = False
        return
    if item.is_scope_early_exit:
        res.scope_check_ok = False
        return

    res.numeric_agreement = False if item.expected_variances else None
    res.citation_coverage = False
    res.numeric_citation_coverage = False if item.expected_variances else None
    res.routing_ok = False
    res.metadata_filter_ok = False
    res.metadata_filter_ratio = 0.0
    res.retrieval_precision_at_10 = 0.0
    _, res.grounding_expected, _ = score_grounding(item, {}, score_chunk_ids=score_chunk_ids)
    res.delta_expected = len(item.expected_deltas)
    res.flag_expected = len(item.expected_flags)
    if item.category == SCOPE_CATEGORY or item.expected_coverage_notice_codes:
        res.scope_check_ok = False


def _score_deterministic_checks(
    res: ItemResult,
    item: GoldenEvalItem,
    response: dict[str, Any],
    *,
    thresholds: EvalThresholds,
    score_chunk_ids: bool,
) -> None:
    """Populate all pure deterministic metrics on `res` for a successful scored RAG turn."""
    res.routing_ok, reasons = score_routing(item, response)
    res.failure_reasons.extend(reasons)

    res.numeric_agreement, reasons = score_numeric_agreement(
        item, response, narrative_pct_tolerance=thresholds.narrative_pct_tolerance
    )
    res.failure_reasons.extend(reasons)

    res.citation_coverage, reasons = score_citation_coverage(item, response)
    res.failure_reasons.extend(reasons)

    res.numeric_citation_coverage, reasons = score_numeric_citation_coverage(item, response)
    res.failure_reasons.extend(reasons)

    res.grounding_matched, res.grounding_expected, missing = score_grounding(
        item, response, score_chunk_ids=score_chunk_ids
    )
    if missing:
        res.failure_reasons.append(f"grounding missed: {missing}")

    res.metadata_filter_ok, res.metadata_filter_ratio, reasons = score_metadata_filter_correctness(
        item, response
    )
    if not res.metadata_filter_ok:
        res.failure_reasons.extend(f"metadata_filter: {r}" for r in reasons)

    res.retrieval_precision_at_10 = score_retrieval_precision_at_10(item, response)

    res.delta_matched, res.delta_expected, missing_deltas = score_wording_delta_accuracy(
        item, response
    )
    if missing_deltas:
        res.failure_reasons.append(f"deltas missed: {missing_deltas}")

    res.flag_matched, res.flag_expected, missing_flags = score_anomaly_flags(item, response)
    if missing_flags:
        res.failure_reasons.append(f"anomaly flags missed: {missing_flags}")

    res.selective_display_ok, reasons = score_selective_display(item, response)
    res.failure_reasons.extend(reasons)

    if item.category == SCOPE_CATEGORY or item.expected_coverage_notice_codes:
        ok, reasons = score_scope_or_coverage(item, response)
        res.scope_check_ok = ok
        res.failure_reasons.extend(reasons)

    res.unsupported_claims_count, unsup_reasons = score_unsupported_claims(response)
    res.failure_reasons.extend(unsup_reasons)


def _score_with_judge(
    res: ItemResult,
    item: GoldenEvalItem,
    response: dict[str, Any],
    *,
    judge_fn: JudgeFn,
) -> None:
    """Invoke the LLM judge and populate quality/faithfulness scores on `res`."""
    try:
        cached_count = len(
            (response.get("react_execution") or {}).get("cached_10k_explanations") or []
        )
        report = judge_fn(
            query=item.query,
            response_markdown=response.get("answer_markdown") or "",
            variance_reports=response.get("variance_reports") or [],
            citations=response.get("citations") or [],
            cached_explanations_count=cached_count,
            reference_explanation=item.reference_explanation,
        )
        judged = report.model_dump(mode="json") if hasattr(report, "model_dump") else dict(report)
        res.faithfulness_score = round(
            (float(judged["numeric_fidelity_score"]) + float(judged["citation_grounding_score"]))
            / 2.0,
            4,
        )
        raw_prec = judged.get("citation_precision_score")
        res.citation_precision_score = (
            float(raw_prec) if raw_prec is not None else float(judged["citation_grounding_score"])
        )
        res.explanation_quality_score = float(judged["explanation_quality_score"])
        res.thematic_consistency_score = float(judged["thematic_consistency_score"])
        res.judge_grade = judged.get("grade")
        res.judge_model = judged.get("evaluator_model")
        if res.judge_grade == "fail":
            res.failure_reasons.append(f"judge grade=fail: {judged.get('rationale', '')[:200]}")
    except Exception as exc:
        res.failure_reasons.append(f"judge raised {type(exc).__name__}: {exc}")


def evaluate_item(
    item: GoldenEvalItem,
    *,
    run_id: str,
    thresholds: EvalThresholds,
    score_chunk_ids: bool,
    turn_fn: TurnFn,
    judge_fn: JudgeFn,
) -> ItemResult:
    res = ItemResult(
        item_id=item.id,
        category=item.category,
        query=item.query,
        is_scope_early_exit=item.is_scope_early_exit,
    )
    f = item.filters
    turn_kwargs: dict[str, Any] = {
        "query": item.query,
        "tickers": f.tickers,
        "gics_sector": f.gics_sector,
        "section_id": f.section_id,
        "start_year": f.start_year,
        "end_year": f.end_year,
        "top_k": f.top_k,
        "session_id": f"golden-eval-{run_id}-{item.id}",
        "user_id": "golden-eval-runner",
    }
    if _supports_kwarg(turn_fn, "skip_background_judge"):
        turn_kwargs["skip_background_judge"] = True

    try:
        response = turn_fn(**turn_kwargs)
    except Exception as exc:
        # One broken item must not abort the run, and its failure must count in every denominator.
        res.error = f"{type(exc).__name__}: {exc}"
        res.failure_reasons.append(f"turn raised {res.error}")
        res.status = "error"
        _mark_scored_failure(res, item, score_chunk_ids=score_chunk_ids)
        if os.environ.get("EDGAR_EVAL_TRACEBACKS"):
            traceback.print_exc()
        return res

    res.status = str(response.get("status"))
    res.intent = response.get("intent")
    res.intent_source = response.get("intent_source")
    res.execution_mode = response.get("execution_mode")
    res.total_latency_ms = response.get("total_latency_ms")
    res.citations_count = len(response.get("citations") or [])
    ttft = (response.get("react_execution") or {}).get("observed_ttft_ms")
    res.ttft_ms = float(ttft) if ttft is not None else None

    if item.is_safety:
        ok, reasons = score_safety(item, response)
        res.safety_blocked = response.get("status") == "blocked"
        res.failure_reasons.extend(reasons)
        res.passed = ok
        return res

    if item.is_scope_early_exit:
        ok, reasons = score_scope_or_coverage(item, response)
        res.scope_check_ok = ok
        res.failure_reasons.extend(reasons)
        res.passed = ok
        return res

    if res.status != "success":
        res.failure_reasons.append(f"unexpected status={res.status} (intent={res.intent})")
        _mark_scored_failure(res, item, score_chunk_ids=score_chunk_ids)
        return res

    _score_deterministic_checks(
        res,
        item,
        response,
        thresholds=thresholds,
        score_chunk_ids=score_chunk_ids,
    )
    _score_with_judge(res, item, response, judge_fn=judge_fn)

    res.passed = (
        bool(res.routing_ok)
        and res.numeric_agreement is not False
        and bool(res.citation_coverage)
        and res.numeric_citation_coverage is not False
        and bool(res.metadata_filter_ok)
        and res.grounding_matched == res.grounding_expected
        and res.delta_matched == res.delta_expected
        and res.flag_matched == res.flag_expected
        and res.selective_display_ok is not False
        and res.scope_check_ok is not False
        and res.unsupported_claims_count == 0
        and res.faithfulness_score is not None
        and res.judge_grade != "fail"
    )
    return res


def _ratio(values: list[bool]) -> float | None:
    return round(sum(1 for v in values if v) / len(values), 4) if values else None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def aggregate_metrics(results: list[ItemResult]) -> dict[str, float | int | None]:
    scored = [r for r in results if r.category != "safety_red_team" and not r.is_scope_early_exit]
    safety = [r for r in results if r.category == "safety_red_team"]
    scope_checks = [r for r in results if r.scope_check_ok is not None]

    g_matched = sum(r.grounding_matched for r in scored)
    g_expected = sum(r.grounding_expected for r in scored)
    d_matched = sum(r.delta_matched for r in scored)
    d_expected = sum(r.delta_expected for r in scored)
    f_matched = sum(r.flag_matched for r in scored)
    f_expected = sum(r.flag_expected for r in scored)

    ttfts = [r.ttft_ms for r in scored if r.ttft_ms is not None]
    latencies = [r.total_latency_ms for r in results if r.total_latency_ms is not None]
    p10_vals = [
        r.retrieval_precision_at_10 for r in scored if r.retrieval_precision_at_10 is not None
    ]
    filter_vals = [r.metadata_filter_ratio for r in scored if r.metadata_filter_ratio is not None]

    def _judge(attr: str) -> float | None:
        # Items whose turn or judge failed score 0 so they drag the mean down instead of vanishing.
        return _mean([getattr(r, attr) if getattr(r, attr) is not None else 0.0 for r in scored])

    return {
        "numeric_agreement": _ratio(
            [bool(r.numeric_agreement) for r in scored if r.numeric_agreement is not None]
        ),
        "citation_coverage": _ratio([bool(r.citation_coverage) for r in scored]),
        "numeric_citation_coverage": _ratio(
            [
                bool(r.numeric_citation_coverage)
                for r in scored
                if r.numeric_citation_coverage is not None
            ]
        ),
        "grounding_recall": round(g_matched / g_expected, 4) if g_expected else None,
        "metadata_filter_correctness": _mean(filter_vals) if filter_vals else None,
        "retrieval_precision_at_10": _mean(p10_vals) if p10_vals else None,
        "wording_delta_accuracy": round(d_matched / d_expected, 4) if d_expected else None,
        "anomaly_flag_recall": round(f_matched / f_expected, 4) if f_expected else None,
        # Only successful scored turns contribute (errored / non-success items stay None).
        "selective_display_accuracy": _ratio(
            [bool(r.selective_display_ok) for r in scored if r.selective_display_ok is not None]
        ),
        "scope_and_coverage_accuracy": _ratio([bool(r.scope_check_ok) for r in scope_checks]),
        "routing_accuracy": _ratio([bool(r.routing_ok) for r in scored]),
        "judge_faithfulness": _judge("faithfulness_score"),
        "judge_citation_precision": _judge("citation_precision_score"),
        "judge_explanation_quality": _judge("explanation_quality_score"),
        "judge_thematic_consistency": _judge("thematic_consistency_score"),
        "unsupported_claims_total": sum(r.unsupported_claims_count for r in scored),
        "safety_compliance": _ratio([r.passed for r in safety]),
        "p95_ttft_ms": percentile(ttfts, 95) if ttfts else None,
        "p95_total_latency_ms": percentile(latencies, 95) if latencies else None,
        "item_pass_rate": _ratio([r.passed for r in results]),
    }


def evaluate_gates(
    metrics: dict[str, float | int | None],
    thresholds: EvalThresholds,
    *,
    skip_ttft_gate: bool = False,
    skip_ttft_reason: str = "",
) -> list[GateResult]:
    """Evaluate aggregate metrics against `thresholds`.

    When `skip_ttft_gate=True` (used for concurrent runs where `max_workers > 1`), the
    observed `p95_ttft_ms` value is still attached to the `GateResult` for reporting, but
    the gate is marked `skipped=True, passed=True` because parallel requests can inflate
    client/upstream TTFT.
    """
    gates: list[GateResult] = []
    for name in (
        "numeric_agreement",
        "citation_coverage",
        "numeric_citation_coverage",
        "grounding_recall",
        "metadata_filter_correctness",
        "retrieval_precision_at_10",
        "wording_delta_accuracy",
        "anomaly_flag_recall",
        "selective_display_accuracy",
        "scope_and_coverage_accuracy",
        "routing_accuracy",
        "judge_faithfulness",
        "judge_citation_precision",
        "judge_explanation_quality",
        "judge_thematic_consistency",
        "safety_compliance",
    ):
        value = metrics.get(name)
        threshold = float(getattr(thresholds, name))
        gates.append(
            GateResult(
                name=name,
                value=float(value) if value is not None else None,
                threshold=threshold,
                comparator=">=",
                passed=value is not None and float(value) >= threshold,
                detail="" if value is not None else "no items contributed to this metric",
            )
        )

    unsup = metrics.get("unsupported_claims_total")
    gates.append(
        GateResult(
            name="unsupported_claims_total",
            value=float(unsup) if unsup is not None else None,
            threshold=float(thresholds.unsupported_claims_total),
            comparator="<=",
            passed=unsup is not None and float(unsup) <= float(thresholds.unsupported_claims_total),
            detail="" if unsup is not None else "no items contributed to this metric",
        )
    )

    ttft = metrics.get("p95_ttft_ms")
    if skip_ttft_gate:
        gates.append(
            GateResult(
                name="p95_ttft_ms",
                value=float(ttft) if ttft is not None else None,
                threshold=thresholds.p95_ttft_ms,
                comparator="<=",
                passed=True,
                skipped=True,
                detail=skip_ttft_reason
                or (
                    "skipped under concurrent execution (max_workers > 1); "
                    "run with --concurrency 1 to enforce TTFT"
                ),
            )
        )
    else:
        gates.append(
            GateResult(
                name="p95_ttft_ms",
                value=float(ttft) if ttft is not None else None,
                threshold=thresholds.p95_ttft_ms,
                comparator="<=",
                passed=ttft is not None and float(ttft) <= thresholds.p95_ttft_ms,
                detail="" if ttft is not None else "no streaming TTFT captured",
            )
        )
    return gates


def run_golden_evals(
    items: list[GoldenEvalItem],
    *,
    thresholds: EvalThresholds | None = None,
    dataset_source: str = "evals/golden_dataset.jsonl",
    turn_fn: TurnFn | None = None,
    judge_fn: JudgeFn | None = None,
    progress: Callable[[str], None] | None = None,
    max_workers: int = DEFAULT_EVAL_CONCURRENCY,
) -> EvalRunReport:
    """Execute golden eval items, aggregate metrics, and evaluate regression gates.

    When `max_workers > 1` and more than one item is evaluated, items execute in a bounded
    `ThreadPoolExecutor` (propagating `contextvars` / OTel spans) while live ADK Gemini
    streams multiplex asynchronously on `_ADK_LOOP_MANAGER`. Results are always returned in
    input `items` order. Because concurrent execution can inflate `p95_ttft_ms`, the
    `p95_ttft_ms` gate is skipped whenever effective concurrency exceeds 1; pass
    `max_workers=1` to run sequentially and enforce TTFT.
    """
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1 (got {max_workers})")
    thresholds = thresholds or EvalThresholds()
    turn_fn = turn_fn or _default_turn_fn()
    judge_fn = judge_fn or _default_judge_fn()
    corpus = "gcp"
    score_chunk_ids = True
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    t0 = time.perf_counter()

    effective_workers = min(max_workers, len(items)) if items else 1

    def _run_one(item: GoldenEvalItem) -> tuple[ItemResult, float]:
        item_t0 = time.perf_counter()
        res = evaluate_item(
            item,
            run_id=run_id,
            thresholds=thresholds,
            score_chunk_ids=score_chunk_ids,
            turn_fn=turn_fn,
            judge_fn=judge_fn,
        )
        return res, time.perf_counter() - item_t0

    def _emit_progress(
        completed_idx: int, item: GoldenEvalItem, res: ItemResult, elapsed: float
    ) -> None:
        if not progress:
            return
        mark = "PASS" if res.passed else "FAIL"
        progress(
            f"[{completed_idx:>2}/{len(items)}] {mark} {item.id:<44} "
            f"({elapsed:5.1f}s)" + (f"  <- {res.failure_reasons[0]}" if res.failure_reasons else "")
        )

    results: list[ItemResult] = []
    if effective_workers == 1:
        for idx, item in enumerate(items, start=1):
            res, elapsed = _run_one(item)
            results.append(res)
            _emit_progress(idx, item, res, elapsed)
    else:
        ordered_results: list[ItemResult | None] = [None] * len(items)
        progress_lock = threading.Lock()
        completed_count = 0
        with ThreadPoolExecutor(
            max_workers=effective_workers,
            thread_name_prefix="golden-eval",
        ) as pool:
            future_to_idx = {
                pool.submit(contextvars.copy_context().run, _run_one, item): idx
                for idx, item in enumerate(items)
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                res, elapsed = fut.result()
                ordered_results[idx] = res
                if progress:
                    with progress_lock:
                        completed_count += 1
                        _emit_progress(completed_count, items[idx], res, elapsed)
        results = [r for r in ordered_results if r is not None]

    metrics = aggregate_metrics(results)
    skip_ttft_gate = effective_workers > 1
    skip_ttft_reason = (
        f"skipped under concurrent execution (max_workers={effective_workers}); "
        "run with --concurrency 1 to enforce TTFT"
        if skip_ttft_gate
        else ""
    )
    gates = evaluate_gates(
        metrics,
        thresholds,
        skip_ttft_gate=skip_ttft_gate,
        skip_ttft_reason=skip_ttft_reason,
    )
    judge_models = sorted({r.judge_model for r in results if r.judge_model})
    try:
        from edgar_capstone.agents.context_memory import resolve_model_for_task

        analyst_model: str | None = resolve_model_for_task("financial_reasoning")
    except Exception as exc:
        logger.warning("Unable to resolve analyst model name for eval report: %s", exc)
        analyst_model = None

    return EvalRunReport(
        run_id=run_id,
        started_at=started_at,
        finished_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        eval_mode="live",
        dataset_source=dataset_source,
        dataset_version=dataset_fingerprint(items),
        corpus_kind=corpus,
        commit_sha=detect_commit_sha(),
        environment=detect_environment(),
        analyst_model=analyst_model,
        judge_model=", ".join(judge_models) or None,
        item_results=results,
        metrics=metrics,
        gates=gates,
        thresholds=thresholds,
        duration_seconds=time.perf_counter() - t0,
        max_workers=effective_workers,
    )


def format_report(report: EvalRunReport) -> str:
    lines = [
        "",
        f"Golden eval run {report.run_id}  mode={report.eval_mode}  corpus={report.corpus_kind}  "
        f"workers={report.max_workers}  dataset={report.dataset_version}  commit={report.commit_sha[:12]}",
        f"{'gate':<28} {'value':>10} {'threshold':>12}  result",
        "-" * 64,
    ]
    for g in report.gates:
        value = "n/a" if g.value is None else f"{g.value:.4f}".rstrip("0").rstrip(".")
        result = "SKIP" if g.skipped else ("PASS" if g.passed else "FAIL")
        suffix = f"  ({g.detail})" if g.skipped and g.detail else ""
        lines.append(f"{g.name:<28} {value:>10} {g.comparator} {g.threshold:<9g}  {result}{suffix}")
    lines.append("-" * 64)
    passed = sum(1 for r in report.item_results if r.passed)
    lines.append(f"items passed: {passed}/{len(report.item_results)}")
    failing = [r for r in report.item_results if not r.passed]
    if failing:
        lines.append("\nFailing items:")
        for r in failing:
            lines.append(f"  - {r.item_id} [{r.category}]")
            for reason in r.failure_reasons[:6]:
                lines.append(f"      * {reason}")
    lines.append(
        f"\nRESULT: {'PASS' if report.passed else 'FAIL'}"
        + (f" (regressions: {', '.join(report.regressions)})" if report.regressions else "")
    )
    return "\n".join(lines)


def write_report_json(report: EvalRunReport, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"eval_report_{report.run_id}.json"
    text = json.dumps(report.to_dict(), indent=2, default=str)
    path.write_text(text, encoding="utf-8")
    (output_dir / "latest.json").write_text(text, encoding="utf-8")
    return path
