"""Golden eval runner: executes every golden item through the RAG pipeline, scores it, and
applies the regression gates from `evals/thresholds.yaml`.

Every run is live: answers come from the Gemini `FinancialAnalystAgent`, retrieval reads
BigQuery / Vertex AI Search / Vertex AI embeddings / GCS, and the Gemini 3.5 Flash judge scores
each answer against its reference explanation. The p95 TTFT gate always applies.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from edgar_capstone.evals.golden import (
    REPO_ROOT,
    GoldenEvalItem,
    dataset_fingerprint,
)
from edgar_capstone.evals.metrics import (
    percentile,
    score_citation_coverage,
    score_grounding,
    score_numeric_agreement,
    score_routing,
    score_safety,
)

DEFAULT_THRESHOLDS_PATH = REPO_ROOT / "evals" / "thresholds.yaml"


@dataclass(frozen=True)
class EvalThresholds:
    numeric_agreement: float = 1.0
    citation_coverage: float = 1.0
    grounding_recall: float = 0.92
    routing_accuracy: float = 1.0
    judge_faithfulness: float = 0.85
    judge_explanation_quality: float = 0.75
    judge_thematic_consistency: float = 0.75
    safety_compliance: float = 1.0
    p95_ttft_ms: float = 3000.0
    narrative_pct_tolerance: float = 0.1
    min_scored_items: int = 20

    @classmethod
    def load(cls, path: Path | str | None = None) -> EvalThresholds:
        resolved = Path(path) if path else DEFAULT_THRESHOLDS_PATH
        if not resolved.exists():
            return cls()
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        gates = raw.get("gates", raw)
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(gates) - known
        if unknown:
            raise ValueError(f"Unknown threshold keys in {resolved}: {sorted(unknown)}")
        return cls(**{k: type(getattr(cls(), k))(v) for k, v in gates.items()})


@dataclass
class ItemResult:
    item_id: str
    category: str
    query: str
    passed: bool = False
    status: str = ""
    intent: str | None = None
    execution_mode: str | None = None
    numeric_agreement: bool | None = None
    citation_coverage: bool | None = None
    routing_ok: bool | None = None
    grounding_matched: int = 0
    grounding_expected: int = 0
    faithfulness_score: float | None = None
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
    metrics: dict[str, float | None]
    gates: list[GateResult]
    thresholds: EvalThresholds
    duration_seconds: float

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
                    "routing_ok": r.routing_ok,
                    "grounding_recall": r.grounding_recall,
                    "faithfulness_score": r.faithfulness_score,
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
    from edgar_capstone.agents.evaluation_and_coding import evaluate_variance_explanation

    return evaluate_variance_explanation


def evaluate_item(
    item: GoldenEvalItem,
    *,
    run_id: str,
    thresholds: EvalThresholds,
    score_chunk_ids: bool,
    turn_fn: TurnFn,
    judge_fn: JudgeFn,
) -> ItemResult:
    res = ItemResult(item_id=item.id, category=item.category, query=item.query)
    f = item.filters
    try:
        response = turn_fn(
            query=item.query,
            tickers=f.tickers,
            gics_sector=f.gics_sector,
            section_id=f.section_id,
            start_year=f.start_year,
            end_year=f.end_year,
            top_k=f.top_k,
            session_id=f"golden-eval-{run_id}-{item.id}",
            user_id="golden-eval-runner",
        )
    except Exception as exc:  # noqa: BLE001 - one broken item must not abort the run
        res.error = f"{type(exc).__name__}: {exc}"
        res.failure_reasons.append(f"turn raised {res.error}")
        res.status = "error"
        if os.environ.get("EDGAR_EVAL_TRACEBACKS"):
            traceback.print_exc()
        return res

    res.status = str(response.get("status"))
    res.intent = response.get("intent")
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

    if res.status != "success":
        res.failure_reasons.append(f"unexpected status={res.status} (intent={res.intent})")
        res.numeric_agreement = False if item.expected_variances else None
        res.citation_coverage = False
        res.routing_ok = False
        _, res.grounding_expected, _ = score_grounding(item, {}, score_chunk_ids=score_chunk_ids)
        return res

    res.routing_ok, reasons = score_routing(item, response)
    res.failure_reasons.extend(reasons)

    res.numeric_agreement, reasons = score_numeric_agreement(
        item, response, narrative_pct_tolerance=thresholds.narrative_pct_tolerance
    )
    res.failure_reasons.extend(reasons)

    res.citation_coverage, reasons = score_citation_coverage(item, response)
    res.failure_reasons.extend(reasons)

    res.grounding_matched, res.grounding_expected, missing = score_grounding(
        item, response, score_chunk_ids=score_chunk_ids
    )
    if missing:
        res.failure_reasons.append(f"grounding missed: {missing}")

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
        res.explanation_quality_score = float(judged["explanation_quality_score"])
        res.thematic_consistency_score = float(judged["thematic_consistency_score"])
        res.judge_grade = judged.get("grade")
        res.judge_model = judged.get("evaluator_model")
        if res.judge_grade == "fail":
            res.failure_reasons.append(f"judge grade=fail: {judged.get('rationale', '')[:200]}")
    except Exception as exc:  # noqa: BLE001
        res.failure_reasons.append(f"judge raised {type(exc).__name__}: {exc}")

    res.passed = (
        bool(res.routing_ok)
        and res.numeric_agreement is not False
        and bool(res.citation_coverage)
        and res.grounding_matched == res.grounding_expected
        and res.faithfulness_score is not None
        and res.judge_grade != "fail"
    )
    return res


def _ratio(values: list[bool]) -> float | None:
    return round(sum(1 for v in values if v) / len(values), 4) if values else None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def aggregate_metrics(results: list[ItemResult]) -> dict[str, float | None]:
    scored = [r for r in results if r.category != "safety_red_team"]
    safety = [r for r in results if r.category == "safety_red_team"]
    g_matched = sum(r.grounding_matched for r in scored)
    g_expected = sum(r.grounding_expected for r in scored)
    ttfts = [r.ttft_ms for r in scored if r.ttft_ms is not None]
    latencies = [r.total_latency_ms for r in results if r.total_latency_ms is not None]

    def _judge(attr: str) -> float | None:
        # Items whose turn or judge failed score 0 so they drag the mean down instead of vanishing.
        return _mean([getattr(r, attr) if getattr(r, attr) is not None else 0.0 for r in scored])

    return {
        "numeric_agreement": _ratio(
            [bool(r.numeric_agreement) for r in scored if r.numeric_agreement is not None]
        ),
        "citation_coverage": _ratio([bool(r.citation_coverage) for r in scored]),
        "grounding_recall": round(g_matched / g_expected, 4) if g_expected else None,
        "routing_accuracy": _ratio([bool(r.routing_ok) for r in scored]),
        "judge_faithfulness": _judge("faithfulness_score"),
        "judge_explanation_quality": _judge("explanation_quality_score"),
        "judge_thematic_consistency": _judge("thematic_consistency_score"),
        "safety_compliance": _ratio([r.passed for r in safety]),
        "p95_ttft_ms": percentile(ttfts, 95) if ttfts else None,
        "p95_total_latency_ms": percentile(latencies, 95) if latencies else None,
        "item_pass_rate": _ratio([r.passed for r in results]),
    }


def evaluate_gates(
    metrics: dict[str, float | None], thresholds: EvalThresholds
) -> list[GateResult]:
    gates: list[GateResult] = []
    for name in (
        "numeric_agreement",
        "citation_coverage",
        "grounding_recall",
        "routing_accuracy",
        "judge_faithfulness",
        "judge_explanation_quality",
        "judge_thematic_consistency",
        "safety_compliance",
    ):
        value = metrics.get(name)
        threshold = float(getattr(thresholds, name))
        gates.append(
            GateResult(
                name=name,
                value=value,
                threshold=threshold,
                comparator=">=",
                passed=value is not None and value >= threshold,
                detail="" if value is not None else "no items contributed to this metric",
            )
        )

    ttft = metrics.get("p95_ttft_ms")
    gates.append(
        GateResult(
            name="p95_ttft_ms",
            value=ttft,
            threshold=thresholds.p95_ttft_ms,
            comparator="<=",
            passed=ttft is not None and ttft <= thresholds.p95_ttft_ms,
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
) -> EvalRunReport:
    thresholds = thresholds or EvalThresholds()
    turn_fn = turn_fn or _default_turn_fn()
    judge_fn = judge_fn or _default_judge_fn()
    corpus = "gcp"
    score_chunk_ids = True
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    t0 = time.perf_counter()

    results: list[ItemResult] = []
    for idx, item in enumerate(items, start=1):
        item_t0 = time.perf_counter()
        res = evaluate_item(
            item,
            run_id=run_id,
            thresholds=thresholds,
            score_chunk_ids=score_chunk_ids,
            turn_fn=turn_fn,
            judge_fn=judge_fn,
        )
        results.append(res)
        if progress:
            mark = "PASS" if res.passed else "FAIL"
            progress(
                f"[{idx:>2}/{len(items)}] {mark} {item.id:<44} "
                f"({time.perf_counter() - item_t0:5.1f}s)"
                + (f"  <- {res.failure_reasons[0]}" if res.failure_reasons else "")
            )

    metrics = aggregate_metrics(results)
    gates = evaluate_gates(metrics, thresholds)
    judge_models = sorted({r.judge_model for r in results if r.judge_model})
    try:
        from edgar_capstone.agents.context_memory import resolve_model_for_task

        analyst_model: str | None = resolve_model_for_task("financial_reasoning")
    except Exception:  # noqa: BLE001
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
    )


def format_report(report: EvalRunReport) -> str:
    lines = [
        "",
        f"Golden eval run {report.run_id}  mode={report.eval_mode}  corpus={report.corpus_kind}  "
        f"dataset={report.dataset_version}  commit={report.commit_sha[:12]}",
        f"{'gate':<28} {'value':>10} {'threshold':>12}  result",
        "-" * 64,
    ]
    for g in report.gates:
        value = "n/a" if g.value is None else f"{g.value:.4f}".rstrip("0").rstrip(".")
        result = "SKIP" if g.skipped else ("PASS" if g.passed else "FAIL")
        lines.append(f"{g.name:<28} {value:>10} {g.comparator} {g.threshold:<9g}  {result}")
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
