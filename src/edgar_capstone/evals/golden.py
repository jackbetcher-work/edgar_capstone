"""Golden eval dataset schema, loading, validation, and expected-value refresh.

Each line of `evals/golden_dataset.jsonl` is one `GoldenEvalItem`: an analyst query mapped to
the deterministic variances it must reproduce, the 10-K sections it must be grounded in, and an
analyst-authored reference explanation used by the Gemini 3.5 Flash judge. Red-team items
(`category="safety_red_team"`) instead assert that Model Armor / the projection filter blocks
the prompt.

The same records are loaded into BigQuery `sec_edgar.golden_eval_set`
(`terraform/schemas/golden_eval_set.json`), so field names here are the BigQuery column names.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from edgar_capstone.common.domain import normalize_edgar_ticker
from edgar_capstone.gcp.paths import resolve_repo_root

REPO_ROOT = resolve_repo_root()
DEFAULT_GOLDEN_DATASET_PATH = REPO_ROOT / "evals" / "golden_dataset.jsonl"

GoldenCategory = Literal[
    "single_company_longitudinal",
    "multi_company_peer_comparison",
    "thematic_shift",
    "anomaly_explanation",
    "out_of_corpus_and_clarification",
    "safety_red_team",
]
CORE_CATEGORIES: tuple[str, ...] = (
    "single_company_longitudinal",
    "multi_company_peer_comparison",
    "thematic_shift",
    "anomaly_explanation",
)
SCOPE_CATEGORY = "out_of_corpus_and_clarification"
SAFETY_CATEGORY = "safety_red_team"

MetricName = Literal["revenue", "operating_income", "net_income", "research_and_development"]
METRIC_TO_REPORT_KEY: dict[str, str] = {
    "revenue": "revenue_variance",
    "operating_income": "operating_income_variance",
    "net_income": "net_income_variance",
    "research_and_development": "research_and_development_variance",
}

QueryIntent = Literal[
    "longitudinal_single_company",
    "multi_company_peer_comparison",
    "thematic_trend_tracking",
    "chunk_health_monitoring",
    "clarification_needed",
    "out_of_corpus",
]
ExpectedStatusType = Literal["success", "blocked", "clarification", "out_of_corpus"]
DeltaChangeType = Literal["added", "removed", "modified"]
BlockType = Literal["prompt_safety", "projection_compliance"]
SectionId = Literal["item_1a", "item_7"]

MIN_SCORED_ITEMS = 20

# `FinancialAnalystAgent` chooses which deterministic engines to invoke, so an item that declares
# no `expected_flags` asserts that `material_variance_flags` stays empty (see
# `metrics.score_selective_display`). Items whose query legitimately permits
# `flag_material_variances` (margin shifts, unusual swings, losses) without asserting a specific
# flag carry this tag: flags are then neither required nor penalized.
FLAGS_OPTIONAL_TAG = "flags_optional"


class ExpectedVariance(BaseModel):
    """One deterministic variance the calculation engine must reproduce for this query."""

    model_config = ConfigDict(extra="forbid")

    ticker: str
    metric: MetricName
    start_year: int
    end_year: int
    start_value: float | None = None
    end_value: float | None = None
    absolute_delta: float | None = None
    percentage_delta: float | None = None
    # When true, the percentage must also appear in the narrative (not just the tool output).
    required_in_narrative: bool = False

    @field_validator("ticker")
    @classmethod
    def _upper_ticker(cls, v: str) -> str:
        return normalize_edgar_ticker(v)

    @property
    def key(self) -> str:
        return f"{self.ticker}:{self.metric}:FY{self.start_year}-FY{self.end_year}"


class ExpectedSection(BaseModel):
    """A 10-K source the answer must retrieve *and* cite. Unset fields act as wildcards."""

    model_config = ConfigDict(extra="forbid")

    ticker: str | None = None
    fiscal_year: int | None = None
    section_id: SectionId | None = None

    @field_validator("ticker")
    @classmethod
    def _upper_ticker(cls, v: str | None) -> str | None:
        return normalize_edgar_ticker(v) if v else None

    def matches(self, ticker: str, fiscal_year: int, section_id: str) -> bool:
        if self.ticker is not None and self.ticker != normalize_edgar_ticker(ticker):
            return False
        if self.fiscal_year is not None and self.fiscal_year != int(fiscal_year):
            return False
        return self.section_id is None or self.section_id == section_id

    @property
    def label(self) -> str:
        return (
            f"{self.ticker or '*'} FY{self.fiscal_year if self.fiscal_year is not None else '*'} "
            f"{self.section_id or '*'}"
        )


class ExpectedDelta(BaseModel):
    """One deterministic year-over-year disclosure wording delta expected for a thematic item."""

    model_config = ConfigDict(extra="forbid")

    ticker: str
    prior_fiscal_year: int
    current_fiscal_year: int
    change_type: DeltaChangeType
    keyword_hint: str | None = None

    @field_validator("ticker")
    @classmethod
    def _upper_ticker(cls, v: str) -> str:
        return normalize_edgar_ticker(v)

    @property
    def label(self) -> str:
        hint = f" ({self.keyword_hint})" if self.keyword_hint else ""
        return (
            f"{self.ticker} FY{self.prior_fiscal_year}->FY{self.current_fiscal_year} "
            f"{self.change_type}{hint}"
        )


class ExpectedFlag(BaseModel):
    """One deterministic material-variance anomaly flag expected for an anomaly item."""

    model_config = ConfigDict(extra="forbid")

    ticker: str
    fiscal_year: int
    metric: str
    rule: str
    flag_id: str | None = None

    @field_validator("ticker")
    @classmethod
    def _upper_ticker(cls, v: str) -> str:
        return normalize_edgar_ticker(v)

    @property
    def label(self) -> str:
        return self.flag_id or f"{self.ticker}_{self.fiscal_year}_{self.metric}_{self.rule}"


class RequestFilters(BaseModel):
    """Explicit UI filter values sent with the query (defaults mirror the Terminal UI)."""

    model_config = ConfigDict(extra="forbid")

    tickers: str = "ALL"
    gics_sector: str = "ALL"
    section_id: str = "ALL"
    start_year: int = 2024
    end_year: int = 2026
    top_k: int = Field(default=8, ge=1, le=25)


class GoldenEvalItem(BaseModel):
    """One golden query-variance-explanation mapping (or a red-team safety probe)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_\-]{2,63}$")
    category: GoldenCategory
    query: str = Field(min_length=8)
    tickers: list[str] = Field(default_factory=list)
    fiscal_years: list[int] = Field(default_factory=list)
    gics_sector: str | None = None
    # Agent-emitted `AnalystReport.intent` (falls back to a deterministic tool-trace / scope
    # inference, see `agents.orchestration.intent.resolve_turn_intent`); scope-gate values for early exits.
    expected_intent: QueryIntent | None = None
    expected_variances: list[ExpectedVariance] = Field(default_factory=list)
    expected_sections: list[ExpectedSection] = Field(default_factory=list)
    expected_deltas: list[ExpectedDelta] = Field(default_factory=list)
    expected_flags: list[ExpectedFlag] = Field(default_factory=list)
    expected_status: ExpectedStatusType | None = None
    expected_coverage_notice_codes: list[str] = Field(default_factory=list)
    # Exact chunk IDs from the live BigQuery `sec_edgar.filing_chunks` corpus.
    expected_chunk_ids: list[str] = Field(default_factory=list)
    reference_explanation: str = ""
    expect_blocked: bool = False
    expected_block_type: BlockType | None = None
    filters: RequestFilters = Field(default_factory=RequestFilters)
    tags: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator("tickers")
    @classmethod
    def _normalize_tickers(cls, v: list[str]) -> list[str]:
        return [normalize_edgar_ticker(t) for t in v if t.strip()]

    @field_validator("fiscal_years")
    @classmethod
    def _sorted_years(cls, v: list[int]) -> list[int]:
        return sorted(set(v))

    @model_validator(mode="after")
    def _check_consistency(self) -> GoldenEvalItem:
        if self.category == SAFETY_CATEGORY:
            if not self.expect_blocked:
                raise ValueError(f"{self.id}: safety_red_team items must set expect_blocked=true")
            if self.expected_variances or self.expected_sections:
                raise ValueError(f"{self.id}: safety items cannot declare grounding expectations")
            return self

        if self.expect_blocked:
            raise ValueError(f"{self.id}: only safety_red_team items may set expect_blocked")
        if self.expected_flags and FLAGS_OPTIONAL_TAG in self.tags:
            raise ValueError(
                f"{self.id}: {FLAGS_OPTIONAL_TAG!r} contradicts declared expected_flags"
            )

        if self.category == SCOPE_CATEGORY:
            if self.expected_status is None:
                raise ValueError(
                    f"{self.id}: out_of_corpus_and_clarification items must set expected_status"
                )
            if self.expected_status == "success":
                if not self.reference_explanation.strip():
                    raise ValueError(
                        f"{self.id}: reference_explanation is required when expected_status='success'"
                    )
                if not self.fiscal_years:
                    raise ValueError(
                        f"{self.id}: fiscal_years is required when expected_status='success'"
                    )
            return self

        if not self.reference_explanation.strip():
            raise ValueError(f"{self.id}: reference_explanation is required for scored items")
        if not self.fiscal_years:
            raise ValueError(f"{self.id}: fiscal_years is required for scored items")
        unknown = {ev.ticker for ev in self.expected_variances} - set(self.tickers)
        if unknown:
            raise ValueError(f"{self.id}: expected_variances reference unknown tickers {unknown}")
        lo, hi = self.fiscal_years[0], self.fiscal_years[-1]
        for ev in self.expected_variances:
            if not (lo <= ev.start_year <= ev.end_year <= hi):
                raise ValueError(f"{self.id}: {ev.key} falls outside fiscal_years FY{lo}-FY{hi}")
        return self

    @property
    def is_safety(self) -> bool:
        return self.category == SAFETY_CATEGORY

    @property
    def is_scope_early_exit(self) -> bool:
        return self.category == SCOPE_CATEGORY and self.expected_status in (
            "clarification",
            "out_of_corpus",
        )

    @property
    def flags_optional(self) -> bool:
        return FLAGS_OPTIONAL_TAG in self.tags

    @property
    def start_year(self) -> int | None:
        return self.fiscal_years[0] if self.fiscal_years else None

    @property
    def end_year(self) -> int | None:
        return self.fiscal_years[-1] if self.fiscal_years else None

    def to_bigquery_row(self, *, dataset_version: str, loaded_at: str) -> dict[str, Any]:
        row = self.model_dump(mode="json")
        row["dataset_version"] = dataset_version
        row["loaded_at"] = loaded_at
        return row

    @classmethod
    def from_bigquery_row(cls, row: dict[str, Any]) -> GoldenEvalItem:
        payload = {k: v for k, v in row.items() if k not in ("dataset_version", "loaded_at")}
        return cls.model_validate(payload)


class GoldenDatasetError(ValueError):
    """Raised when the golden dataset file is malformed or fails coverage requirements."""


def parse_golden_lines(lines: list[str], *, source: str = "<memory>") -> list[GoldenEvalItem]:
    items: list[GoldenEvalItem] = []
    errors: list[str] = []
    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            items.append(GoldenEvalItem.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{source}:{lineno}: {exc}")
    if errors:
        raise GoldenDatasetError("Invalid golden dataset records:\n" + "\n".join(errors))
    return items


def load_golden_dataset(path: Path | str = DEFAULT_GOLDEN_DATASET_PATH) -> list[GoldenEvalItem]:
    dataset_path = Path(path)
    if not dataset_path.is_absolute() and not dataset_path.exists():
        dataset_path = REPO_ROOT / dataset_path
    if not dataset_path.exists():
        raise GoldenDatasetError(f"Golden dataset not found: {dataset_path}")
    return parse_golden_lines(
        dataset_path.read_text(encoding="utf-8").splitlines(), source=str(dataset_path)
    )


def validate_golden_dataset(
    items: list[GoldenEvalItem], *, min_scored_items: int = MIN_SCORED_ITEMS
) -> dict[str, int]:
    """Enforce unique IDs, the 20+ scored-item floor, and coverage of every category."""
    problems: list[str] = []
    dupes = [item_id for item_id, n in Counter(i.id for i in items).items() if n > 1]
    if dupes:
        problems.append(f"duplicate ids: {sorted(dupes)}")

    counts = Counter(i.category for i in items)
    scored = sum(counts[c] for c in CORE_CATEGORIES)
    if scored < min_scored_items:
        problems.append(f"only {scored} scored items; need at least {min_scored_items}")
    required_cats = (*CORE_CATEGORIES, SCOPE_CATEGORY, SAFETY_CATEGORY)
    for cat in required_cats:
        if counts[cat] == 0:
            problems.append(f"no items for category {cat!r}")

    if problems:
        raise GoldenDatasetError("Golden dataset failed validation: " + "; ".join(problems))
    return {cat: counts[cat] for cat in required_cats}


def dataset_fingerprint(items: list[GoldenEvalItem]) -> str:
    """Stable content hash so every eval_results row maps to an exact golden dataset version."""
    canonical = json.dumps(
        sorted((i.model_dump(mode="json") for i in items), key=lambda d: d["id"]),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def dump_golden_dataset(items: list[GoldenEvalItem], path: Path | str) -> None:
    lines = [
        json.dumps(i.model_dump(mode="json", exclude_defaults=False), ensure_ascii=False)
        for i in items
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh_expected_values(
    item: GoldenEvalItem,
    *,
    calculate_fn: Any | None = None,
) -> GoldenEvalItem:
    """Recompute the numeric fields of `expected_variances` from the calculation engine.

    Only the numbers (start/end/delta and the effective years) are overwritten; which tickers,
    metrics, and `required_in_narrative` flags to check stay analyst-authored. Use it to
    bootstrap new items, then review the diff before committing: blindly refreshing would turn
    a numeric regression into a new golden value.
    """
    if item.is_safety or not item.expected_variances:
        return item
    if calculate_fn is None:
        from edgar_capstone.agents.tools import calculate_period_variances

        calculate_fn = calculate_period_variances

    tickers = sorted({ev.ticker for ev in item.expected_variances})
    result = calculate_fn(
        tickers=",".join(tickers), start_year=item.start_year, end_year=item.end_year
    )
    if result.get("status") != "success":
        raise GoldenDatasetError(f"{item.id}: calculation engine failed: {result}")
    reports = {r["ticker"]: r for r in result.get("variance_reports") or []}

    refreshed: list[ExpectedVariance] = []
    for ev in item.expected_variances:
        rep = reports.get(ev.ticker)
        if rep is None:
            raise GoldenDatasetError(f"{item.id}: no variance report for {ev.ticker}")
        metric = rep.get(METRIC_TO_REPORT_KEY[ev.metric]) or {}
        refreshed.append(
            ev.model_copy(
                update={
                    "start_year": int(rep["start_year"]),
                    "end_year": int(rep["end_year"]),
                    "start_value": metric.get("start_value"),
                    "end_value": metric.get("end_value"),
                    "absolute_delta": metric.get("absolute_delta"),
                    "percentage_delta": metric.get("percentage_delta"),
                }
            )
        )
    return item.model_copy(update={"expected_variances": refreshed})
