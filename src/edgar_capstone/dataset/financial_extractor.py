"""XBRL income-statement metric extraction with pre-filtered `IncomeStatementView` caching."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from edgar_capstone.common.domain import canonical_xbrl_tag
from edgar_capstone.dataset.models import MetricProvenance

logger = logging.getLogger(__name__)

_UNSET: Any = object()

# Consolidated top-line revenue concepts checked first on non-dimensional income statements.
CONSOLIDATED_TOTAL_REVENUE_CONCEPTS: tuple[str, ...] = (
    "us-gaap_Revenues",
    "us-gaap_SalesRevenueNet",
    "us-gaap_RevenuesNetOfInterestExpense",
    "us-gaap_TotalRevenuesAndOtherIncome",
    "us-gaap_RealEstateRevenueNet",
    "us-gaap_RegulatedAndUnregulatedOperatingRevenue",
)

# Component / standard-specific revenue concepts (ASC 606, ASC 842 REIT leases, healthcare, insurance).
# When no explicit consolidated total-revenue tag exists (e.g., Equity REITs reporting separate
# OperatingLeaseLeaseIncome and minor ASC 606 fee lines), we select the largest positive line.
COMPONENT_REVENUE_CONCEPTS: tuple[str, ...] = (
    "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap_RevenueFromContractWithCustomerIncludingAssessedTax",
    "us-gaap_OperatingLeaseLeaseIncome",
    "us-gaap_LeaseIncome",
    "us-gaap_HealthCareOrganizationRevenue",
    "us-gaap_PremiumsEarnedNet",
    "us-gaap_InterestAndDividendIncomeOperating",
)

OPERATING_INCOME_CONCEPTS: tuple[str, ...] = ("us-gaap_OperatingIncomeLoss",)

OPERATING_INCOME_FALLBACK_CONCEPTS: tuple[str, ...] = (
    "us-gaap_OperatingIncomeLoss",
    "us-gaap_IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "us-gaap_IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    "us-gaap_IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
    "us-gaap_IncomeLossFromContinuingOperationsBeforeIncomeTaxes",
)

NET_INCOME_FALLBACK_CONCEPTS: tuple[str, ...] = (
    "us-gaap_NetIncomeLoss",
    "us-gaap_ProfitLoss",
    "us-gaap_NetIncomeLossAvailableToCommonStockholdersBasic",
)

# Research & Development expense concepts, in priority order. Issuers that do not report an
# R&D line (banks, REITs, utilities, most retailers) get `None`, never 0.
RESEARCH_AND_DEVELOPMENT_CONCEPTS: tuple[str, ...] = (
    "us-gaap_ResearchAndDevelopmentExpense",
    "us-gaap_ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    "us-gaap_ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost",
)

_RESEARCH_AND_DEVELOPMENT_LABEL_RE = re.compile(
    r"^\s*(?:total\s+)?research(?:,)?\s+(?:and|&)\s+development(?:\s+expenses?)?\s*:?\s*$",
    re.IGNORECASE,
)

_TOTAL_REVENUE_LABEL_RE = re.compile(
    r"^\s*total\s+(?:revenues?|net\s+revenues?|operating\s+revenues?|net\s+sales|sales\s+and\s+revenues?)\s*:?\s*$",
    re.IGNORECASE,
)
# Statement top line for filers that present "Total revenues and other" (e.g., APA, EXE). Used to
# sanity-check the chosen revenue concept: APA tags `us-gaap:Revenues` on a $1.1B sub-line in FY2021
# and reports it at 2x the true total in FY2023-FY2025, while `apa_RevenuesAndOther` is reliable.
_TOTAL_REVENUES_AND_OTHER_LABEL_RE = re.compile(
    r"^\s*total\s+revenues?,?\s+and\s+other(?:\s+income)?\s*:?\s*$",
    re.IGNORECASE,
)
_REVENUE_TOP_LINE_TOLERANCE = (0.5, 1.5)
_OPERATING_INCOME_LABEL_RE = re.compile(
    r"^\s*(?:operating\s+(?:income|profit|loss|income\s*\(loss\)|loss\s*\(income\))|"
    r"(?:income|loss)\s+from\s+operations)\s*:?\s*$",
    re.IGNORECASE,
)
_PRETAX_INCOME_LABEL_RE = re.compile(
    r"^\s*(?:net\s+)?(?:income|earnings|loss|\(loss\)\s*income|income\s*\(loss\)|loss\s*\(income\))\s+"
    r"(?:from\s+continuing\s+operations\s+)?before\s+"
    r"(?:provision\s+for\s+|federal\s+and\s+state\s+)?income\s+taxes"
    r"(?:\s+and\s+[^\n]{1,60})?\s*:?\s*$",
    re.IGNORECASE,
)
_DATE_COLUMN_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _coerce_finite_float(value: Any) -> float | None:
    """Convert numeric values to finite Python float, filtering NaN/Inf/None."""
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def _load_income_statement_df(tenk: Any) -> Any | None:
    """Retrieve the income-statement DataFrame from a `TenK` object once (or return `None`)."""
    fin = getattr(tenk, "financials", None)
    if fin is None:
        return None
    try:
        inc_stmt = fin.income_statement()
        return inc_stmt.to_dataframe() if inc_stmt is not None else None
    except Exception as exc:
        logger.debug("Income statement DataFrame unavailable: %s", exc)
        return None


def _filter_non_dimensional_income_df(
    inc_df: Any,
    *,
    report_date_str: str | None = None,
) -> tuple[Any | None, str | None]:
    """Filter income statement DataFrame to non-dimensional, non-breakdown rows and pick primary FY column."""
    if inc_df is None or getattr(inc_df, "empty", True):
        return None, None

    df = inc_df
    if "dimension" in df.columns:
        df = df[df["dimension"] == False]  # noqa: E712
    if "is_breakdown" in df.columns:
        df = df[df["is_breakdown"] == False]  # noqa: E712
    if "abstract" in df.columns:
        df = df[df["abstract"] == False]  # noqa: E712

    if df.empty or "concept" not in df.columns:
        return None, None

    fy_cols = [c for c in df.columns if "(FY)" in str(c)]
    date_cols = [c for c in df.columns if _DATE_COLUMN_RE.match(str(c))]
    # Full-year "(FY)" columns first, then any other dated period columns (e.g., a fresh-start
    # successor period labelled "2021-12-31" without the "(FY)" suffix).
    candidate_cols = fy_cols + [c for c in date_cols if c not in fy_cols]
    if not candidate_cols:
        return None, None

    if not report_date_str:
        return df, candidate_cols[0]

    report_str = str(report_date_str).strip()
    try:
        target_date: date | None = date.fromisoformat(report_str[:10])
    except ValueError:
        target_date = None

    if target_date is not None:
        for col in candidate_cols:
            m = _DATE_COLUMN_RE.match(str(col))
            if not m:
                continue
            try:
                col_date = date.fromisoformat(m.group(0))
            except ValueError:
                continue
            if abs((col_date - target_date).days) <= 7:
                return df, col

    report_prefix = report_str[:7]  # YYYY-MM
    for col in candidate_cols:
        if str(col).startswith(report_prefix):
            return df, col

    # No column covers the reporting period. Previously this fell back to the first "(FY)"
    # column, which silently attributed PRIOR-YEAR values to the current fiscal year
    # (e.g., EXE FY2021 profile carried FY2020 operating/net income). Refuse instead.
    logger.debug(
        "No income statement column matches period_of_report=%s (columns=%s)",
        report_str,
        candidate_cols,
    )
    return None, None


@dataclass(slots=True)
class IncomeStatementView:
    """Pre-filtered view of an income statement DataFrame.

    Calls `_filter_non_dimensional_income_df()` once per filing instead of repeating
    pandas masking and date-column matching on every concept/label lookup.
    """

    raw_df: Any | None
    non_dim_df: Any | None
    primary_col: str | None
    _non_dim_vals: tuple[float, ...] | None = None
    _dim_vals: tuple[float, ...] | None = None

    @classmethod
    def from_dataframe(
        cls,
        inc_df: Any | None,
        *,
        report_date_str: str | None = None,
    ) -> IncomeStatementView:
        if inc_df is None or getattr(inc_df, "empty", True):
            return cls(raw_df=inc_df, non_dim_df=None, primary_col=None)
        non_dim_df, primary_col = _filter_non_dimensional_income_df(
            inc_df, report_date_str=report_date_str
        )
        return cls(raw_df=inc_df, non_dim_df=non_dim_df, primary_col=primary_col)

    @property
    def is_available(self) -> bool:
        return self.non_dim_df is not None and self.primary_col is not None

    def lookup_concept(
        self,
        concepts: tuple[str, ...],
        *,
        pick_largest_positive: bool = False,
    ) -> tuple[float | None, str | None]:
        """Search pre-filtered non-dimensional income rows for matching concepts."""
        if not self.is_available:
            return None, None
        df = self.non_dim_df
        col = self.primary_col
        candidates: list[tuple[float, str]] = []
        for target_concept in concepts:
            matched = df[df["concept"] == target_concept]
            if matched.empty:
                continue
            for raw_val in matched[col].tolist():
                val = _coerce_finite_float(raw_val)
                if val is not None:
                    tag = canonical_xbrl_tag(target_concept)
                    if not pick_largest_positive:
                        return val, tag
                    candidates.append((val, tag))

        if candidates:
            # Prefer largest positive component when disambiguating REIT lease vs fee lines
            pos_candidates = [c for c in candidates if c[0] > 0]
            if pos_candidates:
                return max(pos_candidates, key=lambda item: item[0])
            return candidates[0]

        return None, None

    def lookup_by_label_or_suffix(
        self,
        *,
        concept_suffixes: tuple[str, ...] = (),
        label_re: re.Pattern[str] | None = None,
    ) -> tuple[float | None, str | None]:
        """Match company-extension XBRL concepts (e.g., `bk_...`) or standardized line labels."""
        if not self.is_available:
            return None, None
        df = self.non_dim_df
        col = self.primary_col

        if concept_suffixes:
            for _, row in df.iterrows():
                concept_str = str(row.get("concept", "") or "")
                if any(concept_str.endswith(sfx) or sfx in concept_str for sfx in concept_suffixes):
                    val = _coerce_finite_float(row.get(col))
                    if val is not None:
                        return val, canonical_xbrl_tag(concept_str)

        if label_re is not None and "label" in df.columns:
            for _, row in df.iterrows():
                label_str = str(row.get("label", "") or "").strip()
                concept_str = str(row.get("concept", "") or "")
                if "PerShare" in concept_str or "Shares" in concept_str:
                    continue
                if label_re.match(label_str):
                    val = _coerce_finite_float(row.get(col))
                    if val is not None:
                        return val, canonical_xbrl_tag(concept_str)

        return None, None

    def is_exclusively_dimensional(self, target_val: float) -> bool:
        """Return True if `target_val` appears only on dimensional/breakdown rows."""
        if (
            self.raw_df is None
            or getattr(self.raw_df, "empty", True)
            or self.primary_col is None
            or self.primary_col not in self.raw_df.columns
        ):
            return False

        if self._non_dim_vals is None:
            if self.non_dim_df is not None:
                vals: list[float] = []
                for raw_val in self.non_dim_df[self.primary_col].tolist():
                    num = _coerce_finite_float(raw_val)
                    if num is not None:
                        vals.append(num)
                self._non_dim_vals = tuple(vals)
            else:
                self._non_dim_vals = ()

        for val in self._non_dim_vals:
            if math.isclose(val, target_val, rel_tol=1e-6, abs_tol=1.0):
                return False

        if self._dim_vals is None:
            has_dim_col = "dimension" in self.raw_df.columns
            has_brk_col = "is_breakdown" in self.raw_df.columns
            if not has_dim_col and not has_brk_col:
                self._dim_vals = ()
            else:
                dim_mask: Any = False
                if has_dim_col:
                    dim_mask = dim_mask | (self.raw_df["dimension"] == True)  # noqa: E712
                if has_brk_col:
                    dim_mask = dim_mask | (self.raw_df["is_breakdown"] == True)  # noqa: E712
                dim_rows = self.raw_df[dim_mask]
                dvals: list[float] = []
                for raw_val in dim_rows[self.primary_col].tolist():
                    num = _coerce_finite_float(raw_val)
                    if num is not None:
                        dvals.append(num)
                self._dim_vals = tuple(dvals)

        return any(
            math.isclose(val, target_val, rel_tol=1e-6, abs_tol=1.0) for val in self._dim_vals
        )


def _resolve_income_view(
    tenk: Any,
    *,
    report_date_str: str | None = None,
    inc_df: Any = _UNSET,
    inc_view: IncomeStatementView | None = None,
) -> IncomeStatementView:
    if inc_view is not None:
        return inc_view
    resolved_df = _load_income_statement_df(tenk) if inc_df is _UNSET else inc_df
    return IncomeStatementView.from_dataframe(resolved_df, report_date_str=report_date_str)


def extract_financial_metrics(
    tenk: Any,
    *,
    report_date_str: str | None = None,
    inc_df: Any = _UNSET,
    inc_view: IncomeStatementView | None = None,
) -> tuple[float | None, float | None, float | None, MetricProvenance]:
    """Extract Revenue, Operating Income, and Net Income (in USD) with XBRL provenance.

    Prioritizes non-dimensional (`dimension=False, is_breakdown=False`) Income Statement
    DataFrame rows whenever available so dimensional segment rows (e.g., APA's Kinetik
    breakdown) and minor ASC 606 ancillary lines on Equity REITs (e.g., CPT fee income vs
    ASC 842 `OperatingLeaseLeaseIncome`) never override consolidated totals.
    """
    provenance = MetricProvenance()
    fin = getattr(tenk, "financials", None)
    if fin is None:
        provenance.notes.append("No XBRL financials object available on filing.")
        return None, None, None, provenance

    view = _resolve_income_view(
        tenk, report_date_str=report_date_str, inc_df=inc_df, inc_view=inc_view
    )

    # --- 1. Revenue ---
    revenue: float | None = None
    if view.is_available:
        # Step 1a: Consolidated total revenue tags
        rev_val, rev_tag = view.lookup_concept(CONSOLIDATED_TOTAL_REVENUE_CONCEPTS)
        # Step 1b: Explicit company-extension "Total revenue(s)" line (e.g., BK's bk_TotalRevenues...)
        if rev_val is None:
            rev_val, rev_tag = view.lookup_by_label_or_suffix(
                concept_suffixes=("_TotalRevenues",),
                label_re=_TOTAL_REVENUE_LABEL_RE,
            )
        # Step 1c: Component revenue tags — pick largest positive (REIT leases vs minor fees)
        if rev_val is None:
            rev_val, rev_tag = view.lookup_concept(
                COMPONENT_REVENUE_CONCEPTS, pick_largest_positive=True
            )
        if rev_val is not None:
            revenue = rev_val
            provenance.revenue = rev_tag

    if revenue is None:
        helper_rev = _coerce_finite_float(fin.get_revenue())
        if helper_rev is not None and not view.is_exclusively_dimensional(helper_rev):
            revenue = helper_rev
            provenance.revenue = "edgartools:Financials.get_revenue"

    # Step 1d: Reconcile against the statement's "Total revenues and other" top line when present.
    if revenue is not None and view.is_available:
        top_line, top_tag = view.lookup_by_label_or_suffix(
            concept_suffixes=("_RevenuesAndOther",),
            label_re=_TOTAL_REVENUES_AND_OTHER_LABEL_RE,
        )
        low, high = _REVENUE_TOP_LINE_TOLERANCE
        if (
            top_line is not None
            and top_line > 0
            and not (low * top_line <= revenue <= high * top_line)
        ):
            provenance.notes.append(
                f"Revenue concept {provenance.revenue}={revenue:,.0f} is outside {low}x-{high}x of the "
                f"statement top line {top_tag} ('Total revenues and other')={top_line:,.0f}; "
                "using the top line (mis-tagged or double-counted revenue concept)."
            )
            revenue = top_line
            provenance.revenue = top_tag

    # --- 2. Operating Income (with Pre-Tax Continuing Operations fallback) ---
    operating_income: float | None = None
    if view.is_available:
        op_val, op_tag = view.lookup_concept(OPERATING_INCOME_CONCEPTS)
        if op_val is None:
            op_val, op_tag = view.lookup_by_label_or_suffix(
                concept_suffixes=("_OperatingIncomeLoss",),
                label_re=_OPERATING_INCOME_LABEL_RE,
            )
        if op_val is not None:
            operating_income = op_val
            provenance.operating_income = op_tag

    if operating_income is None:
        helper_op = _coerce_finite_float(fin.get_operating_income())
        if helper_op is not None and not view.is_exclusively_dimensional(helper_op):
            operating_income = helper_op
            provenance.operating_income = (
                "edgartools:Financials.get_operating_income (us-gaap:OperatingIncomeLoss)"
            )

    if operating_income is None and view.is_available:
        pretax_val, pretax_tag = view.lookup_concept(OPERATING_INCOME_FALLBACK_CONCEPTS[1:])
        if pretax_val is None:
            pretax_val, pretax_tag = view.lookup_by_label_or_suffix(
                concept_suffixes=(
                    "_IncomeLossFromContinuingOperationsBeforeIncomeTaxes",
                    "BeforeEarningsFromAffiliatesIncomeTaxes",
                ),
                label_re=_PRETAX_INCOME_LABEL_RE,
            )
        # Tax-exempt single-step Equity REITs (e.g., BXP) report ProfitLoss with no tax line
        if pretax_val is None:
            tax_val, _ = view.lookup_concept(
                ("us-gaap_IncomeTaxExpenseBenefit", "us-gaap_CurrentIncomeTaxExpenseBenefit")
            )
            if tax_val is None:
                reit_pl, reit_tag = view.lookup_concept(("us-gaap_ProfitLoss",))
                if reit_pl is not None:
                    pretax_val, pretax_tag = reit_pl, reit_tag

        if pretax_val is not None:
            operating_income = pretax_val
            provenance.operating_income = pretax_tag
            if pretax_tag and "OperatingIncomeLoss" not in pretax_tag:
                provenance.notes.append(
                    "Filer does not report standard GAAP OperatingIncomeLoss; "
                    f"populated operating_income from {pretax_tag} (pre-tax continuing operations income)."
                )

    # --- 3. Net Income ---
    net_income: float | None = None
    if view.is_available:
        ni_val, ni_tag = view.lookup_concept(NET_INCOME_FALLBACK_CONCEPTS)
        if ni_val is not None:
            net_income = ni_val
            provenance.net_income = ni_tag

    if net_income is None:
        helper_ni = _coerce_finite_float(fin.get_net_income())
        if helper_ni is not None and not view.is_exclusively_dimensional(helper_ni):
            net_income = helper_ni
            provenance.net_income = "edgartools:Financials.get_net_income (us-gaap:NetIncomeLoss)"

    return revenue, operating_income, net_income, provenance


def extract_research_and_development_expense(
    tenk: Any,
    *,
    report_date_str: str | None = None,
    inc_df: Any = _UNSET,
    inc_view: IncomeStatementView | None = None,
) -> tuple[float | None, str | None]:
    """Extract annual Research & Development expense (USD) and its XBRL concept.

    Reads the non-dimensional income statement rows for the reporting-period column, trying
    `RESEARCH_AND_DEVELOPMENT_CONCEPTS` first and then a company-extension concept / line label
    ("Research and development"). Returns `(None, None)` when the filer reports no R&D line.

    Pass a pre-loaded `inc_df` or `inc_view` to avoid calling `fin.income_statement().to_dataframe()`
    twice per filing when called alongside `extract_financial_metrics`.
    """
    if inc_view is None and inc_df is _UNSET and getattr(tenk, "financials", None) is None:
        return None, None
    view = _resolve_income_view(
        tenk, report_date_str=report_date_str, inc_df=inc_df, inc_view=inc_view
    )
    if not view.is_available:
        return None, None

    rd_val, rd_tag = view.lookup_concept(RESEARCH_AND_DEVELOPMENT_CONCEPTS)
    if rd_val is None:
        rd_val, rd_tag = view.lookup_by_label_or_suffix(
            concept_suffixes=("_ResearchAndDevelopmentExpense",),
            label_re=_RESEARCH_AND_DEVELOPMENT_LABEL_RE,
        )
    if rd_val is None:
        return None, None
    return rd_val, rd_tag


__all__ = [
    "COMPONENT_REVENUE_CONCEPTS",
    "CONSOLIDATED_TOTAL_REVENUE_CONCEPTS",
    "NET_INCOME_FALLBACK_CONCEPTS",
    "OPERATING_INCOME_CONCEPTS",
    "OPERATING_INCOME_FALLBACK_CONCEPTS",
    "RESEARCH_AND_DEVELOPMENT_CONCEPTS",
    "IncomeStatementView",
    "extract_financial_metrics",
    "extract_research_and_development_expense",
]
