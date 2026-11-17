"""Shared lightweight utilities with zero heavy SDK dependencies (`edgartools`/`pandas`/`storage`)."""

from __future__ import annotations

from edgar_capstone.common.cache import BoundedTTLCache, CacheMetrics, parse_10k_cache_key
from edgar_capstone.common.concurrency import run_parallel_with_context
from edgar_capstone.common.domain import (
    CANONICAL_GICS_SECTORS,
    CORE_METRICS,
    METRIC_LABELS,
    RATIO_METRICS,
    RETRIEVAL_TOOL_NAMES,
    SECTION_TITLES,
    MetricName,
    SectionId,
    canonical_xbrl_tag,
    canonicalize_gics_sector,
    format_cik,
    lookup_year_key,
    normalize_edgar_ticker,
    parse_iso_date,
    sha256_text,
)
from edgar_capstone.common.env import (
    FALSY_ENV_VALUES,
    TRUTHY_ENV_VALUES,
    env_bool,
    env_positive_float,
    env_positive_int,
    split_csv_env,
)
from edgar_capstone.common.formatting import (
    fmt_bps,
    fmt_pct,
    fmt_usd,
    margin_bps_delta,
    normalize_quote_text,
    parse_model_output,
    pct_change,
    percentile,
    ratio_pct,
    utc_now_iso,
)
from edgar_capstone.common.jsonl import iter_jsonl, write_json, write_jsonl
from edgar_capstone.common.models import FilingMetadataBase

__all__ = [
    "BoundedTTLCache",
    "CANONICAL_GICS_SECTORS",
    "CORE_METRICS",
    "CacheMetrics",
    "FALSY_ENV_VALUES",
    "FilingMetadataBase",
    "METRIC_LABELS",
    "MetricName",
    "RATIO_METRICS",
    "RETRIEVAL_TOOL_NAMES",
    "SECTION_TITLES",
    "SectionId",
    "TRUTHY_ENV_VALUES",
    "canonical_xbrl_tag",
    "canonicalize_gics_sector",
    "env_bool",
    "env_positive_float",
    "env_positive_int",
    "fmt_bps",
    "fmt_pct",
    "fmt_usd",
    "format_cik",
    "iter_jsonl",
    "lookup_year_key",
    "margin_bps_delta",
    "normalize_edgar_ticker",
    "normalize_quote_text",
    "parse_10k_cache_key",
    "parse_iso_date",
    "parse_model_output",
    "pct_change",
    "percentile",
    "ratio_pct",
    "run_parallel_with_context",
    "sha256_text",
    "split_csv_env",
    "utc_now_iso",
    "write_json",
    "write_jsonl",
]
