"""Unit tests for `edgar_capstone.common` helpers (env, domain, formatting, JSONL, concurrency, cache)."""

from __future__ import annotations

import contextvars
import json
import threading
import time
from datetime import date
from pathlib import Path

import pytest
from pydantic import BaseModel


def test_common_env_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    from edgar_capstone.common.env import (
        env_bool,
        env_positive_float,
        env_positive_int,
        split_csv_env,
    )

    monkeypatch.delenv("TEST_FLAG_REG", raising=False)
    assert env_bool("TEST_FLAG_REG", default=True) is True
    assert env_bool("TEST_FLAG_REG", default=False) is False

    # Fail-closed regression: empty string and garbage must preserve default=True
    for raw in ("", "   ", "invalid", "maybe"):
        monkeypatch.setenv("TEST_FLAG_REG", raw)
        assert env_bool("TEST_FLAG_REG", default=True) is True
        assert env_bool("TEST_FLAG_REG", default=False) is False

    for truthy in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("TEST_FLAG_REG", truthy)
        assert env_bool("TEST_FLAG_REG", default=False) is True

    for falsy in ("0", "false", "FALSE", "no", "off"):
        monkeypatch.setenv("TEST_FLAG_REG", falsy)
        assert env_bool("TEST_FLAG_REG", default=True) is False

    assert split_csv_env(None) == ()
    assert split_csv_env("") == ()
    assert split_csv_env(" a , b, , a , c ") == ("a", "b", "c")

    monkeypatch.delenv("TEST_INT_REG", raising=False)
    assert env_positive_int("TEST_INT_REG", 7) == 7
    monkeypatch.setenv("TEST_INT_REG", "0")
    assert env_positive_int("TEST_INT_REG", 7, minimum=1) == 7
    monkeypatch.setenv("TEST_INT_REG", "notanint")
    assert env_positive_int("TEST_INT_REG", 7) == 7
    monkeypatch.setenv("TEST_INT_REG", "12")
    assert env_positive_int("TEST_INT_REG", 7) == 12

    monkeypatch.delenv("TEST_FLOAT_REG", raising=False)
    assert env_positive_float("TEST_FLOAT_REG", 2.5) == 2.5
    monkeypatch.setenv("TEST_FLOAT_REG", "-1.0")
    assert env_positive_float("TEST_FLOAT_REG", 2.5, minimum=0.0) == 2.5
    monkeypatch.setenv("TEST_FLOAT_REG", "abc")
    assert env_positive_float("TEST_FLOAT_REG", 2.5) == 2.5
    monkeypatch.setenv("TEST_FLOAT_REG", "3.75")
    assert env_positive_float("TEST_FLOAT_REG", 2.5) == 3.75


def test_common_domain_helpers() -> None:
    from edgar_capstone.common.domain import (
        canonical_xbrl_tag,
        canonicalize_gics_sector,
        format_cik,
        lookup_year_key,
        normalize_edgar_ticker,
        parse_iso_date,
        sha256_text,
    )

    assert normalize_edgar_ticker("brk.b") == "BRK-B"
    assert normalize_edgar_ticker("  nvda  ") == "NVDA"

    assert format_cik(1045810) == "0001045810"
    assert format_cik("0000320193") == "0000320193"
    assert format_cik("CIK-320193") == "0000320193"
    assert format_cik(None) is None
    assert format_cik("   ") is None
    assert format_cik("abc") is None

    assert canonical_xbrl_tag("us-gaap_Revenues") == "us-gaap:Revenues"
    assert canonical_xbrl_tag("us-gaap:Revenues") == "us-gaap:Revenues"
    assert canonical_xbrl_tag("dei_DocumentFiscalYearFocus") == "dei:DocumentFiscalYearFocus"

    assert parse_iso_date("2026-01-25") == date(2026, 1, 25)
    assert parse_iso_date("2026-01-25T18:30:00Z") == date(2026, 1, 25)
    assert parse_iso_date("2026-99-99") is None
    assert parse_iso_date("bad-date") is None
    assert parse_iso_date(None) is None

    assert canonicalize_gics_sector("information technology") == "Information Technology"
    assert canonicalize_gics_sector("ALL") is None
    assert canonicalize_gics_sector("*") is None
    assert canonicalize_gics_sector("  ") is None
    assert canonicalize_gics_sector(None) is None
    assert canonicalize_gics_sector("Custom Sector") == "Custom Sector"

    digest = sha256_text("hello")
    assert len(digest) == 64
    assert digest == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    assert lookup_year_key({2025: "int_val"}, 2025) == "int_val"
    assert lookup_year_key({"2025": "str_val"}, 2025) == "str_val"
    assert lookup_year_key({"2024": "other"}, 2025) is None
    assert lookup_year_key(None, 2025) is None


def test_common_formatting_and_parse_model_output() -> None:
    from edgar_capstone.common.formatting import (
        _extract_json_dict,
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

    assert fmt_usd(2_500_000_000_000) == "$2.50T"
    assert fmt_usd(130_500_000_000) == "$130.50B"
    assert fmt_usd(45_250_000) == "$45.25M"
    assert fmt_usd(12_500) == "$12.50K"
    assert fmt_usd(512) == "$512"
    assert fmt_usd(-1_500_000_000) == "-$1.50B"
    assert fmt_usd(None) == "N/A"

    assert fmt_pct(12.345) == "+12.35%"
    assert fmt_pct(-4.5) == "-4.50%"
    assert fmt_pct(0.0) == "0.00%"
    assert fmt_pct(12.345, signed=False) == "12.35%"
    assert fmt_pct(None) == "N/A"

    assert fmt_bps(210.4) == "+210 bps"
    assert fmt_bps(-45) == "-45 bps"
    assert fmt_bps(0) == "0 bps"
    assert fmt_bps(210, signed=False) == "210 bps"
    assert fmt_bps(None) == "N/A"

    assert pct_change(100.0, 125.0) == 25.0
    assert pct_change(-100.0, 50.0) == 150.0
    assert pct_change(0.0, 50.0) is None
    assert pct_change(None, 50.0) is None

    assert ratio_pct(25.0, 100.0) == 25.0
    assert ratio_pct(25.0, 0.0) is None
    assert ratio_pct(None, 100.0) is None

    assert margin_bps_delta(20.0, 22.15) == 215.0
    assert margin_bps_delta(None, 22.0) is None

    assert percentile([], 95) == 0.0
    assert percentile([42.0], 95) == 42.0
    assert percentile([10.0, 20.0, 30.0, 40.0], 50) == 25.0
    assert round(percentile([10.0, 20.0, 30.0, 40.0], 95), 2) == 38.5

    iso = utc_now_iso()
    assert iso.endswith("Z") and len(iso) == 20 and "T" in iso

    raw_quote = "\u201cRevenue\u201d\u00a0grew \u2014 **12\\.5\\%** `YoY`."
    assert normalize_quote_text(raw_quote) == '"revenue" grew - 12.5% yoy.'

    # Direct _extract_json_dict edge cases: blank, unfenced brace extraction, corrupt braces, JSON array
    assert _extract_json_dict("   ") is None
    assert _extract_json_dict('preamble {"name": "inline", "score": 4} trailing') == {
        "name": "inline",
        "score": 4,
    }
    assert _extract_json_dict("preamble {not: valid json} trailing") is None
    assert _extract_json_dict("[1, 2, 3]") is None

    class _Sample(BaseModel):
        name: str
        score: int

    existing = _Sample(name="ok", score=10)
    assert parse_model_output(existing, _Sample) is existing
    assert parse_model_output({"name": "a", "score": 2}, _Sample) == _Sample(name="a", score=2)
    assert parse_model_output(
        {"name": "bad", "score": "not_an_int"},
        _Sample,
        fallback_text='Here is JSON:\n```json\n{"name": "recovered", "score": 9}\n```',
    ) == _Sample(name="recovered", score=9)
    assert parse_model_output(
        'prefix text ```json\n{"name": "fenced", "score": 3}\n``` suffix',
        _Sample,
    ) == _Sample(name="fenced", score=3)
    assert parse_model_output('text {"name": "brace", "score": 7} end', _Sample) == _Sample(
        name="brace", score=7
    )
    assert (
        parse_model_output(
            '{"name": "invalid_score", "score": "nope"}',
            _Sample,
            fallback_text='{"name": "still_bad", "score": "nope"}',
        )
        is None
    )
    assert parse_model_output({"bad": True}, _Sample, fallback_text="no valid json here") is None


def test_common_jsonl_streaming_and_writers(tmp_path: Path) -> None:
    from edgar_capstone.common.jsonl import iter_jsonl, write_json, write_jsonl

    jl_path = tmp_path / "nested" / "records.jsonl"
    written = write_jsonl(jl_path, [{"id": 1}, {"id": 2}])
    assert written == 2
    assert list(iter_jsonl(jl_path)) == [{"id": 1}, {"id": 2}]

    jl_path.write_text('{"id": 1}\n\n{corrupt\n[1, 2]\n{"id": 3}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        list(iter_jsonl(jl_path, skip_corrupt=False))
    assert list(iter_jsonl(jl_path, skip_corrupt=True)) == [{"id": 1}, {"id": 3}]

    json_path = tmp_path / "nested2" / "doc.json"
    write_json(json_path, {"ok": True, "count": 5})
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"ok": True, "count": 5}


def test_common_concurrency_run_parallel_with_context() -> None:
    from edgar_capstone.common.concurrency import run_parallel_with_context

    var: contextvars.ContextVar[str] = contextvars.ContextVar("req_tag", default="unset")
    token = var.set("trace-abc-123")
    main_tid = threading.get_ident()
    try:
        assert run_parallel_with_context([], lambda x: x) == []

        single_tids: list[int] = []

        def _single(x: int) -> tuple[int, str]:
            single_tids.append(threading.get_ident())
            return x * 2, var.get()

        assert run_parallel_with_context([5], _single) == [(10, "trace-abc-123")]
        assert single_tids == [main_tid]

        def _worker(idx: int) -> tuple[int, str]:
            # Reverse sleep pattern verifies output retains input order, not completion order.
            time.sleep(0.005 * (3 - idx))
            return idx, var.get()

        out = run_parallel_with_context([0, 1, 2, 3], _worker, max_workers=4)
        assert out == [
            (0, "trace-abc-123"),
            (1, "trace-abc-123"),
            (2, "trace-abc-123"),
            (3, "trace-abc-123"),
        ]
    finally:
        var.reset(token)


def test_common_cache_bounded_ttl_metrics_and_key_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from edgar_capstone.common.cache import (
        BoundedTTLCache,
        CacheMetrics,
        parse_10k_cache_key,
    )

    now = [100.0]
    monkeypatch.setattr("edgar_capstone.common.cache.time.monotonic", lambda: now[0])

    cache: BoundedTTLCache[str, int] = BoundedTTLCache(max_entries=2, ttl_seconds=10.0)
    cache["a"] = 1
    cache["b"] = 2
    assert cache["a"] == 1  # promotes "a"; "b" becomes LRU
    cache["c"] = 3
    assert cache.snapshot() == {"a": 1, "c": 3}
    assert cache.get("b") is None

    now[0] = 115.0
    assert cache.snapshot() == {}
    assert len(cache) == 0
    with pytest.raises(KeyError):
        _ = cache["a"]

    eternal: BoundedTTLCache[str, int] = BoundedTTLCache(max_entries=3, ttl_seconds=0.0)
    eternal["x"] = 10
    eternal["y"] = 20
    assert list(eternal) == ["x", "y"]
    del eternal["x"]
    assert list(eternal) == ["y"]
    eternal.clear()
    assert len(eternal) == 0

    metrics = CacheMetrics()
    metrics.record_hit(1500)
    metrics.record_miss(400)
    assert metrics.snapshot() == {
        "hits": 1,
        "misses": 1,
        "tokens_served_from_cache": 1500,
        "tokens_loaded_on_miss": 400,
    }
    metrics.reset()
    assert metrics.snapshot() == {
        "hits": 0,
        "misses": 0,
        "tokens_served_from_cache": 0,
        "tokens_loaded_on_miss": 0,
    }

    assert parse_10k_cache_key("BRK.B_2025_item_1a") == ("BRK-B", 2025, "item_1a")
    assert parse_10k_cache_key("NVDA_2026_item_7") == ("NVDA", 2026, "item_7")
    assert parse_10k_cache_key("invalid_key") is None
