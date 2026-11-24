"""Unit tests for dynamic 501-company entity resolution, scope validation, clarification, and coverage notices."""

from __future__ import annotations

from types import SimpleNamespace

from edgar_capstone.agents.orchestration.chat_engine import resolve_query_scope
from edgar_capstone.agents.orchestration.scope_validation import (
    SECTOR_ALIASES,
    EntityCatalog,
    build_coverage_notices,
    extract_removed_tickers,
    is_additive_followup,
    validate_query_scope_and_coverage,
)


def _sample_profiles() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            ticker="NVDA", company_name="NVIDIA CORP", gics_sector="Information Technology"
        ),
        SimpleNamespace(
            ticker="AAPL", company_name="Apple Inc.", gics_sector="Information Technology"
        ),
        SimpleNamespace(
            ticker="MSFT",
            company_name="Microsoft Corporation",
            gics_sector="Information Technology",
        ),
        SimpleNamespace(
            ticker="COST",
            company_name="Costco Wholesale Corporation",
            gics_sector="Consumer Staples",
        ),
        SimpleNamespace(
            ticker="DUK", company_name="Duke Energy Corporation", gics_sector="Utilities"
        ),
        SimpleNamespace(ticker="PLD", company_name="Prologis, Inc.", gics_sector="Real Estate"),
        SimpleNamespace(ticker="LIN", company_name="Linde plc", gics_sector="Materials"),
        SimpleNamespace(
            ticker="ALL", company_name="The Allstate Corporation", gics_sector="Financials"
        ),
        SimpleNamespace(
            ticker="ON",
            company_name="ON Semiconductor Corporation",
            gics_sector="Information Technology",
        ),
    ]


def test_entity_catalog_resolves_tickers_names_and_ignores_ambiguous_words() -> None:
    catalog = EntityCatalog(_sample_profiles())
    assert catalog.contains_ticker("NVDA") is True
    assert catalog.contains_ticker("COST") is True
    assert catalog.contains_ticker("RIVN") is False

    # Resolves company names and tickers beyond any hardcoded mega-cap list (e.g., Costco, Prologis, Duke Energy)
    found, unknown = catalog.detect_tickers_in_query("Compare Costco and Prologis revenue growth")
    assert set(found) == {"COST", "PLD"}
    assert unknown == []

    # Ambiguous English words like "all" or "on" or "cost" must NOT trigger false ticker matches
    found2, unknown2 = catalog.detect_tickers_in_query(
        "What impact did cost pressures have on all companies?"
    )
    assert found2 == []
    assert unknown2 == []

    # Non-S&P 500 entities are flagged as unknown
    found3, unknown3 = catalog.detect_tickers_in_query("Compare Rivian and NVIDIA revenue")
    assert "NVDA" in found3
    assert any("rivian" in u.lower() for u in unknown3)

    # Lowercase "shop", "spot", "snow" must NOT false-positive as non-S&P 500 tickers
    _, unknown4 = catalog.detect_tickers_in_query(
        "Did NVDA spot pricing and snow storms affect shop deliveries?"
    )
    assert unknown4 == []


def test_all_11_gics_sectors_supported_in_sector_aliases() -> None:
    expected_sectors = {
        "Information Technology",
        "Health Care",
        "Financials",
        "Energy",
        "Industrials",
        "Consumer Discretionary",
        "Consumer Staples",
        "Communication Services",
        "Utilities",
        "Real Estate",
        "Materials",
    }
    assert expected_sectors <= {sector for _, sector in SECTOR_ALIASES}

    catalog = EntityCatalog(_sample_profiles())
    for phrase, expected in (
        ("utilities", "Utilities"),
        ("real estate", "Real Estate"),
        ("materials", "Materials"),
    ):
        scope = resolve_query_scope(
            query=f"How did risk factors evolve across {phrase} companies from 2024 to 2025?",
            catalog=catalog,
        )
        assert scope["gics_sector"] == expected


def test_additive_and_subtractive_followup_scope_resolution() -> None:
    catalog = EntityCatalog(_sample_profiles())
    prior_mem = {
        "has_prior_memory": True,
        "remembered_tickers": ["NVDA", "AAPL"],
        "last_start_year": 2024,
        "last_end_year": 2026,
        "last_gics_sector": "ALL",
    }

    assert is_additive_followup("Now add Microsoft to the comparison") is True
    assert is_additive_followup("Compare Microsoft and NVIDIA") is False

    merged = resolve_query_scope(
        query="Now add Microsoft to that comparison",
        memory_context=prior_mem,
        catalog=catalog,
    )
    assert set(merged["tickers"]) == {"NVDA", "AAPL", "MSFT"}
    assert merged["memory_inherited"] is True

    removed = extract_removed_tickers("Drop Apple from this analysis", catalog)
    assert removed == ["AAPL"]

    sub_scope = resolve_query_scope(
        query="Remove Apple and keep comparing",
        memory_context=prior_mem,
        catalog=catalog,
    )
    assert sub_scope["tickers"] == ["NVDA"]


def test_validate_scope_catches_out_of_corpus_and_clarification_without_false_positives() -> None:
    catalog = EntityCatalog(_sample_profiles())

    # 1. Non-S&P 500 company
    s1 = resolve_query_scope(
        query="How did Stripe revenue change from 2024 to 2025?", catalog=catalog
    )
    v1 = validate_query_scope_and_coverage(
        query="How did Stripe revenue change from 2024 to 2025?",
        scope=s1,
        explicit_tickers=s1.get("explicit_tickers", []),
        unknown_entities=s1.get("unknown_entities", []),
        catalog=catalog,
        memory_context={"has_prior_memory": False},
    )
    assert v1["action"] == "out_of_corpus"
    assert v1["reason"] == "non_sp500_issuer"
    assert len(v1["suggested_followups"]) >= 1

    # 2. Unsupported filing type (10-Q)
    s2 = resolve_query_scope(query="Summarize NVDA Q3 2025 10-Q earnings", catalog=catalog)
    v2 = validate_query_scope_and_coverage(
        query="Summarize NVDA Q3 2025 10-Q earnings",
        scope=s2,
        explicit_tickers=s2.get("explicit_tickers", []),
        unknown_entities=s2.get("unknown_entities", []),
        catalog=catalog,
        memory_context={"has_prior_memory": False},
    )
    assert v2["action"] == "out_of_corpus"
    assert v2["reason"] == "unsupported_filing_type"

    # 3. Unsupported metric (EBITDA / EPS)
    s3 = resolve_query_scope(
        query="What was NVDA adjusted EBITDA and EPS in 2025?", catalog=catalog
    )
    v3 = validate_query_scope_and_coverage(
        query="What was NVDA adjusted EBITDA and EPS in 2025?",
        scope=s3,
        explicit_tickers=s3.get("explicit_tickers", []),
        unknown_entities=s3.get("unknown_entities", []),
        catalog=catalog,
        memory_context={"has_prior_memory": False},
    )
    assert v3["action"] == "out_of_corpus"
    assert v3["reason"] == "unsupported_metric"

    # 4. "gross margin" MUST proceed (discussed in MD&A!)
    s4 = resolve_query_scope(
        query="What drove NVIDIA's gross margin and revenue expansion from 2024 to 2026?",
        catalog=catalog,
    )
    v4 = validate_query_scope_and_coverage(
        query="What drove NVIDIA's gross margin and revenue expansion from 2024 to 2026?",
        scope=s4,
        explicit_tickers=s4.get("explicit_tickers", []),
        unknown_entities=s4.get("unknown_entities", []),
        catalog=catalog,
        memory_context={"has_prior_memory": False},
    )
    assert v4["action"] == "proceed"

    # 5. Out-of-range year (<2020)
    s5 = resolve_query_scope(query="How did Apple revenue grow from 2015 to 2018?", catalog=catalog)
    v5 = validate_query_scope_and_coverage(
        query="How did Apple revenue grow from 2015 to 2018?",
        scope=s5,
        explicit_tickers=s5.get("explicit_tickers", []),
        unknown_entities=s5.get("unknown_entities", []),
        catalog=catalog,
        memory_context={"has_prior_memory": False},
    )
    assert v5["action"] == "out_of_corpus"
    assert v5["reason"] == "out_of_range_year"

    # 6. Ambiguous query with no ticker or sector triggers clarification (never silent NVDA fallback!)
    s6 = resolve_query_scope(query="How did revenue change?", catalog=catalog)
    v6 = validate_query_scope_and_coverage(
        query="How did revenue change?",
        scope=s6,
        explicit_tickers=[],
        unknown_entities=[],
        catalog=catalog,
        memory_context={"has_prior_memory": False},
    )
    assert v6["action"] == "clarification"
    assert v6["reason"] == "ambiguous_scope"
    assert len(v6["suggested_followups"]) >= 2


def test_build_coverage_notices_covers_missing_fy2026_calendar_misalignment_and_rd() -> None:
    notices = build_coverage_notices(
        variance_reports=[
            {
                "ticker": "NVDA",
                "company_name": "NVIDIA CORP",
                "start_year": 2024,
                "end_year": 2026,
                "available_fiscal_years": [2024, 2025, 2026],
                "missing_fiscal_years": [],
                "period_end_dates": {"2024": "2024-01-28", "2026": "2026-01-25"},
                "research_and_development_variance": {"start_value": 8.6e9, "end_value": 18.5e9},
            },
            {
                "ticker": "META",
                "company_name": "Meta Platforms, Inc.",
                "start_year": 2024,
                "end_year": 2025,
                "available_fiscal_years": [2024, 2025],
                "missing_fiscal_years": [2026],
                "period_end_dates": {"2024": "2024-12-31", "2025": "2025-12-31"},
                "research_and_development_variance": {"direction": "unavailable"},
            },
        ],
        missing_tickers=["FAKE"],
        query="Compare NVDA and META R&D expense from 2024 to 2026",
        sector_coverage={
            "gics_sector": "Information Technology",
            "issuers_in_sector": 76,
            "issuers_analyzed": 76,
            "issuers_with_theme_matches": 58,
            "batches_processed": 6,
        },
    )
    codes = {n["code"] for n in notices}
    assert "missing_ticker" in codes
    assert "missing_fiscal_year" in codes
    assert "fiscal_calendar_misalignment" in codes
    assert "missing_metric" in codes
    assert "sector_coverage_summary" in codes
    sec_notice = next(n for n in notices if n["code"] == "sector_coverage_summary")
    assert "76/76 Information Technology issuers" in sec_notice["message"]
    assert "58 with theme matches" in sec_notice["message"]
    rd_notice = next(n for n in notices if n["code"] == "missing_metric")
    assert rd_notice["tickers"] == ["META"]
    fy_msg = next(n["message"] for n in notices if n["code"] == "missing_fiscal_year")
    assert "Q1 2027" in fy_msg


def test_dual_class_cik_deduplication_prefers_canonical_ticker() -> None:
    profiles = [
        *_sample_profiles(),
        SimpleNamespace(
            ticker="GOOG",
            cik="0001652044",
            company_name="Alphabet Inc.",
            gics_sector="Communication Services",
        ),
        SimpleNamespace(
            ticker="GOOGL",
            cik="0001652044",
            company_name="Alphabet Inc.",
            gics_sector="Communication Services",
        ),
    ]
    catalog = EntityCatalog(profiles)

    # Natural-language "Alphabet" must resolve to canonical GOOGL only (never duplicating GOOG)
    found, unknown = catalog.detect_tickers_in_query(
        "Compare Microsoft and Alphabet on revenue and R&D from 2024 to 2025."
    )
    assert set(found) == {"MSFT", "GOOGL"}
    assert "GOOG" not in found
    assert unknown == []

    # Explicit ticker GOOG still resolves when explicitly requested
    found_explicit, _ = catalog.detect_tickers_in_query("Compare MSFT and GOOG revenue in 2025")
    assert set(found_explicit) == {"MSFT", "GOOG"}


def test_scope_validation_visa_v_ticker_guards_roman_numerals() -> None:
    """Relocated regression test: standalone 'V' resolves to Visa only in financial/peer context."""
    profiles = [
        SimpleNamespace(
            ticker="NVDA",
            company_name="NVIDIA Corporation",
            gics_sector="Information Technology",
            cik="0001045810",
            fiscal_year=2026,
        ),
        SimpleNamespace(
            ticker="V",
            company_name="Visa Inc.",
            gics_sector="Financials",
            cik="0001403161",
            fiscal_year=2025,
        ),
        SimpleNamespace(
            ticker="MA",
            company_name="Mastercard Incorporated",
            gics_sector="Financials",
            cik="0001141391",
            fiscal_year=2025,
        ),
    ]
    catalog = EntityCatalog(profiles)

    found_part, _ = catalog.detect_tickers_in_query("Explain Part V risk factors for NVDA")
    assert found_part == ["NVDA"]

    found_sec, _ = catalog.detect_tickers_in_query("Summarize Section V of the filing")
    assert "V" not in found_sec

    found_peer, _ = catalog.detect_tickers_in_query("Compare NVDA and V revenue in FY2025")
    assert set(found_peer) == {"NVDA", "V"}

    found_brand, _ = catalog.detect_tickers_in_query("How did Visa revenue change?")
    assert "V" in found_brand

    found_dollar, _ = catalog.detect_tickers_in_query("Show $V operating margin in FY2025")
    assert "V" in found_dollar


def test_detect_tickers_explicit_prefix_dollar_unknown_and_dotted_tickers() -> None:
    """Covers `$UNKNOWN`, `ticker XYZ`, ambiguous lowercase skip, and dotted ticker `BRK.B`."""
    profiles = [
        *_sample_profiles(),
        SimpleNamespace(ticker="", company_name="Blank", gics_sector="Unknown"),
        SimpleNamespace(
            ticker="BRK.B",
            company_name="Berkshire Hathaway Inc.",
            gics_sector="Financials",
            cik="0001067983",
            fiscal_year=2025,
        ),
        SimpleNamespace(
            ticker="V",
            company_name="Visa Inc.",
            gics_sector="Financials",
            cik="0001403161",
            fiscal_year=2025,
        ),
    ]
    catalog = EntityCatalog(profiles)

    found_dollar, unknown_dollar = catalog.detect_tickers_in_query(
        "Compare $NVDA and $FAKES revenue"
    )
    assert found_dollar == ["NVDA"]
    assert "FAKES" in unknown_dollar

    found_exp, unknown_exp = catalog.detect_tickers_in_query(
        "Check ticker NVDA vs ticker ZZZZ and stock all vs ticker BRK.B"
    )
    assert "NVDA" in found_exp
    assert "ZZZZ" in unknown_exp
    # Lowercase ambiguous word after `stock` (`stock all`) must be ignored
    assert "ALL" not in found_exp

    found_dotted, unknown_dotted = catalog.detect_tickers_in_query(
        "How did BRK.B operating income change in FY2025?"
    )
    assert found_dotted == ["BRK-B"] or found_dotted == ["BRK.B"]
    assert unknown_dotted == []

    # Standalone 'V' after pre-context (`vs V`) with topic cue resolves to Visa
    found_pre, _ = catalog.detect_tickers_in_query("Revenue trends for NVDA vs V in FY2025")
    assert set(found_pre) == {"NVDA", "V"}


def _fiscal_calendar_catalog() -> EntityCatalog:
    """CRWD (FYE Jan 31), RF (FYE Dec 31) with period-end dates; MMM without them."""
    profiles = [
        SimpleNamespace(
            ticker="CRWD",
            company_name="CrowdStrike Holdings, Inc.",
            gics_sector="Information Technology",
            fiscal_year=fy,
            period_of_report=f"{fy}-01-31",
        )
        for fy in range(2022, 2027)
    ]
    profiles += [
        SimpleNamespace(
            ticker="RF",
            company_name="Regions Financial Corp",
            gics_sector="Financials",
            fiscal_year=fy,
            period_of_report=f"{fy}-12-31",
        )
        for fy in range(2021, 2026)
    ]
    profiles.append(
        SimpleNamespace(ticker="MMM", company_name="3M Company", gics_sector="Industrials")
    )
    return EntityCatalog(profiles)


def test_fiscal_year_for_date_follows_each_issuer_calendar() -> None:
    from datetime import date

    catalog = _fiscal_calendar_catalog()
    # CrowdStrike's fiscal year ends Jan 31: July 2024 falls in FY2025, Jan 31 2024 in FY2024.
    assert catalog.fiscal_year_for_date("CRWD", date(2024, 7, 19)) == 2025
    assert catalog.fiscal_year_for_date("CRWD", date(2024, 1, 31)) == 2024
    # Calendar-year filer.
    assert catalog.fiscal_year_for_date("RF", date(2023, 3, 10)) == 2023
    # After the latest filed period -> the next, not-yet-filed fiscal year.
    assert catalog.fiscal_year_for_date("RF", date(2026, 3, 1)) == 2026
    # No period-end data -> unknown.
    assert catalog.fiscal_year_for_date("MMM", date(2024, 7, 1)) is None


def test_event_dates_resolve_on_the_issuer_fiscal_calendar() -> None:
    catalog = _fiscal_calendar_catalog()

    # "After <dated event>": prior fiscal year baseline through the latest filings.
    s = resolve_query_scope(
        query="How did CrowdStrike's risk factor disclosure change after the July 2024 "
        "Falcon sensor outage?",
        catalog=catalog,
    )
    assert (s["tickers"], s["start_year"], s["end_year"]) == (["CRWD"], 2024, 2026)

    # Bare dated event: baseline + the fiscal year containing the event.
    s = resolve_query_scope(
        query="What did CrowdStrike disclose about the July 19, 2024 outage?", catalog=catalog
    )
    assert (s["start_year"], s["end_year"]) == (2024, 2025)

    # Calendar-year filer with an explicit day.
    s = resolve_query_scope(
        query="How did Regions Financial's liquidity risk wording change following March 10, 2023?",
        catalog=catalog,
    )
    assert (s["tickers"], s["start_year"], s["end_year"]) == (["RF"], 2022, 2026)

    # No period-end data for the issuer -> calendar year is the fallback.
    s = resolve_query_scope(
        query="How did 3M (MMM) describe PFAS risk after the June 2023 settlement?",
        catalog=catalog,
    )
    assert (s["start_year"], s["end_year"]) == (2022, 2026)


def test_year_only_mentions_keep_existing_window_rules() -> None:
    catalog = _fiscal_calendar_catalog()
    # Bare years are fiscal years: the single-year default stays (yr-1, yr) ...
    s = resolve_query_scope(
        query="How did Regions Financial change its deposit risk wording after the 2023 "
        "bank failures?",
        catalog=catalog,
    )
    assert (s["start_year"], s["end_year"]) == (2022, 2023)
    # ... explicit ranges and "since <year>" are unchanged ...
    s = resolve_query_scope(
        query="How did CrowdStrike's revenue change from 2023 to 2025?", catalog=catalog
    )
    assert (s["start_year"], s["end_year"]) == (2023, 2025)
    s = resolve_query_scope(query="CrowdStrike risks since 2024", catalog=catalog)
    assert (s["start_year"], s["end_year"]) == (2024, 2026)
    # ... and "may" as a verb is not a month.
    s = resolve_query_scope(query="What risks may affect CrowdStrike in 2025?", catalog=catalog)
    assert (s["start_year"], s["end_year"]) == (2024, 2025)


def test_fuzzy_company_name_resolution_handles_typos_without_false_positives() -> None:
    """Common company-name misspellings resolve accurately while non-S&P 500 & prose stay clean."""
    profiles = [
        *_sample_profiles(),
        SimpleNamespace(
            ticker="PG",
            cik="0000080424",
            company_name="PROCTER & GAMBLE Co",
            gics_sector="Consumer Staples",
        ),
        SimpleNamespace(
            ticker="CAT",
            cik="0000018230",
            company_name="CATERPILLAR INC",
            gics_sector="Industrials",
        ),
        SimpleNamespace(
            ticker="WAT",
            cik="0001000697",
            company_name="WATERS CORP /DE/",
            gics_sector="Health Care",
        ),
        SimpleNamespace(
            ticker="GOOG",
            cik="0001652044",
            company_name="Alphabet Inc.",
            gics_sector="Communication Services",
        ),
        SimpleNamespace(
            ticker="GOOGL",
            cik="0001652044",
            company_name="Alphabet Inc.",
            gics_sector="Communication Services",
        ),
    ]
    catalog = EntityCatalog(profiles)

    # Multi-word typos ("Proctor & Gamble", "Proctor and Gamble", "Proctor Gamble") -> PG
    for query in (
        "What was Proctor & Gamble's revenue CAGR from fiscal 2020 to fiscal 2026?",
        "Compare Proctor and Gamble operating income in FY2024 and FY2025",
        "Analyze Proctor Gamble net sales growth",
    ):
        found, unknown = catalog.detect_tickers_in_query(query)
        assert found == ["PG"], f"Failed on {query!r}: got {found}"
        assert unknown == []

    # Peer query combining exact + misspelled names
    found_peer, _ = catalog.detect_tickers_in_query(
        "Compare NVIDIA and Proctor & Gamble revenue growth"
    )
    assert set(found_peer) == {"NVDA", "PG"}

    # Single-word proper-noun typos (len >= 7, edit distance == 1) + dual-class CIK dedup
    found_single, _ = catalog.detect_tickers_in_query(
        "Compare Microsft, Caterpiller, and Alphabett revenue"
    )
    assert set(found_single) == {"MSFT", "CAT", "GOOGL"}
    assert "GOOG" not in found_single

    # Zero false positives on generic prose, lowercase near-matches ("water" vs Waters), & non-S&P 500
    found_prose, unknown_prose = catalog.detect_tickers_in_query(
        "How did public water systems and revenue margins change?"
    )
    assert found_prose == []
    assert unknown_prose == []

    found_oos, unknown_oos = catalog.detect_tickers_in_query(
        "Compare Rivian and Snowflake revenue from 2024 to 2025"
    )
    assert found_oos == []
    assert any("rivian" in u.lower() for u in unknown_oos)
    assert any("snowflake" in u.lower() for u in unknown_oos)


def test_subject_change_detection_sector_inheritance_and_clarification_gate() -> None:
    """Layer 1 scope transitions track subject_changed/topic_epoch, inherit sectors, and honor clarification."""
    catalog = EntityCatalog(_sample_profiles())
    nvda_mem = {
        "has_prior_memory": True,
        "remembered_tickers": ["NVDA"],
        "last_start_year": 2024,
        "last_end_year": 2026,
        "last_gics_sector": "ALL",
        "topic_epoch": 0,
    }

    # Same ticker or overlapping peer comparison keeps topic epoch
    same = resolve_query_scope(
        query="What drove NVIDIA's R&D spend in FY2025?",
        memory_context=nvda_mem,
        catalog=catalog,
    )
    assert same["tickers"] == ["NVDA"]
    assert same["subject_changed"] is False
    assert same["topic_epoch"] == 0

    peer = resolve_query_scope(
        query="Compare NVIDIA and Apple revenue growth",
        memory_context=nvda_mem,
        catalog=catalog,
    )
    assert set(peer["tickers"]) == {"NVDA", "AAPL"}
    assert peer["subject_changed"] is False
    assert peer["topic_epoch"] == 0

    # Disjoint company, sector switch, or broad corpus switch marks subject_changed and increments epoch
    disjoint = resolve_query_scope(
        query="Analyze Costco revenue growth from 2024 to 2025",
        memory_context=nvda_mem,
        catalog=catalog,
    )
    assert disjoint["tickers"] == ["COST"]
    assert disjoint["subject_changed"] is True
    assert disjoint["topic_epoch"] == 1

    sector_switch = resolve_query_scope(
        query="How did risk factors evolve across utilities companies?",
        memory_context=nvda_mem,
        catalog=catalog,
    )
    assert sector_switch["tickers"] == []
    assert sector_switch["gics_sector"] == "Utilities"
    assert sector_switch["subject_changed"] is True
    assert sector_switch["topic_epoch"] == 1

    corpus_switch = resolve_query_scope(
        query="Across all S&P 500 companies, how did tariff wording change?",
        memory_context=nvda_mem,
        catalog=catalog,
    )
    assert corpus_switch["tickers"] == []
    assert corpus_switch["gics_sector"] == "ALL"
    assert corpus_switch["subject_changed"] is True
    assert corpus_switch["topic_epoch"] == 1

    # Sector-scoped memory is inherited on follow-up when remembered_tickers is empty
    sector_mem = {
        "has_prior_memory": True,
        "remembered_tickers": [],
        "last_start_year": 2024,
        "last_end_year": 2025,
        "last_gics_sector": "Utilities",
        "topic_epoch": 1,
    }
    sector_followup = resolve_query_scope(
        query="Which companies saw the largest risk factor expansion?",
        memory_context=sector_mem,
        catalog=catalog,
    )
    assert sector_followup["tickers"] == []
    assert sector_followup["gics_sector"] == "Utilities"
    assert sector_followup["memory_inherited"] is True
    assert sector_followup["subject_changed"] is False
    assert sector_followup["topic_epoch"] == 1

    # Clarification gate MUST NOT be bypassed by stale prior memory when memory_inherited is False
    unresolved_scope = {
        "tickers": [],
        "gics_sector": "ALL",
        "section_id": "ALL",
        "start_year": 2024,
        "end_year": 2026,
        "memory_inherited": False,
    }
    val = validate_query_scope_and_coverage(
        query="What about the defense industry?",
        scope=unresolved_scope,
        explicit_tickers=[],
        unknown_entities=[],
        catalog=catalog,
        memory_context=nvda_mem,
    )
    assert val["action"] == "clarification"
    assert val["reason"] == "ambiguous_scope"
