"""Dynamic S&P 500 entity resolution, scope validation, clarification, and coverage notices.

Replaces hardcoded alias lists and silent fallback guessing with:
1. Full dynamic entity resolution against the ~501-company BigQuery S&P 500 catalog (tickers +
   normalized official names + common brand aliases) and all 11 GICS sectors.
2. Additive / subtractive multi-turn follow-up refinement ("now add Microsoft", "drop Apple").
3. Explicit out-of-corpus detection for non-S&P 500 issuers, unsupported filing types
   (10-Q, 8-K, proxy statements, transcripts), unsupported metrics (EBITDA, EPS, FCF, debt, etc.),
   and out-of-range fiscal years (< 2020 or > 2026).
4. Explicit clarification responses when scope is ambiguous (instead of silently guessing tickers).
5. Deterministic `CoverageNotice` generation for missing fiscal years (e.g. calendar-year filers
   without FY2026 yet), fiscal calendar misalignment between peers, and sector coverage stats.
"""

from __future__ import annotations

import calendar
import re
from datetime import date
from typing import Any

from edgar_capstone.agents.core.schemas import CoverageNotice
from edgar_capstone.common.domain import lookup_year_key, normalize_edgar_ticker

MIN_CORPUS_YEAR = 2020
MAX_CORPUS_YEAR = 2026

SUPPORTED_METRICS_DISPLAY = (
    "Revenue, Operating Income, Net Income, R&D Expense "
    "(plus derived Operating Margin, Net Margin, and R&D Intensity)"
)

# High-precision common brand / colloquial aliases across top S&P 500 issuers
_COMMON_BRAND_ALIASES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bnvidia\b|\bnvda\b", re.IGNORECASE), "NVDA"),
    (re.compile(r"\bapple\b|\baapl\b", re.IGNORECASE), "AAPL"),
    (re.compile(r"\bmicrosoft\b|\bmsft\b", re.IGNORECASE), "MSFT"),
    (re.compile(r"\bbroadcom\b|\bavgo\b", re.IGNORECASE), "AVGO"),
    (re.compile(r"\bsalesforce\b|\bcrm\b", re.IGNORECASE), "CRM"),
    (re.compile(r"\balphabet\b|\bgoogle\b|\bgoogl\b", re.IGNORECASE), "GOOGL"),
    (re.compile(r"\bmeta\s+platforms\b|\bmeta\b|\bfacebook\b", re.IGNORECASE), "META"),
    (re.compile(r"\bamazon\b|\bamzn\b|\baws\b", re.IGNORECASE), "AMZN"),
    (re.compile(r"\btesla\b|\btsla\b", re.IGNORECASE), "TSLA"),
    (re.compile(r"\bwalmart\b|\bwmt\b", re.IGNORECASE), "WMT"),
    (re.compile(r"\bprocter\s*(?:&|and)?\s*gamble\b|\bp&g\b|\bpg\b", re.IGNORECASE), "PG"),
    (re.compile(r"\bjpmorgan\b|\bjp\s+morgan\b|\bjpm\b", re.IGNORECASE), "JPM"),
    (re.compile(r"\bvisa\b", re.IGNORECASE), "V"),
    (re.compile(r"\bunitedhealth\b|\bunh\b", re.IGNORECASE), "UNH"),
    (re.compile(r"\beli\s+lilly\b|\blilly\b|\blly\b", re.IGNORECASE), "LLY"),
    (re.compile(r"\bjohnson\s*(?:&|and)?\s*johnson\b|\bj&j\b|\bjnj\b", re.IGNORECASE), "JNJ"),
    (re.compile(r"\bexxon\s*mobil\b|\bexxon\b|\bxom\b", re.IGNORECASE), "XOM"),
    (re.compile(r"\bcaterpillar\b|\bcat\b", re.IGNORECASE), "CAT"),
    (re.compile(r"\bintel\b|\bintc\b", re.IGNORECASE), "INTC"),
    (re.compile(r"\badvanced\s+micro\s+devices\b|\bamd\b", re.IGNORECASE), "AMD"),
    (re.compile(r"\boracle\b|\borcl\b", re.IGNORECASE), "ORCL"),
    (re.compile(r"\badobe\b|\badbe\b", re.IGNORECASE), "ADBE"),
    (re.compile(r"\bqualcomm\b|\bqcom\b", re.IGNORECASE), "QCOM"),
    (re.compile(r"\bcisco\b|\bcsco\b", re.IGNORECASE), "CSCO"),
    (re.compile(r"\bibm\b|\binternational\s+business\s+machines\b", re.IGNORECASE), "IBM"),
    (re.compile(r"\bnetflix\b|\bnflx\b", re.IGNORECASE), "NFLX"),
    (re.compile(r"\bdisney\b|\bwalt\s+disney\b", re.IGNORECASE), "DIS"),
    (re.compile(r"\bboeing\b", re.IGNORECASE), "BA"),
    (re.compile(r"\bhome\s+depot\b", re.IGNORECASE), "HD"),
    (re.compile(r"\bcostco\b", re.IGNORECASE), "COST"),
    (re.compile(r"\bpfizer\b|\bpfe\b", re.IGNORECASE), "PFE"),
    (re.compile(r"\bmerck\b|\bmrk\b", re.IGNORECASE), "MRK"),
    (re.compile(r"\babbvie\b|\babbv\b", re.IGNORECASE), "ABBV"),
    (re.compile(r"\bgoldman\s+sachs\b", re.IGNORECASE), "GS"),
    (re.compile(r"\bmorgan\s+stanley\b", re.IGNORECASE), "MS"),
    (re.compile(r"\bbank\s+of\s+america\b|\bbofa\b|\bbac\b", re.IGNORECASE), "BAC"),
    (re.compile(r"\bwells\s+fargo\b|\bwfc\b", re.IGNORECASE), "WFC"),
    (re.compile(r"\bcitigroup\b|\bciti\b", re.IGNORECASE), "C"),
    (re.compile(r"\bmastercard\b", re.IGNORECASE), "MA"),
    (re.compile(r"\bberkshire(?:\s+hathaway)?\b", re.IGNORECASE), "BRK-B"),
    (re.compile(r"\bchevron\b|\bcvx\b", re.IGNORECASE), "CVX"),
    (re.compile(r"\bcoca[\s-]cola\b", re.IGNORECASE), "KO"),
    (re.compile(r"\bpepsico\b|\bpepsi\b", re.IGNORECASE), "PEP"),
    (re.compile(r"\bmcdonald'?s\b|\bmcd\b", re.IGNORECASE), "MCD"),
    (re.compile(r"\bstarbucks\b|\bsbux\b", re.IGNORECASE), "SBUX"),
    (re.compile(r"\bnike\b|\bnke\b", re.IGNORECASE), "NKE"),
    (re.compile(r"\bpalantir\b|\bpltr\b", re.IGNORECASE), "PLTR"),
    (re.compile(r"\bservicenow\b", re.IGNORECASE), "NOW"),
    (re.compile(r"\bpalo\s+alto\s+networks\b|\bpanw\b", re.IGNORECASE), "PANW"),
    (re.compile(r"\bcrowdstrike\b|\bcrwd\b", re.IGNORECASE), "CRWD"),
    (re.compile(r"\btexas\s+instruments\b|\btxn\b", re.IGNORECASE), "TXN"),
    (re.compile(r"\bmicron\b", re.IGNORECASE), "MU"),
    (re.compile(r"\bapplied\s+materials\b|\bamat\b", re.IGNORECASE), "AMAT"),
    (re.compile(r"\blam\s+research\b|\blrcx\b", re.IGNORECASE), "LRCX"),
    (re.compile(r"\blockheed\s+martin\b|\blmt\b", re.IGNORECASE), "LMT"),
    (re.compile(r"\bgeneral\s+electric\b|\bge\s+aerospace\b", re.IGNORECASE), "GE"),
    (re.compile(r"\bhoneywell\b|\bhon\b", re.IGNORECASE), "HON"),
    (re.compile(r"\bverizon\b|\bvz\b", re.IGNORECASE), "VZ"),
    (re.compile(r"\bat&t\b", re.IGNORECASE), "T"),
    (re.compile(r"\bcomcast\b|\bcmcsa\b", re.IGNORECASE), "CMCSA"),
    (re.compile(r"\bnext\s*era\b|\bnextera\b", re.IGNORECASE), "NEE"),
    (re.compile(r"\bprologis\b|\bpld\b", re.IGNORECASE), "PLD"),
    (re.compile(r"\blinde\b|\blin\b", re.IGNORECASE), "LIN"),
)

# All 11 GICS sectors
SECTOR_ALIASES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\binformation\s+technology\b|\btech\s+companies\b|\btech\s+sector\b|\btechnology\s+sector\b|\bsemiconductors?\b",
            re.IGNORECASE,
        ),
        "Information Technology",
    ),
    (
        re.compile(r"\bhealth\s*care\b|\bpharma(?:ceuticals?)?\b|\bbiotech\b", re.IGNORECASE),
        "Health Care",
    ),
    (
        re.compile(r"\bfinancials\b|\bfinancial\s+sector\b|\bbanking\b|\bbanks\b", re.IGNORECASE),
        "Financials",
    ),
    (
        re.compile(
            r"\bcommunication\s+services\b|\btelecom(?:munications?)?\s+sector\b|\bmedia\s+sector\b",
            re.IGNORECASE,
        ),
        "Communication Services",
    ),
    (
        re.compile(r"\bconsumer\s+discretionary\b", re.IGNORECASE),
        "Consumer Discretionary",
    ),
    (
        re.compile(r"\bconsumer\s+staples\b", re.IGNORECASE),
        "Consumer Staples",
    ),
    (
        re.compile(
            r"\benergy\s+sector\b|\benergy\s+companies\b|\boil\s+and\s+gas\b", re.IGNORECASE
        ),
        "Energy",
    ),
    (
        re.compile(
            r"\bindustrials\b|\bindustrial\s+sector\b|\bindustrial\s+companies\b", re.IGNORECASE
        ),
        "Industrials",
    ),
    (
        re.compile(r"\butilities\b|\butility\s+sector\b|\butility\s+companies\b", re.IGNORECASE),
        "Utilities",
    ),
    (
        re.compile(r"\breal\s+estate\b|\breits?\b|\breal\s+estate\s+sector\b", re.IGNORECASE),
        "Real Estate",
    ),
    (
        re.compile(
            r"\bmaterials\s+sector\b|\bmaterials\s+companies\b|\bbasic\s+materials\b", re.IGNORECASE
        ),
        "Materials",
    ),
)

# Common English / financial acronyms that are ALSO S&P 500 ticker symbols —
# only match these as tickers when explicitly written as `$TICKER`, via full company name,
# or (for `V`/Visa) in explicit financial/peer comparison context.
_AMBIGUOUS_TICKER_WORDS = frozenset(
    {
        "A",
        "AI",
        "ALL",
        "AM",
        "AN",
        "ARE",
        "AS",
        "AT",
        "BE",
        "BIG",
        "BY",
        "C",
        "CASH",
        "CAT",
        "CEO",
        "CFO",
        "CO",
        "COST",
        "D",
        "DATA",
        "DAY",
        "DE",
        "DO",
        "E",
        "EQ",
        "F",
        "FAST",
        "FIT",
        "FOR",
        "G",
        "GE",
        "GEN",
        "GM",
        "GO",
        "HAS",
        "HD",
        "HE",
        "HIGH",
        "HP",
        "I",
        "IF",
        "IN",
        "INFO",
        "IS",
        "IT",
        "K",
        "KEY",
        "KEYS",
        "L",
        "LOW",
        "M",
        "MA",
        "MAIN",
        "MAR",
        "MAY",
        "ME",
        "MET",
        "MS",
        "MU",
        "MY",
        "N",
        "NET",
        "NEW",
        "NO",
        "NOW",
        "O",
        "OF",
        "ON",
        "ONE",
        "OR",
        "OUT",
        "PAY",
        "PC",
        "PE",
        "PM",
        "POOL",
        "R",
        "RD",
        "REAL",
        "RF",
        "S",
        "SEC",
        "SO",
        "T",
        "TECH",
        "THE",
        "TO",
        "U",
        "UP",
        "US",
        "USA",
        "USD",
        "V",
        "VS",
        "W",
        "WE",
        "WELL",
        "YOY",
        "YUM",
    }
)

# Well-known private, foreign, or non-S&P 500 companies analysts often ask about.
_KNOWN_NON_SP500_ENTITIES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\brivian\b|\brivn\b", re.IGNORECASE), "Rivian (RIVN)"),
    (re.compile(r"\bstripe\b", re.IGNORECASE), "Stripe (private)"),
    (re.compile(r"\bbytedance\b|\btiktok\b", re.IGNORECASE), "ByteDance (private)"),
    (re.compile(r"\bspacex\b", re.IGNORECASE), "SpaceX (private)"),
    (re.compile(r"\bopenai\b", re.IGNORECASE), "OpenAI (private)"),
    (re.compile(r"\banthropic\b", re.IGNORECASE), "Anthropic (private)"),
    (re.compile(r"\bdatabricks\b", re.IGNORECASE), "Databricks (private)"),
    (re.compile(r"\bshopify\b", re.IGNORECASE), "Shopify (SHOP)"),
    (re.compile(r"(?<![A-Za-z0-9\-])SHOP(?![A-Za-z0-9\-])"), "Shopify (SHOP)"),
    (re.compile(r"\bspotify\b", re.IGNORECASE), "Spotify (SPOT)"),
    (re.compile(r"(?<![A-Za-z0-9\-])SPOT(?![A-Za-z0-9\-])"), "Spotify (SPOT)"),
    (re.compile(r"\bsnowflake\b", re.IGNORECASE), "Snowflake (SNOW)"),
    (re.compile(r"(?<![A-Za-z0-9\-])SNOW(?![A-Za-z0-9\-])"), "Snowflake (SNOW)"),
    (re.compile(r"\btsmc\b|\btaiwan\s+semiconductor\b", re.IGNORECASE), "TSMC (TSM)"),
    (re.compile(r"\balibaba\b|\bbaba\b", re.IGNORECASE), "Alibaba (BABA)"),
    (re.compile(r"\btencent\b", re.IGNORECASE), "Tencent"),
    (re.compile(r"\bsamsung\b", re.IGNORECASE), "Samsung Electronics"),
    (re.compile(r"\btoyota\b", re.IGNORECASE), "Toyota Motor (TM)"),
    (re.compile(r"\basml\b", re.IGNORECASE), "ASML Holding (ASML)"),
    (re.compile(r"\bnovo\s+nordisk\b|\bnvo\b", re.IGNORECASE), "Novo Nordisk (NVO)"),
    (re.compile(r"\barm\s+holdings\b", re.IGNORECASE), "Arm Holdings (ARM)"),
    (re.compile(r"\blucid\b|\blcid\b", re.IGNORECASE), "Lucid Group (LCID)"),
    (re.compile(r"\bgamestop\b|\bgme\b", re.IGNORECASE), "GameStop (GME)"),
    (re.compile(r"\broblox\b|\brblx\b", re.IGNORECASE), "Roblox (RBLX)"),
    (re.compile(r"\bnokia\b|\bnok\b", re.IGNORECASE), "Nokia (NOK)"),
    (re.compile(r"\bsap\s+se\b", re.IGNORECASE), "SAP SE (SAP)"),
    (re.compile(r"\bsofi\b", re.IGNORECASE), "SoFi Technologies (SOFI)"),
    (re.compile(r"\bzoom\s+video\b|\bzm\b", re.IGNORECASE), "Zoom (ZM)"),
    (re.compile(r"\bpeloton\b|\bpton\b", re.IGNORECASE), "Peloton (PTON)"),
)

_UNSUPPORTED_FILING_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b10-?q\b|\bform\s+10-?q\b|\bquarterly\s+(?:report|filing|results?)\b", re.IGNORECASE
        ),
        "Form 10-Q (quarterly reports)",
    ),
    (
        re.compile(r"\b8-?k\b|\bform\s+8-?k\b|\bcurrent\s+report\b", re.IGNORECASE),
        "Form 8-K (current event reports)",
    ),
    (
        re.compile(r"\bdef\s*14a\b|\bproxy\s+statement\b", re.IGNORECASE),
        "DEF 14A (proxy statements)",
    ),
    (
        re.compile(r"\bs-?1\b|\bform\s+s-?1\b|\bipo\s+prospectus\b", re.IGNORECASE),
        "Form S-1 (registration statements)",
    ),
    (
        re.compile(r"\b20-?f\b|\bform\s+20-?f\b", re.IGNORECASE),
        "Form 20-F (foreign annual reports)",
    ),
    (
        re.compile(r"\b13-?f\b|\b13-?d\b|\bform\s+4\b", re.IGNORECASE),
        "Form 13F / 13D / Form 4 ownership filings",
    ),
    (
        re.compile(
            r"\bearnings\s+(?:call|transcript|release)\b|\bconference\s+call\b", re.IGNORECASE
        ),
        "earnings call transcripts",
    ),
)

_UNSUPPORTED_METRIC_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:adjusted\s+)?ebitda\b", re.IGNORECASE), "EBITDA"),
    (
        re.compile(
            r"\b(?:diluted\s+|basic\s+|non-?gaap\s+)?(?:eps|earnings\s+per\s+share)\b",
            re.IGNORECASE,
        ),
        "EPS (Earnings Per Share)",
    ),
    (
        re.compile(r"\b(?:free\s+cash\s+flow|fcf|operating\s+cash\s+flow)\b", re.IGNORECASE),
        "Cash Flow / FCF",
    ),
    (
        re.compile(r"\b(?:capex|capital\s+expenditures?)\b", re.IGNORECASE),
        "Capital Expenditures (CapEx)",
    ),
    (
        re.compile(
            r"\b(?:balance\s+sheet|total\s+debt|long-?term\s+debt|debt[\s-]to[\s-]equity|leverage\s+ratio)\b",
            re.IGNORECASE,
        ),
        "Debt / Balance Sheet",
    ),
    (
        re.compile(
            r"\b(?:p/e\s+ratio|price[\s-]to[\s-]earnings|ev/ebitda|valuation\s+multiple|dividend\s+yield|share\s+buybacks?|repurchases?)\b",
            re.IGNORECASE,
        ),
        "Valuation Multiples / Dividends / Buybacks",
    ),
    (
        re.compile(r"\b(?:headcount|employee\s+count|number\s+of\s+employees)\b", re.IGNORECASE),
        "Employee Headcount",
    ),
    (
        re.compile(
            r"\b(?:stock\s+price|share\s+price|market\s+cap(?:italization)?)\b", re.IGNORECASE
        ),
        "Market Price / Market Cap",
    ),
)

_SUPPORTED_TOPIC_RE = re.compile(
    r"\b(?:revenue|sales|top[\s-]?line|operating\s+income|operating\s+margin|operating\s+profit|"
    r"net\s+income|net\s+margin|profit(?:ability)?|r&d|research\s+and\s+development|"
    r"item\s*1a|item\s*7|risk\s+factors?|md&a|wording|disclosure|theme|trend|export\s+control|"
    r"ai\s+risk|cybersecurity|supply\s+chain|tariff|variance|anomal(?:y|ies)|flag)\b",
    re.IGNORECASE,
)

_ANY_4DIGIT_YEAR_RE = re.compile(r"\b(?:FY\s*)?((?:19|20)\d{2})\b", re.IGNORECASE)
_ADDITIVE_FOLLOWUP_RE = re.compile(
    r"\b(?:now\s+add|also\s+(?:add|include|show|compare)|add\s+(?:in\s+)?|and\s+add|"
    r"include\s+[A-Z]|compare\s+(?:that|it|them)?\s*(?:with|to|against))\b",
    re.IGNORECASE,
)
_REMOVE_FOLLOWUP_RE = re.compile(
    r"\b(?:remove|drop|exclude|without)\s+([A-Za-z][A-Za-z0-9&.\- ]{0,30})",
    re.IGNORECASE,
)
_DOLLAR_TICKER_RE = re.compile(r"\$([A-Z]{1,5}(?:[.-][A-Z])?)\b")
_EXPLICIT_TICKER_RE = re.compile(
    r"\b(?:ticker|symbol|stock)\s+([A-Z]{1,5}(?:[.-][A-Z])?)\b", re.IGNORECASE
)
_LEGAL_SUFFIX_RE = re.compile(
    r"\b(?:incorporated|inc\.?|corporation|corp\.?|company|co\.?|holdings?|ltd\.?|plc\.?|"
    r"limited|group|class\s+[a-z]|the|nv|n\.v\.|s\.a\.|ag|se|lp|llc)\b",
    re.IGNORECASE,
)
_COMPANY_NAME_PUNCT_RE = re.compile(r"[/,.\-()]+")

# Precompiled detection regexes
_UPPERCASE_TICKER_RE = re.compile(r"(?<![A-Za-z0-9\-])([A-Z]{2,5}(?:[.-][A-Z])?)(?![A-Za-z0-9\-])")
_STANDALONE_V_RE = re.compile(r"(?<![A-Za-z0-9\-])V(?![A-Za-z0-9\-])")
_ROMAN_NUMERAL_PREFIX_RE = re.compile(
    r"\b(?:part|section|item|title|article|series|schedule|chapter|table|note|"
    r"class|version|phase|stage|tier|gen|level|model|plan|option|type|grade|"
    r"exhibit|appendix)\s+$",
    re.IGNORECASE,
)
_VISA_PRE_CONTEXT_RE = re.compile(
    r"(?:,\s*|\b(?:compare|between|vs\.?|versus|and)\s+)$", re.IGNORECASE
)
_VISA_POST_CONTEXT_RE = re.compile(
    r"^\s+(?:vs\.?|versus|and\s+[A-Z]{2,5}\b|revenue|sales|margin|operating|"
    r"net|income|r&d|10-?k|filings?|risks?|FY\d{4})\b",
    re.IGNORECASE,
)
_CHUNK_HEALTH_RE = re.compile(
    r"\b(?:chunk\s+health|embedding\s+coverage|vector\s+(?:db|database)\s+health)\b",
    re.IGNORECASE,
)
_MENTIONS_10K_RE = re.compile(r"\b10-?k\b|\bannual\s+report\b", re.IGNORECASE)
_ASKED_RD_RE = re.compile(r"\b(?:r&d|research\s+and\s+development)\b", re.IGNORECASE)

# Phrases indicating broad/corpus-wide intent rather than an ambiguous missing-ticker query
_BROAD_CORPUS_PHRASE_RE = re.compile(
    r"\b(?:across\s+all|all\s+s&p\s*500|s&p\s*500\s+companies|entire\s+corpus|"
    r"chunk\s+health|embedding\s+coverage|vector\s+(?:db|database)\s+health)\b",
    re.IGNORECASE,
)


def _parse_iso_date(raw: Any) -> date | None:
    """Parse a `YYYY-MM-DD` string (or pass through a `date`); None when absent or malformed."""
    if isinstance(raw, date):
        return raw
    text = str(raw or "").strip()[:10]
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _normalize_company_name(name: str) -> str:
    cleaned = _COMPANY_NAME_PUNCT_RE.sub(" ", (name or "").lower())
    cleaned = _LEGAL_SUFFIX_RE.sub(" ", cleaned)
    return " ".join(cleaned.split())


def _has_visa_ticker_mention(query: str, *, has_peer_context: bool) -> bool:
    """Detect uppercase standalone `'V'` (Visa) only in explicit financial/peer context.

    Avoids false positives on Roman numerals (`Part V`, `Section V`, `Title V`, `Series V`).
    """
    has_topic = has_peer_context or bool(_SUPPORTED_TOPIC_RE.search(query))
    for match in _STANDALONE_V_RE.finditer(query):
        before = query[: match.start()]
        after = query[match.end() :]
        if _ROMAN_NUMERAL_PREFIX_RE.search(before):
            continue
        if _VISA_POST_CONTEXT_RE.search(after):
            return True
        if has_topic and _VISA_PRE_CONTEXT_RE.search(before):
            return True
    return False


_CANONICAL_DUAL_CLASS_PREFERENCE: dict[str, str] = {
    "GOOG": "GOOGL",
    "FOX": "FOXA",
    "NWS": "NWSA",
}


class EntityCatalog:
    """In-memory lookup built from all ~501 S&P 500 issuers cached from BigQuery."""

    def __init__(self, profiles: list[Any]) -> None:
        self.tickers: set[str] = set()
        self.ticker_to_name: dict[str, str] = {}
        self.ticker_to_sector: dict[str, str] = {}
        self.ticker_to_cik: dict[str, str] = {}
        self.ticker_years: dict[str, set[int]] = {}
        self.profiles_by_key: dict[tuple[str, int], Any] = {}
        # ticker -> fiscal_year -> fiscal period end date, used to map calendar event dates
        # ("the July 2024 outage") onto each issuer's own fiscal calendar.
        self.ticker_period_ends: dict[str, dict[int, date]] = {}
        self._name_patterns: list[tuple[re.Pattern[str], str]] = []

        seen_norm_names: dict[str, str] = {}
        for p in profiles:
            tk = normalize_edgar_ticker(str(getattr(p, "ticker", "") or ""))
            if not tk:
                continue
            self.tickers.add(tk)
            cname = str(getattr(p, "company_name", "") or tk)
            sector = str(getattr(p, "gics_sector", "") or "Unknown")
            cik = str(getattr(p, "cik", "") or "").strip().lstrip("0")
            fy = getattr(p, "fiscal_year", None)
            self.ticker_to_name[tk] = cname
            self.ticker_to_sector[tk] = sector
            if cik:
                self.ticker_to_cik[tk] = cik
            if fy is not None:
                fy_int = int(fy)
                self.ticker_years.setdefault(tk, set()).add(fy_int)
                self.profiles_by_key[(tk, fy_int)] = p
                period_end = _parse_iso_date(getattr(p, "period_of_report", None))
                if period_end is not None:
                    self.ticker_period_ends.setdefault(tk, {})[fy_int] = period_end
            norm_name = _normalize_company_name(cname)
            if len(norm_name) >= 4:
                existing = seen_norm_names.get(norm_name)
                if existing is None or _CANONICAL_DUAL_CLASS_PREFERENCE.get(existing) == tk:
                    seen_norm_names[norm_name] = tk

        # Longest company names first so e.g. "bank of america" matches before "america"
        for norm_name, tk in sorted(
            seen_norm_names.items(), key=lambda kv: len(kv[0]), reverse=True
        ):
            canonical_tk = _CANONICAL_DUAL_CLASS_PREFERENCE.get(tk, tk)
            if canonical_tk in self.tickers:
                tk = canonical_tk
            escaped = re.escape(norm_name).replace(r"\ ", r"\s+")
            self._name_patterns.append((re.compile(rf"\b{escaped}\b", re.IGNORECASE), tk))

    def contains_ticker(self, ticker: str) -> bool:
        return normalize_edgar_ticker(ticker) in self.tickers

    def get_profile(self, ticker: str, fiscal_year: int) -> Any | None:
        """Return the cached profile for `(ticker, fiscal_year)` in O(1), or `None` if absent."""
        return self.profiles_by_key.get((normalize_edgar_ticker(ticker), int(fiscal_year)))

    def fiscal_year_for_date(self, ticker: str, event_date: date) -> int | None:
        """Return the issuer's fiscal year whose period contains `event_date`.

        That is the earliest fiscal year whose `period_of_report` falls on or after the date
        (e.g. CrowdStrike, FYE January 31: July 2024 -> FY2025). A date after the latest filed
        period maps to the next, not-yet-filed fiscal year. Returns None when the catalog has no
        period-end dates for the ticker.
        """
        ends = self.ticker_period_ends.get(normalize_edgar_ticker(ticker))
        if not ends:
            return None
        for fy, period_end in sorted(ends.items(), key=lambda kv: kv[1]):
            if period_end >= event_date:
                return fy
        return max(ends) + 1

    def detect_tickers_in_query(self, query: str) -> tuple[list[str], list[str]]:
        """Return `(detected_known_tickers, unknown_ticker_mentions)` from natural language."""
        found: list[str] = []
        unknown: list[str] = []

        def _add_known(sym: str) -> None:
            norm = normalize_edgar_ticker(sym)
            if norm not in found and (not self.tickers or norm in self.tickers):
                found.append(norm)

        # 1. Explicit `$TICKER` or `ticker XYZ`
        for m in _DOLLAR_TICKER_RE.finditer(query):
            sym = normalize_edgar_ticker(m.group(1))
            if not self.tickers or sym in self.tickers:
                _add_known(sym)
            elif sym not in unknown:
                unknown.append(sym)
        for m in _EXPLICIT_TICKER_RE.finditer(query):
            raw = m.group(1)
            sym = normalize_edgar_ticker(raw)
            if not raw.isalpha() or len(sym) > 5:
                continue
            if sym in _AMBIGUOUS_TICKER_WORDS and not raw.isupper():
                continue
            if not self.tickers or sym in self.tickers:
                _add_known(sym)
            elif sym not in unknown:
                unknown.append(sym)

        # 2. High-precision curated brand aliases (includes "Visa" -> "V")
        for pattern, sym in _COMMON_BRAND_ALIASES:
            if pattern.search(query):
                _add_known(sym)

        # 3. Dynamic S&P 500 uppercase 2-5 char ticker matching (excluding ambiguous words)
        if self.tickers:
            for m in _UPPERCASE_TICKER_RE.finditer(query):
                tok = normalize_edgar_ticker(m.group(1))
                if tok in _AMBIGUOUS_TICKER_WORDS:
                    continue
                if tok in self.tickers:
                    _add_known(tok)

        # 4. Context-guarded standalone uppercase 'V' (Visa) — requires financial/peer context
        if "V" not in found and _has_visa_ticker_mention(query, has_peer_context=bool(found)):
            _add_known("V")

        # 5. Full dynamic S&P 500 company name matching (case-insensitive).
        # Deduplicate dual-class shares by CIK so "Alphabet" resolves to GOOGL, not [GOOG, GOOGL].
        found_ciks = {self.ticker_to_cik[t] for t in found if t in self.ticker_to_cik}
        for pattern, tk in self._name_patterns:
            if tk in found:
                continue
            tk_cik = self.ticker_to_cik.get(tk)
            if tk_cik and tk_cik in found_ciks:
                continue
            if pattern.search(query):
                _add_known(tk)
                if tk_cik:
                    found_ciks.add(tk_cik)

        # 6. Detect well-known private / foreign / non-S&P 500 entities
        for pattern, label in _KNOWN_NON_SP500_ENTITIES:
            if pattern.search(query) and label not in unknown:
                unknown.append(label)

        return found, unknown


# ---------------------------------------------------------------------------
# Scope Validation Rule Helpers
# ---------------------------------------------------------------------------


def _validation_result(
    *,
    action: str,
    reason: str | None = None,
    message: str = "",
    suggested_followups: list[str] | None = None,
    pre_notices: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "action": action,
        "reason": reason,
        "message": message,
        "suggested_followups": list(suggested_followups or []),
        "pre_notices": list(pre_notices or []),
    }


def _check_unsupported_filings(
    query: str, known_tickers: list[str]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    unsupported = [label for pat, label in _UNSUPPORTED_FILING_PATTERNS if pat.search(query)]
    if not unsupported:
        return None, None
    filing_str = ", ".join(unsupported)
    if not _MENTIONS_10K_RE.search(query):
        sample_tk = known_tickers[0] if known_tickers else "NVDA"
        notice = CoverageNotice(
            code="unsupported_filing_type",
            severity="warning",
            message=(
                f"Unsupported filing type requested: {filing_str}. "
                "Only Form 10-K (Item 1A & Item 7) is indexed."
            ),
            details={"requested_filing_types": unsupported},
        ).model_dump(mode="json")
        return (
            _validation_result(
                action="out_of_corpus",
                reason="unsupported_filing_type",
                message=(
                    f"This workspace only indexes annual **SEC Form 10-K filings** (Item 1A Risk Factors "
                    f"and Item 7 MD&A) for S&P 500 companies from **FY{MIN_CORPUS_YEAR}–FY{MAX_CORPUS_YEAR}**. "
                    f"It does not include **{filing_str}**."
                ),
                suggested_followups=[
                    f"Analyze {sample_tk}'s annual 10-K revenue and operating income from FY2024 to FY2026",
                    f"What risk factors changed in {sample_tk}'s Item 1A 10-K filings since FY2024?",
                    "Compare NVIDIA and Apple's R&D spend from 2024 to 2026",
                ],
                pre_notices=[notice],
            ),
            None,
        )
    notice = CoverageNotice(
        code="unsupported_filing_type",
        severity="warning",
        message=(
            f"Only annual Form 10-K filings are indexed; {filing_str} "
            "are excluded from this analysis."
        ),
        details={"requested_filing_types": unsupported},
    ).model_dump(mode="json")
    return None, notice


def _check_unknown_issuers(
    all_unknown: list[str],
    known_tickers: list[str],
    scope: dict[str, Any],
    catalog: EntityCatalog,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not all_unknown:
        return None, None
    unknown_str = ", ".join(all_unknown)
    if not known_tickers and scope.get("gics_sector") == "ALL":
        notice = CoverageNotice(
            code="non_sp500_issuer",
            severity="warning",
            message=f"Issuer(s) outside the S&P 500 corpus: {unknown_str}.",
            tickers=all_unknown,
        ).model_dump(mode="json")
        return (
            _validation_result(
                action="out_of_corpus",
                reason="non_sp500_issuer",
                message=(
                    f"I could not find **{unknown_str}** in the curated **S&P 500** 10-K corpus. "
                    f"This tool covers the {len(catalog.tickers) or 500} S&P 500 constituents "
                    f"(FY{MIN_CORPUS_YEAR}–FY{MAX_CORPUS_YEAR} Form 10-K filings) and does not include "
                    "private companies, non-U.S. primary filers outside the S&P 500, or unindexed tickers."
                ),
                suggested_followups=[
                    "Compare NVIDIA and Apple's R&D spend from 2024 to 2026",
                    "Analyze Tesla's revenue and operating margin trends from FY2024 to FY2026",
                    "How has AI risk wording changed across all tech companies in the S&P 500 since 2024?",
                ],
                pre_notices=[notice],
            ),
            None,
        )
    notice = CoverageNotice(
        code="non_sp500_issuer",
        severity="warning",
        message=(
            f"Excluded out-of-corpus issuer(s) not in the S&P 500 snapshot: {unknown_str}. "
            f"Proceeding with in-corpus ticker(s): {', '.join(known_tickers) or scope.get('gics_sector')}."
        ),
        tickers=all_unknown,
    ).model_dump(mode="json")
    return None, notice


def _check_unsupported_metrics(
    query: str, known_tickers: list[str]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    unsupported = [label for pat, label in _UNSUPPORTED_METRIC_PATTERNS if pat.search(query)]
    if not unsupported:
        return None, None
    metric_str = ", ".join(unsupported)
    if not _SUPPORTED_TOPIC_RE.search(query):
        tk_label = ", ".join(known_tickers) if known_tickers else "NVDA"
        notice = CoverageNotice(
            code="unsupported_metric",
            severity="warning",
            message=(
                f"Unsupported metric(s) requested: {metric_str}. "
                f"Supported deterministic metrics: {SUPPORTED_METRICS_DISPLAY}."
            ),
            details={"unsupported_metrics": unsupported},
        ).model_dump(mode="json")
        return (
            _validation_result(
                action="out_of_corpus",
                reason="unsupported_metric",
                message=(
                    f"**{metric_str}** is not in the deterministic financial metrics table. "
                    "To guarantee 100% mathematical accuracy without LLM estimation, this tool calculates "
                    f"deterministic figures only for **{SUPPORTED_METRICS_DISPLAY}** from XBRL 10-K filings."
                ),
                suggested_followups=[
                    f"Analyze {tk_label}'s revenue, operating income, and net income from FY2024 to FY2026",
                    f"Compare {tk_label} and MSFT R&D spend and operating margin from FY2024 to FY2026",
                    f"Summarize MD&A key performance drivers for {tk_label} in FY2025",
                ],
                pre_notices=[notice],
            ),
            None,
        )
    notice = CoverageNotice(
        code="unsupported_metric",
        severity="warning",
        message=(
            f"Deterministic data is unavailable for {metric_str}. "
            f"Reporting covers supported XBRL metrics ({SUPPORTED_METRICS_DISPLAY}) and 10-K narrative text."
        ),
        details={"unsupported_metrics": unsupported},
    ).model_dump(mode="json")
    return None, notice


def _check_out_of_range_years(
    query: str, known_tickers: list[str]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    all_years = sorted({int(y) for y in _ANY_4DIGIT_YEAR_RE.findall(query)})
    out_of_range = [y for y in all_years if y < MIN_CORPUS_YEAR or y > MAX_CORPUS_YEAR]
    if not out_of_range:
        return None, None
    in_range = [y for y in all_years if MIN_CORPUS_YEAR <= y <= MAX_CORPUS_YEAR]
    yrs_str = ", ".join(f"FY{y}" for y in out_of_range)
    if not in_range:
        tk_label = ", ".join(known_tickers) if known_tickers else "Apple"
        notice = CoverageNotice(
            code="out_of_range_year",
            severity="warning",
            message=(
                f"Requested year(s) {yrs_str} fall outside corpus coverage "
                f"(FY{MIN_CORPUS_YEAR}–FY{MAX_CORPUS_YEAR})."
            ),
            missing_years=out_of_range,
        ).model_dump(mode="json")
        return (
            _validation_result(
                action="out_of_corpus",
                reason="out_of_range_year",
                message=(
                    f"Requested period **{yrs_str}** is outside the curated 10-K corpus range. "
                    f"Indexed SEC Form 10-K filings span **FY{MIN_CORPUS_YEAR} through FY{MAX_CORPUS_YEAR}** "
                    "(FY2026 is currently available for issuers whose fiscal year ended and whose 10-K has been filed)."
                ),
                suggested_followups=[
                    f"Analyze {tk_label}'s revenue and net income from FY2020 to FY2025",
                    f"Analyze {tk_label}'s financial performance from FY2024 to FY2026",
                ],
                pre_notices=[notice],
            ),
            None,
        )
    notice = CoverageNotice(
        code="out_of_range_year",
        severity="warning",
        message=(
            f"Excluded out-of-range fiscal year(s) {yrs_str}; corpus coverage spans "
            f"FY{MIN_CORPUS_YEAR}–FY{MAX_CORPUS_YEAR}."
        ),
        missing_years=out_of_range,
    ).model_dump(mode="json")
    return None, notice


def validate_query_scope_and_coverage(
    *,
    query: str,
    scope: dict[str, Any],
    explicit_tickers: list[str],
    unknown_entities: list[str],
    catalog: EntityCatalog,
    memory_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate whether a query can proceed, requires clarification, or is out of corpus.

    Returns a dictionary:
      - `action`: `"proceed"` | `"clarification"` | `"out_of_corpus"`
      - `reason`: short machine-readable code (or None)
      - `message`: plain-English explanation / clarifying question for non-technical users
      - `suggested_followups`: list of clickable suggested queries
      - `pre_notices`: list of `CoverageNotice` dicts to include even when proceeding
    """
    if _CHUNK_HEALTH_RE.search(query):
        return _validation_result(action="proceed")

    missing_explicit = (
        [t for t in explicit_tickers if not catalog.contains_ticker(t)]
        if (catalog.tickers and explicit_tickers)
        else []
    )
    all_unknown = list(dict.fromkeys([*missing_explicit, *unknown_entities]))
    known_tickers = [
        t for t in scope.get("tickers") or [] if not catalog.tickers or catalog.contains_ticker(t)
    ]

    pre_notices: list[dict[str, Any]] = []
    for block_resp, notice in (
        _check_unsupported_filings(query, known_tickers),
        _check_unknown_issuers(all_unknown, known_tickers, scope, catalog),
        _check_unsupported_metrics(query, known_tickers),
        _check_out_of_range_years(query, known_tickers),
    ):
        if block_resp is not None:
            return block_resp
        if notice is not None:
            pre_notices.append(notice)

    has_tickers = bool(known_tickers)
    has_sector = scope.get("gics_sector", "ALL") not in ("ALL", "", None)
    is_broad_corpus = bool(_BROAD_CORPUS_PHRASE_RE.search(query))
    has_memory = bool(
        isinstance(memory_context, dict)
        and memory_context.get("has_prior_memory")
        and memory_context.get("remembered_tickers")
    )
    if not has_tickers and not has_sector and not is_broad_corpus and not has_memory:
        return _validation_result(
            action="clarification",
            reason="ambiguous_scope",
            message=(
                "Could you clarify **which company (or S&P 500 GICS sector)** you would like me to analyze? "
                "I can examine a single company over time, compare 2+ peers side by side, or track a theme "
                "across an entire GICS sector (e.g., Information Technology, Health Care, Financials, Energy, "
                "Industrials, Utilities, Real Estate, Materials)."
            ),
            suggested_followups=[
                "Analyze NVIDIA's revenue, operating income, and net income from FY2024 to FY2026",
                "Compare NVIDIA and Apple's R&D spend from FY2024 to FY2026",
                "How has AI risk wording changed across all Information Technology companies since 2024?",
                "Compare Microsoft and Amazon's operating margins from FY2024 to FY2025",
            ],
        )

    return _validation_result(action="proceed", pre_notices=pre_notices)


def is_additive_followup(query: str) -> bool:
    """Return True when the user says 'now add Microsoft', 'also include Apple', etc."""
    return bool(_ADDITIVE_FOLLOWUP_RE.search(query))


def extract_removed_tickers(query: str, catalog: EntityCatalog) -> list[str]:
    """Extract tickers the user asked to drop/remove/exclude from the active session scope."""
    m = _REMOVE_FOLLOWUP_RE.search(query)
    if not m:
        return []
    detected, _ = catalog.detect_tickers_in_query(m.group(1))
    return detected


# ---------------------------------------------------------------------------
# Deterministic Coverage Notice Builders
# ---------------------------------------------------------------------------


def _missing_year_notices(variance_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    notices: list[dict[str, Any]] = []
    for rep in variance_reports:
        missing_yrs = [int(y) for y in (rep.get("missing_fiscal_years") or [])]
        if not missing_yrs:
            continue
        tk = str(rep.get("ticker") or "")
        cname = str(rep.get("company_name") or tk)
        avail = [int(y) for y in (rep.get("available_fiscal_years") or [])]
        period_dates = rep.get("period_end_dates") or {}
        latest_fy = avail[-1] if avail else rep.get("end_year")
        latest_period = (
            lookup_year_key(period_dates, int(latest_fy)) if latest_fy is not None else None
        )
        missing_str = ", ".join(f"FY{y}" for y in missing_yrs)
        avail_str = (
            f"FY{avail[0]}–FY{avail[-1]}"
            if len(avail) >= 2
            else (f"FY{avail[0]}" if avail else "none")
        )
        period_note = f" (latest period ended {latest_period})" if latest_period else ""
        fy26_hint = (
            " Most calendar-year filers file FY2026 Form 10-Ks in Q1 2027."
            if 2026 in missing_yrs
            else ""
        )
        notices.append(
            CoverageNotice(
                code="missing_fiscal_year",
                severity="warning",
                message=(
                    f"{cname} ({tk}) does not have a filed 10-K in the corpus for {missing_str}; "
                    f"showing available range {avail_str}{period_note}.{fy26_hint}"
                ),
                tickers=[tk],
                missing_years=missing_yrs,
                details={
                    "available_fiscal_years": avail,
                    "latest_period_of_report": latest_period,
                },
            ).model_dump(mode="json")
        )
    return notices


def _calendar_misalignment_notice(variance_reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(variance_reports) < 2:
        return None
    month_by_ticker: dict[str, tuple[int, str, int]] = {}
    for rep in variance_reports:
        tk = str(rep.get("ticker") or "")
        end_fy = int(rep.get("end_year") or 0)
        p_dates = rep.get("period_end_dates") or {}
        p_str = str(lookup_year_key(p_dates, end_fy) or "").strip()
        if len(p_str) >= 7 and p_str[5:7].isdigit():
            month_by_ticker[tk] = (int(p_str[5:7]), p_str, end_fy)
    if len({m for m, _, _ in month_by_ticker.values()}) < 2:
        return None
    desc_parts = [
        f"{tk}: FY{fy} ended {p_str} ({calendar.month_abbr[m]})"
        for tk, (m, p_str, fy) in sorted(month_by_ticker.items())
    ]
    return CoverageNotice(
        code="fiscal_calendar_misalignment",
        severity="info",
        message=(
            "Fiscal calendar note: compared companies have different fiscal year-ends "
            f"({'; '.join(desc_parts)}). Figures reflect each issuer's own fiscal year."
        ),
        tickers=sorted(month_by_ticker.keys()),
        details={
            "period_ends": {
                tk: {"fiscal_year": fy, "period_of_report": p_str}
                for tk, (_, p_str, fy) in month_by_ticker.items()
            }
        },
    ).model_dump(mode="json")


def _missing_rd_notice(query: str, variance_reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not variance_reports or not _ASKED_RD_RE.search(query):
        return None
    no_rd = [
        str(r["ticker"])
        for r in variance_reports
        if (r.get("research_and_development_variance") or {}).get("direction") == "unavailable"
    ]
    if not no_rd:
        return None
    return CoverageNotice(
        code="missing_metric",
        severity="info",
        message=(
            "No standalone R&D expense line is reported in the XBRL 10-K filings for: "
            f"{', '.join(no_rd)}."
        ),
        tickers=no_rd,
    ).model_dump(mode="json")


def _sector_coverage_notice(sector_coverage: dict[str, Any]) -> dict[str, Any]:
    sec_name = str(sector_coverage.get("gics_sector") or "sector")
    analyzed = int(sector_coverage.get("issuers_analyzed") or 0)
    total = int(sector_coverage.get("issuers_in_sector") or analyzed)
    matched = int(sector_coverage.get("issuers_with_theme_matches") or 0)
    filings_by_yr = sector_coverage.get("filings_by_year") or {}
    yr_parts = [f"FY{y}: {cnt} filings" for y, cnt in sorted(filings_by_yr.items())]
    fy26_missing = int((sector_coverage.get("missing_fy_counts") or {}).get("2026", 0) or 0)
    missing_note = (
        f" ({fy26_missing} issuers have not yet filed FY2026 10-Ks)" if fy26_missing > 0 else ""
    )
    return CoverageNotice(
        code="sector_coverage_summary",
        severity="warning" if fy26_missing > 0 else "info",
        message=(
            f"Full-sector map-reduce analyzed {analyzed}/{total} {sec_name} issuers "
            f"({matched} with theme matches; {', '.join(yr_parts)}){missing_note}."
        ),
        details=dict(sector_coverage),
    ).model_dump(mode="json")


def _profile_coverage_rows(
    profiles: list[Any],
    *,
    tickers: list[str],
    start_year: int,
    end_year: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Variance-report-shaped coverage rows built from the cached BigQuery profile catalog.

    Lets missing-fiscal-year and fiscal-calendar notices be computed for every ticker-scoped
    turn whether or not the agent invoked `calculate_period_variances`.
    """
    lo, hi = sorted((int(start_year), int(end_year)))
    wanted = {normalize_edgar_ticker(t): t for t in tickers}
    by_ticker: dict[str, dict[int, Any]] = {}
    for p in profiles:
        tk = normalize_edgar_ticker(str(getattr(p, "ticker", "") or ""))
        fy = int(getattr(p, "fiscal_year", 0) or 0)
        if tk in wanted and lo <= fy <= hi:
            by_ticker.setdefault(tk, {})[fy] = p
    rows: list[dict[str, Any]] = []
    missing_tickers: list[str] = []
    for norm_tk, display_tk in wanted.items():
        fy_map = by_ticker.get(norm_tk) or {}
        avail = sorted(fy_map)
        if not avail:
            missing_tickers.append(display_tk)
            continue
        latest = fy_map[avail[-1]]
        rows.append(
            {
                "ticker": str(getattr(latest, "ticker", display_tk) or display_tk),
                "company_name": str(getattr(latest, "company_name", "") or display_tk),
                "start_year": avail[0],
                "end_year": avail[-1],
                "available_fiscal_years": avail,
                "missing_fiscal_years": [fy for fy in range(lo, hi + 1) if fy not in fy_map],
                "period_end_dates": {
                    fy: str(getattr(p, "period_of_report", "") or "")
                    for fy, p in fy_map.items()
                    if getattr(p, "period_of_report", None)
                },
            }
        )
    return rows, missing_tickers


def build_coverage_notices(
    *,
    variance_reports: list[dict[str, Any]],
    missing_tickers: list[str] | None = None,
    query: str = "",
    sector_coverage: dict[str, Any] | None = None,
    pre_notices: list[dict[str, Any]] | None = None,
    profiles: list[Any] | None = None,
    tickers: list[str] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
) -> list[dict[str, Any]]:
    """Deterministically build user-facing coverage & fiscal calendar notices.

    When the cached BigQuery `profiles` catalog and the resolved scope (`tickers`, `start_year`,
    `end_year`) are supplied, `missing_ticker`, `missing_fiscal_year`, and
    `fiscal_calendar_misalignment` notices are computed from the catalog, independently of
    whether the agent produced `variance_reports` this turn. Without a catalog (e.g. BigQuery
    unavailable) they fall back to the variance reports.
    """
    notices: list[dict[str, Any]] = list(pre_notices or [])

    coverage_rows: list[dict[str, Any]] = list(variance_reports)
    eff_missing = list(missing_tickers or [])
    if profiles and tickers and start_year is not None and end_year is not None:
        coverage_rows, catalog_missing = _profile_coverage_rows(
            profiles, tickers=list(tickers), start_year=start_year, end_year=end_year
        )
        eff_missing = list(dict.fromkeys([*catalog_missing, *eff_missing]))

    if eff_missing:
        notices.append(
            CoverageNotice(
                code="missing_ticker",
                severity="warning",
                message=(
                    "No financial metric filings were found in the requested fiscal window for: "
                    f"{', '.join(eff_missing)}."
                ),
                tickers=list(eff_missing),
            ).model_dump(mode="json")
        )

    notices.extend(_missing_year_notices(coverage_rows))

    cal_notice = _calendar_misalignment_notice(coverage_rows)
    if cal_notice is not None:
        notices.append(cal_notice)

    notices.extend(
        extend_coverage_notices(
            [], query=query, variance_reports=variance_reports, sector_coverage=sector_coverage
        )
    )
    return notices


def extend_coverage_notices(
    notices: list[dict[str, Any]],
    *,
    query: str = "",
    variance_reports: list[dict[str, Any]] | None = None,
    sector_coverage: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Append notices that depend on agent-invoked tool outputs (R&D gaps, sector coverage).

    Called after tool outputs are hydrated from `session_state`; never duplicates a code.
    """
    out = list(notices)
    codes = {str(n.get("code")) for n in out}
    rd_notice = _missing_rd_notice(query, list(variance_reports or []))
    if rd_notice is not None and "missing_metric" not in codes:
        out.append(rd_notice)
    if sector_coverage and "sector_coverage_summary" not in codes:
        out.append(_sector_coverage_notice(sector_coverage))
    return out


def is_broad_corpus_query(query: str) -> bool:
    """True for corpus-wide phrasing ("across all", "chunk health", ...) that must not inherit
    remembered tickers from session memory (scope resolution gate, not intent classification)."""
    return bool(_BROAD_CORPUS_PHRASE_RE.search(query or ""))
