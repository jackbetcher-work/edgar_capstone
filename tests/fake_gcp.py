"""Deterministic in-memory GCP SDK adapter bundle for offline tests.

Exercises the exact production SDK adapters (`BigQueryAdapter`, `VertexAISearchAdapter`,
`GcsCorpusAdapter`, `VertexAIEmbedder`) against a curated synthetic S&P 500 slice with zero
network traffic, using typed SDK fakes (`FakeBigQueryClient`, `FakeSearchServiceClient`,
`FakeStorageClient`, `FakeGenAiClient`, `FakeFirestoreClient`).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime
from typing import Any

from google.api_core import exceptions as gcp_exceptions
from google.cloud import discoveryengine_v1 as de
from google.genai import types as genai_types
from google.protobuf.struct_pb2 import Struct

from edgar_capstone.agents.adapters import registry as adapter_registry
from edgar_capstone.agents.adapters.registry import GcpToolAdapters, GcpToolConfig
from edgar_capstone.agents.orchestration.chat_engine import _ENTITY_CATALOG_CACHE
from edgar_capstone.agents.tools.sec_tools import (
    _SECTION_CHUNKS_CACHE,
    _SESSION_10K_CONTEXT_CACHE,
)
from edgar_capstone.dataset.extractor import format_combined_10k_markdown

EMBEDDING_DIMS = 768
FAKE_BUCKET = "fde-sec-edgar-sandbox-dev-sec-corpus"


class FakeGcpCredentials:
    """Offline OAuth2 credentials stub for offline SDK adapter tests."""

    def __init__(self, token: str = "fake-gcp-oauth-bearer") -> None:
        self.valid = True
        self.token = token
        self.refresh_calls = 0

    def refresh(self, request: Any) -> None:
        self.refresh_calls += 1
        self.valid = True


def mock_embedding_values(text: str, dims: int = EMBEDDING_DIMS) -> list[float]:
    """Deterministic unit-norm hashed bag-of-words vector mimicking `text-embedding-005`."""
    vec = [0.0] * dims
    for token in re.findall(r"[a-z0-9]+", (text or "").lower()):
        vec[int(hashlib.sha256(token.encode()).hexdigest(), 16) % dims] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    return max(0.0, min(2.0, 1.0 - dot))


# Curated synthetic multi-year profiles (real-world-like S&P 500 figures & calendar ends).
# Includes >= 16 distinct issuers across sectors so `/api/v1/companies` checks pass.
_SYNTHETIC_ISSUERS: dict[str, dict[str, Any]] = {
    "NVDA": {
        "cik": "0001045810",
        "name": "NVIDIA CORP",
        "sector": "Information Technology",
        "sub": "Semiconductors",
        "fy_end_mmdd": "01-25",
        "years": {
            2023: (26.974e9, 4.224e9, 4.368e9, 7.339e9),
            2024: (60.922e9, 32.972e9, 29.760e9, 8.675e9),
            2025: (130.497e9, 81.453e9, 72.880e9, 12.914e9),
            2026: (215.938e9, 130.387e9, 120.067e9, 18.497e9),
        },
        "item_1a": {
            2024: (
                "U.S. government export control regulations imposed licensing requirements on "
                "A100, H100, and A800 data center GPUs shipped to China, restricting our competitive "
                "position in the region and creating supply chain uncertainty."
            ),
            2025: (
                "Expanded U.S. government export control licensing restrictions covering China and "
                "D:5 countries further constrained AI accelerator sales, including modified H20 products, "
                "while wafer foundry and CoWoS packaging lead times tightened."
            ),
            2026: (
                "New U.S. government export control license requirements on H20, Hopper, and Blackwell "
                "AI accelerators shipped to China triggered a $4.5 billion excess inventory and purchase "
                "obligation charge in FY2026, alongside sovereign AI trade tariffs and foundry concentration risks."
            ),
        },
        "item_7": {
            2024: (
                "Fiscal year 2024 revenue rose 125.85% to $60.922 billion driven by Hopper GPU architecture "
                "adoption for generative AI training and cloud data center infrastructure expansion."
            ),
            2025: (
                "Fiscal year 2025 Data Center revenue surged to $115.186 billion on strong Hopper H200 "
                "demand and initial Blackwell production ramps; R&D investment increased to $12.914 billion."
            ),
            2026: (
                "Fiscal year 2026 revenue grew 65.47% to $215.938 billion led by hyperscale cloud "
                "deployment of Blackwell GB200 NVL72 AI infrastructure clusters. We incurred a $4.5 billion "
                "charge for excess inventory and purchase obligations related to H20 export license rules. "
                "Research and development expense grew to $18.497 billion supporting next-generation platforms."
            ),
        },
    },
    "AAPL": {
        "cik": "0000320193",
        "name": "Apple Inc.",
        "sector": "Information Technology",
        "sub": "Technology Hardware, Storage & Peripherals",
        "fy_end_mmdd": "09-28",
        "years": {
            2023: (383.285e9, 114.301e9, 96.995e9, 29.915e9),
            2024: (391.035e9, 123.216e9, 93.736e9, 31.370e9),
            2025: (416.161e9, 133.050e9, 108.400e9, 34.550e9),
            2026: (439.800e9, 141.500e9, 116.200e9, 36.820e9),
        },
        "item_1a": {
            2024: "Global macroeconomic volatility, trade tariffs, and supply chain concentration in Asia could adversely affect hardware production and component availability.",
            2025: "Geopolitical export restrictions, antitrust scrutiny of the App Store, and rapid AI competition present material operational and regulatory risks.",
            2026: "Evolving AI regulation, international tariffs on consumer electronics, and silicon foundry dependency remain key risk factors for Apple Inc.",
        },
        "item_7": {
            2024: "Apple Inc. fiscal 2024 revenue grew 2.02% to $391.035 billion led by Services gross margin expansion and $31.370 billion in R&D investments across Apple Silicon and Apple Intelligence.",
            2025: "Services revenue hit record highs and iPhone upgrades accelerated alongside Apple Intelligence rollout; R&D expense increased to $34.550 billion.",
            2026: "Fiscal 2026 net sales reached $439.800 billion driven by Services expansion, AI silicon integration, and higher research and development spending of $36.820 billion.",
        },
    },
    "MSFT": {
        "cik": "0000789019",
        "name": "MICROSOFT CORP",
        "sector": "Information Technology",
        "sub": "Systems Software",
        "fy_end_mmdd": "06-30",
        "years": {
            2023: (211.915e9, 88.523e9, 72.361e9, 27.195e9),
            2024: (245.122e9, 109.433e9, 88.136e9, 29.510e9),
            2025: (281.724e9, 128.528e9, 101.832e9, 32.488e9),
            2026: (312.400e9, 144.200e9, 114.600e9, 35.900e9),
        },
        "item_1a": {
            2024: "Intense cloud and AI infrastructure competition, cybersecurity threats, and data center capital expenditure risks could pressure operating margins.",
            2025: "Scaling GPU data center capacity, energy supply constraints, and AI model governance regulations create execution risk across Intelligent Cloud.",
            2026: "AI infrastructure depreciation, sovereign cloud compliance rules, and enterprise cybersecurity threats remain significant risk factors.",
        },
        "item_7": {
            2024: "Microsoft Intelligent Cloud revenue expanded on Azure AI workloads and Copilot adoption, while research and development expense rose to $29.510 billion.",
            2025: "Azure cloud and AI services drove fiscal 2025 revenue to $281.724 billion with operating income of $128.528 billion and R&D investment of $32.488 billion.",
            2026: "Fiscal 2026 revenue grew to $312.400 billion supported by Azure AI infrastructure demand and $35.900 billion in research and development investment.",
        },
    },
    "INTC": {
        "cik": "0000050863",
        "name": "INTEL CORP",
        "sector": "Information Technology",
        "sub": "Semiconductors",
        "fy_end_mmdd": "12-28",
        "years": {
            2023: (54.228e9, 0.093e9, 1.689e9, 16.046e9),
            2024: (53.101e9, -11.678e9, -18.756e9, 16.546e9),
            2025: (51.400e9, -2.400e9, -2.500e9, 15.200e9),
        },
        "item_1a": {
            2023: "Process technology execution delays and semiconductor cyclicality could impair manufacturing utilization and foundry profitability.",
            2024: "Substantial foundry capital commitments, asset impairments, and restructuring costs significantly heighten net loss exposure.",
            2025: "Intel Foundry losses, competitive AI accelerator pressure, and manufacturing node yield challenges continue to affect profitability.",
        },
        "item_7": {
            2023: "Intel recorded FY2023 revenue of $54.228 billion and positive net income of $1.689 billion amidst PC market recovery.",
            2024: "Intel net income turned negative to a loss of $(18.756) billion in FY2024 due to $15.9 billion in manufacturing asset impairment, valuation allowances on deferred tax assets, and restructuring charges.",
            2025: "In FY2025, Intel narrowed its net loss to $(2.500) billion as restructuring expenses moderated, though Intel Foundry operating losses and low fab utilization persisted.",
        },
    },
    "AMD": {
        "cik": "0000002488",
        "name": "ADVANCED MICRO DEVICES INC",
        "sector": "Information Technology",
        "sub": "Semiconductors",
        "fy_end_mmdd": "12-28",
        "years": {
            2023: (22.680e9, 0.401e9, 0.854e9, 5.872e9),
            2024: (25.785e9, 1.900e9, 1.641e9, 6.456e9),
            2025: (30.500e9, 3.800e9, 3.200e9, 7.100e9),
        },
    },
    "AVGO": {
        "cik": "0001730168",
        "name": "Broadcom Inc.",
        "sector": "Information Technology",
        "sub": "Semiconductors",
        "fy_end_mmdd": "11-02",
        "years": {
            2024: (51.574e9, 13.462e9, 5.895e9, 9.310e9),
            2025: (61.200e9, 21.000e9, 14.500e9, 10.100e9),
        },
    },
    "ORCL": {
        "cik": "0001341439",
        "name": "ORACLE CORP",
        "sector": "Information Technology",
        "sub": "Systems Software",
        "fy_end_mmdd": "05-31",
        "years": {
            2024: (52.961e9, 15.353e9, 10.467e9, 8.915e9),
            2025: (57.400e9, 17.500e9, 12.100e9, 9.400e9),
        },
    },
    "CRM": {
        "cik": "0001108524",
        "name": "Salesforce, Inc.",
        "sector": "Information Technology",
        "sub": "Application Software",
        "fy_end_mmdd": "01-31",
        "years": {
            2024: (34.857e9, 5.011e9, 4.136e9, 4.906e9),
            2025: (37.895e9, 7.205e9, 6.197e9, 5.493e9),
        },
    },
    "CSCO": {
        "cik": "0000858877",
        "name": "CISCO SYSTEMS, INC.",
        "sector": "Information Technology",
        "sub": "Communications Equipment",
        "fy_end_mmdd": "07-27",
        "years": {
            2024: (53.803e9, 12.170e9, 10.320e9, 7.917e9),
            2025: (56.200e9, 13.400e9, 11.200e9, 8.200e9),
        },
    },
    "AMZN": {
        "cik": "0001018724",
        "name": "AMAZON COM INC",
        "sector": "Consumer Discretionary",
        "sub": "Broadline Retail",
        "fy_end_mmdd": "12-31",
        "years": {
            2024: (637.959e9, 68.593e9, 59.248e9, 88.544e9),
            2025: (704.000e9, 81.200e9, 68.500e9, 96.000e9),
            2026: (765.000e9, 92.400e9, 77.200e9, 104.000e9),
        },
    },
    "LLY": {
        "cik": "0000059478",
        "name": "ELI LILLY & Co",
        "sector": "Health Care",
        "sub": "Pharmaceuticals",
        "fy_end_mmdd": "12-31",
        "years": {
            2024: (45.043e9, 12.899e9, 10.590e9, 10.991e9),
            2025: (58.300e9, 19.200e9, 15.800e9, 12.400e9),
            2026: (69.500e9, 24.800e9, 20.100e9, 13.900e9),
        },
    },
    "JPM": {
        "cik": "0000019617",
        "name": "JPMORGAN CHASE & CO",
        "sector": "Financials",
        "sub": "Diversified Banks",
        "fy_end_mmdd": "12-31",
        "years": {
            2024: (177.556e9, 72.100e9, 58.471e9, None),
            2025: (184.200e9, 75.500e9, 60.800e9, None),
            2026: (192.000e9, 79.100e9, 63.400e9, None),
        },
    },
    "GOOGL": {
        "cik": "0001652044",
        "name": "Alphabet Inc.",
        "sector": "Communication Services",
        "sub": "Interactive Media & Services",
        "fy_end_mmdd": "12-31",
        "years": {
            2024: (350.018e9, 112.390e9, 100.118e9, 49.326e9),
            2025: (385.000e9, 126.000e9, 111.500e9, 53.200e9),
        },
    },
    "META": {
        "cik": "0001326801",
        "name": "Meta Platforms, Inc.",
        "sector": "Communication Services",
        "sub": "Interactive Media & Services",
        "fy_end_mmdd": "12-31",
        "years": {
            2024: (164.501e9, 69.380e9, 62.360e9, 43.873e9),
            2025: (189.000e9, 78.500e9, 69.800e9, 48.900e9),
        },
    },
    "PLD": {
        "cik": "0001045609",
        "name": "Prologis, Inc.",
        "sector": "Real Estate",
        "sub": "Industrial REITs",
        "fy_end_mmdd": "12-31",
        "years": {
            2020: (4.439e9, 1.800e9, 1.473e9, None),
            2023: (8.023e9, 3.600e9, 3.053e9, None),
            2024: (8.202e9, 3.720e9, 3.726e9, None),
            2025: (8.850e9, 4.010e9, 3.920e9, None),
        },
        "item_1a": {
            2020: "Pandemic supply chain disruptions and industrial warehouse leasing demand uncertainty affect occupancy.",
            2024: "Elevated interest rates, refinancing costs, and competition for powered data center land sites impact development returns.",
            2025: "Expanding AI data center power constraints, higher cap rates, and global trade tariff volatility pose leasing risks.",
        },
        "item_7": {
            2020: "Prologis experienced resilient e-commerce logistics warehouse demand across coastal markets.",
            2024: "Prologis initiated multiple hyperscale data center conversions while navigating elevated interest rate refinancing conditions.",
            2025: "Data center development pipeline expanded rapidly to serve AI cloud tenants alongside steady industrial rent growth.",
        },
    },
    "EQIX": {
        "cik": "0001101239",
        "name": "EQUINIX INC",
        "sector": "Real Estate",
        "sub": "Data Center REITs",
        "fy_end_mmdd": "12-31",
        "years": {
            2020: (5.999e9, 1.078e9, 0.370e9, None),
            2024: (8.748e9, 1.380e9, 0.815e9, None),
            2025: (9.420e9, 1.610e9, 1.020e9, None),
        },
    },
    "JNJ": {
        "cik": "0000200406",
        "name": "JOHNSON & JOHNSON",
        "sector": "Health Care",
        "sub": "Pharmaceuticals",
        "fy_end_mmdd": "12-31",
        "years": {
            2024: (88.821e9, 20.400e9, 14.066e9, 17.232e9),
            2025: (92.500e9, 22.100e9, 16.500e9, 17.900e9),
        },
    },
    "XOM": {
        "cik": "0000034088",
        "name": "EXXON MOBIL CORP",
        "sector": "Energy",
        "sub": "Integrated Oil & Gas",
        "fy_end_mmdd": "12-31",
        "years": {
            2024: (349.585e9, 46.200e9, 33.680e9, 0.960e9),
            2025: (361.000e9, 48.100e9, 35.200e9, 1.010e9),
        },
    },
}


def _build_synthetic_corpus() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """Build deterministic profile dicts, chunk dicts, and GCS markdown objects."""
    profiles: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    markdowns: dict[str, str] = {}

    for ticker, meta in _SYNTHETIC_ISSUERS.items():
        cik = meta["cik"]
        name = meta["name"]
        sector = meta["sector"]
        sub = meta["sub"]
        mmdd = meta["fy_end_mmdd"]

        for fy, (rev, op, net, rd) in sorted(meta["years"].items()):
            accn = f"{cik}-{str(fy)[-2:]}-000021"
            period = f"{fy}-{mmdd}"
            filing_date = f"{fy}-02-26" if mmdd.startswith("01") else f"{fy}-11-01"
            sec_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{ticker}_{fy}_10k.htm"
            blob_md = f"unstructured/{ticker}/{fy}_10k.md"
            blob_json = f"structured/{ticker}/{fy}_financials.json"
            gcs_md = f"gs://{FAKE_BUCKET}/{blob_md}"
            gcs_json = f"gs://{FAKE_BUCKET}/{blob_json}"

            item_1a_text = (meta.get("item_1a") or {}).get(
                fy,
                f"{name} FY{fy} risk factors include macroeconomic uncertainty, regulatory compliance, "
                f"competitive pricing pressure, and supply chain dependencies across the {sector} sector.",
            )
            item_7_text = (meta.get("item_7") or {}).get(
                fy,
                f"{name} FY{fy} revenue was ${rev / 1e9:.2f} billion with operating income of "
                f"${op / 1e9:.2f} billion and net income of ${net / 1e9:.2f} billion driven by core segment demand.",
            )

            profiles.append(
                {
                    "profile_id": f"{ticker}_{fy}",
                    "ticker": ticker,
                    "cik": cik,
                    "company_name": name,
                    "gics_sector": sector,
                    "gics_sub_industry": sub,
                    "fiscal_year": fy,
                    "period_of_report": period,
                    "filing_date": filing_date,
                    "accession_number": accn,
                    "currency": "USD",
                    "revenue": rev,
                    "operating_income": op,
                    "net_income": net,
                    "research_and_development": rd,
                    "metric_provenance": {
                        "revenue": "us-gaap:Revenues",
                        "operating_income": "us-gaap:OperatingIncomeLoss",
                        "net_income": "us-gaap:NetIncomeLoss",
                        "research_and_development": (
                            "us-gaap:ResearchAndDevelopmentExpense" if rd is not None else None
                        ),
                        "notes": ["Synthetic test fixture XBRL provenance"],
                    },
                    "extracted_sections": ["item_1a", "item_7"],
                    "unstructured_blob_path": blob_md,
                    "structured_blob_path": blob_json,
                    "unstructured_gcs_uri": gcs_md,
                    "structured_gcs_uri": gcs_json,
                    "sec_filing_url": sec_url,
                }
            )

            markdowns[blob_md] = format_combined_10k_markdown(
                ticker=ticker,
                company_name=name,
                cik=cik,
                gics_sector=sector,
                gics_sub_industry=sub,
                fiscal_year=fy,
                period_of_report=period,
                filing_date=filing_date,
                accession_number=accn,
                sec_filing_url=sec_url,
                item_1a_markdown=item_1a_text,
                item_7_markdown=item_7_text,
            )

            for sec_id, sec_title, body in (
                ("item_1a", "Item 1A. Risk Factors", item_1a_text),
                ("item_7", "Item 7. MD&A", item_7_text),
            ):
                chunk_text = f"[{ticker} | FY{fy} | {sec_title}] {body}"
                chunks.append(
                    {
                        "chunk_id": f"{ticker}_{fy}_{sec_id}_0001",
                        "profile_id": f"{ticker}_{fy}",
                        "ticker": ticker,
                        "company_name": name,
                        "gics_sector": sector,
                        "fiscal_year": fy,
                        "period_of_report": period,
                        "filing_date": filing_date,
                        "accession_number": accn,
                        "section_id": sec_id,
                        "section_title": sec_title,
                        "chunk_index": 1,
                        "chunk_text": chunk_text,
                        "embedding": mock_embedding_values(chunk_text),
                        "sec_filing_url": sec_url,
                        "unstructured_gcs_uri": gcs_md,
                    }
                )

    return profiles, chunks, markdowns


_CACHED_CORPUS: tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]] | None = None


def _get_synthetic_corpus() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    global _CACHED_CORPUS
    if _CACHED_CORPUS is None:
        _CACHED_CORPUS = _build_synthetic_corpus()
    return _CACHED_CORPUS


def _extract_sdk_bq_params(job_config: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if job_config is None:
        return out
    for param in getattr(job_config, "query_parameters", None) or ():
        name = getattr(param, "name", None)
        if not name:
            continue
        if hasattr(param, "values"):
            out[name] = list(param.values)
        elif hasattr(param, "value"):
            out[name] = param.value
    return out


def _filter_items(items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    start_year = int(params["start_year"]) if params.get("start_year") is not None else 1900
    end_year = int(params["end_year"]) if params.get("end_year") is not None else 2100
    tickers = {str(t).upper() for t in (params.get("tickers") or [])}
    single_ticker = str(params["ticker"]).upper() if params.get("ticker") else None
    single_fy = int(params["fiscal_year"]) if params.get("fiscal_year") is not None else None
    gics_sector = str(params["gics_sector"]).lower() if params.get("gics_sector") else None
    sectors = {str(s).lower() for s in (params.get("sectors") or [])}
    section_id = str(params["section_id"]).lower() if params.get("section_id") else None

    matched: list[dict[str, Any]] = []
    for item in items:
        fy = int(item["fiscal_year"])
        if single_fy is not None:
            if fy != single_fy:
                continue
        elif not (start_year <= fy <= end_year):
            continue
        t = str(item["ticker"]).upper()
        if single_ticker and t != single_ticker:
            continue
        if tickers and t not in tickers:
            continue
        sec_val = str(item.get("gics_sector") or "").lower()
        if gics_sector and sec_val != gics_sector:
            continue
        if sectors and sec_val not in sectors:
            continue
        if section_id and "section_id" in item and str(item["section_id"]).lower() != section_id:
            continue
        matched.append(item)
    return matched


def _parse_vais_filter(expr: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    m_start = re.search(r"fiscal_year\s*>=\s*(\d+)", expr)
    m_end = re.search(r"fiscal_year\s*<=\s*(\d+)", expr)
    if m_start:
        params["start_year"] = int(m_start.group(1))
    if m_end:
        params["end_year"] = int(m_end.group(1))
    m_tickers = re.search(r"ticker:\s*ANY\(([^)]+)\)", expr)
    if m_tickers:
        params["tickers"] = re.findall(r'"([^"]+)"', m_tickers.group(1))
    m_sector = re.search(r"gics_sector:\s*ANY\(([^)]+)\)", expr)
    if m_sector:
        vals = re.findall(r'"([^"]+)"', m_sector.group(1))
        if vals:
            params["gics_sector"] = vals[0]
    m_sec = re.search(r"section_id:\s*ANY\(([^)]+)\)", expr)
    if m_sec:
        vals = re.findall(r'"([^"]+)"', m_sec.group(1))
        if vals:
            params["section_id"] = vals[0]
    return params


def _dict_to_struct(data: dict[str, Any]) -> Struct:
    st = Struct()
    st.update(data)
    return st


class _FakeQueryJob:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def result(
        self, *, timeout: float | None = None, max_results: int | None = None
    ) -> list[dict[str, Any]]:
        _ = timeout
        return self._rows[:max_results] if max_results is not None else list(self._rows)


class FakeBigQueryClient:
    """Typed in-memory ``google.cloud.bigquery.Client`` fake serving the synthetic corpus."""

    def __init__(
        self,
        profiles: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        requests_log: list[tuple[str, str]],
    ) -> None:
        self._profiles = profiles
        self._chunks = chunks
        self._log = requests_log
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def query(
        self,
        sql: str,
        *,
        job_config: Any | None = None,
        location: str | None = None,
        timeout: float | None = None,
    ) -> _FakeQueryJob:
        _ = (location, timeout)
        self._log.append(("POST", "bigquery:query"))
        params = _extract_sdk_bq_params(job_config)

        # 1. VECTOR_SEARCH or QUALIFY ROW_NUMBER sector vector search
        if "VECTOR_SEARCH(" in sql or "QUALIFY ROW_NUMBER()" in sql:
            q_emb = [float(x) for x in (params.get("query_embedding") or [0.0] * EMBEDDING_DIMS)]
            pool = _filter_items(self._chunks, params)
            scored = [(c, _cosine_distance(q_emb, c["embedding"])) for c in pool]
            if "QUALIFY ROW_NUMBER()" in sql:
                per_cy = int(params.get("per_company_year") or 3)
                grouped: dict[tuple[str, int], list[tuple[dict[str, Any], float]]] = defaultdict(
                    list
                )
                for item, dist in scored:
                    grouped[(item["ticker"], item["fiscal_year"])].append((item, dist))
                rows_out = []
                for key in sorted(grouped):
                    rows_out.extend(sorted(grouped[key], key=lambda x: x[1])[:per_cy])
            else:
                top_k = int(params.get("top_k") or 10)
                rows_out = sorted(scored, key=lambda x: x[1])[:top_k]

            return _FakeQueryJob(
                [
                    {
                        **{k: v for k, v in c.items() if k != "embedding"},
                        "distance": dist,
                    }
                    for c, dist in rows_out
                ]
            )

        # 2. fetch_section_chunks / batch
        if "filing_chunks" in sql:
            include_emb = ", embedding" in sql
            matched = _filter_items(self._chunks, params)
            return _FakeQueryJob(
                [
                    dict(c) if include_emb else {k: v for k, v in c.items() if k != "embedding"}
                    for c in matched
                ]
            )

        # 3. v_chunk_health_metrics
        if "v_chunk_health_metrics" in sql:
            total_c = len(self._chunks)
            return _FakeQueryJob(
                [
                    {
                        "gics_sector": "ALL_SECTORS",
                        "section_id": "ALL_SECTIONS",
                        "health_status": "HEALTHY",
                        "health_score": 100.0,
                        "total_chunks": total_c,
                        "distinct_profiles": len(self._profiles),
                        "distinct_tickers": len(_SYNTHETIC_ISSUERS),
                        "valid_embedding_count": total_c,
                        "embedding_coverage_ratio": 1.0,
                        "full_section_coverage_ratio": 1.0,
                        "avg_char_count": 420.0,
                        "avg_token_estimate": 105.0,
                        "min_token_estimate": 40,
                        "max_token_estimate": 260,
                        "p50_token_estimate": 100.0,
                        "p95_token_estimate": 220.0,
                        "undersized_chunk_count": 0,
                        "oversized_chunk_count": 0,
                        "missing_metadata_count": 0,
                    }
                ]
            )

        # 4. v_financial_variances_yoy
        if "v_financial_variances_yoy" in sql:
            matched = _filter_items(self._profiles, params)
            by_t: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
            for p in self._profiles:
                by_t[p["ticker"]][p["fiscal_year"]] = p

            def _yoy(
                cur_v: float | None, prev_v: float | None
            ) -> tuple[float | None, float | None]:
                if cur_v is None or prev_v is None:
                    return None, None
                d = cur_v - prev_v
                pct = (d / abs(prev_v) * 100.0) if prev_v else None
                return d, pct

            out_rows: list[dict[str, Any]] = []
            for p in matched:
                prev = by_t[p["ticker"]].get(p["fiscal_year"] - 1)
                r_d, r_p = _yoy(p["revenue"], prev["revenue"] if prev else None)
                o_d, o_p = _yoy(p["operating_income"], prev["operating_income"] if prev else None)
                n_d, n_p = _yoy(p["net_income"], prev["net_income"] if prev else None)
                rd_d, rd_p = _yoy(
                    p["research_and_development"],
                    prev["research_and_development"] if prev else None,
                )
                out_rows.append(
                    {
                        "profile_id": p["profile_id"],
                        "ticker": p["ticker"],
                        "company_name": p["company_name"],
                        "gics_sector": p["gics_sector"],
                        "fiscal_year": p["fiscal_year"],
                        "prev_fiscal_year": prev["fiscal_year"] if prev else None,
                        "period_of_report": date.fromisoformat(p["period_of_report"]),
                        "filing_date": date.fromisoformat(p["filing_date"]),
                        "accession_number": p["accession_number"],
                        "revenue": p["revenue"],
                        "revenue_yoy_delta": r_d,
                        "revenue_yoy_pct": r_p,
                        "operating_income": p["operating_income"],
                        "operating_income_yoy_delta": o_d,
                        "operating_income_yoy_pct": o_p,
                        "net_income": p["net_income"],
                        "net_income_yoy_delta": n_d,
                        "net_income_yoy_pct": n_p,
                        "research_and_development": p["research_and_development"],
                        "research_and_development_yoy_delta": rd_d,
                        "research_and_development_yoy_pct": rd_p,
                        "sec_filing_url": p["sec_filing_url"],
                    }
                )
            return _FakeQueryJob(out_rows)

        # 5. sector_benchmarks (GROUP BY 1, 2)
        if "GROUP BY 1, 2" in sql:
            matched = _filter_items(self._profiles, params)
            buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
            for p in matched:
                buckets[(p["gics_sector"], p["fiscal_year"])].append(p)
            out_rows = []
            for (sec, fy), items in sorted(buckets.items()):
                revs = [i["revenue"] for i in items if i["revenue"] is not None]
                ops = [i["operating_income"] for i in items if i["operating_income"] is not None]
                nets = [i["net_income"] for i in items if i["net_income"] is not None]
                rds = [
                    i["research_and_development"]
                    for i in items
                    if i["research_and_development"] is not None
                ]
                out_rows.append(
                    {
                        "gics_sector": sec,
                        "fiscal_year": fy,
                        "company_count": len(items),
                        "tickers": sorted({i["ticker"] for i in items}),
                        "total_sector_revenue": round(sum(revs), 2),
                        "avg_company_revenue": round(sum(revs) / len(revs), 2) if revs else 0.0,
                        "total_sector_operating_income": round(sum(ops), 2),
                        "total_sector_net_income": round(sum(nets), 2),
                        "total_sector_research_and_development": (
                            round(sum(rds), 2) if rds else None
                        ),
                    }
                )
            return _FakeQueryJob(out_rows)

        # 6. Default: financial_metrics fetch_profiles (returns native date/datetime types!)
        matched = _filter_items(self._profiles, params)
        matched.sort(key=lambda p: (p["ticker"], p["fiscal_year"]))
        return _FakeQueryJob(
            [
                {
                    **p,
                    "period_of_report": date.fromisoformat(p["period_of_report"]),
                    "filing_date": date.fromisoformat(p["filing_date"]),
                    "ingested_at": datetime(2026, 1, 15, 0, 0, 0, tzinfo=UTC),
                }
                for p in matched
            ]
        )


class FakeSearchServiceClient:
    """Typed in-memory ``discoveryengine_v1.SearchServiceClient`` fake returning real protos."""

    def __init__(self, chunks: list[dict[str, Any]], requests_log: list[tuple[str, str]]) -> None:
        self._chunks = chunks
        self._log = requests_log

    def search(self, *, request: Any, timeout: float | None = None) -> de.SearchResponse:
        _ = timeout
        serving_config = str(getattr(request, "serving_config", ""))
        self._log.append(("POST", f"discoveryengine:search:{serving_config}"))
        query = str(getattr(request, "query", "") or "")
        q_vec = mock_embedding_values(query)
        filt = _parse_vais_filter(str(getattr(request, "filter", "") or ""))
        page_size = int(getattr(request, "page_size", 10) or 10)
        pool = _filter_items(self._chunks, filt)
        ranked = sorted(pool, key=lambda c: _cosine_distance(q_vec, c["embedding"]))[:page_size]

        results = []
        for rank, c in enumerate(ranked):
            struct_data = _dict_to_struct(
                {
                    "chunk_id": c["chunk_id"],
                    "profile_id": c["profile_id"],
                    "ticker": c["ticker"],
                    "company_name": c["company_name"],
                    "gics_sector": c["gics_sector"],
                    "fiscal_year": c["fiscal_year"],
                    "period_of_report": c["period_of_report"],
                    "filing_date": c["filing_date"],
                    "accession_number": c["accession_number"],
                    "section_id": c["section_id"],
                    "section_title": c["section_title"],
                    "sec_filing_url": c["sec_filing_url"],
                    "unstructured_gcs_uri": c["unstructured_gcs_uri"],
                    "chunk_text": c["chunk_text"],
                }
            )
            derived_data = _dict_to_struct(
                {
                    "extractive_segments": [
                        {
                            "content": c["chunk_text"],
                            "relevanceScore": max(0.35, 0.92 - 0.04 * rank),
                            "section_id": c["section_id"],
                            "section_title": c["section_title"],
                        }
                    ]
                }
            )
            results.append(
                de.SearchResponse.SearchResult(
                    id=c["chunk_id"],
                    document=de.Document(
                        id=c["chunk_id"],
                        struct_data=struct_data,
                        derived_struct_data=derived_data,
                    ),
                )
            )
        return de.SearchResponse(results=results, total_size=len(results))


class _FakeBlob:
    def __init__(
        self,
        bucket: str,
        name: str,
        markdowns: dict[str, str],
        requests_log: list[tuple[str, str]],
    ) -> None:
        self._bucket = bucket
        self._name = name
        self._markdowns = markdowns
        self._log = requests_log

    def download_as_text(
        self,
        *,
        encoding: str = "utf-8",
        retry: Any = None,
        timeout: float | None = None,
    ) -> str:
        _ = (encoding, retry, timeout)
        self._log.append(("GET", f"gcs://{self._bucket}/{self._name}"))
        if self._name not in self._markdowns:
            raise gcp_exceptions.NotFound(f"No such object: {self._name}")
        return self._markdowns[self._name]


class _FakeBucket:
    def __init__(
        self, name: str, markdowns: dict[str, str], requests_log: list[tuple[str, str]]
    ) -> None:
        self._name = name
        self._markdowns = markdowns
        self._log = requests_log

    def blob(self, object_name: str) -> _FakeBlob:
        return _FakeBlob(self._name, object_name, self._markdowns, self._log)


class FakeStorageClient:
    """Typed in-memory ``google.cloud.storage.Client`` fake serving synthetic 10-K Markdown."""

    def __init__(self, markdowns: dict[str, str], requests_log: list[tuple[str, str]]) -> None:
        self._markdowns = markdowns
        self._log = requests_log

    def bucket(self, name: str) -> _FakeBucket:
        return _FakeBucket(name, self._markdowns, self._log)


class _FakeGenAiModels:
    def __init__(self, requests_log: list[tuple[str, str]]) -> None:
        self._log = requests_log

    def embed_content(
        self,
        *,
        model: str,
        contents: Sequence[str] | str,
        config: Any | None = None,
    ) -> genai_types.EmbedContentResponse:
        self._log.append(("POST", f"genai:embed_content:{model}"))
        items = [contents] if isinstance(contents, str) else list(contents)
        dims = int(getattr(config, "output_dimensionality", None) or EMBEDDING_DIMS)
        return genai_types.EmbedContentResponse(
            embeddings=[
                genai_types.ContentEmbedding(values=mock_embedding_values(str(t), dims=dims))
                for t in items
            ]
        )


class FakeGenAiClient:
    """Typed in-memory ``google.genai.Client`` fake for deterministic embeddings."""

    def __init__(self, requests_log: list[tuple[str, str]]) -> None:
        self.models = _FakeGenAiModels(requests_log)


# ---------------------------------------------------------------------------
# Reusable in-memory Firestore SDK fake (feedback + transactional sessions)
# ---------------------------------------------------------------------------


class _FakeDocSnapshot:
    def __init__(self, doc_id: str, data: dict[str, Any] | None) -> None:
        self.id = doc_id
        self._data = dict(data) if data is not None else None
        self.exists = data is not None

    def to_dict(self) -> dict[str, Any] | None:
        return dict(self._data) if self._data is not None else None


class _FakeFirestoreTransaction:
    """Minimal ``google.cloud.firestore.Transaction`` compatible with ``@firestore.transactional``."""

    def __init__(self, client: FakeFirestoreClient) -> None:
        self._client = client
        self._id: bytes | None = None
        self._max_attempts = 5
        self._read_only = False
        self._writes: list[tuple[str, str, dict[str, Any]]] = []

    @property
    def in_progress(self) -> bool:
        return self._id is not None

    def _begin(self, retry_id: bytes | None = None) -> None:
        _ = retry_id
        self._id = b"fake-tx"
        self._writes.clear()

    def _rollback(self) -> None:
        self._writes.clear()
        self._id = None

    def _commit(self) -> list[Any]:
        for coll, doc_id, data in self._writes:
            self._client._collections[coll][doc_id] = dict(data)
            self._client.calls.append(("TX_SET", f"{coll}/{doc_id}"))
        self._writes.clear()
        self._id = None
        return []

    def _clean_up(self) -> None:
        self._writes.clear()
        self._id = None

    def set(self, doc_ref: _FakeFirestoreDocRef, document_data: dict[str, Any]) -> None:
        self._writes.append((doc_ref._collection_name, doc_ref.id, dict(document_data)))


class _FakeFirestoreDocRef:
    def __init__(self, client: FakeFirestoreClient, collection_name: str, doc_id: str) -> None:
        self._client = client
        self._collection_name = collection_name
        self.id = doc_id

    def create(self, document_data: dict[str, Any], *, timeout: float | None = None) -> None:
        _ = timeout
        self._client._maybe_raise("create")
        store = self._client._collections[self._collection_name]
        if self.id in store:
            raise gcp_exceptions.AlreadyExists(f"Document {self.id} already exists")
        store[self.id] = dict(document_data)
        self._client.calls.append(("CREATE", f"{self._collection_name}/{self.id}"))

    def set(
        self,
        document_data: dict[str, Any],
        *,
        merge: bool = False,
        timeout: float | None = None,
    ) -> None:
        _ = timeout
        self._client._maybe_raise("set")
        store = self._client._collections[self._collection_name]
        if merge and self.id in store:
            store[self.id] = {**store[self.id], **document_data}
        else:
            store[self.id] = dict(document_data)
        self._client.calls.append(("SET", f"{self._collection_name}/{self.id}"))

    def get(
        self,
        *,
        transaction: Any | None = None,
        timeout: float | None = None,
    ) -> _FakeDocSnapshot:
        _ = timeout
        self._client._maybe_raise("get")
        verb = "TX_GET" if transaction is not None else "GET"
        self._client.calls.append((verb, f"{self._collection_name}/{self.id}"))
        data = self._client._collections[self._collection_name].get(self.id)
        return _FakeDocSnapshot(self.id, data)

    def delete(self, *, timeout: float | None = None) -> None:
        _ = timeout
        self._client._maybe_raise("delete")
        self._client.calls.append(("DELETE", f"{self._collection_name}/{self.id}"))
        self._client._collections[self._collection_name].pop(self.id, None)


class _FakeFirestoreQuery:
    def __init__(
        self,
        client: FakeFirestoreClient,
        collection_name: str,
        *,
        filters: list[tuple[str, str, Any]] | None = None,
        order_field: str | None = None,
        descending: bool = False,
        limit_val: int | None = None,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._filters = list(filters or [])
        self._order_field = order_field
        self._descending = descending
        self._limit = limit_val

    def where(self, *, filter: Any) -> _FakeFirestoreQuery:  # noqa: A002
        return _FakeFirestoreQuery(
            self._client,
            self._collection_name,
            filters=[*self._filters, (filter.field_path, filter.op_string, filter.value)],
            order_field=self._order_field,
            descending=self._descending,
            limit_val=self._limit,
        )

    def order_by(self, field_path: str, *, direction: str = "ASCENDING") -> _FakeFirestoreQuery:
        return _FakeFirestoreQuery(
            self._client,
            self._collection_name,
            filters=self._filters,
            order_field=field_path,
            descending=str(direction).upper() == "DESCENDING",
            limit_val=self._limit,
        )

    def limit(self, count: int) -> _FakeFirestoreQuery:
        return _FakeFirestoreQuery(
            self._client,
            self._collection_name,
            filters=self._filters,
            order_field=self._order_field,
            descending=self._descending,
            limit_val=count,
        )

    def stream(self, *, timeout: float | None = None) -> list[_FakeDocSnapshot]:
        _ = timeout
        self._client._maybe_raise("stream")
        self._client.calls.append(("STREAM", self._collection_name))
        items = list(self._client._collections[self._collection_name].items())
        for field_path, op, val in self._filters:
            if op == "==":
                items = [(doc_id, data) for doc_id, data in items if data.get(field_path) == val]
        if self._order_field:
            items.sort(
                key=lambda pair: pair[1].get(self._order_field) or 0,
                reverse=self._descending,
            )
        if self._limit is not None:
            items = items[: self._limit]
        return [_FakeDocSnapshot(doc_id, data) for doc_id, data in items]


class _FakeFirestoreCollection(_FakeFirestoreQuery):
    def document(self, doc_id: str) -> _FakeFirestoreDocRef:
        return _FakeFirestoreDocRef(self._client, self._collection_name, doc_id)


class FakeFirestoreClient:
    """In-memory ``google.cloud.firestore.Client`` fake with queries and transactions."""

    def __init__(self, *, error_to_raise: Exception | None = None) -> None:
        self._collections: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        self.calls: list[tuple[str, str]] = []
        self.error_to_raise = error_to_raise

    def _maybe_raise(self, op: str) -> None:
        _ = op
        if self.error_to_raise is not None:
            raise self.error_to_raise

    def collection(self, name: str) -> _FakeFirestoreCollection:
        return _FakeFirestoreCollection(self, name)

    def transaction(self) -> _FakeFirestoreTransaction:
        return _FakeFirestoreTransaction(self)


def create_fake_gcp_adapters() -> tuple[GcpToolAdapters, list[tuple[str, str]]]:
    """Create a `GcpToolAdapters` instance wired to deterministic typed SDK fakes."""
    profiles, chunks, markdowns = _get_synthetic_corpus()
    log: list[tuple[str, str]] = []
    cfg = GcpToolConfig(
        project_id="iron-radio-514217-q7",
        location="us-central1",
        bq_dataset_id="sec_edgar",
        sec_corpus_bucket=FAKE_BUCKET,
    )
    adapters = GcpToolAdapters(
        cfg,
        bq_client=FakeBigQueryClient(profiles, chunks, log),
        search_client=FakeSearchServiceClient(chunks, log),
        storage_client=FakeStorageClient(markdowns, log),
        genai_client=FakeGenAiClient(log),
        credentials=FakeGcpCredentials(),
    )
    return adapters, log


@contextmanager
def preserve_gcp_caches() -> Iterator[None]:
    """Save and restore live GCP adapter singleton + warm caches around a mocked test.

    Prevents tests that inject custom mock adapters from wiping warm live caches
    (`_ENV_GCP_ADAPTERS_CACHE`, `_ENTITY_CATALOG_CACHE`, `_SECTION_CHUNKS_CACHE`,
    `_SESSION_10K_CONTEXT_CACHE`) populated by live smoke tests in the same pytest session.
    """
    with adapter_registry._ADAPTERS_LOCK:
        prev_custom = adapter_registry._CUSTOM_GCP_ADAPTERS
    with _ENTITY_CATALOG_CACHE._lock:
        prev_catalog = _ENTITY_CATALOG_CACHE._cached
        _ENTITY_CATALOG_CACHE._cached = None
    prev_section_chunks = {k: _SECTION_CHUNKS_CACHE.get(k) for k in list(_SECTION_CHUNKS_CACHE)}
    prev_session_10k = {
        k: _SESSION_10K_CONTEXT_CACHE.get(k) for k in list(_SESSION_10K_CONTEXT_CACHE)
    }
    _SECTION_CHUNKS_CACHE.clear()
    _SESSION_10K_CONTEXT_CACHE.clear()
    try:
        yield
    finally:
        with adapter_registry._ADAPTERS_LOCK:
            adapter_registry._CUSTOM_GCP_ADAPTERS = prev_custom
        with _ENTITY_CATALOG_CACHE._lock:
            _ENTITY_CATALOG_CACHE._cached = prev_catalog
        _SECTION_CHUNKS_CACHE.clear()
        for k, v in prev_section_chunks.items():
            if v is not None:
                _SECTION_CHUNKS_CACHE[k] = v
        _SESSION_10K_CONTEXT_CACHE.clear()
        for k, v in prev_session_10k.items():
            if v is not None:
                _SESSION_10K_CONTEXT_CACHE[k] = v


@contextmanager
def use_fake_gcp_adapters() -> Iterator[tuple[GcpToolAdapters, list[tuple[str, str]]]]:
    """Temporarily install `create_fake_gcp_adapters()` while preserving any warm live caches."""
    adapters, log = create_fake_gcp_adapters()
    with preserve_gcp_caches():
        with adapter_registry._ADAPTERS_LOCK:
            adapter_registry._CUSTOM_GCP_ADAPTERS = adapters
        try:
            yield adapters, log
        finally:
            adapters.close()
