"""Deterministic in-memory GCP adapter bundle backed by `httpx.MockTransport`.

Exercises the exact production REST clients (`BigQueryAdapter`, `VertexAISearchAdapter`,
`GcsCorpusAdapter`, `VertexAIEmbedder`) against a curated synthetic S&P 500 slice with zero
network traffic. Use `use_fake_gcp_adapters()` (or the `fake_gcp_adapters` pytest fixture)
to avoid wiping warm live GCP caches when running alongside live smoke tests.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.parse
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx

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
    """Offline OAuth2 credentials stub for `httpx.MockTransport` tests."""

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


_PROFILE_SCHEMA = [
    {"name": "profile_id", "type": "STRING", "mode": "REQUIRED"},
    {"name": "ticker", "type": "STRING", "mode": "REQUIRED"},
    {"name": "cik", "type": "STRING", "mode": "REQUIRED"},
    {"name": "company_name", "type": "STRING", "mode": "REQUIRED"},
    {"name": "gics_sector", "type": "STRING", "mode": "NULLABLE"},
    {"name": "gics_sub_industry", "type": "STRING", "mode": "NULLABLE"},
    {"name": "fiscal_year", "type": "INTEGER", "mode": "REQUIRED"},
    {"name": "period_of_report", "type": "DATE", "mode": "NULLABLE"},
    {"name": "filing_date", "type": "DATE", "mode": "NULLABLE"},
    {"name": "accession_number", "type": "STRING", "mode": "REQUIRED"},
    {"name": "currency", "type": "STRING", "mode": "NULLABLE"},
    {"name": "revenue", "type": "FLOAT", "mode": "NULLABLE"},
    {"name": "operating_income", "type": "FLOAT", "mode": "NULLABLE"},
    {"name": "net_income", "type": "FLOAT", "mode": "NULLABLE"},
    {"name": "research_and_development", "type": "FLOAT", "mode": "NULLABLE"},
    {
        "name": "metric_provenance",
        "type": "RECORD",
        "mode": "NULLABLE",
        "fields": [
            {"name": "revenue", "type": "STRING", "mode": "NULLABLE"},
            {"name": "operating_income", "type": "STRING", "mode": "NULLABLE"},
            {"name": "net_income", "type": "STRING", "mode": "NULLABLE"},
            {"name": "research_and_development", "type": "STRING", "mode": "NULLABLE"},
            {"name": "notes", "type": "STRING", "mode": "REPEATED"},
        ],
    },
    {"name": "extracted_sections", "type": "STRING", "mode": "REPEATED"},
    {"name": "unstructured_blob_path", "type": "STRING", "mode": "NULLABLE"},
    {"name": "structured_blob_path", "type": "STRING", "mode": "NULLABLE"},
    {"name": "unstructured_gcs_uri", "type": "STRING", "mode": "NULLABLE"},
    {"name": "structured_gcs_uri", "type": "STRING", "mode": "NULLABLE"},
    {"name": "sec_filing_url", "type": "STRING", "mode": "NULLABLE"},
]

_CHUNK_BASE_SCHEMA = [
    {"name": "chunk_id", "type": "STRING", "mode": "REQUIRED"},
    {"name": "profile_id", "type": "STRING", "mode": "REQUIRED"},
    {"name": "ticker", "type": "STRING", "mode": "REQUIRED"},
    {"name": "company_name", "type": "STRING", "mode": "REQUIRED"},
    {"name": "gics_sector", "type": "STRING", "mode": "NULLABLE"},
    {"name": "fiscal_year", "type": "INTEGER", "mode": "REQUIRED"},
    {"name": "period_of_report", "type": "DATE", "mode": "NULLABLE"},
    {"name": "filing_date", "type": "DATE", "mode": "NULLABLE"},
    {"name": "accession_number", "type": "STRING", "mode": "REQUIRED"},
    {"name": "section_id", "type": "STRING", "mode": "REQUIRED"},
    {"name": "section_title", "type": "STRING", "mode": "NULLABLE"},
    {"name": "chunk_index", "type": "INTEGER", "mode": "REQUIRED"},
    {"name": "chunk_text", "type": "STRING", "mode": "REQUIRED"},
    {"name": "sec_filing_url", "type": "STRING", "mode": "NULLABLE"},
    {"name": "unstructured_gcs_uri", "type": "STRING", "mode": "NULLABLE"},
]


def _bq_val(val: Any) -> dict[str, Any]:
    if val is None:
        return {"v": None}
    if isinstance(val, bool):
        return {"v": "true" if val else "false"}
    if isinstance(val, list):
        return {"v": [{"v": str(item) if item is not None else None} for item in val]}
    return {"v": str(val)}


def _encode_profile_row(p: dict[str, Any]) -> dict[str, Any]:
    prov = p["metric_provenance"]
    return {
        "f": [
            _bq_val(p["profile_id"]),
            _bq_val(p["ticker"]),
            _bq_val(p["cik"]),
            _bq_val(p["company_name"]),
            _bq_val(p["gics_sector"]),
            _bq_val(p["gics_sub_industry"]),
            _bq_val(p["fiscal_year"]),
            _bq_val(p["period_of_report"]),
            _bq_val(p["filing_date"]),
            _bq_val(p["accession_number"]),
            _bq_val(p["currency"]),
            _bq_val(p["revenue"]),
            _bq_val(p["operating_income"]),
            _bq_val(p["net_income"]),
            _bq_val(p["research_and_development"]),
            {
                "v": {
                    "f": [
                        _bq_val(prov.get("revenue")),
                        _bq_val(prov.get("operating_income")),
                        _bq_val(prov.get("net_income")),
                        _bq_val(prov.get("research_and_development")),
                        _bq_val(prov.get("notes", [])),
                    ]
                }
            },
            _bq_val(p["extracted_sections"]),
            _bq_val(p["unstructured_blob_path"]),
            _bq_val(p["structured_blob_path"]),
            _bq_val(p["unstructured_gcs_uri"]),
            _bq_val(p["structured_gcs_uri"]),
            _bq_val(p["sec_filing_url"]),
        ]
    }


def _extract_bq_params(body: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for param in body.get("queryParameters") or []:
        name = param.get("name")
        pval = param.get("parameterValue") or {}
        ptype = (param.get("parameterType") or {}).get("type", "")
        if ptype == "ARRAY":
            out[name] = [item.get("value") for item in (pval.get("arrayValues") or [])]
        else:
            out[name] = pval.get("value")
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


def build_fake_gcp_transport() -> tuple[httpx.MockTransport, list[tuple[str, str]]]:
    """Return `(MockTransport, request_log)` serving BigQuery, Vertex AI Search, Embeddings, and GCS."""
    profiles, chunks, markdowns = _get_synthetic_corpus()
    requests_log: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        requests_log.append((request.method, url))

        # 1. Vertex AI text-embedding-005:predict
        if "aiplatform.googleapis.com" in url and ":predict" in url:
            body = json.loads(request.content.decode("utf-8"))
            preds = [
                {"embeddings": {"values": mock_embedding_values(inst.get("content", ""))}}
                for inst in (body.get("instances") or [])
            ]
            return httpx.Response(200, json={"predictions": preds})

        # 2. Discovery Engine / Vertex AI Search :search
        if "discoveryengine.googleapis.com" in url and ":search" in url:
            body = json.loads(request.content.decode("utf-8"))
            query = str(body.get("query") or "")
            q_vec = mock_embedding_values(query)
            filt = _parse_vais_filter(str(body.get("filter") or ""))
            pool = _filter_items(chunks, filt)
            ranked = sorted(
                pool,
                key=lambda c: _cosine_distance(q_vec, c["embedding"]),
            )[: int(body.get("pageSize") or 10)]
            results = []
            for rank, c in enumerate(ranked):
                results.append(
                    {
                        "id": c["chunk_id"],
                        "document": {
                            "id": c["chunk_id"],
                            "structData": {
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
                            },
                            "derivedStructData": {
                                "extractive_segments": [
                                    {
                                        "content": c["chunk_text"],
                                        "relevanceScore": max(0.35, 0.92 - 0.04 * rank),
                                        "section_id": c["section_id"],
                                        "section_title": c["section_title"],
                                    }
                                ]
                            },
                        },
                    }
                )
            return httpx.Response(200, json={"results": results, "totalSize": len(results)})

        # 3. GCS object read (alt=media)
        if "storage.googleapis.com" in url and "/o/" in url:
            encoded_obj = request.url.path.split("/o/", 1)[1]
            obj_name = urllib.parse.unquote(encoded_obj)
            if obj_name not in markdowns:
                return httpx.Response(404, text=f"No such object: {obj_name}")
            return httpx.Response(200, text=markdowns[obj_name])

        # 4. BigQuery REST v2 queries
        if "bigquery.googleapis.com" in url and "/queries" in url:
            body = json.loads(request.content.decode("utf-8"))
            sql = str(body.get("query") or "")
            params = _extract_bq_params(body)

            # 4a. VECTOR_SEARCH or QUALIFY ROW_NUMBER sector vector search
            if "VECTOR_SEARCH(" in sql or "QUALIFY ROW_NUMBER()" in sql:
                q_emb = [
                    float(x) for x in (params.get("query_embedding") or [0.0] * EMBEDDING_DIMS)
                ]
                pool = _filter_items(chunks, params)
                scored = [(c, _cosine_distance(q_emb, c["embedding"])) for c in pool]
                if "QUALIFY ROW_NUMBER()" in sql:
                    per_cy = int(params.get("per_company_year") or 3)
                    grouped: dict[tuple[str, int], list[tuple[dict[str, Any], float]]] = (
                        defaultdict(list)
                    )
                    for item, dist in scored:
                        grouped[(item["ticker"], item["fiscal_year"])].append((item, dist))
                    rows_out = []
                    for key in sorted(grouped):
                        top = sorted(grouped[key], key=lambda x: x[1])[:per_cy]
                        rows_out.extend(top)
                else:
                    top_k = int(params.get("top_k") or 10)
                    rows_out = sorted(scored, key=lambda x: x[1])[:top_k]

                schema = [*_CHUNK_BASE_SCHEMA, {"name": "distance", "type": "FLOAT"}]
                bq_rows = [
                    {
                        "f": [
                            _bq_val(c["chunk_id"]),
                            _bq_val(c["profile_id"]),
                            _bq_val(c["ticker"]),
                            _bq_val(c["company_name"]),
                            _bq_val(c["gics_sector"]),
                            _bq_val(c["fiscal_year"]),
                            _bq_val(c["period_of_report"]),
                            _bq_val(c["filing_date"]),
                            _bq_val(c["accession_number"]),
                            _bq_val(c["section_id"]),
                            _bq_val(c["section_title"]),
                            _bq_val(c["chunk_index"]),
                            _bq_val(c["chunk_text"]),
                            _bq_val(c["sec_filing_url"]),
                            _bq_val(c["unstructured_gcs_uri"]),
                            _bq_val(dist),
                        ]
                    }
                    for c, dist in rows_out
                ]
                return httpx.Response(
                    200,
                    json={
                        "jobComplete": True,
                        "schema": {"fields": schema},
                        "rows": bq_rows,
                    },
                )

            # 4b. fetch_section_chunks
            if "filing_chunks" in sql:
                include_emb = ", embedding" in sql
                matched = _filter_items(chunks, params)
                schema = [
                    {"name": "chunk_id", "type": "STRING"},
                    {"name": "profile_id", "type": "STRING"},
                    {"name": "ticker", "type": "STRING"},
                    {"name": "company_name", "type": "STRING"},
                    {"name": "gics_sector", "type": "STRING"},
                    {"name": "fiscal_year", "type": "INTEGER"},
                    {"name": "period_of_report", "type": "DATE"},
                    {"name": "filing_date", "type": "DATE"},
                    {"name": "accession_number", "type": "STRING"},
                    {"name": "section_id", "type": "STRING"},
                    {"name": "section_title", "type": "STRING"},
                    {"name": "chunk_index", "type": "INTEGER"},
                    {"name": "chunk_text", "type": "STRING"},
                ]
                if include_emb:
                    schema.append({"name": "embedding", "type": "FLOAT", "mode": "REPEATED"})
                schema.extend(
                    [
                        {"name": "sec_filing_url", "type": "STRING"},
                        {"name": "unstructured_gcs_uri", "type": "STRING"},
                    ]
                )
                bq_rows = []
                for c in matched:
                    fields = [
                        _bq_val(c["chunk_id"]),
                        _bq_val(c["profile_id"]),
                        _bq_val(c["ticker"]),
                        _bq_val(c["company_name"]),
                        _bq_val(c["gics_sector"]),
                        _bq_val(c["fiscal_year"]),
                        _bq_val(c["period_of_report"]),
                        _bq_val(c["filing_date"]),
                        _bq_val(c["accession_number"]),
                        _bq_val(c["section_id"]),
                        _bq_val(c["section_title"]),
                        _bq_val(c["chunk_index"]),
                        _bq_val(c["chunk_text"]),
                    ]
                    if include_emb:
                        fields.append(_bq_val(c["embedding"]))
                    fields.extend(
                        [
                            _bq_val(c["sec_filing_url"]),
                            _bq_val(c["unstructured_gcs_uri"]),
                        ]
                    )
                    bq_rows.append({"f": fields})
                return httpx.Response(
                    200,
                    json={"jobComplete": True, "schema": {"fields": schema}, "rows": bq_rows},
                )

            # 4c. v_chunk_health_metrics
            if "v_chunk_health_metrics" in sql:
                ch_schema = [
                    {"name": "gics_sector", "type": "STRING"},
                    {"name": "section_id", "type": "STRING"},
                    {"name": "health_status", "type": "STRING"},
                    {"name": "health_score", "type": "FLOAT"},
                    {"name": "total_chunks", "type": "INTEGER"},
                    {"name": "distinct_profiles", "type": "INTEGER"},
                    {"name": "distinct_tickers", "type": "INTEGER"},
                    {"name": "valid_embedding_count", "type": "INTEGER"},
                    {"name": "embedding_coverage_ratio", "type": "FLOAT"},
                    {"name": "full_section_coverage_ratio", "type": "FLOAT"},
                    {"name": "avg_char_count", "type": "FLOAT"},
                    {"name": "avg_token_estimate", "type": "FLOAT"},
                    {"name": "min_token_estimate", "type": "INTEGER"},
                    {"name": "max_token_estimate", "type": "INTEGER"},
                    {"name": "p50_token_estimate", "type": "FLOAT"},
                    {"name": "p95_token_estimate", "type": "FLOAT"},
                    {"name": "undersized_chunk_count", "type": "INTEGER"},
                    {"name": "oversized_chunk_count", "type": "INTEGER"},
                    {"name": "missing_metadata_count", "type": "INTEGER"},
                ]
                total_c = len(chunks)
                ch_row = {
                    "f": [
                        _bq_val("ALL_SECTORS"),
                        _bq_val("ALL_SECTIONS"),
                        _bq_val("HEALTHY"),
                        _bq_val(100.0),
                        _bq_val(total_c),
                        _bq_val(len(profiles)),
                        _bq_val(len(_SYNTHETIC_ISSUERS)),
                        _bq_val(total_c),
                        _bq_val(1.0),
                        _bq_val(1.0),
                        _bq_val(420.0),
                        _bq_val(105.0),
                        _bq_val(40),
                        _bq_val(260),
                        _bq_val(100.0),
                        _bq_val(220.0),
                        _bq_val(0),
                        _bq_val(0),
                        _bq_val(0),
                    ]
                }
                return httpx.Response(
                    200,
                    json={"jobComplete": True, "schema": {"fields": ch_schema}, "rows": [ch_row]},
                )

            # 4d. v_financial_variances_yoy
            if "v_financial_variances_yoy" in sql:
                matched = _filter_items(profiles, params)
                by_t: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
                for p in profiles:
                    by_t[p["ticker"]][p["fiscal_year"]] = p
                yoy_schema = [
                    {"name": "profile_id", "type": "STRING"},
                    {"name": "ticker", "type": "STRING"},
                    {"name": "company_name", "type": "STRING"},
                    {"name": "gics_sector", "type": "STRING"},
                    {"name": "fiscal_year", "type": "INTEGER"},
                    {"name": "prev_fiscal_year", "type": "INTEGER"},
                    {"name": "period_of_report", "type": "DATE"},
                    {"name": "filing_date", "type": "DATE"},
                    {"name": "accession_number", "type": "STRING"},
                    {"name": "revenue", "type": "FLOAT"},
                    {"name": "revenue_yoy_delta", "type": "FLOAT"},
                    {"name": "revenue_yoy_pct", "type": "FLOAT"},
                    {"name": "operating_income", "type": "FLOAT"},
                    {"name": "operating_income_yoy_delta", "type": "FLOAT"},
                    {"name": "operating_income_yoy_pct", "type": "FLOAT"},
                    {"name": "net_income", "type": "FLOAT"},
                    {"name": "net_income_yoy_delta", "type": "FLOAT"},
                    {"name": "net_income_yoy_pct", "type": "FLOAT"},
                    {"name": "research_and_development", "type": "FLOAT"},
                    {"name": "research_and_development_yoy_delta", "type": "FLOAT"},
                    {"name": "research_and_development_yoy_pct", "type": "FLOAT"},
                    {"name": "sec_filing_url", "type": "STRING"},
                ]
                bq_rows = []
                for p in matched:
                    prev = by_t[p["ticker"]].get(p["fiscal_year"] - 1)

                    def _yoy(cur_v: float | None, prev_v: float | None):
                        if cur_v is None or prev_v is None:
                            return None, None
                        d = cur_v - prev_v
                        pct = (d / abs(prev_v) * 100.0) if prev_v else None
                        return d, pct

                    rd_d, rd_p = _yoy(
                        p["research_and_development"],
                        prev["research_and_development"] if prev else None,
                    )
                    r_d, r_p = _yoy(p["revenue"], prev["revenue"] if prev else None)
                    o_d, o_p = _yoy(
                        p["operating_income"], prev["operating_income"] if prev else None
                    )
                    n_d, n_p = _yoy(p["net_income"], prev["net_income"] if prev else None)
                    bq_rows.append(
                        {
                            "f": [
                                _bq_val(p["profile_id"]),
                                _bq_val(p["ticker"]),
                                _bq_val(p["company_name"]),
                                _bq_val(p["gics_sector"]),
                                _bq_val(p["fiscal_year"]),
                                _bq_val(prev["fiscal_year"] if prev else None),
                                _bq_val(p["period_of_report"]),
                                _bq_val(p["filing_date"]),
                                _bq_val(p["accession_number"]),
                                _bq_val(p["revenue"]),
                                _bq_val(r_d),
                                _bq_val(r_p),
                                _bq_val(p["operating_income"]),
                                _bq_val(o_d),
                                _bq_val(o_p),
                                _bq_val(p["net_income"]),
                                _bq_val(n_d),
                                _bq_val(n_p),
                                _bq_val(p["research_and_development"]),
                                _bq_val(rd_d),
                                _bq_val(rd_p),
                                _bq_val(p["sec_filing_url"]),
                            ]
                        }
                    )
                return httpx.Response(
                    200,
                    json={"jobComplete": True, "schema": {"fields": yoy_schema}, "rows": bq_rows},
                )

            # 4e. sector_benchmarks (GROUP BY 1, 2)
            if "GROUP BY 1, 2" in sql:
                matched = _filter_items(profiles, params)
                buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
                for p in matched:
                    buckets[(p["gics_sector"], p["fiscal_year"])].append(p)
                sb_schema = [
                    {"name": "gics_sector", "type": "STRING"},
                    {"name": "fiscal_year", "type": "INTEGER"},
                    {"name": "company_count", "type": "INTEGER"},
                    {"name": "tickers", "type": "STRING", "mode": "REPEATED"},
                    {"name": "total_sector_revenue", "type": "FLOAT"},
                    {"name": "avg_company_revenue", "type": "FLOAT"},
                    {"name": "total_sector_operating_income", "type": "FLOAT"},
                    {"name": "total_sector_net_income", "type": "FLOAT"},
                    {"name": "total_sector_research_and_development", "type": "FLOAT"},
                ]
                bq_rows = []
                for (sec, fy), items in sorted(buckets.items()):
                    revs = [i["revenue"] for i in items if i["revenue"] is not None]
                    ops = [
                        i["operating_income"] for i in items if i["operating_income"] is not None
                    ]
                    nets = [i["net_income"] for i in items if i["net_income"] is not None]
                    rds = [
                        i["research_and_development"]
                        for i in items
                        if i["research_and_development"] is not None
                    ]
                    bq_rows.append(
                        {
                            "f": [
                                _bq_val(sec),
                                _bq_val(fy),
                                _bq_val(len(items)),
                                _bq_val(sorted({i["ticker"] for i in items})),
                                _bq_val(round(sum(revs), 2)),
                                _bq_val(round(sum(revs) / len(revs), 2) if revs else 0.0),
                                _bq_val(round(sum(ops), 2)),
                                _bq_val(round(sum(nets), 2)),
                                _bq_val(round(sum(rds), 2) if rds else None),
                            ]
                        }
                    )
                return httpx.Response(
                    200,
                    json={"jobComplete": True, "schema": {"fields": sb_schema}, "rows": bq_rows},
                )

            # 4f. default: financial_metrics fetch_profiles
            matched = _filter_items(profiles, params)
            matched.sort(key=lambda p: (p["ticker"], p["fiscal_year"]))
            return httpx.Response(
                200,
                json={
                    "jobComplete": True,
                    "schema": {"fields": _PROFILE_SCHEMA},
                    "rows": [_encode_profile_row(p) for p in matched],
                },
            )

        return httpx.Response(404, text=f"Unhandled fake GCP URL: {request.method} {url}")

    return httpx.MockTransport(handler), requests_log


def create_fake_gcp_adapters() -> tuple[GcpToolAdapters, list[tuple[str, str]]]:
    """Create a `GcpToolAdapters` instance wired to the deterministic `MockTransport`."""
    transport, log = build_fake_gcp_transport()
    client = httpx.Client(transport=transport)
    cfg = GcpToolConfig(
        project_id="iron-radio-514217-q7",
        location="us-central1",
        bq_dataset_id="sec_edgar",
        sec_corpus_bucket=FAKE_BUCKET,
    )
    adapters = GcpToolAdapters(cfg, http_client=client, credentials=FakeGcpCredentials())
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
