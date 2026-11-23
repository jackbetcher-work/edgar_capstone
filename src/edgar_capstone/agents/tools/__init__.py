"""ADK tools, turn-scoped tool state, and deterministic SEC 10-K analytical engines."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from edgar_capstone.agents.tools.sec_tools import (
        calculate_period_variances,
        get_active_gcp_tool_adapters,
        get_full_10k_section,
        get_gcp_adapters_status_metadata,
        load_cached_10k_section,
        query_analytics_view,
        search_10k_chunks_rag,
    )

_EXPORTS: dict[str, tuple[str, str]] = {
    "calculate_period_variances": (
        "edgar_capstone.agents.tools.sec_tools",
        "calculate_period_variances",
    ),
    "get_active_gcp_tool_adapters": (
        "edgar_capstone.agents.tools.sec_tools",
        "get_active_gcp_tool_adapters",
    ),
    "get_full_10k_section": ("edgar_capstone.agents.tools.sec_tools", "get_full_10k_section"),
    "get_gcp_adapters_status_metadata": (
        "edgar_capstone.agents.tools.sec_tools",
        "get_gcp_adapters_status_metadata",
    ),
    "load_cached_10k_section": (
        "edgar_capstone.agents.tools.sec_tools",
        "load_cached_10k_section",
    ),
    "query_analytics_view": ("edgar_capstone.agents.tools.sec_tools", "query_analytics_view"),
    "search_10k_chunks_rag": ("edgar_capstone.agents.tools.sec_tools", "search_10k_chunks_rag"),
}

__all__ = [
    "calculate_period_variances",
    "get_active_gcp_tool_adapters",
    "get_full_10k_section",
    "get_gcp_adapters_status_metadata",
    "load_cached_10k_section",
    "query_analytics_view",
    "search_10k_chunks_rag",
]


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
