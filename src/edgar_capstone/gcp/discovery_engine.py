"""Shared Discovery Engine / Vertex AI Search host and resource path builders."""

from __future__ import annotations

DEFAULT_VERTEX_AI_SEARCH_LOCATION = "global"
DEFAULT_VERTEX_AI_SEARCH_COLLECTION = "default_collection"
DEFAULT_VERTEX_AI_SEARCH_ENGINE_ID = "sec-edgar-10k-hybrid-search"
DEFAULT_VERTEX_AI_SEARCH_DATASTORE_ID = "sec-edgar-10k-corpus-ds"


def discovery_engine_host(location: str = DEFAULT_VERTEX_AI_SEARCH_LOCATION) -> str:
    """Return the regional or global Discovery Engine API endpoint hostname."""
    loc = (location or DEFAULT_VERTEX_AI_SEARCH_LOCATION).strip().lower()
    if loc == "global":
        return "discoveryengine.googleapis.com"
    return f"{loc}-discoveryengine.googleapis.com"


def vertex_search_datastore_path(
    project_id: str,
    *,
    data_store_id: str = DEFAULT_VERTEX_AI_SEARCH_DATASTORE_ID,
    location: str = DEFAULT_VERTEX_AI_SEARCH_LOCATION,
    collection: str = DEFAULT_VERTEX_AI_SEARCH_COLLECTION,
) -> str:
    """Build the canonical ``projects/.../collections/.../dataStores/...`` resource path."""
    loc = (location or DEFAULT_VERTEX_AI_SEARCH_LOCATION).strip().lower()
    coll = (collection or DEFAULT_VERTEX_AI_SEARCH_COLLECTION).strip()
    return f"projects/{project_id}/locations/{loc}/collections/{coll}/dataStores/{data_store_id}"


def vertex_search_engine_path(
    project_id: str,
    *,
    engine_id: str = DEFAULT_VERTEX_AI_SEARCH_ENGINE_ID,
    location: str = DEFAULT_VERTEX_AI_SEARCH_LOCATION,
    collection: str = DEFAULT_VERTEX_AI_SEARCH_COLLECTION,
) -> str:
    """Build the canonical ``projects/.../collections/.../engines/...`` resource path."""
    loc = (location or DEFAULT_VERTEX_AI_SEARCH_LOCATION).strip().lower()
    coll = (collection or DEFAULT_VERTEX_AI_SEARCH_COLLECTION).strip()
    return f"projects/{project_id}/locations/{loc}/collections/{coll}/engines/{engine_id}"
