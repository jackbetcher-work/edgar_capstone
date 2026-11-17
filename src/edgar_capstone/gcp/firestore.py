"""Shared Firestore Native REST v1 wire codec and base collection store."""

from __future__ import annotations

import re
import threading
import urllib.parse
from typing import Any

import httpx

from edgar_capstone.gcp.bigquery import validate_bq_identifier
from edgar_capstone.gcp.rest_client import GcpNotFoundError, GcpRestClient

DEFAULT_FIRESTORE_DATABASE_ID = "(default)"
_ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")


def encode_firestore_value(value: Any, *, field_name: str = "") -> dict[str, Any]:
    """Encode a Python value into Firestore REST v1 ``Value`` JSON wire representation."""
    if value is None:
        return {"nullValue": None}
    if isinstance(value, bool):
        return {"booleanValue": value}
    if isinstance(value, int):
        return {"integerValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        if field_name.endswith("_at") and _ISO_UTC_RE.match(value):
            return {"timestampValue": value}
        return {"stringValue": value}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [encode_firestore_value(v) for v in value]}}
    if isinstance(value, dict):
        return {
            "mapValue": {
                "fields": {
                    str(k): encode_firestore_value(v, field_name=str(k)) for k, v in value.items()
                }
            }
        }
    return {"stringValue": str(value)}


def encode_firestore_fields(record: dict[str, Any]) -> dict[str, Any]:
    return {str(k): encode_firestore_value(v, field_name=str(k)) for k, v in record.items()}


def decode_firestore_value(val_obj: dict[str, Any]) -> Any:
    """Decode a Firestore REST v1 ``Value`` JSON object back into native Python types."""
    if "nullValue" in val_obj:
        return None
    if "booleanValue" in val_obj:
        return bool(val_obj["booleanValue"])
    if "integerValue" in val_obj:
        return int(val_obj["integerValue"])
    if "doubleValue" in val_obj:
        return float(val_obj["doubleValue"])
    if "timestampValue" in val_obj:
        return str(val_obj["timestampValue"])
    if "stringValue" in val_obj:
        return str(val_obj["stringValue"])
    if "arrayValue" in val_obj:
        arr = val_obj["arrayValue"] or {}
        return [decode_firestore_value(item) for item in (arr.get("values") or [])]
    if "mapValue" in val_obj:
        mp = val_obj["mapValue"] or {}
        return {str(k): decode_firestore_value(v) for k, v in (mp.get("fields") or {}).items()}
    return None


def decode_firestore_document(doc: dict[str, Any]) -> dict[str, Any]:
    """Decode a Firestore REST v1 ``Document`` payload into a flat Python dictionary."""
    fields = doc.get("fields") or {}
    decoded = {str(k): decode_firestore_value(v) for k, v in fields.items()}
    if doc.get("name") and "document_name" not in decoded:
        decoded["document_name"] = str(doc["name"])
    return decoded


class FirestoreCollectionStore(GcpRestClient):
    """Shared base class for Firestore Native REST v1 collection stores.

    Unifies project ID validation, URL-encoded database/collection path building,
    pooled HTTP transport, document CRUD, and ``:runQuery`` execution.
    """

    def __init__(
        self,
        *,
        project_id: str,
        database_id: str = DEFAULT_FIRESTORE_DATABASE_ID,
        collection_id: str,
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
        default_timeout: float = 15.0,
    ) -> None:
        super().__init__(
            http_client=http_client,
            credentials=credentials,
            default_timeout=default_timeout,
            max_connections=32,
            max_keepalive_connections=16,
        )
        self.project_id = validate_bq_identifier(project_id, "project_id")
        self.database_id = (database_id or DEFAULT_FIRESTORE_DATABASE_ID).strip()
        self.collection_id = collection_id.strip()
        self._lock = threading.Lock()
        self._last_error: str | None = None

    def _documents_base_url(self) -> str:
        proj_q = urllib.parse.quote(self.project_id, safe="")
        db_q = urllib.parse.quote(self.database_id, safe="()")
        return f"https://firestore.googleapis.com/v1/projects/{proj_q}/databases/{db_q}/documents"

    @property
    def documents_base_url(self) -> str:
        return self._documents_base_url()

    def collection_url(self) -> str:
        coll_q = urllib.parse.quote(self.collection_id, safe="")
        return f"{self._documents_base_url()}/{coll_q}"

    def document_url(self, doc_id: str) -> str:
        doc_q = urllib.parse.quote(doc_id.strip(), safe="")
        return f"{self.collection_url()}/{doc_q}"

    def _firestore_request_json(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        resp = self._send_request(
            method,
            url,
            json_body=json_body,
            params=params,
            not_found_as_dedicated_error=True,
        )
        if not resp.content:
            return {}
        return resp.json()

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        try:
            raw = self._firestore_request_json("GET", self.document_url(doc_id))
            return decode_firestore_document(raw)
        except GcpNotFoundError:
            return None

    def run_structured_query(self, structured_query: dict[str, Any]) -> list[dict[str, Any]]:
        url = f"{self._documents_base_url()}:runQuery"
        rows = self._firestore_request_json(
            "POST", url, json_body={"structuredQuery": structured_query}
        )
        if not isinstance(rows, list):
            return []
        docs: list[dict[str, Any]] = []
        for row in rows:
            doc = row.get("document") if isinstance(row, dict) else None
            if isinstance(doc, dict):
                docs.append(decode_firestore_document(doc))
        return docs
