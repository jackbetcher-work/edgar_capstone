"""Live Google Cloud Storage adapter for unstructured SEC 10-K Markdown filings."""

from __future__ import annotations

import urllib.parse
from typing import Any

import httpx

from edgar_capstone.agents.adapters.registry import (
    GcpAdapterErrorTranslationMixin,
    GcpToolConfig,
)
from edgar_capstone.gcp.rest_client import GcpRestClient


class GcsCorpusAdapter(GcpAdapterErrorTranslationMixin, GcpRestClient):
    """Live Google Cloud Storage object reader for unstructured 10-K Markdown filings."""

    def __init__(
        self,
        config: GcpToolConfig,
        *,
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
    ) -> None:
        super().__init__(http_client=http_client, credentials=credentials, default_timeout=30.0)
        self.default_bucket = config.sec_corpus_bucket

    def parse_gcs_uri(self, uri_or_blob_path: str) -> tuple[str, str]:
        """Split `gs://bucket/path/to/blob` or relative blob path into `(bucket, object_name)`."""
        cleaned = uri_or_blob_path.strip()
        if cleaned.startswith("gs://"):
            without_scheme = cleaned[5:]
            slash_idx = without_scheme.find("/")
            if slash_idx <= 0:
                raise ValueError(f"Invalid GCS URI: {uri_or_blob_path!r}")
            return without_scheme[:slash_idx], without_scheme[slash_idx + 1 :]
        return self.default_bucket, cleaned.lstrip("/")

    def fetch_markdown(self, uri_or_blob_path: str) -> str:
        """Download UTF-8 10-K Markdown content from Google Cloud Storage."""
        bucket, object_name = self.parse_gcs_uri(uri_or_blob_path)
        encoded_obj = urllib.parse.quote(object_name, safe="")
        url = (
            f"https://storage.googleapis.com/storage/v1/b/"
            f"{urllib.parse.quote(bucket, safe='')}/o/{encoded_obj}"
        )
        return self._request_text("GET", url, params={"alt": "media"})
