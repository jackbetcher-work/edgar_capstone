"""Live Google Cloud Storage SDK adapter for unstructured SEC 10-K Markdown filings."""

from __future__ import annotations

import threading
from types import TracebackType
from typing import Any

from edgar_capstone.agents.adapters.registry import (
    GcpToolConfig,
    translate_gcp_adapter_error,
)


class GcsCorpusAdapter:
    """Live ``google-cloud-storage`` reader for unstructured 10-K Markdown filings."""

    def __init__(
        self,
        config: GcpToolConfig,
        *,
        storage_client: Any | None = None,
        credentials: Any | None = None,
    ) -> None:
        self.project_id = config.project_id
        self.default_bucket = config.sec_corpus_bucket
        self._injected_client = storage_client
        self._owned_client: Any | None = None
        self._credentials = credentials
        self._client_lock = threading.Lock()

    def _get_client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        with self._client_lock:
            if self._owned_client is None:
                from google.cloud import storage

                kwargs: dict[str, Any] = {"project": self.project_id}
                if self._credentials is not None:
                    kwargs["credentials"] = self._credentials
                self._owned_client = storage.Client(**kwargs)
            return self._owned_client

    def close(self) -> None:
        """Close the internally managed ``storage.Client`` (leaves injected clients open)."""
        with self._client_lock:
            if self._owned_client is not None:
                close_fn = getattr(self._owned_client, "close", None)
                if callable(close_fn):
                    close_fn()
                self._owned_client = None

    def __enter__(self) -> GcsCorpusAdapter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

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
        """Download UTF-8 10-K Markdown content from Google Cloud Storage with integrity checks."""
        from google.cloud.storage.retry import DEFAULT_RETRY

        bucket, object_name = self.parse_gcs_uri(uri_or_blob_path)
        client = self._get_client()
        try:
            blob = client.bucket(bucket).blob(object_name)
            return str(blob.download_as_text(encoding="utf-8", retry=DEFAULT_RETRY, timeout=30.0))
        except Exception as exc:
            raise translate_gcp_adapter_error(
                exc, context=f"GCS gs://{bucket}/{object_name}"
            ) from exc
