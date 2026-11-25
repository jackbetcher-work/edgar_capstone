"""Dense vector embedding generation for SEC 10-K RAG chunks via ``google-genai`` on Vertex AI.

There is no offline or synthetic embedding backend: ingestion and query embedding always call
Vertex AI ``text-embedding-005`` via the official ``google-genai`` SDK and fail loudly when
unavailable, so every stored vector is a real model embedding.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import TracebackType
from typing import Any

from edgar_capstone.common.domain import sha256_text
from edgar_capstone.common.jsonl import iter_jsonl
from edgar_capstone.gcp.defaults import DEFAULT_LOCATION
from edgar_capstone.gcp.errors import (
    GcpClientError,
    GcpRateLimitError,
    GcpTimeoutError,
    translate_gcp_sdk_error,
)
from edgar_capstone.vector_db.models import FilingChunk

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "text-embedding-005"
DEFAULT_EMBEDDING_DIMENSIONS = 768
# text-embedding-005 accepts at most 20,000 input tokens per request. SEC Markdown
# averages ~4 characters per token, so 60,000 characters leaves comfortable headroom.
DEFAULT_MAX_BATCH_CHARS = 60_000
# Decimal places kept for Vertex AI embeddings in manifests.
EMBEDDING_ROUND_DECIMALS = 6


class _BatchTooLargeError(RuntimeError):
    """Raised when Vertex AI rejects a multi-instance batch for exceeding the token limit."""


def _retry_backoff_seconds(attempt: int) -> float:
    """Exponential backoff with jitter, capped at 30 seconds."""
    return min(30.0, 1.5 * (2 ** (attempt - 1))) * (0.75 + random.random() * 0.5)


class VertexAIEmbedder:
    """Client for generating dense embeddings via ``google-genai`` on Vertex AI.

    Requests are packed by instance count (``batch_size``), total characters
    (``max_batch_chars``), and matching document ``title`` so each batch stays under the
    20,000-token per-request limit while passing section titles natively via
    ``EmbedContentConfig.title``. Oversized batches are split in half recursively.
    """

    def __init__(
        self,
        *,
        project_id: str,
        location: str = DEFAULT_LOCATION,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        batch_size: int = 50,
        max_retries: int = 5,
        genai_client: Any | None = None,
        credentials: Any | None = None,
        max_batch_chars: int = DEFAULT_MAX_BATCH_CHARS,
        max_workers: int = 1,
    ) -> None:
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.dimensions = dimensions
        self.batch_size = max(1, min(batch_size, 250))
        self.max_retries = max(1, int(max_retries))
        self.max_batch_chars = max(1, max_batch_chars)
        self.max_workers = max(1, max_workers)
        self._injected_client = genai_client
        self._owned_client: Any | None = None
        self._credentials = credentials
        self._client_lock = threading.Lock()

    def _get_client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        with self._client_lock:
            if self._owned_client is None:
                from google import genai
                from google.genai import types as genai_types

                kwargs: dict[str, Any] = {
                    "vertexai": True,
                    "project": self.project_id,
                    "location": self.location,
                    "http_options": genai_types.HttpOptions(timeout=60_000),
                }
                if self._credentials is not None:
                    kwargs["credentials"] = self._credentials
                self._owned_client = genai.Client(**kwargs)
            return self._owned_client

    def close(self) -> None:
        """Close the internally managed ``genai.Client`` (leaves injected clients open)."""
        with self._client_lock:
            if self._owned_client is not None:
                close_fn = getattr(self._owned_client, "close", None)
                if callable(close_fn):
                    close_fn()
                self._owned_client = None

    def __enter__(self) -> VertexAIEmbedder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _plan_batches(
        self,
        texts: list[str],
        titles: list[str] | None = None,
    ) -> list[tuple[int, int]]:
        """Split ``texts`` into ``(start, end)`` ranges bounded by count, chars, and title."""
        batches: list[tuple[int, int]] = []
        start = 0
        chars = 0
        current_title = titles[0] if (titles and len(titles) > 0) else None
        for idx, text in enumerate(texts):
            size = len(text)
            item_title = titles[idx] if (titles and idx < len(titles)) else None
            if idx > start and (
                idx - start >= self.batch_size
                or chars + size > self.max_batch_chars
                or (titles is not None and item_title != current_title)
            ):
                batches.append((start, idx))
                start, chars, current_title = idx, 0, item_title
            chars += size
        if start < len(texts):
            batches.append((start, len(texts)))
        return batches

    def _predict_batch(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
        title: str | None = None,
    ) -> list[list[float]]:
        """Invoke ``client.models.embed_content`` for a single homogeneous batch of texts."""
        from google.genai import types as genai_types

        config_kwargs: dict[str, Any] = {
            "task_type": task_type,
            "output_dimensionality": self.dimensions,
            "auto_truncate": True,
        }
        if title and task_type == "RETRIEVAL_DOCUMENT":
            config_kwargs["title"] = title
        config = genai_types.EmbedContentConfig(**config_kwargs)

        client = self._get_client()
        last_translated: GcpClientError | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = client.models.embed_content(
                    model=self.model_name,
                    contents=texts,
                    config=config,
                )
            except Exception as exc:
                msg_lower = str(exc).lower()
                translated = translate_gcp_sdk_error(
                    exc, context=f"Vertex AI embedding ({self.model_name})"
                )
                last_translated = translated

                if translated.status_code == 400 and len(texts) > 1 and "token" in msg_lower:
                    raise _BatchTooLargeError(str(exc)[:400]) from exc

                if isinstance(translated, (GcpRateLimitError, GcpTimeoutError)):
                    if attempt < self.max_retries:
                        explicit_delay = (
                            translated.retry_after_seconds
                            if isinstance(translated, GcpRateLimitError)
                            else None
                        )
                        backoff = (
                            explicit_delay
                            if explicit_delay is not None
                            else _retry_backoff_seconds(attempt)
                        )
                        logger.warning(
                            "Vertex AI embedding transient error %s (attempt %d/%d); retrying in %.1fs...",
                            type(exc).__name__,
                            attempt,
                            self.max_retries,
                            backoff,
                        )
                        time.sleep(backoff)
                        continue
                    raise translated from exc
                raise translated from exc

            embeddings = list(getattr(response, "embeddings", None) or ())
            vectors = [
                [float(v) for v in (getattr(emb, "values", None) or ())] for emb in embeddings
            ]
            if len(vectors) != len(texts):
                raise GcpClientError(
                    f"Vertex AI returned {len(vectors)} embeddings for {len(texts)} inputs"
                )
            return vectors

        assert last_translated is not None
        raise last_translated

    def _embed_batch_adaptive(
        self,
        texts: list[str],
        *,
        task_type: str,
        title: str | None,
    ) -> list[list[float]]:
        """Embed one batch, halving it recursively if the API rejects it as too many tokens."""
        try:
            return self._predict_batch(texts, task_type=task_type, title=title)
        except _BatchTooLargeError:
            mid = len(texts) // 2
            logger.info("Embedding batch of %d exceeded the token limit; splitting.", len(texts))
            left = self._embed_batch_adaptive(texts[:mid], task_type=task_type, title=title)
            right = self._embed_batch_adaptive(texts[mid:], task_type=task_type, title=title)
            return left + right

    def embed_texts(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
        titles: list[str] | None = None,
        on_batch_complete: Callable[[int, int, list[list[float]]], None] | None = None,
    ) -> list[list[float]]:
        """Embed a list of texts in count-, char-, and title-bounded batches."""
        batches = self._plan_batches(texts, titles=titles)
        results: list[list[float] | None] = [None] * len(texts)

        def run(bounds: tuple[int, int]) -> tuple[int, int, list[list[float]]]:
            start, end = bounds
            batch_title = (
                titles[start] if (titles and start < len(titles) and titles[start]) else None
            )
            vecs = self._embed_batch_adaptive(
                texts[start:end],
                task_type=task_type,
                title=batch_title,
            )
            return start, end, vecs

        def record(start: int, end: int, vecs: list[list[float]]) -> None:
            results[start:end] = vecs
            if on_batch_complete is not None:
                on_batch_complete(start, end, vecs)

        if self.max_workers == 1 or len(batches) <= 1:
            for bounds in batches:
                record(*run(bounds))
        else:
            _ = self._get_client()
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = [pool.submit(run, bounds) for bounds in batches]
                try:
                    for fut in as_completed(futures):
                        record(*fut.result())
                except BaseException:
                    for f in futures:
                        f.cancel()
                    raise

        missing = [i for i, vec in enumerate(results) if vec is None]
        if missing:
            raise GcpClientError(f"Vertex AI embedding left {len(missing)} inputs without vectors")
        return [vec for vec in results if vec is not None]


def load_embedding_checkpoint(
    checkpoint_path: Path,
    *,
    model_name: str,
    dimensions: int,
) -> dict[str, tuple[str, list[float]]]:
    """Load `chunk_id -> (sha256_text, embedding)` rows from a Vertex AI embedding checkpoint."""
    if not checkpoint_path.exists():
        return {}
    cached: dict[str, tuple[str, list[float]]] = {}
    for row in iter_jsonl(checkpoint_path, skip_corrupt=True):
        emb = row.get("embedding")
        if (
            row.get("model") == model_name
            and isinstance(emb, list)
            and len(emb) == dimensions
            and row.get("chunk_id")
            and row.get("text_sha256")
        ):
            cached[str(row["chunk_id"])] = (str(row["text_sha256"]), emb)
    return cached


def populate_chunk_embeddings(
    chunks: list[FilingChunk],
    *,
    project_id: str,
    location: str = DEFAULT_LOCATION,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    batch_size: int = 50,
    existing_embeddings: dict[str, tuple[str, list[float]]] | None = None,
    vertex_embedder: VertexAIEmbedder | None = None,
    max_workers: int = 1,
    max_batch_chars: int = DEFAULT_MAX_BATCH_CHARS,
    checkpoint_path: Path | None = None,
) -> tuple[list[FilingChunk], str]:
    """Populate `embedding` on each `FilingChunk` with Vertex AI and return `(chunks, source)`."""
    if not chunks:
        return [], "vertex_ai"

    checkpointed: dict[str, tuple[str, list[float]]] = {}
    if checkpoint_path is not None:
        checkpointed = load_embedding_checkpoint(
            checkpoint_path, model_name=model_name, dimensions=dimensions
        )
        if checkpointed:
            logger.info(
                "Loaded %d Vertex AI embeddings from checkpoint %s",
                len(checkpointed),
                checkpoint_path,
            )

    hashes = [sha256_text(c.chunk_text) for c in chunks]
    to_embed_indices: list[int] = []
    reused_from_checkpoint = 0
    for idx, chunk in enumerate(chunks):
        text_hash = hashes[idx]
        if chunk.chunk_id in checkpointed:
            cached_hash, cached_vec = checkpointed[chunk.chunk_id]
            if cached_hash == text_hash:
                chunk.embedding = cached_vec
                chunk.embedding_model = model_name
                reused_from_checkpoint += 1
                continue
        if existing_embeddings and chunk.chunk_id in existing_embeddings:
            cached_hash, cached_vec = existing_embeddings[chunk.chunk_id]
            if cached_hash == text_hash and len(cached_vec) == dimensions:
                chunk.embedding = cached_vec
                chunk.embedding_model = model_name
                continue
        to_embed_indices.append(idx)
    checkpointed.clear()

    if not to_embed_indices:
        logger.info("Reused all %d cached chunk embeddings.", len(chunks))
        return chunks, "vertex_ai" if reused_from_checkpoint == len(chunks) else "cached"

    owns_embedder = vertex_embedder is None
    embedder = vertex_embedder or VertexAIEmbedder(
        project_id=project_id,
        location=location,
        model_name=model_name,
        dimensions=dimensions,
        batch_size=batch_size,
        max_retries=6,
        max_batch_chars=max_batch_chars,
        max_workers=max_workers,
    )
    texts = [chunks[i].chunk_text for i in to_embed_indices]
    titles = [
        f"{chunks[i].ticker} FY{chunks[i].fiscal_year} 10-K {chunks[i].section_title}"
        for i in to_embed_indices
    ]
    total = len(texts)
    progress = {"done": 0, "next_log": 0.0, "started": time.monotonic()}
    ckpt_fh = None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_fh = checkpoint_path.open("a", encoding="utf-8")

    def on_batch_complete(start: int, end: int, vecs: list[list[float]]) -> None:
        for offset, vec in enumerate(vecs):
            i = to_embed_indices[start + offset]
            rounded = [round(v, EMBEDDING_ROUND_DECIMALS) for v in vec]
            chunks[i].embedding = rounded
            chunks[i].embedding_model = model_name
            if ckpt_fh is not None:
                ckpt_fh.write(
                    json.dumps(
                        {
                            "chunk_id": chunks[i].chunk_id,
                            "text_sha256": hashes[i],
                            "model": model_name,
                            "embedding": rounded,
                        }
                    )
                    + "\n"
                )
        if ckpt_fh is not None:
            ckpt_fh.flush()
        progress["done"] += end - start
        now = time.monotonic()
        if now >= progress["next_log"] or progress["done"] == total:
            elapsed = max(now - progress["started"], 1e-6)
            rate = progress["done"] / elapsed
            eta = (total - progress["done"]) / rate if rate > 0 else float("inf")
            logger.info(
                "Vertex AI embeddings: %d/%d (%.1f%%) | %.0f chunks/s | ETA %.0fs",
                progress["done"],
                total,
                100.0 * progress["done"] / total,
                rate,
                eta,
            )
            progress["next_log"] = now + 30.0

    try:
        logger.info(
            "Embedding %d chunks with Vertex AI %s (%d reused from checkpoint)...",
            total,
            model_name,
            reused_from_checkpoint,
        )
        embedder.embed_texts(
            texts,
            task_type="RETRIEVAL_DOCUMENT",
            titles=titles,
            on_batch_complete=on_batch_complete,
        )
    finally:
        if ckpt_fh is not None:
            ckpt_fh.close()
        if owns_embedder:
            embedder.close()
    return chunks, "vertex_ai"
