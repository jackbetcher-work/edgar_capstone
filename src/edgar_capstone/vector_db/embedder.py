"""Dense vector embedding generation for SEC 10-K RAG chunks via Vertex AI `text-embedding-005`.

There is no offline or synthetic embedding backend: ingestion always calls Vertex AI and
fails loudly when it is unavailable, so every stored vector is a real model embedding.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

from edgar_capstone.common.domain import sha256_text
from edgar_capstone.common.jsonl import iter_jsonl
from edgar_capstone.gcp.defaults import DEFAULT_LOCATION
from edgar_capstone.gcp.rest_client import GcpRestClient
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


class VertexAIEmbedder(GcpRestClient):
    """Client for generating dense embeddings via Vertex AI (`text-embedding-005`).

    Inherits thread-safe ADC bearer authentication, empty-token guards, and pooled
    `httpx.Client` lifecycle management (`close()` / context manager) from `GcpRestClient`.

    Requests are packed by both instance count (`batch_size`) and total characters
    (`max_batch_chars`) so a batch stays under the per-request input-token limit
    (20,000 tokens for `text-embedding-005`). A batch the API still rejects as too
    large is split in half and retried. With `max_workers > 1`, batches are sent
    concurrently over the shared thread-safe HTTP pool and results are returned in
    input order.
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
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
        max_batch_chars: int = DEFAULT_MAX_BATCH_CHARS,
        max_workers: int = 1,
    ) -> None:
        workers = max(1, max_workers)
        super().__init__(
            http_client=http_client,
            credentials=credentials,
            default_timeout=60.0,
            max_connections=max(32, workers * 2),
            max_keepalive_connections=max(16, workers),
        )
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.dimensions = dimensions
        self.batch_size = max(1, min(batch_size, 250))
        self.max_retries = max_retries
        self.max_batch_chars = max(1, max_batch_chars)
        self.max_workers = workers

    def _get_access_token(self) -> str:
        """Obtain a valid OAuth2 bearer token from Application Default Credentials."""
        return self._get_bearer_token()

    def _plan_batches(self, texts: list[str]) -> list[tuple[int, int]]:
        """Split `texts` into `(start, end)` ranges bounded by count and total characters."""
        batches: list[tuple[int, int]] = []
        start = 0
        chars = 0
        for idx, text in enumerate(texts):
            size = len(text)
            if idx > start and (
                idx - start >= self.batch_size or chars + size > self.max_batch_chars
            ):
                batches.append((start, idx))
                start, chars = idx, 0
            chars += size
        if start < len(texts):
            batches.append((start, len(texts)))
        return batches

    def _predict_batch(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
        titles: list[str] | None = None,
    ) -> list[list[float]]:
        """Invoke the Vertex AI `predict` REST endpoint for a single batch of texts."""
        url = (
            f"https://{self.location}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project_id}/locations/{self.location}/"
            f"publishers/google/models/{self.model_name}:predict"
        )
        instances: list[dict[str, Any]] = []
        for idx, text in enumerate(texts):
            inst: dict[str, Any] = {
                "content": text,
                "task_type": task_type,
            }
            if titles and idx < len(titles) and titles[idx]:
                inst["title"] = titles[idx]
            instances.append(inst)

        payload: dict[str, Any] = {
            "instances": instances,
            "parameters": {
                "outputDimensionality": self.dimensions,
                "autoTruncate": True,
            },
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._raw_request("POST", url, json_body=payload)
            except httpx.TransportError as exc:
                if attempt < self.max_retries:
                    backoff = _retry_backoff_seconds(attempt)
                    logger.warning(
                        "Vertex AI embedding transport error %s (attempt %d/%d); retrying in %.1fs...",
                        type(exc).__name__,
                        attempt,
                        self.max_retries,
                        backoff,
                    )
                    time.sleep(backoff)
                    continue
                raise
            if resp.status_code == 200:
                data = resp.json()
                predictions = data.get("predictions", [])
                vectors: list[list[float]] = []
                for pred in predictions:
                    emb_obj = pred.get("embeddings", pred)
                    values = emb_obj.get("values", [])
                    vectors.append([float(v) for v in values])
                if len(vectors) != len(texts):
                    raise RuntimeError(
                        f"Vertex AI returned {len(vectors)} embeddings for {len(texts)} inputs"
                    )
                return vectors

            if resp.status_code == 400 and len(texts) > 1 and "token" in resp.text.lower():
                raise _BatchTooLargeError(resp.text[:400])

            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                backoff = _retry_backoff_seconds(attempt)
                logger.warning(
                    "Vertex AI embedding HTTP %d (attempt %d/%d); retrying in %.1fs...",
                    resp.status_code,
                    attempt,
                    self.max_retries,
                    backoff,
                )
                time.sleep(backoff)
                continue

            raise RuntimeError(
                f"Vertex AI embedding request failed (HTTP {resp.status_code}): {resp.text[:400]}"
            )

        raise RuntimeError("Exhausted retries calling Vertex AI embedding API")

    def _embed_batch_adaptive(
        self,
        texts: list[str],
        *,
        task_type: str,
        titles: list[str] | None,
    ) -> list[list[float]]:
        """Embed one batch, halving it recursively if the API rejects it as too many tokens."""
        try:
            return self._predict_batch(texts, task_type=task_type, titles=titles)
        except _BatchTooLargeError:
            mid = len(texts) // 2
            logger.info("Embedding batch of %d exceeded the token limit; splitting.", len(texts))
            left = self._embed_batch_adaptive(
                texts[:mid], task_type=task_type, titles=titles[:mid] if titles else None
            )
            right = self._embed_batch_adaptive(
                texts[mid:], task_type=task_type, titles=titles[mid:] if titles else None
            )
            return left + right

    def embed_texts(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
        titles: list[str] | None = None,
        on_batch_complete: Callable[[int, int, list[list[float]]], None] | None = None,
    ) -> list[list[float]]:
        """Embed a list of texts in count- and size-bounded batches.

        `on_batch_complete(start, end, vectors)` is called after each batch finishes
        (from the calling thread), which lets callers checkpoint progress.
        Thread-safe for concurrent callers: never mutates `self._http_client`.
        """
        batches = self._plan_batches(texts)
        results: list[list[float] | None] = [None] * len(texts)

        def run(bounds: tuple[int, int]) -> tuple[int, int, list[list[float]]]:
            start, end = bounds
            vecs = self._embed_batch_adaptive(
                texts[start:end],
                task_type=task_type,
                titles=titles[start:end] if titles else None,
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
            # Ensure the shared pooled httpx.Client is initialized safely under lock before fan-out.
            _ = self._get_http_client()
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
            raise RuntimeError(f"Vertex AI embedding left {len(missing)} inputs without vectors")
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
    """Populate `embedding` on each `FilingChunk` with Vertex AI and return `(chunks, source)`.

    `source` is `"vertex_ai"` when any vector was freshly generated or resumed from the
    Vertex AI checkpoint, or `"cached"` when every vector was reused from a prior manifest.
    Any Vertex AI failure propagates to the caller; there is no synthetic fallback.

    If `existing_embeddings` is provided (mapping `chunk_id -> (sha256_text, embedding)`),
    unchanged chunks with valid dimension embeddings are reused without recomputation.

    If `checkpoint_path` is provided, every Vertex AI batch is appended to that JSONL
    file as it completes, and rows already in it (same model and dimensions) are
    reused, so an interrupted run resumes instead of re-embedding everything.
    """
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
