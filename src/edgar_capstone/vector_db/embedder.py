"""Dense vector embedding generation for SEC 10-K RAG chunks (Vertex AI & deterministic)."""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from typing import Any, Literal

import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest
import httpx
import numpy as np

from edgar_capstone.vector_db.models import FilingChunk

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "text-embedding-005"
DEFAULT_EMBEDDING_DIMENSIONS = 768
_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def generate_deterministic_embedding(
    text: str,
    *,
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
) -> list[float]:
    """Generate a unit-normalized deterministic feature-hashed embedding vector.

    Uses signed unigram and bigram feature hashing so passages sharing financial
    and risk terminology exhibit positive cosine similarity in offline/local tests.
    """
    if dimensions <= 0:
        raise ValueError(f"dimensions must be positive, got {dimensions}")

    tokens = _TOKEN_RE.findall(text.lower())
    vec = np.zeros(dimensions, dtype=np.float64)

    if not tokens:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=32).digest()
        idx = int.from_bytes(digest[:4], "little") % dimensions
        vec[idx] = 1.0
        return [round(float(x), 6) for x in vec]

    features: list[str] = list(tokens)
    if len(tokens) > 1:
        features.extend(f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1))

    for feat in features:
        h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        val = int.from_bytes(h, "little")
        bucket = val % dimensions
        sign = 1.0 if ((val >> 32) & 1) == 0 else -1.0
        vec[bucket] += sign

    norm = float(np.linalg.norm(vec))
    if norm > 0.0 and math.isfinite(norm):
        vec = vec / norm
    else:
        vec[0] = 1.0

    return [round(float(x), 6) for x in vec]


class VertexAIEmbedder:
    """Client for generating dense embeddings via Vertex AI (`text-embedding-005`)."""

    def __init__(
        self,
        *,
        project_id: str,
        location: str = "us-central1",
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        batch_size: int = 50,
        max_retries: int = 3,
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
    ) -> None:
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.dimensions = dimensions
        self.batch_size = max(1, min(batch_size, 250))
        self.max_retries = max_retries
        self._http_client = http_client
        self._credentials = credentials

    def _get_access_token(self) -> str:
        """Obtain a valid OAuth2 bearer token from Application Default Credentials."""
        if self._credentials is None:
            self._credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        if not self._credentials.valid:
            self._credentials.refresh(GoogleAuthRequest())
        return str(self._credentials.token)

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

        client = self._http_client or httpx.Client(timeout=60.0)
        close_client = self._http_client is None
        try:
            for attempt in range(1, self.max_retries + 1):
                token = self._get_access_token()
                resp = client.post(
                    url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                )
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

                if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    backoff = min(16.0, 1.5 * (2 ** (attempt - 1)))
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
        finally:
            if close_client:
                client.close()

        raise RuntimeError("Exhausted retries calling Vertex AI embedding API")

    def embed_texts(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
        titles: list[str] | None = None,
    ) -> list[list[float]]:
        """Embed a list of texts in batches."""
        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            batch_titles = titles[start : start + self.batch_size] if titles else None
            batch_vecs = self._predict_batch(
                batch_texts,
                task_type=task_type,
                titles=batch_titles,
            )
            all_vectors.extend(batch_vecs)
        return all_vectors


def populate_chunk_embeddings(
    chunks: list[FilingChunk],
    *,
    project_id: str,
    location: str = "us-central1",
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    backend: Literal["vertex_ai", "deterministic", "auto"] = "auto",
    batch_size: int = 50,
    existing_embeddings: dict[str, tuple[str, list[float]]] | None = None,
    vertex_embedder: VertexAIEmbedder | None = None,
) -> tuple[list[FilingChunk], str]:
    """Populate `embedding` on each `FilingChunk` and return `(chunks, resolved_backend)`.

    If `existing_embeddings` is provided (mapping `chunk_id -> (sha256_text, embedding)`),
    unchanged chunks with valid dimension embeddings are reused without recomputation.
    """
    if not chunks:
        return [], backend if backend != "auto" else "deterministic"

    to_embed_indices: list[int] = []
    for idx, chunk in enumerate(chunks):
        text_hash = hashlib.sha256(chunk.chunk_text.encode("utf-8")).hexdigest()
        if existing_embeddings and chunk.chunk_id in existing_embeddings:
            cached_hash, cached_vec = existing_embeddings[chunk.chunk_id]
            if cached_hash == text_hash and len(cached_vec) == dimensions:
                chunk.embedding = cached_vec
                chunk.embedding_model = model_name
                continue
        to_embed_indices.append(idx)

    if not to_embed_indices:
        logger.info("Reused all %d cached chunk embeddings.", len(chunks))
        return chunks, "cached"

    resolved_backend = backend
    if backend in ("vertex_ai", "auto"):
        embedder = vertex_embedder or VertexAIEmbedder(
            project_id=project_id,
            location=location,
            model_name=model_name,
            dimensions=dimensions,
            batch_size=batch_size,
        )
        try:
            texts = [chunks[i].chunk_text for i in to_embed_indices]
            titles = [
                f"{chunks[i].ticker} FY{chunks[i].fiscal_year} 10-K {chunks[i].section_title}"
                for i in to_embed_indices
            ]
            vectors = embedder.embed_texts(
                texts,
                task_type="RETRIEVAL_DOCUMENT",
                titles=titles,
            )
            for i, vec in zip(to_embed_indices, vectors, strict=True):
                chunks[i].embedding = vec
                chunks[i].embedding_model = model_name
            return chunks, "vertex_ai"
        except Exception as exc:
            if backend == "vertex_ai":
                raise
            logger.warning(
                "Vertex AI live embedding unavailable (%s); falling back to deterministic %d-d embeddings.",
                exc,
                dimensions,
            )
            resolved_backend = "deterministic"

    for i in to_embed_indices:
        chunks[i].embedding = generate_deterministic_embedding(
            chunks[i].chunk_text,
            dimensions=dimensions,
        )
        chunks[i].embedding_model = model_name

    return chunks, resolved_backend
