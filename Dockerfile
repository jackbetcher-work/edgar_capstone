# syntax=docker/dockerfile:1
# ==============================================================================
# Stage 0: Frontend Builder — Compile React + TypeScript Terminal UI bundle
# ==============================================================================
FROM node:22-slim AS frontend-builder

ARG NPM_REGISTRY="https://registry.npmjs.org/"

WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund --registry="${NPM_REGISTRY}"

COPY frontend/ ./
RUN npm run build && test -f /frontend/dist/index.html

# ==============================================================================
# Stage 1: Builder — Build wheel & install Python runtime dependencies
# ==============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

COPY pyproject.toml requirements.txt ./
COPY src/ ./src/
COPY prompts/ ./prompts/
COPY scripts/ ./scripts/

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install .

# ==============================================================================
# Stage 2: Runtime — Minimal non-root container for Google Cloud Run
# No secrets or keys are baked into this image; all sensitive credentials are
# injected at runtime from Google Cloud Secret Manager.
# ==============================================================================
FROM python:3.11-slim AS runtime

ARG COMMIT_SHA="unknown"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8080 \
    FRONTEND_DIST_DIR="/app/frontend/dist" \
    GCP_PROJECT_ID="iron-radio-514217-q7" \
    GCP_PROJECT_NAME="fde-sec-edgar-sandbox-dev" \
    GCP_REGION="us-central1" \
    RUNTIME_SERVICE_ACCOUNT_ID="sec-analyst-sa" \
    COMMIT_SHA="${COMMIT_SHA}" \
    EDGAR_DATA_ROOT="/app/data" \
    EDGAR_DATASET_DIR="/app/data/curated_dataset" \
    SEC_CORPUS_BUCKET="fde-sec-edgar-sandbox-dev-sec-corpus" \
    EDGAR_CORPUS_HYDRATION="off" \
    EDGAR_CORPUS_HYDRATION_SCOPES="chunks,unstructured"

RUN groupadd --gid 10001 appgroup && \
    useradd --uid 10001 --gid appgroup --shell /sbin/nologin --create-home appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appgroup /opt/venv /opt/venv
COPY --from=frontend-builder --chown=appuser:appgroup /frontend/dist ./frontend/dist
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup prompts/ ./prompts/
COPY --chown=appuser:appgroup looker/ ./looker/
COPY --chown=appuser:appgroup scripts/ ./scripts/
COPY --chown=appuser:appgroup data/sp500_constituents.csv ./data/sp500_constituents.csv
COPY --chown=appuser:appgroup data/curated_dataset/manifests/bigquery_financials.jsonl ./data/curated_dataset/manifests/bigquery_financials.jsonl
COPY --chown=appuser:appgroup data/curated_dataset/manifests/chunk_health_report.json ./data/curated_dataset/manifests/chunk_health_report.json
COPY --chown=appuser:appgroup data/curated_dataset/manifests/dataset_catalog.json ./data/curated_dataset/manifests/dataset_catalog.json
COPY --chown=appuser:appgroup data/curated_dataset/manifests/vector_db_catalog.json ./data/curated_dataset/manifests/vector_db_catalog.json

# ------------------------------------------------------------------------------
# Container corpus packaging
# - Only lightweight, git-tracked manifests are baked into the image. The gitignored
#   heavyweight artifacts (manifests/bigquery_filing_chunks.jsonl, unstructured/*.md)
#   are excluded via .dockerignore and served from BigQuery / Vertex AI Search / GCS
#   in Cloud Run.
# - For offline smoke tests, set EDGAR_CORPUS_HYDRATION=auto|required to download
#   them from gs://${SEC_CORPUS_BUCKET}/ into the writable dataset dir on startup.
# - EDGAR_DATASET_DIR pins the corpus location because the package is installed
#   non-editable into /opt/venv (source-relative paths do not reach /app/data).
# ------------------------------------------------------------------------------
RUN mkdir -p /app/data/curated_dataset/manifests \
             /app/data/curated_dataset/structured \
             /app/data/curated_dataset/unstructured && \
    chown -R appuser:appgroup /app/data && \
    python -c "from edgar_capstone.dataset.corpus_hydration import describe_local_corpus as d; s = d(); assert s['profile_source'] != 'missing', s; print('Packaged corpus:', s)"

USER appuser

EXPOSE 8080

CMD ["sh", "-c", "uvicorn edgar_capstone.server:app --host 0.0.0.0 --port ${PORT:-8080}"]
