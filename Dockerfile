# syntax=docker/dockerfile:1
# ==============================================================================
# Stage 1: Builder — Build wheel & install runtime dependencies
# ==============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

COPY pyproject.toml requirements.txt ./
COPY src/ ./src/
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
    GCP_PROJECT_ID="iron-radio-514217-q7" \
    GCP_REGION="us-central1" \
    COMMIT_SHA="${COMMIT_SHA}"

RUN groupadd --gid 10001 appgroup && \
    useradd --uid 10001 --gid appgroup --shell /sbin/nologin --create-home appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appgroup /opt/venv /opt/venv
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup scripts/ ./scripts/
COPY --chown=appuser:appgroup data/sp500_constituents.csv ./data/sp500_constituents.csv

USER appuser

EXPOSE 8080

CMD ["sh", "-c", "uvicorn edgar_capstone.server:app --host 0.0.0.0 --port ${PORT:-8080}"]
