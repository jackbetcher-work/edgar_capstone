.DEFAULT_GOAL := help

VENV := .venv
PYTHON := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
BACKEND_HOST ?= 127.0.0.1
BACKEND_PORT ?= 8000
FRONTEND_PORT ?= 5173

# Container packaging & smoke-test settings
DOCKER ?= docker
IMAGE ?= sec-edgar-analyst:local
SMOKE_PORT ?= 8080
SMOKE_CONTAINER ?= sec-edgar-analyst-smoke
ADC_FILE ?= $(HOME)/.config/gcloud/application_default_credentials.json
# SMOKE_LIVE_DATA=true makes the smoke tests also read live BigQuery data (needs ADC).
SMOKE_LIVE_DATA ?= false
SMOKE_ARGS := $(if $(filter true 1 yes,$(SMOKE_LIVE_DATA)),--live-data,)
# On native Linux, ADC files are mode 0600 owned by the host user; match UID/GID inside the container.
DOCKER_ADC_USER := $(if $(filter Linux,$(shell uname -s)),--user $$(id -u):$$(id -g) -e HOME=/tmp,)

# Live GCP project / Vertex AI settings. There is no offline mode: the backend, the test
# suite, and the evals read BigQuery / Vertex AI Search / GCS and call Gemini on Vertex AI.
# Developer shells often export an unrelated GCP_PROJECT_ID, so these targets pin it.
CORPUS_PROJECT_ID ?= iron-radio-514217-q7
GCP_LOCATION ?= us-central1
GEMINI_LOCATION ?= global
SEC_CORPUS_BUCKET ?= fde-sec-edgar-sandbox-dev-sec-corpus
VERTEX_AI_SEARCH_DATASTORE_ID ?= sec-edgar-10k-corpus-ds
VERTEX_SEARCH_RECONCILIATION ?= INCREMENTAL
LIVE_GCP_ENV := GCP_PROJECT_ID=$(CORPUS_PROJECT_ID) GOOGLE_CLOUD_PROJECT=$(CORPUS_PROJECT_ID) \
	GOOGLE_CLOUD_LOCATION=$(GCP_LOCATION) GOOGLE_GENAI_USE_VERTEXAI=true \
	EDGAR_GEMINI_LOCATION=$(GEMINI_LOCATION)
# Tests, evals, and smoke tests keep telemetry + feedback in memory (no prod telemetry writes).
LIVE_TEST_ENV := $(LIVE_GCP_ENV) EDGAR_TELEMETRY_BACKEND=local EDGAR_FEEDBACK_BACKEND=local
DOCKER_LIVE_TEST_ENV := $(addprefix -e ,$(LIVE_TEST_ENV))

# Golden eval suite (scripts/run_evals.py): live Gemini + the Flash judge on Vertex AI.
# EVAL_WRITE_BQ=true streams the run to BigQuery sec_edgar.eval_results.
EVAL_DATASET ?= evals/golden_dataset.jsonl
EVAL_WRITE_BQ ?= false
EVAL_ARGS ?=

.PHONY: help install dev start backend frontend build lint format test check clean smoke docker-build docker-smoke publish-financials index-vertex-search backfill-rd eval-all load-golden-dataset refresh-golden-expectations

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage: make <target>\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 } END {printf "\n"}' $(MAKEFILE_LIST)

install: ## Install backend (.venv) and frontend dependencies
	@test -d $(VENV) || python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]" || $(VENV)/bin/pip install -r requirements.txt
	cd frontend && npm ci --no-audit --no-fund

dev: ## Run FastAPI backend (:8000, live GCP) and Vite Terminal UI (:5173) together
	@echo "Starting FastAPI (http://$(BACKEND_HOST):$(BACKEND_PORT)) + Terminal UI (http://127.0.0.1:$(FRONTEND_PORT))..."
	@trap 'kill 0' INT TERM EXIT; \
		$(LIVE_GCP_ENV) $(PYTHON) -m uvicorn --app-dir src edgar_capstone.server:app --reload --host $(BACKEND_HOST) --port $(BACKEND_PORT) & \
		(cd frontend && npm run dev -- --host 127.0.0.1 --port $(FRONTEND_PORT)) & \
		wait

start: build ## Build production UI bundle and serve unified single-origin app via FastAPI (:8000, live GCP)
	$(LIVE_GCP_ENV) $(PYTHON) -m uvicorn --app-dir src edgar_capstone.server:app --host $(BACKEND_HOST) --port $(BACKEND_PORT)

backend: ## Run FastAPI backend server only (live GCP)
	$(LIVE_GCP_ENV) $(PYTHON) -m uvicorn --app-dir src edgar_capstone.server:app --reload --host $(BACKEND_HOST) --port $(BACKEND_PORT)

frontend: ## Run Vite React dev server only
	cd frontend && npm run dev -- --host 127.0.0.1 --port $(FRONTEND_PORT)

build: ## Build production frontend bundle and validate backend
	$(PYTHON) -m compileall -q src
	cd frontend && npm run build

lint: ## Run the CI lint/format gates: ruff check + format --check, terraform fmt/validate, ESLint
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
	@if command -v terraform >/dev/null 2>&1; then \
		terraform fmt -check -recursive terraform/ && \
		terraform -chdir=terraform init -backend=false -input=false >/dev/null && \
		terraform -chdir=terraform validate -no-color; \
	else \
		echo "terraform not installed; skipping terraform fmt/validate (still enforced in CI)"; \
	fi
	cd frontend && npm run lint

format: ## Auto-fix Python lint issues and apply ruff formatting + terraform fmt
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .
	@if command -v terraform >/dev/null 2>&1; then terraform fmt -recursive terraform/; fi

check: lint test ## Run every CI gate locally (lint + format checks, secret scan, pytest, vitest)

test: ## Run backend (secret scan + pytest with >=80% coverage against live GCP) and frontend (lint + vitest) checks
	$(PYTHON) scripts/verify_no_secrets.py --repo-root .
	$(LIVE_TEST_ENV) $(PYTHON) -m pytest -q --cov=edgar_capstone --cov-report=term-missing --cov-fail-under=80
	cd frontend && npm run lint && npm test

eval-all: ## Run the golden eval regression gates on live Gemini + GCP data (EVAL_WRITE_BQ=true)
	$(LIVE_TEST_ENV) $(PYTHON) scripts/run_evals.py --dataset $(EVAL_DATASET) \
		$(if $(filter true 1 yes,$(EVAL_WRITE_BQ)),--write-bigquery --project-id $(CORPUS_PROJECT_ID),) $(EVAL_ARGS)

load-golden-dataset: ## Validate evals/golden_dataset.jsonl and load it into BigQuery sec_edgar.golden_eval_set
	$(PYTHON) scripts/load_golden_dataset.py --project-id $(CORPUS_PROJECT_ID) --dataset $(EVAL_DATASET)

refresh-golden-expectations: ## Recompute golden expected_variances numbers from the local engine (review the diff!)
	$(PYTHON) scripts/refresh_golden_expectations.py --dataset $(EVAL_DATASET)

backfill-rd: ## Backfill R&D expense onto staged profiles from SEC XBRL companyfacts, rewrite manifests, upload to GCS
	$(PYTHON) scripts/backfill_rd_metric.py --project-id $(CORPUS_PROJECT_ID) --bucket $(SEC_CORPUS_BUCKET)

publish-financials: ## Reload BigQuery sec_edgar.financial_metrics from the GCS manifests/bigquery_financials.jsonl
	$(PYTHON) scripts/publish_corpus.py financial-metrics --project-id $(CORPUS_PROJECT_ID) --bucket $(SEC_CORPUS_BUCKET)

index-vertex-search: ## Import vertex_ai_search_metadata.jsonl into Vertex AI Search (documents:import) + smoke query
	$(PYTHON) scripts/publish_corpus.py vertex-search --project-id $(CORPUS_PROJECT_ID) --bucket $(SEC_CORPUS_BUCKET) \
		--data-store-id $(VERTEX_AI_SEARCH_DATASTORE_ID) --reconciliation-mode $(VERTEX_SEARCH_RECONCILIATION)

smoke: ## Smoke-test the app in-process against the local checkout (no Docker; SMOKE_LIVE_DATA=true reads BigQuery)
	$(LIVE_TEST_ENV) $(PYTHON) scripts/container_smoke_test.py --in-process \
		$(if $(wildcard frontend/dist/index.html),,--skip-ui) $(SMOKE_ARGS)

docker-build: ## Build the production container image locally ($(IMAGE))
	$(DOCKER) build --build-arg COMMIT_SHA=$$(git rev-parse --short HEAD 2>/dev/null || echo local) -t $(IMAGE) .

docker-smoke: docker-build ## Run the image and smoke-test /health + UI over HTTP (SMOKE_LIVE_DATA=true mounts local ADC)
	@if [ -n "$(SMOKE_ARGS)" ]; then test -f "$(ADC_FILE)" || { echo "Missing ADC file $(ADC_FILE); run 'gcloud auth application-default login'"; exit 1; }; fi
	@$(DOCKER) rm -f $(SMOKE_CONTAINER) >/dev/null 2>&1 || true
	$(DOCKER) run -d --name $(SMOKE_CONTAINER) -p $(SMOKE_PORT):8080 $(DOCKER_LIVE_TEST_ENV) \
		$(if $(SMOKE_ARGS),$(DOCKER_ADC_USER) -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/adc.json -v "$(ADC_FILE)":/tmp/keys/adc.json:ro,) \
		$(IMAGE)
	@status=0; \
	$(PYTHON) scripts/container_smoke_test.py --base-url http://127.0.0.1:$(SMOKE_PORT) --wait-seconds 90 $(SMOKE_ARGS) || status=$$?; \
	if [ $$status -ne 0 ]; then $(DOCKER) logs --tail 60 $(SMOKE_CONTAINER); fi; \
	$(DOCKER) rm -f $(SMOKE_CONTAINER) >/dev/null; exit $$status

clean: ## Remove build artifacts and caches
	rm -rf frontend/dist .pytest_cache .ruff_cache
	find src tests scripts -type d -name "__pycache__" -exec rm -rf {} +
