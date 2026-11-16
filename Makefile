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
HYDRATION_SCOPES ?= chunks,unstructured
HYDRATION_TICKERS ?=
ADC_FILE ?= $(HOME)/.config/gcloud/application_default_credentials.json
CORPUS_PROJECT_ID ?= iron-radio-514217-q7
SEC_CORPUS_BUCKET ?= fde-sec-edgar-sandbox-dev-sec-corpus
VERTEX_AI_SEARCH_DATASTORE_ID ?= sec-edgar-10k-corpus-ds
VERTEX_SEARCH_RECONCILIATION ?= INCREMENTAL
OFFLINE_ENV := -e EDGAR_EXECUTION_MODE=deterministic -e EDGAR_DATA_BACKEND=local \
	-e EDGAR_TELEMETRY_BACKEND=local -e EDGAR_FEEDBACK_BACKEND=local \
	-e EDGAR_PROMPT_STORE_BACKEND=local -e EDGAR_MODEL_ARMOR_MODE=disabled

.PHONY: help install dev start backend frontend build lint format test check clean hydrate-corpus smoke docker-build docker-smoke docker-smoke-hydrated publish-financials index-vertex-search backfill-rd

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage: make <target>\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 } END {printf "\n"}' $(MAKEFILE_LIST)

install: ## Install backend (.venv) and frontend dependencies
	@test -d $(VENV) || python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]" || $(VENV)/bin/pip install -r requirements.txt
	cd frontend && npm ci --no-audit --no-fund

dev: ## Run FastAPI backend (:8000) and Vite Terminal UI (:5173) together
	@echo "Starting FastAPI (http://$(BACKEND_HOST):$(BACKEND_PORT)) + Terminal UI (http://127.0.0.1:$(FRONTEND_PORT))..."
	@trap 'kill 0' INT TERM EXIT; \
		$(PYTHON) -m uvicorn --app-dir src edgar_capstone.server:app --reload --host $(BACKEND_HOST) --port $(BACKEND_PORT) & \
		(cd frontend && npm run dev -- --host 127.0.0.1 --port $(FRONTEND_PORT)) & \
		wait

start: build ## Build production UI bundle and serve unified single-origin app via FastAPI (:8000)
	$(PYTHON) -m uvicorn --app-dir src edgar_capstone.server:app --host $(BACKEND_HOST) --port $(BACKEND_PORT)

backend: ## Run FastAPI backend server only
	$(PYTHON) -m uvicorn --app-dir src edgar_capstone.server:app --reload --host $(BACKEND_HOST) --port $(BACKEND_PORT)

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

test: ## Run backend (pytest + secret scan) and frontend (lint + vitest) checks
	$(PYTHON) scripts/verify_no_secrets.py --repo-root .
	$(PYTHON) -m pytest -q
	cd frontend && npm run lint && npm test

hydrate-corpus: ## Download gitignored chunks manifest + 10-K Markdown from GCS into data/curated_dataset
	$(PYTHON) scripts/hydrate_corpus.py --bucket $(SEC_CORPUS_BUCKET) --project-id $(CORPUS_PROJECT_ID) --scopes $(HYDRATION_SCOPES) $(if $(HYDRATION_TICKERS),--tickers $(HYDRATION_TICKERS),)

backfill-rd: ## Backfill R&D expense onto staged profiles from SEC XBRL companyfacts, rewrite manifests, upload to GCS
	$(PYTHON) scripts/backfill_rd_metric.py --project-id $(CORPUS_PROJECT_ID) --bucket $(SEC_CORPUS_BUCKET)

publish-financials: ## Reload BigQuery sec_edgar.financial_metrics from the GCS manifests/bigquery_financials.jsonl
	$(PYTHON) scripts/publish_corpus.py financial-metrics --project-id $(CORPUS_PROJECT_ID) --bucket $(SEC_CORPUS_BUCKET)

index-vertex-search: ## Import vertex_ai_search_metadata.jsonl into Vertex AI Search (documents:import) + smoke query
	$(PYTHON) scripts/publish_corpus.py vertex-search --project-id $(CORPUS_PROJECT_ID) --bucket $(SEC_CORPUS_BUCKET) \
		--data-store-id $(VERTEX_AI_SEARCH_DATASTORE_ID) --reconciliation-mode $(VERTEX_SEARCH_RECONCILIATION)

smoke: ## Run the offline API smoke test in-process against the local checkout (no Docker)
	EDGAR_EXECUTION_MODE=deterministic EDGAR_DATA_BACKEND=local EDGAR_TELEMETRY_BACKEND=local \
	EDGAR_FEEDBACK_BACKEND=local EDGAR_PROMPT_STORE_BACKEND=local EDGAR_MODEL_ARMOR_MODE=disabled \
	$(PYTHON) scripts/container_smoke_test.py --in-process --expect-corpus baked

docker-build: ## Build the production container image locally ($(IMAGE))
	$(DOCKER) build --build-arg COMMIT_SHA=$$(git rev-parse --short HEAD 2>/dev/null || echo local) -t $(IMAGE) .

docker-smoke: docker-build ## Run the image offline (no GCP) and smoke-test the baked corpus over HTTP
	@$(DOCKER) rm -f $(SMOKE_CONTAINER) >/dev/null 2>&1 || true
	$(DOCKER) run -d --name $(SMOKE_CONTAINER) -p $(SMOKE_PORT):8080 $(OFFLINE_ENV) $(IMAGE)
	@status=0; \
	$(PYTHON) scripts/container_smoke_test.py --base-url http://127.0.0.1:$(SMOKE_PORT) --wait-seconds 90 --expect-corpus baked || status=$$?; \
	if [ $$status -ne 0 ]; then $(DOCKER) logs --tail 60 $(SMOKE_CONTAINER); fi; \
	$(DOCKER) rm -f $(SMOKE_CONTAINER) >/dev/null; exit $$status

docker-smoke-hydrated: docker-build ## Run the image with GCS corpus hydration (local ADC) and assert real chunks + 10-K Markdown
	@test -f "$(ADC_FILE)" || { echo "Missing ADC file $(ADC_FILE); run 'gcloud auth application-default login'"; exit 1; }
	@$(DOCKER) rm -f $(SMOKE_CONTAINER) >/dev/null 2>&1 || true
	$(DOCKER) run -d --name $(SMOKE_CONTAINER) -p $(SMOKE_PORT):8080 $(OFFLINE_ENV) \
		-e EDGAR_CORPUS_HYDRATION=required \
		-e EDGAR_CORPUS_HYDRATION_SCOPES=$(HYDRATION_SCOPES) \
		-e EDGAR_CORPUS_HYDRATION_TICKERS=$(HYDRATION_TICKERS) \
		-e SEC_CORPUS_BUCKET=$(SEC_CORPUS_BUCKET) \
		-e GOOGLE_CLOUD_PROJECT=$(CORPUS_PROJECT_ID) \
		-e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/adc.json \
		-v "$(ADC_FILE)":/tmp/keys/adc.json:ro \
		$(IMAGE)
	@status=0; \
	$(PYTHON) scripts/container_smoke_test.py --base-url http://127.0.0.1:$(SMOKE_PORT) --wait-seconds 600 --expect-corpus hydrated || status=$$?; \
	if [ $$status -ne 0 ]; then $(DOCKER) logs --tail 60 $(SMOKE_CONTAINER); fi; \
	$(DOCKER) rm -f $(SMOKE_CONTAINER) >/dev/null; exit $$status

clean: ## Remove build artifacts and caches
	rm -rf frontend/dist .pytest_cache .ruff_cache
	find src tests scripts -type d -name "__pycache__" -exec rm -rf {} +
