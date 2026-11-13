.DEFAULT_GOAL := help

VENV := .venv
PYTHON := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
BACKEND_HOST ?= 127.0.0.1
BACKEND_PORT ?= 8000
FRONTEND_PORT ?= 5173

.PHONY: help install dev start backend frontend build test clean

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage: make <target>\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2 } END {printf "\n"}' $(MAKEFILE_LIST)

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

test: ## Run backend (pytest + secret scan) and frontend (lint + vitest) checks
	$(PYTHON) scripts/verify_no_secrets.py --repo-root .
	$(PYTHON) -m pytest -q
	cd frontend && npm run lint && npm test

clean: ## Remove build artifacts and caches
	rm -rf frontend/dist .pytest_cache
	find src tests scripts -type d -name "__pycache__" -exec rm -rf {} +
