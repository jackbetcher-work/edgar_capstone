# API Integration Guide

The machine-readable OpenAPI 3.1 contract exported directly from FastAPI lives at [`docs/openapi.json`](openapi.json) (and is served dynamically at `/openapi.json` and `/docs` when running locally).

---

## 1. Authentication & Authorization

All `/api/v1/*` endpoints require authentication except `/api/v1/auth/config`, `/api/v1/auth/sso/exchange`, and `/api/v1/auth/sso/dev-token` (local dev only; disabled on Cloud Run where `IAP_STRICT_MODE=true`).

- **Production (Cloud Run IAP):** Direct Cloud Run IAP verifies the user and injects the signed `X-Goog-IAP-JWT-Assertion` header. Alternatively, send an authorized Google Workspace OIDC / IAP JWT via `Authorization: Bearer <jwt>`.
- **Local Development:** Mint an ephemeral ES256 dev token:
  ```bash
  TOKEN=$(curl -s -X POST http://127.0.0.1:8000/api/v1/auth/sso/dev-token \
    -H "Content-Type: application/json" \
    -d '{"email":"analyst@fde-workspace.corp","name":"Analyst","groups":["sec-edgar-analysts"]}' \
    | jq -r '.access_token')
  ```
- **RBAC Roles:** `admin` (`sec-edgar-admins`), `analyst` (`sec-edgar-analysts`), and `viewer` (`sec-edgar-viewers` / `sec-edgar-auditors`). Endpoints enforce granular permissions (`chat:query`, `filings:read`, `audit:read`, `rbac:admin`).

---

## 2. Primary Endpoints

| Method & Path | Permission | Purpose |
| :--- | :--- | :--- |
| `GET /health` | Public (probe) | Readiness, active models, backend adapters, session memory, and tracing status |
| `POST /api/v1/chat/stream` | `chat:query` | Primary conversational Hybrid RAG endpoint using Server-Sent Events (SSE) |
| `POST /api/v1/chat` | `chat:query` | Synchronous JSON variant of the same RAG pipeline |
| `GET /api/v1/companies` | `filings:read` | Curated S&P 500 catalog, GICS sectors, and available fiscal-year bounds |
| `GET /api/v1/section` | `filings:read` | Full Item 1A or Item 7 Markdown section from GCS for the split-pane viewer |
| `GET` · `DELETE /api/v1/sessions[/{session_id}]` | `chat:query` | List or delete the caller's tenant-isolated Firestore conversation sessions |
| `POST` · `GET /api/v1/feedback` | `chat:query` | Submit or retrieve analyst thumbs-up/down ratings and comments |
| `GET /api/v1/telemetry/sessions` | `audit:read` | Session, tool-usage, latency, and cost telemetry summaries |
| `GET /api/v1/auth/audit` | `audit:read` | Authentication, RBAC denial, guardrail block, and 10-K access audit trail |
| `GET /api/v1/admin/rbac/policies` | `rbac:admin` | Inspect active RBAC policy and custom runtime service account config |

---

## 3. SSE Streaming Contract (`POST /api/v1/chat/stream`)

### Request Body (`ChatQueryRequest`)
```json
{
  "query": "Compare NVIDIA and Apple's R&D spend from 2024 to 2026.",
  "tickers": "ALL",
  "gics_sector": "ALL",
  "section_id": "ALL",
  "start_year": 2024,
  "end_year": 2026,
  "top_k": 8,
  "session_id": "optional-client-session-id"
}
```
- `start_year` and `end_year` must be within `2015–2035` with `start_year <= end_year`.
- `tickers`, `gics_sector`, and `section_id` (`item_1a`, `item_7`, or `ALL`) default to `"ALL"` and are automatically resolved from the query and session memory when omitted.

### Streamed Events (`text/event-stream`)
The server emits named SSE events and sends `: keepalive` comment frames every 15 seconds to keep proxies open:

1. **`event: progress`** — Emitted immediately (`stage: "connected"`) and at each pipeline milestone (`starting`, `scope_resolved`, `retrieving`, `agent_tool`, `sector_retrieval`, `sector_map_start`, `sector_map_progress`, `synthesizing`, `verifying`, `intent_resolved`, `complete`):
   ```json
   {"stage": "agent_tool", "message": "Computing deterministic period-over-period variances (NVDA, AAPL)...", "meta": {"tool": "calculate_period_variances"}, "elapsed_ms": 1420.5}
   ```
2. **`event: result`** — Final canonical turn payload containing:
   - `status`: `"success"` | `"clarification"` | `"out_of_corpus"` | `"blocked"`
   - `answer_verified`: boolean (`false` if withheld by claim verification or blocked by guardrails)
   - `structured_report`: verified `AnalystReport` JSON (`intent`, `headline`, `key_metrics`, `findings`)
   - `answer_markdown`: deterministically rendered Markdown fallback with inline 10-K & XBRL tags
   - `variance_reports`, `material_variance_flags`, `disclosure_deltas`, `sector_coverage`, `coverage_notices`
   - `citations`: deduplicated `GroundedCitation[]` with `chunk_id`, `sec_filing_url`, and excerpts
   - `report_verification`, `fact_check`, `evaluation`, `model_armor`, `observability` (`trace_id`, `cloud_trace_url`)
3. **`event: error`** — Emitted if an upstream GCP/Vertex AI failure (`live_adk_unavailable`) or unexpected server error occurs mid-stream.

---

## 4. Response Headers & Error Codes

- **Correlation headers (every response):** `X-Trace-Id`, `X-Request-Id`, and `Server-Timing: app;dur=<ms>`.
- **Status codes:**
  - `401 Unauthorized` — Missing, expired, revoked, or invalid JWT/IAP assertion.
  - `403 Forbidden` — Authenticated principal lacks required RBAC permission or allowed Workspace domain/group.
  - `422 Unprocessable Entity` — Request schema validation failure (e.g., `start_year > end_year`).
  - `503 Service Unavailable` — Upstream GCP data plane or Vertex AI Gemini runtime unavailable after retries.

---

## 5. Regenerating the OpenAPI Spec

Whenever routes or Pydantic schemas change, refresh `docs/openapi.json`:

```bash
.venv/bin/python -c "import json; from pathlib import Path; from edgar_capstone.server import app; Path('docs/openapi.json').write_text(json.dumps(app.openapi(), indent=2) + '\n')"
```
