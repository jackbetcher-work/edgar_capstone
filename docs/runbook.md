# Operational Runbook — SEC EDGAR Natural Language Analyst

Operational procedures, SLO targets, deployment/rollback commands, incident triage playbooks, and routine maintenance for the `sec-edgar-analyst` Cloud Run service in project `iron-radio-514217-q7` (`us-central1`).

---

## 1. Service Overview & SLOs

| Property | Value |
| :--- | :--- |
| **GCP Project / Region** | `iron-radio-514217-q7` / `us-central1` (Gemini endpoint: `global`) |
| **Cloud Run Service** | `sec-edgar-analyst` (`min=1`, `max=10`, concurrency `8`, `2 vCPU / 4Gi`, timeout `300s`) |
| **Runtime Identity** | `sec-analyst-sa@iron-radio-514217-q7.iam.gserviceaccount.com` |
| **Access Boundary** | Direct Cloud Run IAP (`iap_enabled = true`, `IAP_STRICT_MODE = true`) |
| **Health Probe** | `GET /health` (startup probe: 10 s period, 5 s timeout, 24 failures max) |
| **TTFT SLO** | `p95_ttft_ms <= 3,000 ms` (`DEFAULT_TTFT_SLO_MS`) |
| **API Latency Alert** | `api_latency_p95` > `30,000 ms` over 5 min on `/api/v1/chat(/stream)?` |
| **Error Rate Alert** | `api_5xx_errors` >= `5` HTTP 5xx responses over 5 min |
| **Security Alert** | `security_audit_blocks` > `20` denied (`401`/`403`) or guardrail-blocked events over 5 min |

Quick console links (via Terraform outputs):

```bash
terraform -chdir=terraform output -raw app_url
terraform -chdir=terraform output -raw llm_observability_dashboard_url
```

---

## 2. Deployment, Canary Rollout & Rollback

### Separation of Concerns
- **Terraform (`terraform/cicd_and_secrets.tf`)** is the single source of truth for Cloud Run configuration, scaling, IAM, IAP, and environment variables.
- **GitHub Actions (`.github/workflows/deploy.yml`)** performs **image-only staged rollouts**:
  1. **Test stage:** secret scan (`verify_no_secrets.py`), `ruff`, `terraform fmt/validate`, live `pytest` (`>=80%` coverage), frontend ESLint/Vitest/build.
  2. **Build stage (`main` only):** multi-stage Docker build, in-process container smoke test (`--live-data`), push to Artifact Registry.
  3. **Optional Eval stage (`main` only):** runs `scripts/run_evals.py --write-bigquery` when `EVAL_GATE_MODE` / `eval_mode` is `advisory` or `blocking` (defaults to `skip`).
  4. **Canary Deploy stage (`main` only):**
     - Captures the current serving stable revision (`prev_revision`).
     - Deploys the new image with `--no-traffic --tag canary` so Cloud Run validates the `/health` startup probe before routing any live user traffic.
     - Shifts **10%** canary traffic to the new revision (`90%` remains on `prev_revision`) for a soak window, verifies revision `Ready=True`, expected image, `/health` probe, and `iapEnabled=true`.
     - Promotes the new revision to **100%** traffic and removes the temporary `canary` tag.
     - **Automated Rollback:** if any verification or promotion step fails, `rollback_cloud_run` immediately restores `100%` traffic to `prev_revision`.

### Manual Rollback Procedure (Emergency)

1. List recent revisions and current traffic split:
   ```bash
   gcloud run revisions list \
     --project iron-radio-514217-q7 \
     --region us-central1 \
     --service sec-edgar-analyst \
     --limit 10
   ```
2. Route 100% of traffic immediately back to the last known good revision (`<GOOD_REVISION>`):
   ```bash
   gcloud run services update-traffic sec-edgar-analyst \
     --project iron-radio-514217-q7 \
     --region us-central1 \
     --to-revisions="<GOOD_REVISION>=100"
   ```
3. Confirm traffic and IAP integrity:
   ```bash
   gcloud run services describe sec-edgar-analyst \
     --project iron-radio-514217-q7 \
     --region us-central1 \
     --format="yaml(status.traffic,status.latestReadyRevisionName)"
   ```

---

## 3. Incident Triage Playbooks

Every API response includes `X-Trace-Id`, `X-Request-Id`, and `Server-Timing`. Always start triage with the trace ID (see [`docs/observability.md`](observability.md) for full trace/log/BigQuery join recipes).

### Playbook A: High Latency / TTFT SLO Breach (`api_latency_p95`)
1. Open the **Cloud Monitoring Dashboard** (`llm_observability_dashboard`) and click trace exemplars on `sec_edgar/api_request_latencies` or `sec_edgar/llm_ttft_ms`.
2. Inspect the Cloud Trace waterfall to isolate the slow span:
   - **`adk.run_agent_stream` / `call_llm` (Vertex AI):** Check Vertex AI `global` endpoint status or quota throttling (`HTTP 429` retry events on spans). Verify context cache hit rate in `v_telemetry_cost_and_latency`.
   - **`tool.search_10k_chunks_rag` / BigQuery / Discovery Engine:** Check BigQuery job queuing or Discovery Engine latency. Confirm `EntityCatalogCache` isn't cold-looping on errors.
   - **`guardrail.inspect_user_prompt` (Model Armor):** Check regional `us-central1` Model Armor RPC latency.
3. Mitigations:
   - Lower `EDGAR_ASYNC_JUDGE_SAMPLE_RATE` (e.g., `0.1` or `0.0`) via Terraform if background judging competes for Vertex AI quota.
   - Raise Cloud Run `max_instance_count` or `min_instance_count` in `terraform/cicd_and_secrets.tf` if cold starts dominate.

### Playbook B: Elevated HTTP 5xx Errors (`api_5xx_errors`)
1. Query Cloud Logging for 5xx access logs and unhandled exceptions:
   ```text
   resource.type="cloud_run_revision"
   resource.labels.service_name="sec-edgar-analyst"
   (jsonPayload.http_status>=500 OR severity>=ERROR)
   ```
2. Common causes:
   - **HTTP `503` (`live_adk_unavailable`):** Upstream Vertex AI, BigQuery, or Firestore outage/quota exhaustion after retries (`resilience.py`). Check GCP Service Health and Cloud Run IAM bindings on `sec-analyst-sa`.
   - **Mid-stream SSE abort (`stream_aborted: true`):** Inspect `http_request_failed` stack trace in Cloud Error Reporting using the correlated `trace_id`.
   - **Bad recent release:** Execute the **Manual Rollback Procedure** in Section 2.

### Playbook C: Security / Guardrail Spike (`security_audit_blocks`) or Model Armor Outage
1. Inspect recent audit events in BigQuery:
   ```sql
   SELECT recorded_at, event_type, status, user_email, identity_provider, detail, trace_id
   FROM `iron-radio-514217-q7.sec_edgar.security_audit_logs`
   WHERE DATE(recorded_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
     AND (status LIKE 'denied%' OR status LIKE 'blocked%')
   ORDER BY recorded_at DESC
   LIMIT 50;
   ```
2. Differentiate cause by `event_type`:
   - **`model_armor_fail_closed` / `model_armor_response_fail_closed`:** Model Armor RPC failed and the service **intentionally failed closed** (`fail_closed=True`). Check `modelarmor.googleapis.com` and `dlp.googleapis.com` health/quotas in `us-central1`, and verify `sec-analyst-sa` holds `roles/modelarmor.user` and `roles/dlp.user`. Never silently disable guardrails in production without security sign-off.
   - **`api_authorization_denied` (`denied_401` / `denied_403`):** Verify `INTERNAL_IAP_AUDIENCE`, `ALLOWED_WORKSPACE_DOMAINS`, and `iap_authorized_principals` in Terraform.
   - **`model_armor_prompt_blocked` / `projection_filter_*_blocked`:** Review whether a legitimate analyst query triggered a false positive in `prompts/v1/model_armor_policy.yaml` or an active abuse/red-team event is occurring.

### Playbook D: Corpus / Retrieval Quality Degradation
1. Query BigQuery chunk health views:
   ```sql
   SELECT gics_sector, section_id, total_chunks, embedding_coverage_ratio,
          full_section_coverage_ratio, health_score, health_status
   FROM `iron-radio-514217-q7.sec_edgar.v_chunk_health_metrics`
   WHERE health_status != 'HEALTHY'
   ORDER BY health_score ASC;
   ```
2. If specific filings show `UNINDEXED` or `EMBEDDING_ANOMALY` in `sec_edgar.v_filing_chunk_coverage`, re-run the corpus refresh workflow in Section 4.

---

## 4. Routine Operations

### Granting or Revoking User Access (IAP / RBAC)
1. Edit `iap_authorized_principals` (must be `group:<group>@<domain>` or `domain:<domain>`; individual `user:` bindings are rejected) and `iap_default_app_role` in `terraform/variables.tf` or your `.tfvars` file.
2. Apply Terraform:
   ```bash
   terraform -chdir=terraform plan
   terraform -chdir=terraform apply
   ```
3. To revoke a user immediately, remove them from the authorized Cloud Identity group (or remove the group/domain from `iap_authorized_principals` and apply). Note: `POST /api/v1/auth/logout` revokes bearer tokens in-memory per instance; enterprise revocation should happen at the IdP/IAP group level.

### Refreshing the 10-K Corpus & Search Indexes
```bash
# 1. Extract/update filings & embeddings (requires `pip install -e ".[ingest]"`)
.venv/bin/python scripts/build_sec_dataset.py --sp500 --no-create-bucket
.venv/bin/python scripts/build_vector_db.py

# 2. Backfill R&D, publish BigQuery financial_metrics, and re-index Vertex AI Search
make backfill-rd publish-financials index-vertex-search
```

### Prompt Updates & Regression Evaluation
1. Update the versioned prompt YAML under `prompts/` and update `prompts/registry.yaml`.
2. Run the full lint/test suite and live 30-query golden evaluation gates:
   ```bash
   make check
   EVAL_CONCURRENCY=1 EVAL_WRITE_BQ=true make eval-all
   ```
3. Confirm all 18 gates pass in `build/evals/latest.json` before merging to `main`.

---

## 5. Backup, Recovery & Disaster Recovery

| Asset | Durability / Protection | Recovery Procedure |
| :--- | :--- | :--- |
| **Terraform State** (`gs://iron-radio-514217-q7-tfstate`) | GCS object versioning enabled | Restore prior object version via `gsutil cp gs://.../default.tfstate#<gen> ...`. |
| **10-K Corpus Bucket** (`fde-sec-edgar-sandbox-dev-sec-corpus`) | Uniform access, object versioning, 7-day soft delete (`terraform/storage.tf`) | Restore soft-deleted or noncurrent object versions from GCS, or rebuild deterministically from SEC EDGAR via `scripts/build_sec_dataset.py`. |
| **BigQuery Tables** (`financial_metrics`, `filing_chunks`, telemetry, evals) | Native 7-day BigQuery time-travel (`FOR SYSTEM_TIME AS OF`); reloadable from GCS manifests | Query time-travel snapshot to restore accidental overwrites, or reload from `manifests/bigquery_*.jsonl` (`make publish-financials`, `scripts/build_vector_db.py --upload-only`). |
| **Firestore** (`analyst_sessions`, `analyst_feedback`) | Managed regional replication (`us-central1`) | Enable scheduled Firestore managed exports to GCS / PITR before enterprise production launch (RPO/RTO target: <1 hr). |
| **Container Images** (`sec-edgar-repo` GAR) | Immutable commit-SHA tags (`IMAGE_URI:<sha>`) | Roll back to any prior SHA via Cloud Run traffic shift or `gcloud run services update --image=...:<sha>`. |
