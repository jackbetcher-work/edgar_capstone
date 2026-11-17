# ------------------------------------------------------------------------------
# Golden Eval Dataset & Eval Results (`make eval-all`, scripts/run_evals.py)
# ------------------------------------------------------------------------------

# 1. Golden query-variance-explanation mappings, loaded from evals/golden_dataset.jsonl
#    by scripts/load_golden_dataset.py (WRITE_TRUNCATE, so it mirrors the committed file).
resource "google_bigquery_table" "golden_eval_set" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.sec_edgar.dataset_id
  table_id            = "golden_eval_set"
  description         = "Golden eval dataset: analyst queries mapped to expected deterministic variances, grounding 10-K sections, reference explanations, and Model Armor red-team probes."
  deletion_protection = var.deletion_protection
  labels              = var.labels

  schema = file("${path.module}/schemas/golden_eval_set.json")

  clustering = ["category"]
}

# 2. One row per eval run: gate metrics, thresholds outcome, and per-item results.
resource "google_bigquery_table" "eval_results" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.sec_edgar.dataset_id
  table_id            = "eval_results"
  description         = "Golden eval regression runs (numeric agreement, citation coverage, grounding recall, Gemini 3.5 Flash judge scores, safety compliance, p95 TTFT) keyed by commit SHA."
  deletion_protection = var.deletion_protection
  labels              = var.labels

  schema = file("${path.module}/schemas/eval_results.json")

  time_partitioning {
    type  = "DAY"
    field = "recorded_at"
  }

  clustering = ["eval_mode", "environment", "passed"]
}

# 3. CI runners stream eval_results rows and read the golden set. eval_results reuses the
#    append-only custom role (tables.get + tables.updateData) defined in telemetry.tf.
locals {
  eval_ci_principals = merge(
    { github_actions = "serviceAccount:${google_service_account.cicd_runner.email}" },
    local.cloudbuild_enabled ? {
      cloud_build = "serviceAccount:${google_service_account.cloudbuild_runner[0].email}"
    } : {}
  )
}

resource "google_bigquery_table_iam_member" "eval_results_ci_append" {
  for_each = local.eval_ci_principals

  project    = var.project_id
  dataset_id = google_bigquery_dataset.sec_edgar.dataset_id
  table_id   = google_bigquery_table.eval_results.table_id
  role       = google_project_iam_custom_role.bigquery_telemetry_sink_writer.name
  member     = each.value
}

resource "google_bigquery_table_iam_member" "golden_eval_set_ci_reader" {
  for_each = local.eval_ci_principals

  project    = var.project_id
  dataset_id = google_bigquery_dataset.sec_edgar.dataset_id
  table_id   = google_bigquery_table.golden_eval_set.table_id
  role       = "roles/bigquery.dataViewer"
  member     = each.value
}
