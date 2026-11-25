# ------------------------------------------------------------------------------
# 1. Artifact Registry Repository for Container Images Built on Push
# ------------------------------------------------------------------------------
resource "google_artifact_registry_repository" "sec_edgar_repo" {
  project       = var.project_id
  location      = var.region
  repository_id = var.artifact_registry_repo_id
  description   = "Docker container repository for the SEC EDGAR Natural Language Analyst service."
  format        = "DOCKER"
  labels        = var.labels

  depends_on = [google_project_service.required_apis]
}

# ------------------------------------------------------------------------------
# 2. CI/CD Runner Service Account & Keyless GitHub Workload Identity Federation
# ------------------------------------------------------------------------------
resource "google_service_account" "cicd_runner" {
  project      = var.project_id
  account_id   = var.cicd_service_account_id
  display_name = "SEC EDGAR GitHub CI/CD Pipeline Runner"
  description  = "Keyless Workload Identity Federation service account for test, build, and deploy stages."

  depends_on = [google_project_service.required_apis]
}

resource "google_project_iam_member" "cicd_artifact_registry_writer" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.cicd_runner.email}"
}

resource "google_project_iam_member" "cicd_cloud_run_developer" {
  project = var.project_id
  role    = "roles/run.developer"
  member  = "serviceAccount:${google_service_account.cicd_runner.email}"
}

resource "google_service_account_iam_member" "cicd_act_as_runtime_sa" {
  service_account_id = (
    var.create_analyst_service_account
    ? google_service_account.sec_analyst[0].name
    : "projects/${var.project_id}/serviceAccounts/${local.runtime_service_account_email}"
  )
  role   = "roles/iam.serviceAccountUser"
  member = "serviceAccount:${google_service_account.cicd_runner.email}"
}

resource "google_iam_workload_identity_pool" "github_pool" {
  project                   = var.project_id
  workload_identity_pool_id = "sec-edgar-github-pool"
  display_name              = "SEC EDGAR GitHub WIF Pool"
  description               = "Keyless OIDC authentication pool for GitHub push deployment pipelines."

  depends_on = [google_project_service.required_apis]
}

resource "google_iam_workload_identity_pool_provider" "github_provider" {
  project                            = var.project_id
  workload_identity_pool_id          = google_iam_workload_identity_pool.github_pool.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-oidc-provider"
  display_name                       = "GitHub Actions OIDC Provider"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.actor"      = "assertion.actor"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
  }

  attribute_condition = "assertion.repository == '${var.github_repository}'"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account_iam_member" "github_wif_impersonation" {
  service_account_id = google_service_account.cicd_runner.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github_pool.name}/attribute.repository/${var.github_repository}"
}

# ------------------------------------------------------------------------------
# 3. Google Cloud Run v2 Service with Direct IAP & Single Runtime SA
# ------------------------------------------------------------------------------
# Terraform is the single source of truth for Cloud Run configuration & env vars;
# GitHub Actions pushes new images via `gcloud run services update --image=...`.
resource "google_cloud_run_v2_service" "sec_edgar_analyst" {
  # google-beta: `iap_enabled` is still a beta-only field in provider 6.x.
  provider = google-beta
  count    = var.enable_cloud_run_service ? 1 : 0

  project             = var.project_id
  name                = var.cloud_run_service_name
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  iap_enabled         = true
  launch_stage        = "BETA"
  deletion_protection = var.deletion_protection
  labels              = var.labels

  template {
    service_account                  = local.runtime_service_account_email
    max_instance_request_concurrency = 8
    timeout                          = "300s"

    scaling {
      min_instance_count = 1
      max_instance_count = 10
    }

    containers {
      image = var.cloud_run_initial_image

      resources {
        limits = {
          cpu    = "2"
          memory = "4Gi"
        }
      }

      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }

      env {
        name  = "GCP_PROJECT_NAME"
        value = var.project_name
      }

      env {
        name  = "GCP_REGION"
        value = var.region
      }

      env {
        name  = "GOOGLE_CLOUD_LOCATION"
        value = var.region
      }

      env {
        name  = "GOOGLE_GENAI_USE_VERTEXAI"
        value = "true"
      }

      env {
        name  = "RUNTIME_SERVICE_ACCOUNT_ID"
        value = var.analyst_service_account_id
      }

      env {
        name  = "RUNTIME_SERVICE_ACCOUNT"
        value = local.runtime_service_account_email
      }

      env {
        name  = "BIGQUERY_DATASET_ID"
        value = var.dataset_id
      }

      env {
        name  = "SEC_CORPUS_BUCKET"
        value = local.sec_corpus_bucket_name
      }

      env {
        name  = "EDGAR_ANALYST_MODEL"
        value = var.financial_reasoning_model
      }

      env {
        name  = "EDGAR_FLASH_MODEL"
        value = var.coding_and_evaluation_model
      }

      env {
        name  = "EDGAR_ASYNC_JUDGE_SAMPLE_RATE"
        value = tostring(var.async_judge_sample_rate)
      }

      env {
        name  = "EDGAR_LITE_MODEL"
        value = var.compaction_and_mapping_model
      }

      env {
        name  = "EDGAR_GEMINI_LOCATION"
        value = var.gemini_location
      }

      env {
        name  = "EDGAR_MODEL_ARMOR_MODE"
        value = "gcp"
      }

      env {
        name  = "MODEL_ARMOR_TEMPLATE_ID"
        value = var.model_armor_template_id
      }

      env {
        name  = "EDGAR_TELEMETRY_BACKEND"
        value = "gcp"
      }

      env {
        name  = "OTEL_SERVICE_NAME"
        value = var.cloud_run_service_name
      }

      env {
        name  = "OTEL_SEMCONV_STABILITY_OPT_IN"
        value = "gen_ai_latest_experimental"
      }

      env {
        name  = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
        value = "SPAN_AND_EVENT"
      }

      env {
        name  = "OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT"
        value = "true"
      }

      env {
        name  = "ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"
        value = "true"
      }

      env {
        name  = "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY"
        value = "true"
      }

      env {
        name  = "EDGAR_TRACE_CAPTURE_CONTENT"
        value = "true"
      }

      env {
        name  = "EDGAR_FEEDBACK_BACKEND"
        value = "gcp"
      }

      env {
        name  = "BIGQUERY_LLM_TELEMETRY_TABLE"
        value = google_bigquery_table.llm_telemetry_events.table_id
      }

      env {
        name  = "BIGQUERY_SESSION_TELEMETRY_TABLE"
        value = google_bigquery_table.session_telemetry_events.table_id
      }

      env {
        name  = "BIGQUERY_AUDIT_LOGS_TABLE"
        value = google_bigquery_table.security_audit_logs.table_id
      }

      env {
        name  = "FIRESTORE_DATABASE_ID"
        value = var.firestore_database_id
      }

      env {
        name  = "FIRESTORE_FEEDBACK_COLLECTION"
        value = var.firestore_feedback_collection
      }

      env {
        name  = "IAP_STRICT_MODE"
        value = "true"
      }

      env {
        name  = "IAP_ASSERTION_DEFAULT_ROLE"
        value = var.iap_default_app_role
      }

      env {
        name  = "INTERNAL_IAP_AUDIENCE"
        value = local.iap_jwt_audience != null ? local.iap_jwt_audience : ""
      }

      env {
        name  = "ALLOWED_WORKSPACE_DOMAINS"
        value = join(",", var.allowed_workspace_domains)
      }
    }
  }

  lifecycle {
    ignore_changes = [
      template[0].containers[0].image,
      template[0].containers[0].startup_probe,
      client,
      client_version,
    ]
  }

  depends_on = [
    google_artifact_registry_repository.sec_edgar_repo,
  ]
}
