# ------------------------------------------------------------------------------
# 1. Google Cloud Secret Manager — Managed Secrets (Zero Plaintext Values in IaC)
# ------------------------------------------------------------------------------
locals {
  managed_secret_ids = toset([
    "sec-edgar-user-agent",
    "sec-edgar-gemini-api-key",
    "sec-edgar-iap-audience",
  ])
}

resource "google_secret_manager_secret" "managed_secrets" {
  for_each = local.managed_secret_ids

  project   = var.project_id
  secret_id = each.value
  labels    = var.labels

  replication {
    auto {}
  }

  depends_on = [google_project_service.required_apis]
}

# Optional bootstrap placeholder versions for greenfield projects so Cloud Run
# secret_key_ref (version = "latest") mounts succeed in a single-stage apply.
resource "google_secret_manager_secret_version" "bootstrap_placeholders" {
  # The IAP audience is derived from Terraform-managed infrastructure, so it gets a
  # real value (see iap_audience below) instead of a placeholder.
  for_each = (
    var.seed_placeholder_secret_versions
    ? (local.iap_jwt_audience != null ? setsubtract(local.managed_secret_ids, ["sec-edgar-iap-audience"]) : local.managed_secret_ids)
    : toset([])
  )

  secret      = google_secret_manager_secret.managed_secrets[each.key].id
  secret_data = "REPLACE_ME_IN_SECRET_MANAGER"

  lifecycle {
    ignore_changes = [secret_data]
  }
}

# Keep the IAP JWT audience secret in sync with the IAP-protected resource (the
# Cloud Run service in iap_direct mode, the backend service in load_balancer mode).
# If it changes, a new version is written automatically and Cloud Run (pinned to
# this version below) rolls a new revision to pick it up.
resource "google_secret_manager_secret_version" "iap_audience" {
  count = local.iap_jwt_audience != null ? 1 : 0

  secret      = google_secret_manager_secret.managed_secrets["sec-edgar-iap-audience"].id
  secret_data = local.iap_jwt_audience
}

# Grant least-privilege Secret Accessor role exclusively to the custom Cloud Run runtime SA
resource "google_secret_manager_secret_iam_member" "runtime_secret_accessor" {
  for_each = local.managed_secret_ids

  project   = var.project_id
  secret_id = google_secret_manager_secret.managed_secrets[each.key].secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${local.runtime_service_account_email}"
}

# ------------------------------------------------------------------------------
# 2. Artifact Registry Repository for Container Images Built on Push
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
# 3. CI/CD Runner Service Account & Keyless GitHub Workload Identity Federation
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

resource "google_project_iam_member" "cicd_cloud_run_admin" {
  project = var.project_id
  role    = "roles/run.admin"
  member  = "serviceAccount:${google_service_account.cicd_runner.email}"
}

resource "google_secret_manager_secret_iam_member" "cicd_secret_accessor" {
  for_each = local.managed_secret_ids

  project   = var.project_id
  secret_id = google_secret_manager_secret.managed_secrets[each.key].secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cicd_runner.email}"
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
# 4. Google Cloud Run v2 Service with Custom SA, Cloud Identity RBAC, & Secrets
# ------------------------------------------------------------------------------
# The `sec-edgar-analyst` service was created outside Terraform (e.g. by an earlier
# `gcloud run deploy`), so adopt it into state instead of failing with HTTP 409.
# Import blocks are idempotent: once the resource is in state this is a no-op.
import {
  for_each = var.enable_cloud_run_service ? toset(["0"]) : toset([])
  to       = google_cloud_run_v2_service.sec_edgar_analyst[0]
  id       = "projects/${var.project_id}/locations/${var.region}/services/${var.cloud_run_service_name}"
}

resource "google_cloud_run_v2_service" "sec_edgar_analyst" {
  # google-beta: `iap_enabled` is still a beta-only field in provider 6.x.
  provider = google-beta
  count    = var.enable_cloud_run_service ? 1 : 0

  project  = var.project_id
  name     = var.cloud_run_service_name
  location = var.region
  # iap_direct: the run.app URL is reachable, but IAP requires a Google sign-in from
  # an iap_authorized_principals member on every request (no anonymous access).
  # load_balancer: only the external HTTPS LB (with IAP on its backend) can reach it.
  ingress             = local.cloud_run_direct_iap ? "INGRESS_TRAFFIC_ALL" : "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"
  iap_enabled         = local.cloud_run_direct_iap
  launch_stage        = local.cloud_run_direct_iap ? "BETA" : null
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

    # Direct VPC egress so the service can reach the Vector Search PSC endpoint IP.
    # PRIVATE_RANGES_ONLY keeps Google API traffic on its normal path.
    dynamic "vpc_access" {
      for_each = local.vector_search_psc ? [1] : []
      content {
        egress = "PRIVATE_RANGES_ONLY"
        network_interfaces {
          network    = google_compute_network.sec_edgar[0].id
          subnetwork = google_compute_subnetwork.sec_edgar_private[0].id
        }
      }
    }

    containers {
      # Bootstraps with a lightweight public Cloud Run placeholder image so initial
      # `terraform apply` succeeds in a single stage before CI/CD pushes to GAR.
      # Subsequent image updates deployed by Cloud Build / GitHub Actions are preserved
      # via `lifecycle.ignore_changes` below.
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
        name  = "EDGAR_EXECUTION_MODE"
        value = "auto"
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
        name  = "SDP_INSPECT_TEMPLATE_ID"
        value = var.enable_model_armor ? google_data_loss_prevention_inspect_template.sec_edgar_sdp_inspect[0].id : ""
      }

      env {
        name  = "SDP_DEIDENTIFY_TEMPLATE_ID"
        value = var.enable_model_armor ? google_data_loss_prevention_deidentify_template.sec_edgar_sdp_deidentify[0].id : ""
      }

      env {
        name  = "EDGAR_PROMPT_STORE_BACKEND"
        value = var.prompt_store_backend
      }

      env {
        name  = "EDGAR_TELEMETRY_BACKEND"
        value = "gcp"
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
        name  = "VERTEX_VECTOR_SEARCH_ENDPOINT_ID"
        value = var.enable_vertex_vector_search ? google_vertex_ai_index_endpoint.sec_10k_chunks_endpoint[0].id : ""
      }

      env {
        name  = "VERTEX_VECTOR_SEARCH_PUBLIC_DOMAIN"
        value = var.enable_vertex_vector_search && !local.vector_search_psc ? google_vertex_ai_index_endpoint.sec_10k_chunks_endpoint[0].public_endpoint_domain_name : ""
      }

      env {
        name  = "VERTEX_VECTOR_SEARCH_PSC_ADDRESS"
        value = local.vector_search_psc_endpoint ? google_compute_address.vector_search_psc[0].address : ""
      }

      env {
        name = "VERTEX_VECTOR_SEARCH_DEPLOYED_INDEX_ID"
        # Uses the variable (not the resource) so Cloud Run updates don't block on
        # the 20-40 minute index deployment.
        value = var.enable_vertex_vector_search && var.deploy_vertex_vector_index ? var.vector_search_deployed_index_id : ""
      }

      env {
        name  = "CLOUD_IDENTITY_DOMAIN"
        value = var.cloud_identity_domain
      }

      env {
        name  = "ADMIN_CLOUD_IDENTITY_GROUPS"
        value = join(",", sort(local.rbac_admin_group_emails))
      }

      env {
        name  = "ANALYST_CLOUD_IDENTITY_GROUPS"
        value = join(",", sort(local.rbac_analyst_group_emails))
      }

      env {
        name  = "VIEWER_CLOUD_IDENTITY_GROUPS"
        value = join(",", sort(local.rbac_viewer_group_emails))
      }

      env {
        name  = "ALLOWED_WORKSPACE_DOMAINS"
        value = join(",", var.allowed_workspace_domains)
      }

      env {
        name  = "OKTA_ISSUER_URI"
        value = var.okta_issuer_uri
      }

      env {
        name  = "OKTA_CLIENT_ID"
        value = var.okta_client_id
      }

      env {
        name = "SEC_EDGAR_USER_AGENT"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.managed_secrets["sec-edgar-user-agent"].secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "GEMINI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.managed_secrets["sec-edgar-gemini-api-key"].secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "INTERNAL_IAP_AUDIENCE"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.managed_secrets["sec-edgar-iap-audience"].secret_id
            version = length(google_secret_manager_secret_version.iap_audience) > 0 ? google_secret_manager_secret_version.iap_audience[0].version : "latest"
          }
        }
      }
    }
  }

  lifecycle {
    ignore_changes = [
      template[0].containers[0].image,
      client,
      client_version,
    ]
  }

  depends_on = [
    google_secret_manager_secret_iam_member.runtime_secret_accessor,
    google_secret_manager_secret_version.bootstrap_placeholders,
    google_artifact_registry_repository.sec_edgar_repo,
  ]
}
