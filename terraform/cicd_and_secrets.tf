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
resource "google_cloud_run_v2_service" "sec_edgar_analyst" {
  count = var.enable_cloud_run_service ? 1 : 0

  project             = var.project_id
  name                = var.cloud_run_service_name
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"
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
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_registry_repo_id}/${var.cloud_run_service_name}:latest"

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
        name  = "GCP_PROJECT_NAME"
        value = var.project_name
      }

      env {
        name  = "GCP_REGION"
        value = var.region
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
        value = var.sec_corpus_bucket
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
        name  = "IAP_STRICT_MODE"
        value = "true"
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
            version = "latest"
          }
        }
      }
    }
  }

  depends_on = [
    google_secret_manager_secret_iam_member.runtime_secret_accessor,
    google_artifact_registry_repository.sec_edgar_repo,
  ]
}
