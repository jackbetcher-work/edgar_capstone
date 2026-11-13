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

# Grant least-privilege Secret Accessor role to the Cloud Run runtime service account
resource "google_secret_manager_secret_iam_member" "runtime_secret_accessor" {
  for_each = var.create_analyst_service_account ? local.managed_secret_ids : toset([])

  project   = var.project_id
  secret_id = google_secret_manager_secret.managed_secrets[each.key].secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.sec_analyst[0].email}"
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
  count = var.create_analyst_service_account ? 1 : 0

  service_account_id = google_service_account.sec_analyst[0].name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.cicd_runner.email}"
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
# 4. Google Cloud Run v2 Service with Native Secret Manager Volume/Env Mounts
# ------------------------------------------------------------------------------
resource "google_cloud_run_v2_service" "sec_edgar_analyst" {
  count = var.enable_cloud_run_service && var.create_analyst_service_account ? 1 : 0

  project             = var.project_id
  name                = var.cloud_run_service_name
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"
  deletion_protection = var.deletion_protection
  labels              = var.labels

  template {
    service_account                  = google_service_account.sec_analyst[0].email
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
        name  = "GCP_REGION"
        value = var.region
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
        name  = "IAP_STRICT_MODE"
        value = "true"
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
