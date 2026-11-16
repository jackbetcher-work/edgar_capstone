# ------------------------------------------------------------------------------
# Cloud Build CI/CD triggers (2nd-gen GitHub repository connection)
# ------------------------------------------------------------------------------
# Merges to `main` start `cloudbuild.yaml` in `deploy` mode:
#   lint (ruff, terraform fmt/validate, ESLint) + tests -> build + smoke test
#   -> push to Artifact Registry -> deploy to Cloud Run.
# Pull requests into `main` run the same file in `verify` mode, which runs every
# gate but skips the image push and Cloud Run deploy.
#
# One-time manual prerequisites (credentials are never stored in Terraform):
#   1. Install the "Google Cloud Build" GitHub App on the repository and note the
#      installation ID (github.com/settings/installations/<ID>).
#   2. Store a GitHub OAuth token / PAT (scopes: repo, read:user, read:org) in
#      Secret Manager as `var.cloudbuild_github_oauth_token_secret_id`.
#   3. Set `enable_cloudbuild_trigger = true` and
#      `cloudbuild_github_app_installation_id = <ID>` in terraform.tfvars.
# To reuse a connection created in the console instead, set
# `create_cloudbuild_github_connection = false` and point
# `cloudbuild_github_connection_name` at it.

locals {
  cloudbuild_enabled           = var.enable_cloudbuild_trigger
  cloudbuild_create_connection = local.cloudbuild_enabled && var.create_cloudbuild_github_connection

  github_repository_owner = split("/", var.github_repository)[0]
  github_repository_name  = split("/", var.github_repository)[1]

  cloudbuild_connection_id = (
    local.cloudbuild_create_connection
    ? google_cloudbuildv2_connection.github[0].id
    : "projects/${var.project_id}/locations/${var.region}/connections/${var.cloudbuild_github_connection_name}"
  )

  cloudbuild_oauth_token_secret_version = (
    "projects/${var.project_id}/secrets/${var.cloudbuild_github_oauth_token_secret_id}/versions/${var.cloudbuild_github_oauth_token_secret_version}"
  )

  # Values passed to cloudbuild.yaml so the pipeline always matches this Terraform config.
  cloudbuild_common_substitutions = {
    _REGION            = var.region
    _PROJECT_NAME      = var.project_name
    _GAR_REPOSITORY    = google_artifact_registry_repository.sec_edgar_repo.repository_id
    _SERVICE_NAME      = var.cloud_run_service_name
    _RUNTIME_SA        = local.runtime_service_account_email
    _BIGQUERY_DATASET  = var.dataset_id
    _SEC_CORPUS_BUCKET = local.sec_corpus_bucket_name
    _ANALYST_MODEL     = var.financial_reasoning_model
    _FLASH_MODEL       = var.coding_and_evaluation_model
    _GEMINI_LOCATION   = var.gemini_location
  }

  # Doc-only changes never trigger a build.
  cloudbuild_ignored_files = ["**/*.md", "docs/**", "LICENSE"]
}

# --- Dedicated least-privilege Cloud Build service account --------------------
resource "google_service_account" "cloudbuild_runner" {
  count = local.cloudbuild_enabled ? 1 : 0

  project      = var.project_id
  account_id   = var.cloudbuild_service_account_id
  display_name = "SEC EDGAR Cloud Build Pipeline Runner"
  description  = "Runs the Terraform-managed Cloud Build triggers: lint, test, build, push, and Cloud Run deploy."

  depends_on = [google_project_service.required_apis]
}

resource "google_project_iam_member" "cloudbuild_log_writer" {
  count = local.cloudbuild_enabled ? 1 : 0

  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.cloudbuild_runner[0].email}"
}

# Deploys new revisions to the existing service; cannot change IAM policy.
resource "google_project_iam_member" "cloudbuild_run_developer" {
  count = local.cloudbuild_enabled ? 1 : 0

  project = var.project_id
  role    = "roles/run.developer"
  member  = "serviceAccount:${google_service_account.cloudbuild_runner[0].email}"
}

# Push access scoped to the single application image repository.
resource "google_artifact_registry_repository_iam_member" "cloudbuild_repo_writer" {
  count = local.cloudbuild_enabled ? 1 : 0

  project    = var.project_id
  location   = google_artifact_registry_repository.sec_edgar_repo.location
  repository = google_artifact_registry_repository.sec_edgar_repo.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.cloudbuild_runner[0].email}"
}

# Required to deploy Cloud Run revisions that run as the runtime service account.
resource "google_service_account_iam_member" "cloudbuild_act_as_runtime_sa" {
  count = local.cloudbuild_enabled ? 1 : 0

  service_account_id = (
    var.create_analyst_service_account
    ? google_service_account.sec_analyst[0].name
    : "projects/${var.project_id}/serviceAccounts/${local.runtime_service_account_email}"
  )
  role   = "roles/iam.serviceAccountUser"
  member = "serviceAccount:${google_service_account.cloudbuild_runner[0].email}"
}

# cloudbuild.yaml `availableSecrets` reads these for build/deploy-time checks.
resource "google_secret_manager_secret_iam_member" "cloudbuild_secret_accessor" {
  for_each = local.cloudbuild_enabled ? local.managed_secret_ids : toset([])

  project   = var.project_id
  secret_id = google_secret_manager_secret.managed_secrets[each.key].secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cloudbuild_runner[0].email}"
}

# --- GitHub connection + repository link --------------------------------------
# The Cloud Build service agent reads the GitHub OAuth token when it creates
# the connection.
resource "google_project_service_identity" "cloudbuild_service_agent" {
  provider = google-beta
  count    = local.cloudbuild_create_connection ? 1 : 0

  project = var.project_id
  service = "cloudbuild.googleapis.com"

  depends_on = [google_project_service.required_apis]
}

resource "google_secret_manager_secret_iam_member" "cloudbuild_agent_github_token_accessor" {
  count = local.cloudbuild_create_connection ? 1 : 0

  project   = var.project_id
  secret_id = var.cloudbuild_github_oauth_token_secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_project_service_identity.cloudbuild_service_agent[0].email}"
}

resource "google_cloudbuildv2_connection" "github" {
  count = local.cloudbuild_create_connection ? 1 : 0

  project  = var.project_id
  location = var.region
  name     = var.cloudbuild_github_connection_name

  github_config {
    app_installation_id = var.cloudbuild_github_app_installation_id

    authorizer_credential {
      oauth_token_secret_version = local.cloudbuild_oauth_token_secret_version
    }
  }

  lifecycle {
    precondition {
      condition     = var.cloudbuild_github_app_installation_id > 0
      error_message = "Set cloudbuild_github_app_installation_id to the Google Cloud Build GitHub App installation ID (or set create_cloudbuild_github_connection = false to reuse an existing connection)."
    }
  }

  depends_on = [google_secret_manager_secret_iam_member.cloudbuild_agent_github_token_accessor]
}

resource "google_cloudbuildv2_repository" "edgar_capstone" {
  count = local.cloudbuild_enabled ? 1 : 0

  project           = var.project_id
  location          = var.region
  name              = local.github_repository_name
  parent_connection = local.cloudbuild_connection_id
  remote_uri        = "https://github.com/${local.github_repository_owner}/${local.github_repository_name}.git"
}

# --- Triggers -----------------------------------------------------------------
resource "google_cloudbuild_trigger" "main_deploy" {
  count = local.cloudbuild_enabled ? 1 : 0

  project     = var.project_id
  location    = var.region
  name        = "${var.cloud_run_service_name}-main-deploy"
  description = "Merges to main: lint + test gates, container build + smoke test, push to Artifact Registry, deploy to Cloud Run."
  tags        = ["sec-edgar", "deploy"]

  service_account = google_service_account.cloudbuild_runner[0].id
  filename        = "cloudbuild.yaml"
  ignored_files   = local.cloudbuild_ignored_files

  repository_event_config {
    repository = google_cloudbuildv2_repository.edgar_capstone[0].id

    push {
      branch = var.cloudbuild_trigger_branch_regex
    }
  }

  substitutions = merge(local.cloudbuild_common_substitutions, {
    _PIPELINE_MODE = "deploy"
  })

  include_build_logs = "INCLUDE_BUILD_LOGS_WITH_STATUS"

  depends_on = [
    google_project_iam_member.cloudbuild_log_writer,
    google_project_iam_member.cloudbuild_run_developer,
    google_artifact_registry_repository_iam_member.cloudbuild_repo_writer,
    google_service_account_iam_member.cloudbuild_act_as_runtime_sa,
    google_secret_manager_secret_iam_member.cloudbuild_secret_accessor,
  ]
}

resource "google_cloudbuild_trigger" "pull_request_verify" {
  count = local.cloudbuild_enabled && var.enable_cloudbuild_pr_trigger ? 1 : 0

  project     = var.project_id
  location    = var.region
  name        = "${var.cloud_run_service_name}-pr-verify"
  description = "Pull requests into main: lint + test gates and container build + smoke test (no push, no deploy)."
  tags        = ["sec-edgar", "verify"]

  service_account = google_service_account.cloudbuild_runner[0].id
  filename        = "cloudbuild.yaml"
  ignored_files   = local.cloudbuild_ignored_files

  repository_event_config {
    repository = google_cloudbuildv2_repository.edgar_capstone[0].id

    pull_request {
      branch = var.cloudbuild_trigger_branch_regex
      # PRs from outside the repo need a `/gcbrun` comment from a collaborator
      # before they can run with the pipeline's secrets.
      comment_control = "COMMENTS_ENABLED_FOR_EXTERNAL_CONTRIBUTORS_ONLY"
    }
  }

  substitutions = merge(local.cloudbuild_common_substitutions, {
    _PIPELINE_MODE = "verify"
  })

  include_build_logs = "INCLUDE_BUILD_LOGS_WITH_STATUS"

  depends_on = [
    google_project_iam_member.cloudbuild_log_writer,
    google_secret_manager_secret_iam_member.cloudbuild_secret_accessor,
  ]
}
