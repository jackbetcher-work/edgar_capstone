variable "project_id" {
  description = "Google Cloud Project ID (Argolis sandbox project named fde-sec-edgar-sandbox-dev)."
  type        = string
  default     = "iron-radio-514217-q7"
}

variable "project_name" {
  description = "Logical Google Cloud Project Name (used for custom service account alias metadata: sec-analyst-sa@<project_name>.iam.gserviceaccount.com)."
  type        = string
  default     = "fde-sec-edgar-sandbox-dev"
}

variable "region" {
  description = "GCP region for BigQuery dataset, Cloud Resource Connection, and GCS alignment."
  type        = string
  default     = "us-central1"
}

variable "dataset_id" {
  description = "BigQuery dataset ID for SEC 10-K structured financial metrics and metadata-filtered RAG."
  type        = string
  default     = "sec_edgar"
}

variable "sec_corpus_bucket" {
  description = "GCS bucket name holding the curated SEC 10-K Data Profiles, manifests, and vector embeddings."
  type        = string
  default     = "fde-sec-edgar-sandbox-dev-sec-corpus"

  validation {
    condition     = length(trimspace(var.sec_corpus_bucket)) > 0 && !startswith(lower(trimspace(var.sec_corpus_bucket)), "gs://") && length(regexall("/", trimspace(var.sec_corpus_bucket))) == 0
    error_message = "sec_corpus_bucket must be a bare GCS bucket name without a 'gs://' prefix or slashes."
  }
}

variable "create_sec_corpus_bucket" {
  description = "Whether Terraform should create and manage the GCS bucket holding the curated SEC 10-K corpus and manifests."
  type        = bool
  default     = true
}

variable "sec_corpus_bucket_versioning" {
  description = "Whether to enable GCS object versioning on the managed SEC 10-K corpus bucket."
  type        = bool
  default     = true
}

variable "vertex_connection_id" {
  description = "BigQuery Cloud Resource Connection ID for Vertex AI embeddings & GCS Object Tables."
  type        = string
  default     = "vertex-ai-rag-conn"
}

variable "analyst_service_account_id" {
  description = "Account ID for the custom SEC Analyst runtime service account (sec-analyst-sa@<project>.iam.gserviceaccount.com)."
  type        = string
  default     = "sec-analyst-sa"

  validation {
    condition     = var.analyst_service_account_id == "sec-analyst-sa"
    error_message = "Application must run under custom service account ID 'sec-analyst-sa' (sec-analyst-sa@<project>.iam.gserviceaccount.com)."
  }
}

variable "create_analyst_service_account" {
  description = "Whether to create and grant least-privilege BigQuery/GCS IAM roles to sec-analyst-sa."
  type        = bool
  default     = true
}

variable "create_agent_service_accounts" {
  description = "Whether to provision granular per-agent service accounts enforcing Least Privilege for S2S communication."
  type        = bool
  default     = true
}

variable "seed_native_financial_metrics" {
  description = "Whether to trigger a BigQuery load job seeding native financial_metrics from GCS manifests/bigquery_financials.jsonl (enable once manifests are uploaded)."
  type        = bool
  default     = false
}

variable "seed_native_filing_chunks" {
  description = "Whether to trigger a BigQuery load job seeding native filing_chunks from GCS manifests/bigquery_filing_chunks.jsonl (enable once vector manifests are uploaded, or populate via scripts/build_vector_db.py)."
  type        = bool
  default     = false
}

variable "seed_vertex_vector_index" {
  description = "Whether to attach gs://<sec_corpus_bucket>/vector_search/embeddings as contents_delta_uri on the Vertex AI Vector Search index (enable once embeddings exist in GCS)."
  type        = bool
  default     = false
}

variable "embedding_model" {
  description = "Vertex AI embedding model used for 10-K chunk embeddings."
  type        = string
  default     = "text-embedding-005"
}

variable "financial_reasoning_model" {
  description = "Gemini model used by FinancialAnalystAgent and RootOrchestrator for complex multi-period financial reasoning (the reference spec names gemini-2.5-pro, which this project cannot access)."
  type        = string
  default     = "gemini-3.8-flash"
}

variable "gemini_location" {
  description = "Vertex AI location used for Gemini model calls and context caches (EDGAR_GEMINI_LOCATION). The configured models are only served from the global endpoint in this project; BigQuery, GCS, Model Armor, and embeddings stay in var.region."
  type        = string
  default     = "global"
}

variable "coding_and_evaluation_model" {
  description = "Gemini Flash model used for LLM-as-a-Judge evaluation, coding assistance, and context compaction."
  type        = string
  default     = "gemini-3.5-flash"
}

variable "embedding_dimensions" {
  description = "Dimensionality of dense vector embeddings stored in filing_chunks and Vertex AI Vector Search."
  type        = number
  default     = 768
}

variable "enable_vertex_vector_search" {
  description = "Whether to provision the Vertex AI Vector Search index and index endpoint for metadata-filtered ANN retrieval."
  type        = bool
  default     = true
}

variable "enable_vertex_ai_search" {
  description = "Whether to provision the Vertex AI Search (Discovery Engine) unstructured datastore and search engine."
  type        = bool
  default     = true
}

variable "enable_model_armor" {
  description = "Whether to provision Google Cloud Model Armor and Cloud SDP/DLP inspect & de-identify templates."
  type        = bool
  default     = true
}

variable "model_armor_template_id" {
  description = "Regional Google Cloud Model Armor template ID enforcing prompt-injection, RAI, malicious URI, and SDP guardrails."
  type        = string
  default     = "sec-edgar-analyst-armor"
}

variable "prompt_store_backend" {
  description = "Versioned prompt loader backend mode ('auto', 'local', 'gcs', or 'vertex')."
  type        = string
  default     = "auto"
}

variable "enable_bigquery_logging_sink" {
  description = "Whether to provision the Cloud Logging project sink routing structured telemetry and security audit logs to BigQuery."
  type        = bool
  default     = true
}

variable "enable_firestore_feedback" {
  description = "Whether to provision the Firestore Native database for persistent analyst thumbs-up/down feedback and comments."
  type        = bool
  default     = true
}

variable "firestore_database_id" {
  description = "Firestore Native database ID used to persist analyst feedback ratings and audit comments."
  type        = string
  default     = "(default)"
}

variable "firestore_feedback_collection" {
  description = "Firestore collection name storing persisted analyst feedback documents."
  type        = string
  default     = "analyst_feedback"
}

variable "artifact_registry_repo_id" {
  description = "Artifact Registry repository ID for SEC EDGAR Analyst container images."
  type        = string
  default     = "sec-edgar-repo"
}

variable "cloud_run_service_name" {
  description = "Google Cloud Run service name for the SEC EDGAR Analyst FastAPI & ADK backend."
  type        = string
  default     = "sec-edgar-analyst"
}

variable "enable_cloud_run_service" {
  description = "Whether to manage the Cloud Run v2 service resource in Terraform."
  type        = bool
  default     = true
}

variable "cloud_run_initial_image" {
  description = "Bootstrap container image used on initial Cloud Run v2 service creation so Terraform succeeds in a single stage before CI/CD pushes the first image to Artifact Registry."
  type        = string
  default     = "us-docker.pkg.dev/cloudrun/container/hello"
}

variable "seed_placeholder_secret_versions" {
  description = "Whether to create initial placeholder versions for managed Secret Manager secrets on brand-new projects so Cloud Run secret mounts succeed in a single pass (leave false when secret versions already exist)."
  type        = bool
  default     = false
}

variable "cicd_service_account_id" {
  description = "Account ID for the keyless GitHub Actions / Cloud Build CI/CD service account."
  type        = string
  default     = "sec-edgar-cicd-sa"
}

variable "github_repository" {
  description = "GitHub repository (owner/name) trusted by Workload Identity Federation for push deployments."
  type        = string
  default     = "jackbetcher-work/edgar_capstone"
}

variable "deletion_protection" {
  description = "Enable Terraform deletion protection on BigQuery tables and datasets (set true in production)."
  type        = bool
  default     = false
}

variable "enable_iap_backend_service" {
  description = "Whether to provision the Identity-Aware Proxy (IAP) Serverless NEG and Backend Service protecting Cloud Run."
  type        = bool
  default     = true
}

variable "allowed_workspace_domains" {
  description = "Allowed Google Workspace hosted domains (hd) federated via Okta SSO for JWT bearer verification."
  type        = list(string)
  default     = []
}

variable "okta_issuer_uri" {
  description = "Optional Okta OIDC issuer URI integrated with Google Workspace SSO (e.g. https://org.okta.com/oauth2/default)."
  type        = string
  default     = ""
}

variable "okta_client_id" {
  description = "Optional Okta OIDC client ID accepted as a programmatic JWT bearer audience."
  type        = string
  default     = ""
}

variable "iap_authorized_principals" {
  description = "Cloud Identity / Google Workspace SSO group principals granted roles/iap.httpsResourceAccessor (e.g. group:sec-edgar-analysts@example.com)."
  type        = list(string)
  default     = []

  validation {
    condition     = alltrue([for p in var.iap_authorized_principals : !startswith(lower(trimspace(p)), "user:")])
    error_message = "Individual user: bindings are prohibited under Cloud Identity Group RBAC; use group:<group-id>@<domain>."
  }
}

variable "cloud_identity_domain" {
  description = "Primary Google Workspace / Cloud Identity domain managing RBAC security groups."
  type        = string
  default     = "fde-workspace.corp"
}

variable "cloud_identity_customer_id" {
  description = "Optional Google Cloud Identity / Workspace customer ID (e.g. C01234567) when provisioning google_cloud_identity_group resources."
  type        = string
  default     = ""
}

variable "manage_cloud_identity_groups" {
  description = "Whether Terraform should create the Cloud Identity RBAC groups (requires cloud_identity_customer_id and Workspace group admin permissions)."
  type        = bool
  default     = false
}

variable "enable_cloud_identity_group_iam" {
  description = "Whether to bind Cloud Identity RBAC group principals (admins, analysts, viewers) onto IAP, BigQuery, Discovery Engine, and Cloud Run."
  type        = bool
  default     = false
}

variable "cloud_identity_rbac_groups" {
  description = "Group-based RBAC access definitions managed via Google Cloud Identity Groups."
  type = object({
    admins   = list(string)
    analysts = list(string)
    viewers  = list(string)
  })
  default = {
    admins   = ["sec-edgar-admins@fde-workspace.corp"]
    analysts = ["sec-edgar-analysts@fde-workspace.corp"]
    viewers  = ["sec-edgar-viewers@fde-workspace.corp"]
  }

  validation {
    condition = alltrue([
      for g in concat(
        var.cloud_identity_rbac_groups.admins,
        var.cloud_identity_rbac_groups.analysts,
        var.cloud_identity_rbac_groups.viewers
      ) : !startswith(lower(trimspace(g)), "user:") && length(regexall("@", g)) == 1
    ])
    error_message = "Every RBAC definition must be a Cloud Identity Group email (<group>@<domain> or group:<group>@<domain>); individual user: bindings are prohibited."
  }
}

variable "labels" {
  description = "Standard resource labels applied across BigQuery resources."
  type        = map(string)
  default = {
    project     = "sec-edgar-analyst"
    environment = "dev"
    managed_by  = "terraform"
  }
}

# ------------------------------------------------------------------------------
# HTTPS Load Balancer
# ------------------------------------------------------------------------------
variable "enable_https_load_balancer" {
  description = "Whether to provision the global external HTTPS load balancer in front of the IAP backend service."
  type        = bool
  default     = true
}

variable "lb_domain" {
  description = "FQDN for the Google-managed TLS certificate. Leave empty to use <lb-ip>.nip.io (no DNS setup needed). If set, create an A record pointing to the lb_ip_address output."
  type        = string
  default     = ""
}

# ------------------------------------------------------------------------------
# Vertex AI Vector Search Index Deployment
# ------------------------------------------------------------------------------
variable "deploy_vertex_vector_index" {
  description = "Whether to deploy the Vector Search index to the index endpoint (incurs always-on node cost)."
  type        = bool
  default     = true
}

variable "vector_search_deployed_index_id" {
  description = "Deployed index ID on the endpoint (letters, numbers, underscores; must start with a letter)."
  type        = string
  default     = "deployed_sec_10k_chunks_v1"

  validation {
    condition     = can(regex("^[A-Za-z][A-Za-z0-9_]*$", var.vector_search_deployed_index_id))
    error_message = "deployed_index_id must start with a letter and contain only letters, numbers, and underscores."
  }
}

variable "vector_search_machine_type" {
  description = "Machine type serving the deployed index (e2-standard-2 is the cheapest option for SHARD_SIZE_SMALL)."
  type        = string
  default     = "e2-standard-2"
}

variable "vector_search_min_replicas" {
  description = "Minimum replicas for the deployed index."
  type        = number
  default     = 1
}

variable "vector_search_max_replicas" {
  description = "Maximum replicas for the deployed index."
  type        = number
  default     = 1
}

# ------------------------------------------------------------------------------
# Cloud Build CI/CD triggers (cloudbuild.tf)
# ------------------------------------------------------------------------------
variable "enable_cloudbuild_trigger" {
  description = "Whether to manage the Cloud Build GitHub connection, repository link, service account, and main/PR triggers. Requires the one-time GitHub App install and OAuth token secret described in cloudbuild.tf."
  type        = bool
  default     = false
}

variable "enable_cloudbuild_pr_trigger" {
  description = "Whether to also create the pull-request trigger that runs cloudbuild.yaml in verify mode (lint, test, build, smoke test; no push or deploy)."
  type        = bool
  default     = true
}

variable "create_cloudbuild_github_connection" {
  description = "Create the Cloud Build 2nd-gen GitHub connection in Terraform. Set false to reuse an existing connection named cloudbuild_github_connection_name (e.g. one created in the console)."
  type        = bool
  default     = true
}

variable "cloudbuild_github_connection_name" {
  description = "Name of the Cloud Build 2nd-gen GitHub connection in var.region."
  type        = string
  default     = "sec-edgar-github"
}

variable "cloudbuild_github_app_installation_id" {
  description = "Installation ID of the Google Cloud Build GitHub App on the repository owner (github.com/settings/installations/<ID>). Required when create_cloudbuild_github_connection is true."
  type        = number
  default     = 0
}

variable "cloudbuild_github_oauth_token_secret_id" {
  description = "Secret Manager secret ID holding the GitHub OAuth token / PAT used by the Cloud Build connection. Created out of band so the token never enters Terraform state."
  type        = string
  default     = "sec-edgar-github-oauth-token"
}

variable "cloudbuild_github_oauth_token_secret_version" {
  description = "Version of the GitHub OAuth token secret referenced by the Cloud Build connection."
  type        = string
  default     = "1"
}

variable "cloudbuild_trigger_branch_regex" {
  description = "RE2 branch regex: pushes to matching branches deploy, and pull requests targeting them are verified."
  type        = string
  default     = "^main$"
}

variable "cloudbuild_service_account_id" {
  description = "Account ID for the dedicated Cloud Build trigger service account."
  type        = string
  default     = "sec-edgar-cloudbuild-sa"
}
