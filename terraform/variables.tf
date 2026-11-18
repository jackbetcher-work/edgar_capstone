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
  description = "Whether to attach gs://<sec_corpus_bucket>/vector_search/embeddings as contents_delta_uri on the Vertex AI Vector Search index. The full-index embeddings are published, so this defaults to true; set false only for a fresh project whose bucket has no embeddings yet."
  type        = bool
  default     = true
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
  description = "Whether to provision the Identity-Aware Proxy (IAP) Serverless NEG and Backend Service protecting Cloud Run. Only used when cloud_run_access_mode = \"load_balancer\"."
  type        = bool
  default     = true
}

variable "cloud_run_access_mode" {
  description = <<-EOT
    How people reach the Cloud Run app. Neither mode allows anonymous access.
      iap_direct    (default) IAP is turned on for the Cloud Run service itself. Share the
                    run.app URL; every request needs a Google sign-in from a principal in
                    iap_authorized_principals. No load balancer, no static IP, no cert.
      load_balancer Cloud Run ingress is internal-and-cloud-load-balancing behind the global
                    external HTTPS load balancer + IAP backend service (load_balancer.tf).
    IAP can't be enabled on both a load balancer backend and the Cloud Run service at once.
  EOT
  type        = string
  default     = "iap_direct"

  validation {
    condition     = contains(["iap_direct", "load_balancer"], var.cloud_run_access_mode)
    error_message = "cloud_run_access_mode must be \"iap_direct\" or \"load_balancer\"."
  }
}

variable "iap_default_app_role" {
  description = "App role (IAP_ASSERTION_DEFAULT_ROLE) given to users who sign in through IAP and match no Cloud Identity RBAC group. IAP has already checked iap_authorized_principals. Use \"analyst\", \"viewer\", or \"\" to deny. Admin always requires a group."
  type        = string
  default     = "analyst"

  validation {
    condition     = contains(["", "analyst", "viewer"], var.iap_default_app_role)
    error_message = "iap_default_app_role must be \"analyst\", \"viewer\", or \"\"."
  }
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
  description = "Principals granted roles/iap.httpsResourceAccessor, i.e. who can open the app. Use group:<group>@<domain> or domain:<workspace-domain> (everyone in that Workspace/Cloud Identity domain). The org policy iam.allowedPolicyMemberDomains decides which outside domains (e.g. google.com) are allowed."
  type        = list(string)
  default     = ["domain:jackbetcher.altostrat.com"]

  validation {
    condition     = alltrue([for p in var.iap_authorized_principals : startswith(lower(trimspace(p)), "group:") || startswith(lower(trimspace(p)), "domain:")])
    error_message = "Use group:<group-id>@<domain> or domain:<workspace-domain>; individual user: bindings are prohibited under Cloud Identity Group RBAC."
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
  description = "Standard resource labels applied across BigQuery, Cloud Run, and monitoring resources. Keys must stay unique after hyphen->underscore normalization: Cloud Logging copies Cloud Run labels onto LogEntry.labels, and the BigQuery log sink maps both `managed_by` and `managed-by` to the same `labels.managed_by` column (table_invalid_schema)."
  type        = map(string)
  default = {
    project     = "sec-edgar-analyst"
    environment = "dev"
    managed_by  = "terraform"
  }

  validation {
    condition     = length(distinct([for k in keys(var.labels) : replace(k, "-", "_")])) == length(keys(var.labels))
    error_message = "Label keys must be unique after replacing '-' with '_' (e.g. do not set both managed_by and managed-by); colliding keys break the Cloud Logging -> BigQuery sink schema."
  }

  validation {
    condition     = !contains(keys(var.labels), "managed-by") && !contains(keys(var.labels), "commit-sha")
    error_message = "Use underscore label keys only: `managed-by` / `commit-sha` are reserved for removal by the CI/CD deploy steps (they collide with managed_by / commit_sha in the BigQuery log sink)."
  }
}

# ------------------------------------------------------------------------------
# HTTPS Load Balancer
# ------------------------------------------------------------------------------
variable "enable_https_load_balancer" {
  description = "Whether to provision the global external HTTPS load balancer in front of the IAP backend service. Only used when cloud_run_access_mode = \"load_balancer\"."
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
  description = "Whether to deploy the Vector Search index to the index endpoint (incurs always-on node cost). Disabled by default because runtime hybrid retrieval uses BigQuery VECTOR_SEARCH + Vertex AI Search directly."
  type        = bool
  default     = false
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

# ------------------------------------------------------------------------------
# VPC Service Controls (vpc_service_controls.tf)
# ------------------------------------------------------------------------------
variable "enable_vpc_service_controls" {
  description = "Whether to manage the VPC Service Controls perimeter around this project. Needs an access policy: set vpc_sc_access_policy_id (delegated scoped policy) or create_vpc_sc_scoped_access_policy = true (needs org-level Access Context Manager permissions)."
  type        = bool
  default     = false
}

variable "vpc_sc_access_policy_id" {
  description = "Numeric ID (or accessPolicies/<id>) of an existing Access Context Manager policy to put the perimeter in, typically a policy scoped to this project that an org admin created and delegated to you with roles/accesscontextmanager.policyAdmin."
  type        = string
  default     = ""
}

variable "create_vpc_sc_scoped_access_policy" {
  description = "Create a new Access Context Manager policy scoped to this project. Requires accesscontextmanager.policies.create on the organization."
  type        = bool
  default     = false
}

variable "vpc_sc_enforce" {
  description = "false (default): the perimeter is dry-run only (use_explicit_dry_run_spec); violations are logged, nothing is blocked. true: enforce the same configuration."
  type        = bool
  default     = false
}

variable "vpc_sc_restricted_services" {
  description = "Google APIs protected by the perimeter."
  type        = list(string)
  default = [
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "bigquery.googleapis.com",
    "discoveryengine.googleapis.com",
    "dlp.googleapis.com",
    "firestore.googleapis.com",
    "modelarmor.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
  ]
}

variable "vpc_sc_admin_identities" {
  description = "Human operators allowed to call restricted APIs from outside the perimeter (Terraform, gcloud, the console, local scripts). Use user:<email> or serviceAccount:<email>."
  type        = list(string)
  default     = ["user:admin@jackbetcher.altostrat.com"]
}

variable "vpc_sc_app_ingress_access_levels" {
  description = "Access levels whose callers may reach run.googleapis.com (i.e. invoke the app through IAP). VPC-SC can't match IAM identities on Cloud Run invocations, so the default [\"*\"] admits any source and leaves who-gets-in to IAP. Swap in an IP-based access level to also restrict by network."
  type        = list(string)
  default     = ["*"]
}

variable "vector_search_endpoint_access" {
  description = "How the Vector Search index endpoint is reached: private_service_connect (no public domain; Cloud Run queries a PSC forwarding-rule IP over Direct VPC egress) or public (REST findNeighbors on the public endpoint domain)."
  type        = string
  default     = "private_service_connect"

  validation {
    condition     = contains(["private_service_connect", "public"], var.vector_search_endpoint_access)
    error_message = "vector_search_endpoint_access must be \"private_service_connect\" or \"public\"."
  }
}

variable "private_subnet_cidr" {
  description = "CIDR of the private subnet used for Cloud Run Direct VPC egress and the Vector Search PSC endpoint IP."
  type        = string
  default     = "10.20.0.0/24"
}

# ------------------------------------------------------------------------------
# Observability Alerting (observability.tf)
# ------------------------------------------------------------------------------
variable "monitoring_notification_channels" {
  description = "Optional Cloud Monitoring notification channel resource names (projects/<id>/notificationChannels/<id>) attached to the latency, 5xx error, and security audit block alert policies."
  type        = list(string)
  default     = []
}

variable "api_latency_p95_alert_threshold_ms" {
  description = "p95 server-side latency threshold (ms) for the /api/v1/chat alert policy on logging.googleapis.com/user/sec_edgar/api_request_latencies."
  type        = number
  default     = 30000
}

variable "api_5xx_alert_threshold" {
  description = "Number of HTTP 5xx responses from sec-edgar-analyst within a 5-minute window that triggers the server-error alert policy (log-based metric sec_edgar/api_request_latencies, http_status >= 500)."
  type        = number
  default     = 5
}

variable "security_block_alert_threshold" {
  description = "Number of denied (401/403) or guardrail-blocked security audit events within a 5-minute window that triggers the elevated auth/guardrail block alert policy (log-based metric sec_edgar/security_audit_events)."
  type        = number
  default     = 20
}

# ------------------------------------------------------------------------------
# Cloud Logging -> BigQuery operational log export
# ------------------------------------------------------------------------------
variable "ops_logs_dataset_id" {
  description = "Dedicated BigQuery dataset receiving the Cloud Logging project sink (sec-edgar-telemetry-audit-bq-sink). Kept separate from var.dataset_id so the sink writer identity never has write access to curated financial tables."
  type        = string
  default     = "sec_edgar_ops_logs"
}

variable "ops_logs_retention_days" {
  description = "Default partition expiration (days) for the auto-created, partitioned Cloud Logging export tables in var.ops_logs_dataset_id. 0 keeps partitions forever."
  type        = number
  default     = 90

  validation {
    condition     = var.ops_logs_retention_days >= 0
    error_message = "ops_logs_retention_days must be >= 0."
  }
}
