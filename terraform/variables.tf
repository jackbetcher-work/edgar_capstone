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
  description = "Gemini Flash model used by NarrativeEvaluatorAgent for LLM-as-a-Judge evaluation (runs with ThinkingLevel.LOW)."
  type        = string
  default     = "gemini-3.8-flash"
}

variable "async_judge_sample_rate" {
  description = "Probability in [0.0, 1.0] that a verified interactive turn triggers the background async LLM-as-a-Judge (EDGAR_ASYNC_JUDGE_SAMPLE_RATE). 1.0 evaluates every verified turn; 0.0 disables live background judging."
  type        = number
  default     = 1.0

  validation {
    condition     = var.async_judge_sample_rate >= 0.0 && var.async_judge_sample_rate <= 1.0
    error_message = "async_judge_sample_rate must be between 0.0 and 1.0 inclusive."
  }
}

variable "compaction_and_mapping_model" {
  description = "Lightweight Gemini Flash-Lite model used for SectorMapSummarizerAgent map-reduce batches and ADK LlmEventSummarizer compaction (EDGAR_LITE_MODEL)."
  type        = string
  default     = "gemini-3.5-flash-lite"
}

variable "embedding_dimensions" {
  description = "Dimensionality of dense vector embeddings stored in BigQuery filing_chunks."
  type        = number
  default     = 768
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

variable "cicd_service_account_id" {
  description = "Account ID for the keyless GitHub Actions CI/CD service account."
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

variable "iap_default_app_role" {
  description = "App role (IAP_ASSERTION_DEFAULT_ROLE) given to users who sign in through Direct Cloud Run IAP. Use \"analyst\", \"viewer\", or \"\" to deny."
  type        = string
  default     = "analyst"

  validation {
    condition     = contains(["", "analyst", "viewer"], var.iap_default_app_role)
    error_message = "iap_default_app_role must be \"analyst\", \"viewer\", or \"\"."
  }
}

variable "allowed_workspace_domains" {
  description = "Optional allowed Google Workspace hosted domains (hd) for JWT verification."
  type        = list(string)
  default     = []
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
