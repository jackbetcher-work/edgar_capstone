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
  description = "Existing GCS bucket holding the curated SEC 10-K Data Profiles and manifests."
  type        = string
  default     = "fde-sec-edgar-sandbox-dev-sec-corpus"
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
  description = "Whether to trigger a BigQuery load job seeding native financial_metrics from GCS manifests/bigquery_financials.jsonl."
  type        = bool
  default     = true
}

variable "seed_native_filing_chunks" {
  description = "Whether to trigger a BigQuery load job seeding native filing_chunks from GCS manifests/bigquery_filing_chunks.jsonl."
  type        = bool
  default     = true
}

variable "embedding_model" {
  description = "Vertex AI embedding model used for 10-K chunk embeddings."
  type        = string
  default     = "text-embedding-005"
}

variable "financial_reasoning_model" {
  description = "Gemini Pro model used by FinancialAnalystAgent and RootOrchestrator for complex multi-period financial reasoning."
  type        = string
  default     = "gemini-2.5-pro"
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
