variable "project_id" {
  description = "Google Cloud Project ID (Argolis sandbox project named fde-sec-edgar-sandbox-dev)."
  type        = string
  default     = "iron-radio-514217-q7"
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
  description = "Account ID for the SEC Analyst runtime service account referenced in project_reference.md."
  type        = string
  default     = "sec-analyst-sa"
}

variable "create_analyst_service_account" {
  description = "Whether to create and grant least-privilege BigQuery/GCS IAM roles to sec-analyst-sa."
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
  description = "Cloud Identity / Google Workspace SSO principals granted roles/iap.httpsResourceAccessor (e.g. group:sec-edgar-analysts@example.com)."
  type        = list(string)
  default     = []
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
