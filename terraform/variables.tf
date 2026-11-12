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

variable "deletion_protection" {
  description = "Enable Terraform deletion protection on BigQuery tables and datasets (set true in production)."
  type        = bool
  default     = false
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
