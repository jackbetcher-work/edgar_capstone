# ------------------------------------------------------------------------------
# BigQuery Dataset (us-central1, aligned with GCS corpus bucket)
# ------------------------------------------------------------------------------
resource "google_bigquery_dataset" "sec_edgar" {
  project                    = var.project_id
  dataset_id                 = var.dataset_id
  friendly_name              = "SEC EDGAR 10-K Hybrid RAG & Financial Metrics"
  description                = "Curated SEC 10-K structured financial metrics, filing metadata, and vector chunk embeddings for metadata-filtered RAG."
  location                   = var.region
  delete_contents_on_destroy = !var.deletion_protection
  labels                     = var.labels

  depends_on = [google_project_service.required_apis]
}

# ------------------------------------------------------------------------------
# BigQuery Cloud Resource Connection (Vertex AI Embeddings & GCS Object Tables)
# ------------------------------------------------------------------------------
resource "google_bigquery_connection" "vertex_ai" {
  project       = var.project_id
  connection_id = var.vertex_connection_id
  location      = var.region
  friendly_name = "SEC EDGAR Vertex AI & GCS RAG Connection"
  description   = "Enables BigQuery ML.GENERATE_EMBEDDING via Vertex AI and Object Tables over the SEC 10-K GCS bucket."

  cloud_resource {}

  depends_on = [google_project_service.required_apis]
}

# Allow Google-managed BigQuery connection service account (bqcx-*) to propagate
# to global IAM / Cloud Storage before binding bucket-level IAM policies.
resource "time_sleep" "wait_for_bq_connection_sa" {
  depends_on      = [google_bigquery_connection.vertex_ai]
  create_duration = "15s"
}

resource "google_project_iam_member" "bq_connection_vertex_user" {
  project    = var.project_id
  role       = "roles/aiplatform.user"
  member     = "serviceAccount:${google_bigquery_connection.vertex_ai.cloud_resource[0].service_account_id}"
  depends_on = [time_sleep.wait_for_bq_connection_sa]
}

resource "google_storage_bucket_iam_member" "bq_connection_corpus_viewer" {
  bucket     = var.sec_corpus_bucket
  role       = "roles/storage.objectViewer"
  member     = "serviceAccount:${google_bigquery_connection.vertex_ai.cloud_resource[0].service_account_id}"
  depends_on = [time_sleep.wait_for_bq_connection_sa]
}

# ------------------------------------------------------------------------------
# 1. Native Structured Financial Metrics Table (Partitioned & Clustered)
# ------------------------------------------------------------------------------
resource "google_bigquery_table" "financial_metrics" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.sec_edgar.dataset_id
  table_id            = "financial_metrics"
  description         = "Native partitioned and clustered SEC 10-K structured financial profiles (Revenue, Operating Income, Net Income, and metadata)."
  deletion_protection = var.deletion_protection
  labels              = var.labels

  schema = file("${path.module}/schemas/financial_metrics.json")

  range_partitioning {
    field = "fiscal_year"
    range {
      start    = 2015
      end      = 2036
      interval = 1
    }
  }

  clustering = ["gics_sector", "ticker"]
}

# ------------------------------------------------------------------------------
# 2. GCS External Table over manifests/bigquery_financials.jsonl
# ------------------------------------------------------------------------------
resource "google_bigquery_table" "financial_metrics_ext" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.sec_edgar.dataset_id
  table_id            = "financial_metrics_ext"
  description         = "Live BigQuery external table backed directly by gs://${var.sec_corpus_bucket}/manifests/bigquery_financials.jsonl."
  deletion_protection = var.deletion_protection
  labels              = var.labels

  external_data_configuration {
    autodetect            = false
    source_format         = "NEWLINE_DELIMITED_JSON"
    source_uris           = ["gs://${var.sec_corpus_bucket}/manifests/bigquery_financials.jsonl"]
    ignore_unknown_values = false
    schema                = file("${path.module}/schemas/financial_metrics.json")
  }
}

# ------------------------------------------------------------------------------
# 3. Seed Native financial_metrics from GCS Manifest (auto-refreshes on update)
# ------------------------------------------------------------------------------
data "google_storage_bucket_object" "financials_manifest" {
  count  = var.seed_native_financial_metrics ? 1 : 0
  bucket = var.sec_corpus_bucket
  name   = "manifests/bigquery_financials.jsonl"
}

resource "google_bigquery_job" "seed_financial_metrics" {
  count    = var.seed_native_financial_metrics ? 1 : 0
  project  = var.project_id
  location = var.region
  job_id   = "seed_financial_metrics_gen_${data.google_storage_bucket_object.financials_manifest[0].generation}"
  labels   = var.labels

  load {
    source_uris = [
      "gs://${var.sec_corpus_bucket}/manifests/bigquery_financials.jsonl"
    ]

    destination_table {
      project_id = google_bigquery_table.financial_metrics.project
      dataset_id = google_bigquery_table.financial_metrics.dataset_id
      table_id   = google_bigquery_table.financial_metrics.table_id
    }

    source_format     = "NEWLINE_DELIMITED_JSON"
    write_disposition = "WRITE_TRUNCATE"
    autodetect        = false
  }
}

# ------------------------------------------------------------------------------
# 4. GCS Object Table over Unstructured 10-K Markdown Corpus
# ------------------------------------------------------------------------------
resource "google_bigquery_table" "unstructured_10k_objects" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.sec_edgar.dataset_id
  table_id            = "unstructured_10k_objects"
  description         = "BigQuery Object Table indexing unstructured 10-K Markdown documents in gs://${var.sec_corpus_bucket}/unstructured/."
  deletion_protection = var.deletion_protection
  labels              = var.labels

  external_data_configuration {
    connection_id   = google_bigquery_connection.vertex_ai.name
    autodetect      = false
    object_metadata = "SIMPLE"
    source_uris     = ["gs://${var.sec_corpus_bucket}/unstructured/*"]
  }

  depends_on = [google_storage_bucket_iam_member.bq_connection_corpus_viewer]
}

# ------------------------------------------------------------------------------
# 5. Filing Chunks & Embeddings Table for Metadata-Filtered RAG (VECTOR_SEARCH)
# ------------------------------------------------------------------------------
resource "google_bigquery_table" "filing_chunks" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.sec_edgar.dataset_id
  table_id            = "filing_chunks"
  description         = "Chunked 10-K sections (Item 1A & Item 7) with metadata columns and FLOAT64[] embeddings for metadata-filtered BigQuery VECTOR_SEARCH."
  deletion_protection = var.deletion_protection
  labels              = var.labels

  schema = file("${path.module}/schemas/filing_chunks.json")

  range_partitioning {
    field = "fiscal_year"
    range {
      start    = 2015
      end      = 2036
      interval = 1
    }
  }

  clustering = ["gics_sector", "ticker", "section_id"]
}

# ------------------------------------------------------------------------------
# 6. Deterministic Period-over-Period Financial Variance View
# ------------------------------------------------------------------------------
resource "google_bigquery_table" "v_financial_variances_yoy" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.sec_edgar.dataset_id
  table_id            = "v_financial_variances_yoy"
  description         = "Deterministic year-over-year variance view (USD delta and %) for Revenue, Operating Income, and Net Income paired with GCS/SEC URIs."
  deletion_protection = var.deletion_protection
  labels              = var.labels

  view {
    use_legacy_sql = false
    query          = <<-SQL
      WITH ordered AS (
        SELECT
          profile_id,
          ticker,
          cik,
          company_name,
          gics_sector,
          gics_sub_industry,
          fiscal_year,
          period_of_report,
          filing_date,
          accession_number,
          currency,
          revenue,
          operating_income,
          net_income,
          LAG(fiscal_year) OVER (PARTITION BY ticker ORDER BY fiscal_year) AS prev_fiscal_year,
          LAG(revenue) OVER (PARTITION BY ticker ORDER BY fiscal_year) AS prev_revenue,
          LAG(operating_income) OVER (PARTITION BY ticker ORDER BY fiscal_year) AS prev_operating_income,
          LAG(net_income) OVER (PARTITION BY ticker ORDER BY fiscal_year) AS prev_net_income,
          unstructured_gcs_uri,
          structured_gcs_uri,
          sec_filing_url
        FROM `${var.project_id}.${google_bigquery_dataset.sec_edgar.dataset_id}.${google_bigquery_table.financial_metrics_ext.table_id}`
      )
      SELECT
        profile_id,
        ticker,
        cik,
        company_name,
        gics_sector,
        gics_sub_industry,
        fiscal_year,
        prev_fiscal_year,
        period_of_report,
        filing_date,
        accession_number,
        currency,
        revenue,
        prev_revenue,
        IF(prev_fiscal_year = fiscal_year - 1, revenue - prev_revenue, NULL) AS revenue_yoy_delta,
        IF(prev_fiscal_year = fiscal_year - 1 AND prev_revenue != 0,
           SAFE_DIVIDE(revenue - prev_revenue, ABS(prev_revenue)) * 100.0, NULL) AS revenue_yoy_pct,
        operating_income,
        prev_operating_income,
        IF(prev_fiscal_year = fiscal_year - 1, operating_income - prev_operating_income, NULL) AS operating_income_yoy_delta,
        IF(prev_fiscal_year = fiscal_year - 1 AND prev_operating_income != 0,
           SAFE_DIVIDE(operating_income - prev_operating_income, ABS(prev_operating_income)) * 100.0, NULL) AS operating_income_yoy_pct,
        net_income,
        prev_net_income,
        IF(prev_fiscal_year = fiscal_year - 1, net_income - prev_net_income, NULL) AS net_income_yoy_delta,
        IF(prev_fiscal_year = fiscal_year - 1 AND prev_net_income != 0,
           SAFE_DIVIDE(net_income - prev_net_income, ABS(prev_net_income)) * 100.0, NULL) AS net_income_yoy_pct,
        unstructured_gcs_uri,
        structured_gcs_uri,
        sec_filing_url
      FROM ordered
    SQL
  }
}
