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
# 5b. GCS External Table over manifests/bigquery_filing_chunks.jsonl
# ------------------------------------------------------------------------------
resource "google_bigquery_table" "filing_chunks_ext" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.sec_edgar.dataset_id
  table_id            = "filing_chunks_ext"
  description         = "Live BigQuery external table backed directly by gs://${var.sec_corpus_bucket}/manifests/bigquery_filing_chunks.jsonl."
  deletion_protection = var.deletion_protection
  labels              = var.labels

  external_data_configuration {
    autodetect            = false
    source_format         = "NEWLINE_DELIMITED_JSON"
    source_uris           = ["gs://${var.sec_corpus_bucket}/manifests/bigquery_filing_chunks.jsonl"]
    ignore_unknown_values = false
    schema                = file("${path.module}/schemas/filing_chunks.json")
  }
}

# ------------------------------------------------------------------------------
# 5c. Seed Native filing_chunks from GCS Manifest (auto-refreshes on update)
# ------------------------------------------------------------------------------
data "google_storage_bucket_object" "filing_chunks_manifest" {
  count  = var.seed_native_filing_chunks ? 1 : 0
  bucket = var.sec_corpus_bucket
  name   = "manifests/bigquery_filing_chunks.jsonl"
}

resource "google_bigquery_job" "seed_filing_chunks" {
  count    = var.seed_native_filing_chunks ? 1 : 0
  project  = var.project_id
  location = var.region
  job_id   = "seed_filing_chunks_gen_${data.google_storage_bucket_object.filing_chunks_manifest[0].generation}"
  labels   = var.labels

  load {
    source_uris = [
      "gs://${var.sec_corpus_bucket}/manifests/bigquery_filing_chunks.jsonl"
    ]

    destination_table {
      project_id = google_bigquery_table.filing_chunks.project
      dataset_id = google_bigquery_table.filing_chunks.dataset_id
      table_id   = google_bigquery_table.filing_chunks.table_id
    }

    source_format     = "NEWLINE_DELIMITED_JSON"
    write_disposition = "WRITE_TRUNCATE"
    autodetect        = false
  }
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

# ------------------------------------------------------------------------------
# 7. Per-Filing Chunk Coverage & Health View
# ------------------------------------------------------------------------------
resource "google_bigquery_table" "v_filing_chunk_coverage" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.sec_edgar.dataset_id
  table_id            = "v_filing_chunk_coverage"
  description         = "Per-filing diagnostic view tracking Item 1A and Item 7 chunk counts, embedding validity, and coverage completeness across all 10-K filings."
  deletion_protection = var.deletion_protection
  labels              = var.labels

  view {
    use_legacy_sql = false
    query          = <<-SQL
      WITH chunk_rollup AS (
        SELECT
          profile_id,
          COUNT(*) AS total_chunks,
          COUNTIF(section_id = 'item_1a') AS item_1a_chunks,
          COUNTIF(section_id = 'item_7') AS item_7_chunks,
          COUNTIF(ARRAY_LENGTH(embedding) = ${var.embedding_dimensions}) AS valid_embedding_chunks,
          COUNTIF(ARRAY_LENGTH(embedding) = 0 OR embedding IS NULL) AS missing_embedding_chunks,
          COUNTIF(ARRAY_LENGTH(embedding) > 0 AND ARRAY_LENGTH(embedding) != ${var.embedding_dimensions}) AS dimension_mismatch_chunks,
          COUNTIF(char_count < 100 OR TRIM(chunk_text) = '') AS undersized_chunks,
          COUNTIF(char_count > 8000 OR token_estimate > 2048) AS oversized_chunks,
          ROUND(AVG(char_count), 2) AS avg_char_count,
          ROUND(AVG(token_estimate), 2) AS avg_token_estimate,
          MAX(ingested_at) AS last_ingested_at
        FROM `${var.project_id}.${google_bigquery_dataset.sec_edgar.dataset_id}.${google_bigquery_table.filing_chunks.table_id}`
        GROUP BY profile_id
      )
      SELECT
        f.profile_id,
        f.ticker,
        f.cik,
        f.company_name,
        f.gics_sector,
        f.gics_sub_industry,
        f.fiscal_year,
        f.filing_date,
        f.accession_number,
        COALESCE(c.total_chunks, 0) AS total_chunks,
        COALESCE(c.item_1a_chunks, 0) AS item_1a_chunks,
        COALESCE(c.item_7_chunks, 0) AS item_7_chunks,
        COALESCE(c.valid_embedding_chunks, 0) AS valid_embedding_chunks,
        COALESCE(c.missing_embedding_chunks, 0) AS missing_embedding_chunks,
        COALESCE(c.dimension_mismatch_chunks, 0) AS dimension_mismatch_chunks,
        COALESCE(c.undersized_chunks, 0) AS undersized_chunks,
        COALESCE(c.oversized_chunks, 0) AS oversized_chunks,
        c.avg_char_count,
        c.avg_token_estimate,
        c.last_ingested_at,
        CASE
          WHEN COALESCE(c.total_chunks, 0) = 0 THEN 'UNINDEXED'
          WHEN COALESCE(c.missing_embedding_chunks, 0) > 0 OR COALESCE(c.dimension_mismatch_chunks, 0) > 0 THEN 'EMBEDDING_ANOMALY'
          WHEN COALESCE(c.item_1a_chunks, 0) = 0 OR COALESCE(c.item_7_chunks, 0) = 0 THEN 'PARTIAL_SECTIONS'
          ELSE 'HEALTHY'
        END AS filing_chunk_status
      FROM `${var.project_id}.${google_bigquery_dataset.sec_edgar.dataset_id}.${google_bigquery_table.financial_metrics_ext.table_id}` AS f
      LEFT JOIN chunk_rollup AS c
        ON f.profile_id = c.profile_id
    SQL
  }
}

# ------------------------------------------------------------------------------
# 8. RAG Vector Database Chunk Health Monitoring View
# ------------------------------------------------------------------------------
resource "google_bigquery_table" "v_chunk_health_metrics" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.sec_edgar.dataset_id
  table_id            = "v_chunk_health_metrics"
  description         = "Chunk health monitoring view reporting embedding coverage, dimension consistency, token distributions, section completeness, and composite health score across the RAG vector database."
  deletion_protection = var.deletion_protection
  labels              = var.labels

  view {
    use_legacy_sql = false
    query          = <<-SQL
      WITH expected_filings AS (
        SELECT
          COALESCE(gics_sector, 'Unknown') AS gics_sector,
          COUNT(DISTINCT profile_id) AS expected_profiles
        FROM `${var.project_id}.${google_bigquery_dataset.sec_edgar.dataset_id}.${google_bigquery_table.financial_metrics_ext.table_id}`
        GROUP BY 1
      ),
      total_expected AS (
        SELECT SUM(expected_profiles) AS total_expected_profiles
        FROM expected_filings
      ),
      chunk_base AS (
        SELECT
          chunk_id,
          profile_id,
          ticker,
          cik,
          COALESCE(gics_sector, 'Unknown') AS gics_sector,
          section_id,
          section_title,
          chunk_text,
          char_count,
          token_estimate,
          embedding,
          ARRAY_LENGTH(embedding) AS embedding_dim,
          sec_filing_url,
          unstructured_gcs_uri,
          ingested_at
        FROM `${var.project_id}.${google_bigquery_dataset.sec_edgar.dataset_id}.${google_bigquery_table.filing_chunks.table_id}`
      ),
      grouped_metrics AS (
        SELECT
          IF(GROUPING(gics_sector) = 1, 'ALL_SECTORS', gics_sector) AS gics_sector,
          IF(GROUPING(section_id) = 1, 'ALL_SECTIONS', section_id) AS section_id,
          COUNT(*) AS total_chunks,
          COUNT(DISTINCT profile_id) AS distinct_profiles,
          COUNT(DISTINCT ticker) AS distinct_tickers,
          COUNTIF(embedding_dim = ${var.embedding_dimensions}) AS valid_embedding_count,
          COUNTIF(embedding IS NULL OR embedding_dim = 0) AS missing_embedding_count,
          COUNTIF(embedding_dim > 0 AND embedding_dim != ${var.embedding_dimensions}) AS dimension_mismatch_count,
          ROUND(SAFE_DIVIDE(COUNTIF(embedding_dim = ${var.embedding_dimensions}), COUNT(*)), 4) AS embedding_coverage_ratio,
          ROUND(AVG(char_count), 2) AS avg_char_count,
          ROUND(AVG(token_estimate), 2) AS avg_token_estimate,
          MIN(token_estimate) AS min_token_estimate,
          MAX(token_estimate) AS max_token_estimate,
          APPROX_QUANTILES(token_estimate, 100)[OFFSET(50)] AS p50_token_estimate,
          APPROX_QUANTILES(token_estimate, 100)[OFFSET(95)] AS p95_token_estimate,
          COUNTIF(char_count < 100 OR TRIM(chunk_text) = '') AS undersized_chunk_count,
          COUNTIF(char_count > 8000 OR token_estimate > 2048) AS oversized_chunk_count,
          COUNTIF(
            ticker IS NULL OR cik IS NULL OR gics_sector IS NULL OR gics_sector = 'Unknown'
            OR section_title IS NULL OR sec_filing_url IS NULL OR unstructured_gcs_uri IS NULL
          ) AS missing_metadata_count,
          MAX(ingested_at) AS latest_ingested_at
        FROM chunk_base
        GROUP BY ROLLUP(gics_sector, section_id)
      )
      SELECT
        g.gics_sector,
        g.section_id,
        g.total_chunks,
        g.distinct_profiles,
        g.distinct_tickers,
        IF(g.gics_sector = 'ALL_SECTORS', te.total_expected_profiles, ef.expected_profiles) AS expected_profiles,
        GREATEST(
          0,
          COALESCE(IF(g.gics_sector = 'ALL_SECTORS', te.total_expected_profiles, ef.expected_profiles), 0) - g.distinct_profiles
        ) AS missing_profiles_count,
        ROUND(
          SAFE_DIVIDE(
            g.distinct_profiles,
            IF(g.gics_sector = 'ALL_SECTORS', te.total_expected_profiles, ef.expected_profiles)
          ),
          4
        ) AS filing_coverage_ratio,
        g.valid_embedding_count,
        g.missing_embedding_count,
        g.dimension_mismatch_count,
        g.embedding_coverage_ratio,
        g.avg_char_count,
        g.avg_token_estimate,
        g.min_token_estimate,
        g.max_token_estimate,
        g.p50_token_estimate,
        g.p95_token_estimate,
        g.undersized_chunk_count,
        g.oversized_chunk_count,
        g.missing_metadata_count,
        g.latest_ingested_at,
        ROUND(
          45.0 * COALESCE(g.embedding_coverage_ratio, 0.0)
          + 30.0 * COALESCE(
              SAFE_DIVIDE(
                g.distinct_profiles,
                IF(g.gics_sector = 'ALL_SECTORS', te.total_expected_profiles, ef.expected_profiles)
              ),
              0.0
            )
          + 15.0 * (1.0 - LEAST(1.0, SAFE_DIVIDE(g.undersized_chunk_count + g.oversized_chunk_count, g.total_chunks)))
          + 10.0 * (1.0 - LEAST(1.0, SAFE_DIVIDE(g.missing_metadata_count, g.total_chunks))),
          2
        ) AS health_score,
        CASE
          WHEN g.total_chunks = 0 THEN 'UNHEALTHY'
          WHEN g.embedding_coverage_ratio >= 0.99
               AND SAFE_DIVIDE(g.distinct_profiles, IF(g.gics_sector = 'ALL_SECTORS', te.total_expected_profiles, ef.expected_profiles)) >= 0.95
               AND (g.undersized_chunk_count + g.oversized_chunk_count + g.missing_metadata_count) = 0
            THEN 'HEALTHY'
          WHEN g.embedding_coverage_ratio >= 0.90 THEN 'DEGRADED'
          ELSE 'UNHEALTHY'
        END AS health_status
      FROM grouped_metrics AS g
      CROSS JOIN total_expected AS te
      LEFT JOIN expected_filings AS ef
        ON g.gics_sector = ef.gics_sector
    SQL
  }
}
