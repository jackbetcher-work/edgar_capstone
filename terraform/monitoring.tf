# ------------------------------------------------------------------------------
# Custom Cloud Monitoring Metric Descriptors for Vector DB Chunk Health
# ------------------------------------------------------------------------------
resource "google_monitoring_metric_descriptor" "chunk_health_score" {
  project      = var.project_id
  type         = "custom.googleapis.com/sec_edgar/chunk_health/health_score"
  metric_kind  = "GAUGE"
  value_type   = "DOUBLE"
  unit         = "1"
  display_name = "SEC EDGAR Vector DB Chunk Health Score"
  description  = "Composite 0-100 health score of the SEC 10-K RAG vector database chunks and embeddings."

  labels {
    key         = "dataset_id"
    value_type  = "STRING"
    description = "BigQuery dataset ID"
  }
  labels {
    key         = "table_id"
    value_type  = "STRING"
    description = "BigQuery chunks table ID"
  }
  labels {
    key         = "health_status"
    value_type  = "STRING"
    description = "Evaluated health status (HEALTHY, DEGRADED, UNHEALTHY)"
  }

  depends_on = [google_project_service.required_apis]
}

resource "google_monitoring_metric_descriptor" "embedding_coverage_ratio" {
  project      = var.project_id
  type         = "custom.googleapis.com/sec_edgar/chunk_health/embedding_coverage_ratio"
  metric_kind  = "GAUGE"
  value_type   = "DOUBLE"
  unit         = "1"
  display_name = "SEC EDGAR Chunk Embedding Coverage Ratio"
  description  = "Fraction (0.0-1.0) of 10-K chunks with valid, expected-dimension dense vector embeddings."

  labels {
    key         = "dataset_id"
    value_type  = "STRING"
    description = "BigQuery dataset ID"
  }
  labels {
    key         = "table_id"
    value_type  = "STRING"
    description = "BigQuery chunks table ID"
  }
  labels {
    key         = "embedding_model"
    value_type  = "STRING"
    description = "Vertex AI embedding model identifier"
  }

  depends_on = [google_project_service.required_apis]
}

resource "google_monitoring_metric_descriptor" "filing_coverage_ratio" {
  project      = var.project_id
  type         = "custom.googleapis.com/sec_edgar/chunk_health/filing_coverage_ratio"
  metric_kind  = "GAUGE"
  value_type   = "DOUBLE"
  unit         = "1"
  display_name = "SEC EDGAR 10-K Dual-Section Filing Coverage Ratio"
  description  = "Fraction (0.0-1.0) of curated 10-K filings that have both Item 1A and Item 7 chunks indexed."

  labels {
    key         = "dataset_id"
    value_type  = "STRING"
    description = "BigQuery dataset ID"
  }
  labels {
    key         = "table_id"
    value_type  = "STRING"
    description = "BigQuery chunks table ID"
  }

  depends_on = [google_project_service.required_apis]
}

resource "google_monitoring_metric_descriptor" "total_chunks" {
  project      = var.project_id
  type         = "custom.googleapis.com/sec_edgar/chunk_health/total_chunks"
  metric_kind  = "GAUGE"
  value_type   = "INT64"
  unit         = "{chunks}"
  display_name = "SEC EDGAR Total Indexed RAG Chunks"
  description  = "Total number of indexed 10-K chunks broken down by GICS sector and 10-K section."

  labels {
    key         = "dataset_id"
    value_type  = "STRING"
    description = "BigQuery dataset ID"
  }
  labels {
    key         = "table_id"
    value_type  = "STRING"
    description = "BigQuery chunks table ID"
  }
  labels {
    key         = "gics_sector"
    value_type  = "STRING"
    description = "GICS Sector classification (or 'all')"
  }
  labels {
    key         = "section_id"
    value_type  = "STRING"
    description = "10-K Section ID (item_1a, item_7, or 'all')"
  }

  depends_on = [google_project_service.required_apis]
}

resource "google_monitoring_metric_descriptor" "token_count_mean" {
  project      = var.project_id
  type         = "custom.googleapis.com/sec_edgar/chunk_health/token_count_mean"
  metric_kind  = "GAUGE"
  value_type   = "DOUBLE"
  unit         = "{tokens}"
  display_name = "SEC EDGAR Mean Chunk Token Count"
  description  = "Average estimated token count per chunk by 10-K section (item_1a vs item_7)."

  labels {
    key         = "dataset_id"
    value_type  = "STRING"
    description = "BigQuery dataset ID"
  }
  labels {
    key         = "table_id"
    value_type  = "STRING"
    description = "BigQuery chunks table ID"
  }
  labels {
    key         = "section_id"
    value_type  = "STRING"
    description = "10-K Section ID (item_1a or item_7)"
  }

  depends_on = [google_project_service.required_apis]
}

resource "google_monitoring_metric_descriptor" "anomaly_chunk_count" {
  project      = var.project_id
  type         = "custom.googleapis.com/sec_edgar/chunk_health/anomaly_chunk_count"
  metric_kind  = "GAUGE"
  value_type   = "INT64"
  unit         = "{anomalies}"
  display_name = "SEC EDGAR Vector DB Chunk Anomaly Count"
  description  = "Count of anomalous chunks by anomaly_type (missing_embedding, dimension_mismatch, undersized_chunk, oversized_chunk, missing_metadata)."

  labels {
    key         = "dataset_id"
    value_type  = "STRING"
    description = "BigQuery dataset ID"
  }
  labels {
    key         = "table_id"
    value_type  = "STRING"
    description = "BigQuery chunks table ID"
  }
  labels {
    key         = "anomaly_type"
    value_type  = "STRING"
    description = "Category of chunk anomaly"
  }

  depends_on = [google_project_service.required_apis]
}

# ------------------------------------------------------------------------------
# Google Cloud Monitoring Dashboard — RAG Vector Database Chunk Health View
# ------------------------------------------------------------------------------
resource "google_monitoring_dashboard" "chunk_health_dashboard" {
  project = var.project_id

  dashboard_json = jsonencode({
    displayName = "SEC EDGAR RAG Vector Database — Chunk Health & Ingestion Monitoring"
    mosaicLayout = {
      columns = 12
      tiles = [
        # Row 1: KPI Scorecards (4 columns each)
        {
          xPos   = 0
          yPos   = 0
          width  = 4
          height = 4
          widget = {
            title = "Overall Chunk DB Health Score (0-100)"
            scorecard = {
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"custom.googleapis.com/sec_edgar/chunk_health/health_score\" resource.type=\"global\""
                  aggregation = {
                    alignmentPeriod  = "300s"
                    perSeriesAligner = "ALIGN_MEAN"
                  }
                }
              }
              gaugeView = {
                lowerBound = 0.0
                upperBound = 100.0
              }
              thresholds = [
                {
                  label     = "Unhealthy"
                  value     = 80.0
                  color     = "RED"
                  direction = "BELOW"
                },
                {
                  label     = "Degraded"
                  value     = 95.0
                  color     = "YELLOW"
                  direction = "BELOW"
                }
              ]
            }
          }
        },
        {
          xPos   = 4
          yPos   = 0
          width  = 4
          height = 4
          widget = {
            title = "Embedding Coverage Ratio (768-d Valid)"
            scorecard = {
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"custom.googleapis.com/sec_edgar/chunk_health/embedding_coverage_ratio\" resource.type=\"global\""
                  aggregation = {
                    alignmentPeriod  = "300s"
                    perSeriesAligner = "ALIGN_MEAN"
                  }
                }
              }
              sparkChartView = {
                sparkChartType = "SPARK_LINE"
              }
              thresholds = [
                {
                  label     = "Critical Missing Embeddings"
                  value     = 0.90
                  color     = "RED"
                  direction = "BELOW"
                },
                {
                  label     = "Incomplete Embeddings"
                  value     = 0.99
                  color     = "YELLOW"
                  direction = "BELOW"
                }
              ]
            }
          }
        },
        {
          xPos   = 8
          yPos   = 0
          width  = 4
          height = 4
          widget = {
            title = "10-K Dual-Section Filing Coverage (Item 1A + Item 7)"
            scorecard = {
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"custom.googleapis.com/sec_edgar/chunk_health/filing_coverage_ratio\" resource.type=\"global\""
                  aggregation = {
                    alignmentPeriod  = "300s"
                    perSeriesAligner = "ALIGN_MEAN"
                  }
                }
              }
              sparkChartView = {
                sparkChartType = "SPARK_LINE"
              }
              thresholds = [
                {
                  label     = "Missing Filings"
                  value     = 0.95
                  color     = "YELLOW"
                  direction = "BELOW"
                }
              ]
            }
          }
        },

        # Row 2: Chunk Counts by GICS Sector & 10-K Section
        {
          xPos   = 0
          yPos   = 4
          width  = 6
          height = 4
          widget = {
            title = "Indexed RAG Chunks by GICS Sector"
            xyChart = {
              dataSets = [
                {
                  plotType       = "STACKED_BAR"
                  legendTemplate = "$${metric.labels.gics_sector}"
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "metric.type=\"custom.googleapis.com/sec_edgar/chunk_health/total_chunks\" resource.type=\"global\" metric.labels.section_id=\"all\""
                      aggregation = {
                        alignmentPeriod    = "300s"
                        perSeriesAligner   = "ALIGN_MAX"
                        crossSeriesReducer = "REDUCE_SUM"
                        groupByFields      = ["metric.label.gics_sector"]
                      }
                    }
                  }
                }
              ]
              yAxis = {
                label = "Chunks"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          xPos   = 6
          yPos   = 4
          width  = 6
          height = 4
          widget = {
            title = "Indexed RAG Chunks by 10-K Section (Item 1A vs Item 7)"
            xyChart = {
              dataSets = [
                {
                  plotType       = "LINE"
                  legendTemplate = "$${metric.labels.section_id}"
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "metric.type=\"custom.googleapis.com/sec_edgar/chunk_health/total_chunks\" resource.type=\"global\" metric.labels.gics_sector=\"all\""
                      aggregation = {
                        alignmentPeriod    = "300s"
                        perSeriesAligner   = "ALIGN_MAX"
                        crossSeriesReducer = "REDUCE_SUM"
                        groupByFields      = ["metric.label.section_id"]
                      }
                    }
                  }
                }
              ]
              yAxis = {
                label = "Chunks"
                scale = "LINEAR"
              }
            }
          }
        },

        # Row 3: Chunk Anomaly Breakdown & Mean Token Size by Section
        {
          xPos   = 0
          yPos   = 8
          width  = 6
          height = 4
          widget = {
            title = "Chunk Health Anomalies by Category (Embedding / Size / Metadata)"
            xyChart = {
              dataSets = [
                {
                  plotType       = "STACKED_BAR"
                  legendTemplate = "$${metric.labels.anomaly_type}"
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "metric.type=\"custom.googleapis.com/sec_edgar/chunk_health/anomaly_chunk_count\" resource.type=\"global\""
                      aggregation = {
                        alignmentPeriod    = "300s"
                        perSeriesAligner   = "ALIGN_MAX"
                        crossSeriesReducer = "REDUCE_SUM"
                        groupByFields      = ["metric.label.anomaly_type"]
                      }
                    }
                  }
                }
              ]
              yAxis = {
                label = "Anomalous Chunks"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          xPos   = 6
          yPos   = 8
          width  = 6
          height = 4
          widget = {
            title = "Mean Token Count per Chunk by 10-K Section"
            xyChart = {
              dataSets = [
                {
                  plotType       = "LINE"
                  legendTemplate = "$${metric.labels.section_id}"
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "metric.type=\"custom.googleapis.com/sec_edgar/chunk_health/token_count_mean\" resource.type=\"global\""
                      aggregation = {
                        alignmentPeriod    = "300s"
                        perSeriesAligner   = "ALIGN_MEAN"
                        crossSeriesReducer = "REDUCE_MEAN"
                        groupByFields      = ["metric.label.section_id"]
                      }
                    }
                  }
                }
              ]
              yAxis = {
                label = "Estimated Tokens / Chunk"
                scale = "LINEAR"
              }
            }
          }
        },

        # Row 4: BigQuery Vector Search Query Volume & Structured Health Logs
        {
          xPos   = 0
          yPos   = 12
          width  = 6
          height = 4
          widget = {
            title = "BigQuery Dataset Query Execution Count (sec_edgar)"
            xyChart = {
              dataSets = [
                {
                  plotType = "LINE"
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "metric.type=\"bigquery.googleapis.com/query/count\" resource.type=\"bigquery_project\""
                      aggregation = {
                        alignmentPeriod  = "300s"
                        perSeriesAligner = "ALIGN_RATE"
                      }
                    }
                  }
                }
              ]
              yAxis = {
                label = "Queries / sec"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          xPos   = 6
          yPos   = 12
          width  = 6
          height = 4
          widget = {
            title = "Vector Database Chunk Health Evaluation Logs"
            logsPanel = {
              filter = "jsonPayload.event_type=\"sec_edgar_chunk_health_evaluation\""
              resourceNames = [
                "projects/${var.project_id}"
              ]
            }
          }
        }
      ]
    }
  })

  depends_on = [
    google_monitoring_metric_descriptor.chunk_health_score,
    google_monitoring_metric_descriptor.embedding_coverage_ratio,
    google_monitoring_metric_descriptor.filing_coverage_ratio,
    google_monitoring_metric_descriptor.total_chunks,
    google_monitoring_metric_descriptor.token_count_mean,
    google_monitoring_metric_descriptor.anomaly_chunk_count,
  ]
}
