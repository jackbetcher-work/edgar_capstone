# ------------------------------------------------------------------------------
# Vertex AI Vector Search (Matching Engine) Index for Metadata-Filtered RAG
# ------------------------------------------------------------------------------
resource "google_vertex_ai_index" "sec_10k_chunks" {
  count = var.enable_vertex_vector_search ? 1 : 0

  project             = var.project_id
  region              = var.region
  display_name        = "sec-edgar-10k-chunks-index"
  description         = "Metadata-tagged SEC 10-K (Item 1A & Item 7) vector index backed by gs://${var.sec_corpus_bucket}/vector_search/embeddings."
  index_update_method = "BATCH_UPDATE"
  labels              = var.labels

  metadata {
    contents_delta_uri = "gs://${var.sec_corpus_bucket}/vector_search/embeddings"

    config {
      dimensions                  = var.embedding_dimensions
      approximate_neighbors_count = 150
      distance_measure_type       = "COSINE_DISTANCE"
      shard_size                  = "SHARD_SIZE_SMALL"

      algorithm_config {
        tree_ah_config {
          leaf_node_embedding_count    = 500
          leaf_nodes_to_search_percent = 7
        }
      }
    }
  }

  depends_on = [google_project_service.required_apis]
}

resource "google_vertex_ai_index_endpoint" "sec_10k_chunks_endpoint" {
  count = var.enable_vertex_vector_search ? 1 : 0

  project                 = var.project_id
  region                  = var.region
  display_name            = "sec-edgar-10k-chunks-endpoint"
  description             = "Public endpoint for metadata-filtered ANN retrieval over SEC 10-K Item 1A and Item 7 chunks."
  public_endpoint_enabled = true
  labels                  = var.labels

  depends_on = [google_project_service.required_apis]
}

# ------------------------------------------------------------------------------
# Vertex AI Search (Discovery Engine) Data Store & Hybrid Search Engine
# ------------------------------------------------------------------------------
resource "google_discovery_engine_data_store" "sec_10k_corpus" {
  count = var.enable_vertex_ai_search ? 1 : 0

  project                     = var.project_id
  location                    = "global"
  data_store_id               = "sec-edgar-10k-corpus-ds"
  display_name                = "SEC EDGAR 10-K Unstructured Filings Corpus"
  industry_vertical           = "GENERIC"
  content_config              = "CONTENT_REQUIRED"
  solution_types              = ["SOLUTION_TYPE_SEARCH"]
  create_advanced_site_search = false

  depends_on = [google_project_service.required_apis]
}

resource "google_discovery_engine_search_engine" "sec_10k_hybrid_search" {
  count = var.enable_vertex_ai_search ? 1 : 0

  project        = var.project_id
  location       = "global"
  collection_id  = "default_collection"
  engine_id      = "sec-edgar-10k-hybrid-search"
  display_name   = "SEC EDGAR 10-K Hybrid RAG Search App"
  data_store_ids = [google_discovery_engine_data_store.sec_10k_corpus[0].data_store_id]

  search_engine_config {
    search_tier    = "SEARCH_TIER_ENTERPRISE"
    search_add_ons = ["SEARCH_ADD_ON_LLM"]
  }

  depends_on = [google_discovery_engine_data_store.sec_10k_corpus]
}
