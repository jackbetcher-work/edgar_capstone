# ------------------------------------------------------------------------------
# Vertex AI Vector Search (Matching Engine) Index for Metadata-Filtered RAG
# ------------------------------------------------------------------------------
resource "google_vertex_ai_index" "sec_10k_chunks" {
  count = var.enable_vertex_vector_search ? 1 : 0

  project             = var.project_id
  region              = var.region
  display_name        = "sec-edgar-10k-chunks-index"
  description         = "Metadata-tagged SEC 10-K (Item 1A & Item 7) vector index backed by gs://${local.sec_corpus_bucket_name}/vector_search/embeddings."
  index_update_method = "BATCH_UPDATE"
  labels              = var.labels

  metadata {
    contents_delta_uri = (
      var.seed_vertex_vector_index
      ? "gs://${local.sec_corpus_bucket_name}/vector_search/embeddings"
      : null
    )

    config {
      dimensions                  = var.embedding_dimensions
      approximate_neighbors_count = 150
      distance_measure_type       = "COSINE_DISTANCE"
      # Vertex AI requires UNIT_L2_NORM feature normalization for COSINE_DISTANCE indexes.
      feature_norm_type = "UNIT_L2_NORM"
      shard_size        = "SHARD_SIZE_SMALL"

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
  description             = local.vector_search_psc ? "Private Service Connect endpoint for metadata-filtered ANN retrieval over SEC 10-K Item 1A and Item 7 chunks." : "Public endpoint for metadata-filtered ANN retrieval over SEC 10-K Item 1A and Item 7 chunks."
  public_endpoint_enabled = !local.vector_search_psc
  labels                  = var.labels

  # PSC-only endpoint: no public domain, connections accepted from this project only.
  # Switching an existing public endpoint to PSC replaces the endpoint (and redeploys
  # the index); see network.tf for the consumer side.
  dynamic "private_service_connect_config" {
    for_each = local.vector_search_psc ? [1] : []
    content {
      enable_private_service_connect = true
      project_allowlist              = [var.project_id]
    }
  }

  depends_on = [google_project_service.required_apis]
}

# Deploys the index onto the endpoint so it can serve queries (gRPC over PSC by
# default, REST findNeighbors in public mode). Deployment typically takes 20-40 minutes. SHARD_SIZE_SMALL indexes
# can be served on e2-standard-2 machines.
resource "google_vertex_ai_index_endpoint_deployed_index" "sec_10k_chunks_deployed" {
  count = var.enable_vertex_vector_search && var.deploy_vertex_vector_index ? 1 : 0

  index_endpoint    = google_vertex_ai_index_endpoint.sec_10k_chunks_endpoint[0].id
  index             = google_vertex_ai_index.sec_10k_chunks[0].id
  deployed_index_id = var.vector_search_deployed_index_id
  display_name      = "sec-edgar-10k-chunks-deployed"

  dedicated_resources {
    machine_spec {
      machine_type = var.vector_search_machine_type
    }
    min_replica_count = var.vector_search_min_replicas
    max_replica_count = var.vector_search_max_replicas
  }

  timeouts {
    create = "90m"
    update = "90m"
    delete = "60m"
  }
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

  # The API populates a server-side default `document_processing_config` that is
  # not declared here. Without ignoring it, every plan forces a destroy/recreate,
  # which fails (HTTP 400) while the data store is attached to the search engine
  # and would wipe the indexed corpus if it succeeded.
  lifecycle {
    ignore_changes = [document_processing_config]
  }

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
