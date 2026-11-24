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
