# ------------------------------------------------------------------------------
# Private networking: VPC + Private Service Connect for Vertex AI Vector Search
# ------------------------------------------------------------------------------
# With vector_search_endpoint_access = "private_service_connect" (the default):
#   - the index endpoint has no public domain; it only accepts PSC connections
#     from this project (project_allowlist);
#   - a regional internal IP in the private subnet fronts the deployed index's
#     service attachment through a PSC forwarding rule;
#   - Cloud Run reaches that IP through Direct VPC egress (PRIVATE_RANGES_ONLY,
#     so Google API calls keep their normal path) and queries over gRPC
#     (MatchService on <ip>:10000) using VERTEX_VECTOR_SEARCH_PSC_ADDRESS.
#
# The forwarding rule needs the deployed index (deploy_vertex_vector_index), whose
# service attachment only exists after the 20-40 minute deployment finishes.

locals {
  vector_search_psc          = var.enable_vertex_vector_search && var.vector_search_endpoint_access == "private_service_connect"
  vector_search_psc_endpoint = local.vector_search_psc && var.deploy_vertex_vector_index
}

resource "google_compute_network" "sec_edgar" {
  count = local.vector_search_psc ? 1 : 0

  project                 = var.project_id
  name                    = "${var.cloud_run_service_name}-vpc"
  description             = "Private network for Cloud Run Direct VPC egress and Vector Search Private Service Connect."
  auto_create_subnetworks = false
  routing_mode            = "REGIONAL"

  depends_on = [google_project_service.required_apis]
}

resource "google_compute_subnetwork" "sec_edgar_private" {
  count = local.vector_search_psc ? 1 : 0

  project                  = var.project_id
  name                     = "${var.cloud_run_service_name}-private"
  region                   = var.region
  network                  = google_compute_network.sec_edgar[0].id
  ip_cidr_range            = var.private_subnet_cidr
  private_ip_google_access = true
}

# Internal IP the app connects to. Independent of the deployed index, so Cloud Run
# can be configured with it without waiting for index deployment.
resource "google_compute_address" "vector_search_psc" {
  count = local.vector_search_psc_endpoint ? 1 : 0

  project      = var.project_id
  name         = "${var.cloud_run_service_name}-vector-search-psc"
  region       = var.region
  subnetwork   = google_compute_subnetwork.sec_edgar_private[0].id
  address_type = "INTERNAL"
  description  = "Private Service Connect endpoint IP for the SEC 10-K Vector Search deployed index."
}

resource "google_compute_forwarding_rule" "vector_search_psc" {
  count = local.vector_search_psc_endpoint ? 1 : 0

  project               = var.project_id
  name                  = "${var.cloud_run_service_name}-vector-search-psc"
  region                = var.region
  network               = google_compute_network.sec_edgar[0].id
  ip_address            = google_compute_address.vector_search_psc[0].id
  target                = google_vertex_ai_index_endpoint_deployed_index.sec_10k_chunks_deployed[0].private_endpoints[0].service_attachment
  load_balancing_scheme = "" # required for Private Service Connect consumer endpoints
}
