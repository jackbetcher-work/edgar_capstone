# ------------------------------------------------------------------------------
# Identity-Aware Proxy (IAP) & Google Workspace Okta/SSO Access Boundary
# ------------------------------------------------------------------------------

data "google_project" "current" {
  project_id = var.project_id

  depends_on = [google_project_service.crm_api]
}

resource "google_compute_region_network_endpoint_group" "sec_edgar_serverless_neg" {
  count = var.enable_iap_backend_service && var.enable_cloud_run_service && var.create_analyst_service_account ? 1 : 0

  project               = var.project_id
  name                  = "${var.cloud_run_service_name}-serverless-neg"
  network_endpoint_type = "SERVERLESS"
  region                = var.region

  cloud_run {
    service = google_cloud_run_v2_service.sec_edgar_analyst[0].name
  }

  depends_on = [google_project_service.required_apis]
}

resource "google_compute_backend_service" "sec_edgar_iap_backend" {
  count = var.enable_iap_backend_service && var.enable_cloud_run_service && var.create_analyst_service_account ? 1 : 0

  project               = var.project_id
  name                  = "${var.cloud_run_service_name}-iap-backend"
  description           = "IAP-protected HTTPS backend service fronting Cloud Run with Google Workspace Okta/SSO."
  load_balancing_scheme = "EXTERNAL_MANAGED"
  protocol              = "HTTPS"

  backend {
    group = google_compute_region_network_endpoint_group.sec_edgar_serverless_neg[0].id
  }

  iap {
    enabled = true
  }

  log_config {
    enable      = true
    sample_rate = 1.0
  }

  depends_on = [google_project_service.required_apis]
}

# Restrict Cloud Run direct invocation to the Google-managed IAP Service Agent
resource "google_cloud_run_v2_service_iam_member" "iap_service_agent_invoker" {
  count = var.enable_iap_backend_service && var.enable_cloud_run_service && var.create_analyst_service_account ? 1 : 0

  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.sec_edgar_analyst[0].name
  role     = "roles/run.invoker"
  member   = "serviceAccount:service-${data.google_project.current.number}@gcp-sa-iap.iam.gserviceaccount.com"
}

# Grant IAP HTTPS Resource Accessor to Google Workspace / Okta SSO Cloud Identity groups
resource "google_iap_web_backend_service_iam_member" "workspace_okta_authorized_accessors" {
  for_each = (
    var.enable_iap_backend_service && var.enable_cloud_run_service && var.create_analyst_service_account
    ? toset(var.iap_authorized_principals)
    : toset([])
  )

  project             = var.project_id
  web_backend_service = google_compute_backend_service.sec_edgar_iap_backend[0].name
  role                = "roles/iap.httpsResourceAccessor"
  member              = each.value
}
