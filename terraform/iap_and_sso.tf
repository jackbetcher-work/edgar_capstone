# ------------------------------------------------------------------------------
# Identity-Aware Proxy (IAP) & Google Workspace Okta/SSO Access Boundary
# ------------------------------------------------------------------------------
# Two ways to put IAP in front of Cloud Run, picked by var.cloud_run_access_mode:
#   iap_direct    IAP is enabled on the Cloud Run service (iap_enabled = true in
#                 cicd_and_secrets.tf). Users open the run.app URL and sign in with
#                 Google; only iap_authorized_principals get through.
#   load_balancer IAP is enabled on a serverless-NEG backend service behind the
#                 external HTTPS load balancer in load_balancer.tf.
# Either way Cloud Run only accepts invocations from the IAP service agent.

data "google_project" "current" {
  project_id = var.project_id

  depends_on = [google_project_service.crm_api]
}

locals {
  cloud_run_direct_iap = var.enable_cloud_run_service && var.cloud_run_access_mode == "iap_direct"
  cloud_run_lb_iap = (
    var.enable_cloud_run_service
    && var.enable_iap_backend_service
    && var.cloud_run_access_mode == "load_balancer"
  )
  cloud_run_behind_iap = local.cloud_run_direct_iap || local.cloud_run_lb_iap

  # Expected IAP JWT `aud` claim; stored in the sec-edgar-iap-audience secret.
  iap_jwt_audience = (
    local.cloud_run_direct_iap
    ? "/projects/${data.google_project.current.number}/locations/${var.region}/services/${var.cloud_run_service_name}"
    : (
      local.cloud_run_lb_iap
      ? "/projects/${data.google_project.current.number}/global/backendServices/${google_compute_backend_service.sec_edgar_iap_backend[0].generated_id}"
      : null
    )
  )

  effective_iap_group_principals = setunion(
    toset(var.iap_authorized_principals),
    local.bind_cloud_identity_groups ? local.all_rbac_group_members : toset([]),
  )
}

resource "google_compute_region_network_endpoint_group" "sec_edgar_serverless_neg" {
  count = local.cloud_run_lb_iap ? 1 : 0

  project               = var.project_id
  name                  = "${var.cloud_run_service_name}-serverless-neg"
  network_endpoint_type = "SERVERLESS"
  region                = var.region

  cloud_run {
    # Reference by name (not the resource) to avoid a dependency cycle:
    # Cloud Run -> IAP audience secret -> backend service -> NEG -> Cloud Run.
    service = var.cloud_run_service_name
  }

  depends_on = [google_project_service.required_apis]
}

resource "google_compute_backend_service" "sec_edgar_iap_backend" {
  count = local.cloud_run_lb_iap ? 1 : 0

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

# The IAP service agent (service-<PROJECT_NUMBER>@gcp-sa-iap.iam.gserviceaccount.com)
# is NOT created automatically when iap.googleapis.com is enabled; it must be
# provisioned explicitly before it can be granted roles.
resource "google_project_service_identity" "iap_service_agent" {
  provider = google-beta
  count    = local.cloud_run_behind_iap ? 1 : 0

  project = var.project_id
  service = "iap.googleapis.com"

  depends_on = [google_project_service.required_apis]
}

# Newly created service agents take a short time to propagate through IAM.
resource "time_sleep" "wait_for_iap_service_agent" {
  count = local.cloud_run_behind_iap ? 1 : 0

  create_duration = "60s"

  depends_on = [google_project_service_identity.iap_service_agent]
}

# Restrict Cloud Run direct invocation to the Google-managed IAP Service Agent
resource "google_cloud_run_v2_service_iam_member" "iap_service_agent_invoker" {
  count = local.cloud_run_behind_iap ? 1 : 0

  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.sec_edgar_analyst[0].name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_project_service_identity.iap_service_agent[0].email}"

  depends_on = [time_sleep.wait_for_iap_service_agent]
}

# Who can open the app (iap_direct): IAP HTTPS Resource Accessor on the Cloud Run service.
resource "google_iap_web_cloud_run_service_iam_member" "cloud_run_authorized_accessors" {
  for_each = local.cloud_run_direct_iap ? local.effective_iap_group_principals : toset([])

  project                = var.project_id
  location               = var.region
  cloud_run_service_name = google_cloud_run_v2_service.sec_edgar_analyst[0].name
  role                   = "roles/iap.httpsResourceAccessor"
  member                 = each.value

  depends_on = [google_cloud_identity_group.rbac_groups]
}

# Who can open the app (load_balancer): IAP HTTPS Resource Accessor on the backend service.
resource "google_iap_web_backend_service_iam_member" "workspace_okta_authorized_accessors" {
  for_each = local.cloud_run_lb_iap ? local.effective_iap_group_principals : toset([])

  project             = var.project_id
  web_backend_service = google_compute_backend_service.sec_edgar_iap_backend[0].name
  role                = "roles/iap.httpsResourceAccessor"
  member              = each.value

  depends_on = [google_cloud_identity_group.rbac_groups]
}
