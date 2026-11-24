# ------------------------------------------------------------------------------
# Identity-Aware Proxy (IAP) — Direct Cloud Run Access Boundary
# ------------------------------------------------------------------------------
# IAP is enabled directly on the Cloud Run service (iap_enabled = true in
# cicd_and_secrets.tf). Users open the run.app URL and sign in with Google;
# only iap_authorized_principals get through, and Cloud Run only accepts
# invocations from the IAP service agent.

data "google_project" "current" {
  project_id = var.project_id

  depends_on = [google_project_service.crm_api]
}

locals {
  cloud_run_direct_iap = var.enable_cloud_run_service

  # Expected IAP JWT `aud` claim passed to Cloud Run as INTERNAL_IAP_AUDIENCE.
  iap_jwt_audience = (
    local.cloud_run_direct_iap
    ? "/projects/${data.google_project.current.number}/locations/${var.region}/services/${var.cloud_run_service_name}"
    : null
  )

  effective_iap_group_principals = toset(var.iap_authorized_principals)
}

# The IAP service agent (service-<PROJECT_NUMBER>@gcp-sa-iap.iam.gserviceaccount.com)
# is NOT created automatically when iap.googleapis.com is enabled; it must be
# provisioned explicitly before it can be granted roles.
resource "google_project_service_identity" "iap_service_agent" {
  provider = google-beta
  count    = local.cloud_run_direct_iap ? 1 : 0

  project = var.project_id
  service = "iap.googleapis.com"

  depends_on = [google_project_service.required_apis]
}

# Newly created service agents take a short time to propagate through IAM.
resource "time_sleep" "wait_for_iap_service_agent" {
  count = local.cloud_run_direct_iap ? 1 : 0

  create_duration = "60s"

  depends_on = [google_project_service_identity.iap_service_agent]
}

# Restrict Cloud Run direct invocation to the Google-managed IAP Service Agent
resource "google_cloud_run_v2_service_iam_member" "iap_service_agent_invoker" {
  count = local.cloud_run_direct_iap ? 1 : 0

  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.sec_edgar_analyst[0].name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_project_service_identity.iap_service_agent[0].email}"

  depends_on = [time_sleep.wait_for_iap_service_agent]
}

# Who can open the app: IAP HTTPS Resource Accessor on the Cloud Run service.
resource "google_iap_web_cloud_run_service_iam_member" "cloud_run_authorized_accessors" {
  for_each = local.cloud_run_direct_iap ? local.effective_iap_group_principals : toset([])

  project                = var.project_id
  location               = var.region
  cloud_run_service_name = google_cloud_run_v2_service.sec_edgar_analyst[0].name
  role                   = "roles/iap.httpsResourceAccessor"
  member                 = each.value
}
