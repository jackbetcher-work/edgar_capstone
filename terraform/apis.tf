resource "google_project_service" "crm_api" {
  project            = var.project_id
  service            = "cloudresourcemanager.googleapis.com"
  disable_on_destroy = false
}

locals {
  required_services = toset([
    "aiplatform.googleapis.com",
    "bigquery.googleapis.com",
    "bigqueryconnection.googleapis.com",
    "bigquerystorage.googleapis.com",
    "iam.googleapis.com",
    "serviceusage.googleapis.com",
    "storage.googleapis.com",
  ])
}

resource "google_project_service" "required_apis" {
  for_each = local.required_services

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false

  depends_on = [google_project_service.crm_api]
}
