# ------------------------------------------------------------------------------
# Identity, Cloud Identity Group RBAC, Custom Service Account, & Agent Identities
# ------------------------------------------------------------------------------

locals {
  runtime_service_account_email              = var.create_analyst_service_account ? google_service_account.sec_analyst[0].email : "${var.analyst_service_account_id}@${var.project_id}.iam.gserviceaccount.com"
  runtime_service_account_project_name_email = "${var.analyst_service_account_id}@${var.project_name}.iam.gserviceaccount.com"

  bind_cloud_identity_groups = var.enable_cloud_identity_group_iam || var.manage_cloud_identity_groups

  rbac_admin_group_emails = toset([
    for g in var.cloud_identity_rbac_groups.admins : lower(trimprefix(trimspace(g), "group:"))
  ])
  rbac_analyst_group_emails = toset([
    for g in var.cloud_identity_rbac_groups.analysts : lower(trimprefix(trimspace(g), "group:"))
  ])
  rbac_viewer_group_emails = toset([
    for g in var.cloud_identity_rbac_groups.viewers : lower(trimprefix(trimspace(g), "group:"))
  ])

  rbac_admin_members   = toset([for email in local.rbac_admin_group_emails : "group:${email}"])
  rbac_analyst_members = toset([for email in local.rbac_analyst_group_emails : "group:${email}"])
  rbac_viewer_members  = toset([for email in local.rbac_viewer_group_emails : "group:${email}"])
  all_rbac_group_members = setunion(
    local.rbac_admin_members,
    local.rbac_analyst_members,
    local.rbac_viewer_members,
  )

  managed_cloud_identity_group_defs = merge(
    {
      for email in local.rbac_admin_group_emails : email => {
        email        = email
        display_name = "SEC EDGAR Platform Admins"
        description  = "Cloud Identity RBAC group for SEC EDGAR platform administration and audit governance."
      }
    },
    {
      for email in local.rbac_analyst_group_emails : email => {
        email        = email
        display_name = "SEC EDGAR Financial Analysts"
        description  = "Cloud Identity RBAC group for SEC 10-K Hybrid RAG synthesis and variance analysis."
      }
    },
    {
      for email in local.rbac_viewer_group_emails : email => {
        email        = email
        display_name = "SEC EDGAR Compliance Viewers"
        description  = "Cloud Identity RBAC group for read-only 10-K filing and audit inspection."
      }
    },
  )

  # Granular per-agent service accounts enforcing Least Privilege for S2S communication
  agent_identities = {
    root_orchestrator = {
      agent_name   = "RootOrchestrator"
      account_id   = "root-orchestrator-agent-sa"
      display_name = "ADK RootOrchestrator Agent Identity"
      description  = "Per-agent S2S identity for intent routing and delegating tasks to specialist sub-agents."
      project_roles = [
        "roles/aiplatform.user",
        "roles/logging.logWriter",
        "roles/monitoring.metricWriter",
      ]
    }
    financial_analyst = {
      agent_name   = "FinancialAnalystAgent"
      account_id   = "financial-analyst-agent-sa"
      display_name = "ADK FinancialAnalystAgent Identity"
      description  = "Per-agent S2S identity scoped to read-only BigQuery financial metrics and Hybrid RAG search."
      project_roles = [
        "roles/bigquery.jobUser",
        "roles/bigquery.connectionUser",
        "roles/discoveryengine.viewer",
        "roles/aiplatform.user",
        "roles/logging.logWriter",
        "roles/monitoring.metricWriter",
      ]
    }
    context_cache = {
      agent_name   = "ContextCacheManager"
      account_id   = "context-cache-agent-sa"
      display_name = "ADK ContextCacheAgent Identity"
      description  = "Per-agent S2S identity scoped strictly to reading curated 10-K Markdown sections from GCS."
      project_roles = [
        "roles/aiplatform.user",
        "roles/logging.logWriter",
      ]
    }
    model_armor = {
      agent_name   = "ModelArmorGuardrail"
      account_id   = "model-armor-guardrail-sa"
      display_name = "Model Armor Guardrail Agent Identity"
      description  = "Isolated S2S safety guardrail identity with zero access to BigQuery, GCS, or secrets."
      project_roles = [
        "roles/aiplatform.user",
        "roles/logging.logWriter",
      ]
    }
  }

  agent_project_role_bindings = {
    for pair in flatten([
      for agent_key, spec in local.agent_identities : [
        for role in spec.project_roles : {
          key       = "${agent_key}__${replace(role, "roles/", "")}"
          agent_key = agent_key
          role      = role
        }
      ]
    ]) : pair.key => pair
  }

  # Directed S2S impersonation graph (caller -> callee) enforcing strict Least Privilege
  agent_s2s_delegation_edges = {
    "root_to_financial" = {
      caller = "root_orchestrator"
      callee = "financial_analyst"
    }
    "root_to_cache" = {
      caller = "root_orchestrator"
      callee = "context_cache"
    }
    "root_to_armor" = {
      caller = "root_orchestrator"
      callee = "model_armor"
    }
    "financial_to_cache" = {
      caller = "financial_analyst"
      callee = "context_cache"
    }
    "financial_to_armor" = {
      caller = "financial_analyst"
      callee = "model_armor"
    }
  }
}

# ------------------------------------------------------------------------------
# 1. Custom Application Runtime Service Account (sec-analyst-sa@<project>.iam.gserviceaccount.com)
# ------------------------------------------------------------------------------
resource "google_service_account" "sec_analyst" {
  count = var.create_analyst_service_account ? 1 : 0

  project      = var.project_id
  account_id   = var.analyst_service_account_id
  display_name = "SEC EDGAR Natural Language Analyst Service Account"
  description  = "Custom runtime identity (${var.analyst_service_account_id}@${var.project_name}.iam.gserviceaccount.com) for the Cloud Run application; default Compute SAs are prohibited."

  depends_on = [google_project_service.required_apis]
}

resource "google_project_iam_member" "sec_analyst_bq_job_user" {
  count = var.create_analyst_service_account ? 1 : 0

  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.sec_analyst[0].email}"
}

resource "google_project_iam_member" "sec_analyst_bq_connection_user" {
  count = var.create_analyst_service_account ? 1 : 0

  project = var.project_id
  role    = "roles/bigquery.connectionUser"
  member  = "serviceAccount:${google_service_account.sec_analyst[0].email}"
}

resource "google_bigquery_dataset_iam_member" "sec_analyst_dataset_viewer" {
  count = var.create_analyst_service_account ? 1 : 0

  project    = var.project_id
  dataset_id = google_bigquery_dataset.sec_edgar.dataset_id
  role       = "roles/bigquery.dataViewer"
  member     = "serviceAccount:${google_service_account.sec_analyst[0].email}"
}

resource "google_storage_bucket_iam_member" "sec_analyst_corpus_viewer" {
  count = var.create_analyst_service_account ? 1 : 0

  bucket = var.sec_corpus_bucket
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.sec_analyst[0].email}"
}

resource "google_project_iam_member" "sec_analyst_vertex_user" {
  count = var.create_analyst_service_account ? 1 : 0

  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.sec_analyst[0].email}"
}

resource "google_project_iam_member" "sec_analyst_discoveryengine_viewer" {
  count = var.create_analyst_service_account ? 1 : 0

  project = var.project_id
  role    = "roles/discoveryengine.viewer"
  member  = "serviceAccount:${google_service_account.sec_analyst[0].email}"
}

resource "google_project_iam_member" "sec_analyst_monitoring_metric_writer" {
  count = var.create_analyst_service_account ? 1 : 0

  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.sec_analyst[0].email}"
}

resource "google_project_iam_member" "sec_analyst_log_writer" {
  count = var.create_analyst_service_account ? 1 : 0

  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.sec_analyst[0].email}"
}

resource "google_project_iam_member" "sec_analyst_trace_agent" {
  count = var.create_analyst_service_account ? 1 : 0

  project = var.project_id
  role    = "roles/cloudtrace.agent"
  member  = "serviceAccount:${google_service_account.sec_analyst[0].email}"
}

# ------------------------------------------------------------------------------
# 2. Role-Based Access Control (RBAC) via Google Cloud Identity Groups
# ------------------------------------------------------------------------------
resource "google_cloud_identity_group" "rbac_groups" {
  for_each = (
    var.manage_cloud_identity_groups && var.cloud_identity_customer_id != ""
    ? local.managed_cloud_identity_group_defs
    : {}
  )

  display_name         = each.value.display_name
  description          = each.value.description
  initial_group_config = "EMPTY"
  parent               = "customers/${trimprefix(var.cloud_identity_customer_id, "customers/")}"

  group_key {
    id = each.value.email
  }

  labels = {
    "cloudidentity.googleapis.com/groups.discussion_forum" = ""
  }

  depends_on = [google_project_service.required_apis]
}

resource "google_project_iam_member" "rbac_admins_run_viewer" {
  for_each = local.bind_cloud_identity_groups ? local.rbac_admin_members : toset([])

  project = var.project_id
  role    = "roles/run.viewer"
  member  = each.value
}

resource "google_project_iam_member" "rbac_admins_log_viewer" {
  for_each = local.bind_cloud_identity_groups ? local.rbac_admin_members : toset([])

  project = var.project_id
  role    = "roles/logging.privateLogViewer"
  member  = each.value
}

resource "google_project_iam_member" "rbac_admins_monitoring_viewer" {
  for_each = local.bind_cloud_identity_groups ? local.rbac_admin_members : toset([])

  project = var.project_id
  role    = "roles/monitoring.viewer"
  member  = each.value
}

resource "google_bigquery_dataset_iam_member" "rbac_admins_dataset_metadata_viewer" {
  for_each = local.bind_cloud_identity_groups ? local.rbac_admin_members : toset([])

  project    = var.project_id
  dataset_id = google_bigquery_dataset.sec_edgar.dataset_id
  role       = "roles/bigquery.metadataViewer"
  member     = each.value
}

resource "google_bigquery_dataset_iam_member" "rbac_analysts_dataset_viewer" {
  for_each = local.bind_cloud_identity_groups ? local.rbac_analyst_members : toset([])

  project    = var.project_id
  dataset_id = google_bigquery_dataset.sec_edgar.dataset_id
  role       = "roles/bigquery.dataViewer"
  member     = each.value
}

resource "google_project_iam_member" "rbac_analysts_discoveryengine_viewer" {
  for_each = local.bind_cloud_identity_groups ? local.rbac_analyst_members : toset([])

  project = var.project_id
  role    = "roles/discoveryengine.viewer"
  member  = each.value
}

resource "google_bigquery_dataset_iam_member" "rbac_viewers_dataset_metadata_viewer" {
  for_each = local.bind_cloud_identity_groups ? local.rbac_viewer_members : toset([])

  project    = var.project_id
  dataset_id = google_bigquery_dataset.sec_edgar.dataset_id
  role       = "roles/bigquery.metadataViewer"
  member     = each.value
}

# ------------------------------------------------------------------------------
# 3. Granular Per-Agent Identities & Least-Privilege S2S IAM Policies
# ------------------------------------------------------------------------------
resource "google_service_account" "agent_identities" {
  for_each = var.create_agent_service_accounts ? local.agent_identities : {}

  project      = var.project_id
  account_id   = each.value.account_id
  display_name = each.value.display_name
  description  = each.value.description

  depends_on = [google_project_service.required_apis]
}

resource "google_project_iam_member" "agent_least_privilege_project_roles" {
  for_each = var.create_agent_service_accounts ? local.agent_project_role_bindings : {}

  project = var.project_id
  role    = each.value.role
  member  = "serviceAccount:${google_service_account.agent_identities[each.value.agent_key].email}"
}

# FinancialAnalystAgent ONLY: Dataset-scoped read-only access to SEC BigQuery tables
resource "google_bigquery_dataset_iam_member" "financial_analyst_agent_dataset_viewer" {
  count = var.create_agent_service_accounts ? 1 : 0

  project    = var.project_id
  dataset_id = google_bigquery_dataset.sec_edgar.dataset_id
  role       = "roles/bigquery.dataViewer"
  member     = "serviceAccount:${google_service_account.agent_identities["financial_analyst"].email}"
}

# ContextCacheAgent ONLY: Bucket-scoped read-only access to curated 10-K Markdown objects
resource "google_storage_bucket_iam_member" "context_cache_agent_corpus_viewer" {
  count = var.create_agent_service_accounts ? 1 : 0

  bucket = var.sec_corpus_bucket
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.agent_identities["context_cache"].email}"
}

# Allow RootOrchestrator and FinancialAnalystAgent to make authenticated S2S calls to Cloud Run
resource "google_cloud_run_v2_service_iam_member" "agent_s2s_cloud_run_invokers" {
  for_each = (
    var.enable_cloud_run_service && var.create_agent_service_accounts
    ? toset(["root_orchestrator", "financial_analyst", "context_cache"])
    : toset([])
  )

  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.sec_edgar_analyst[0].name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.agent_identities[each.value].email}"
}

# Host runtime SA (sec-analyst-sa) may mint short-lived S2S OIDC / downscoped OAuth tokens for each agent SA
resource "google_service_account_iam_member" "runtime_to_agent_s2s_token_creator" {
  for_each = var.create_agent_service_accounts ? local.agent_identities : {}

  service_account_id = google_service_account.agent_identities[each.key].name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "serviceAccount:${local.runtime_service_account_email}"
}

# Directed agent-to-agent S2S token minting (RootOrchestrator -> FinancialAnalyst/ContextCache/ModelArmor, FinancialAnalyst -> ContextCache/ModelArmor)
resource "google_service_account_iam_member" "agent_to_agent_s2s_token_creator" {
  for_each = var.create_agent_service_accounts ? local.agent_s2s_delegation_edges : {}

  service_account_id = google_service_account.agent_identities[each.value.callee].name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "serviceAccount:${google_service_account.agent_identities[each.value.caller].email}"
}
