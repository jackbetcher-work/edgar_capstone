# ------------------------------------------------------------------------------
# VPC Service Controls: service perimeter around the project
# ------------------------------------------------------------------------------
# Puts the project in a perimeter so the data and model APIs (BigQuery, GCS,
# Vertex AI, Discovery Engine, Firestore, Secret Manager, Model Armor, DLP,
# Artifact Registry, Cloud Run) only answer callers inside it, plus the
# identities and sources the ingress rules below allow.
#
# Rollout: vpc_sc_enforce = false (default) writes the configuration as an explicit
# dry-run spec. Nothing is blocked; would-be denials show up in Cloud Logging:
#   protoPayload.metadata.@type="type.googleapis.com/google.cloud.audit.VpcServiceControlAuditMetadata"
#   protoPayload.metadata.dryRun=true
# Once the log is clean, set vpc_sc_enforce = true to enforce the same configuration.
#
# Access policy: a perimeter lives in an Access Context Manager policy, which is an
# organization-level resource. Either
#   * have an org admin create a policy scoped to this project and grant you
#     roles/accesscontextmanager.policyAdmin on it, then set vpc_sc_access_policy_id, or
#   * set create_vpc_sc_scoped_access_policy = true if you hold
#     accesscontextmanager.policies.create on the organization.
#
# Cloud Run note: without Direct VPC egress, Cloud Run's calls to Google APIs aren't
# treated as coming from inside the perimeter, so the runtime and agent service
# accounts are allowed by identity below. Routing egress through a VPC with
# restricted.googleapis.com (gap 2, private networking) removes the need for that.

locals {
  vpc_sc_enabled = var.enable_vpc_service_controls

  vpc_sc_policy_id = (
    var.create_vpc_sc_scoped_access_policy
    ? try(google_access_context_manager_access_policy.sec_edgar_scoped[0].name, "")
    : trimprefix(trimspace(var.vpc_sc_access_policy_id), "accessPolicies/")
  )

  vpc_sc_project_resource = "projects/${data.google_project.current.number}"

  # Identities allowed to call restricted APIs from any source: operators, CI/CD,
  # the Cloud Run runtime and per-agent SAs, and the Cloud Logging -> BigQuery sink.
  vpc_sc_trusted_identities = sort(distinct(compact(concat(
    var.vpc_sc_admin_identities,
    ["serviceAccount:${google_service_account.cicd_runner.email}"],
    local.cloudbuild_enabled ? ["serviceAccount:${google_service_account.cloudbuild_runner[0].email}"] : [],
    ["serviceAccount:${local.runtime_service_account_email}"],
    [for sa in google_service_account.agent_identities : "serviceAccount:${sa.email}"],
    var.enable_bigquery_logging_sink ? [google_logging_project_sink.telemetry_and_audit_to_bigquery[0].writer_identity] : [],
  ))))
}

resource "google_access_context_manager_access_policy" "sec_edgar_scoped" {
  count = local.vpc_sc_enabled && var.create_vpc_sc_scoped_access_policy ? 1 : 0

  parent = "organizations/${data.google_project.current.org_id}"
  title  = "sec-edgar-${var.project_id}"
  scopes = [local.vpc_sc_project_resource]

  depends_on = [google_project_service.required_apis]
}

resource "google_access_context_manager_service_perimeter" "sec_edgar" {
  count = local.vpc_sc_enabled ? 1 : 0

  parent         = "accessPolicies/${local.vpc_sc_policy_id}"
  name           = "accessPolicies/${local.vpc_sc_policy_id}/servicePerimeters/sec_edgar_perimeter"
  title          = "sec_edgar_perimeter"
  description    = "Restricts SEC EDGAR Analyst data and model APIs in ${var.project_id} to the perimeter."
  perimeter_type = "PERIMETER_TYPE_REGULAR"

  # Dry run: the configuration lives in `spec` and `status` stays empty (enforces nothing).
  # Enforced: the same configuration moves to `status`.
  use_explicit_dry_run_spec = !var.vpc_sc_enforce

  dynamic "spec" {
    for_each = var.vpc_sc_enforce ? [] : [1]
    content {
      resources           = [local.vpc_sc_project_resource]
      restricted_services = var.vpc_sc_restricted_services

      ingress_policies {
        title = "operators-ci-and-runtime-identities"
        ingress_from {
          identities = local.vpc_sc_trusted_identities
          sources {
            access_level = "*"
          }
        }
        ingress_to {
          resources = ["*"]
          operations {
            service_name = "*"
          }
        }
      }

      # IAP-authenticated users invoking the app. VPC-SC can't evaluate IAM identities
      # on Cloud Run invocations (only ANY_IDENTITY works), so IAP decides who gets in.
      ingress_policies {
        title = "iap-app-users-invoke-cloud-run"
        ingress_from {
          identity_type = "ANY_IDENTITY"
          dynamic "sources" {
            for_each = var.vpc_sc_app_ingress_access_levels
            content {
              access_level = sources.value
            }
          }
        }
        ingress_to {
          resources = ["*"]
          operations {
            service_name = "run.googleapis.com"
            method_selectors {
              method = "*"
            }
          }
        }
      }
    }
  }

  dynamic "status" {
    for_each = var.vpc_sc_enforce ? [1] : []
    content {
      resources           = [local.vpc_sc_project_resource]
      restricted_services = var.vpc_sc_restricted_services

      ingress_policies {
        title = "operators-ci-and-runtime-identities"
        ingress_from {
          identities = local.vpc_sc_trusted_identities
          sources {
            access_level = "*"
          }
        }
        ingress_to {
          resources = ["*"]
          operations {
            service_name = "*"
          }
        }
      }

      ingress_policies {
        title = "iap-app-users-invoke-cloud-run"
        ingress_from {
          identity_type = "ANY_IDENTITY"
          dynamic "sources" {
            for_each = var.vpc_sc_app_ingress_access_levels
            content {
              access_level = sources.value
            }
          }
        }
        ingress_to {
          resources = ["*"]
          operations {
            service_name = "run.googleapis.com"
            method_selectors {
              method = "*"
            }
          }
        }
      }
    }
  }

  lifecycle {
    precondition {
      condition     = local.vpc_sc_policy_id != ""
      error_message = "enable_vpc_service_controls needs an access policy: set vpc_sc_access_policy_id to a delegated scoped policy, or create_vpc_sc_scoped_access_policy = true (needs org-level Access Context Manager permissions)."
    }
  }

  depends_on = [google_project_service.required_apis]
}
