# ------------------------------------------------------------------------------
# Managed Google Cloud Model Armor & Sensitive Data Protection (SDP / Cloud DLP)
# ------------------------------------------------------------------------------

resource "google_data_loss_prevention_inspect_template" "sec_edgar_sdp_inspect" {
  count = var.enable_model_armor ? 1 : 0

  parent       = "projects/${var.project_id}/locations/${var.region}"
  template_id  = "sec-edgar-sdp-inspect"
  display_name = "SEC EDGAR Analyst Sensitive Data & Projection Inspect Template"
  description  = "Inspects prompts and synthesized 10-K reports for PII, credentials, confidential MNPI, and ungrounded financial projection markers."

  inspect_config {
    min_likelihood = "LIKELY"

    info_types {
      name = "US_SOCIAL_SECURITY_NUMBER"
    }
    info_types {
      name = "CREDIT_CARD_NUMBER"
    }
    info_types {
      name = "EMAIL_ADDRESS"
    }
    info_types {
      name = "PHONE_NUMBER"
    }
    info_types {
      name = "GCP_API_KEY"
    }
    info_types {
      name = "AUTH_TOKEN"
    }

    custom_info_types {
      info_type {
        name = "CONFIDENTIAL_MNPI_OR_PROJECTION_LEAK"
      }
      likelihood = "VERY_LIKELY"
      dictionary {
        word_list {
          words = [
            "CONFIDENTIAL MNPI",
            "MATERIAL NON-PUBLIC INFORMATION",
            "NON-PUBLIC FINANCIAL PROJECTION",
            "INTERNAL EMBARGOED DRAFT",
            "UNRELEASED EARNINGS PREVIEW",
            "12-MONTH PRICE TARGET",
            "STOCK PRICE FORECAST",
          ]
        }
      }
    }
  }

  depends_on = [google_project_service.required_apis]
}

resource "google_data_loss_prevention_deidentify_template" "sec_edgar_sdp_deidentify" {
  count = var.enable_model_armor ? 1 : 0

  parent       = "projects/${var.project_id}/locations/${var.region}"
  template_id  = "sec-edgar-sdp-deidentify"
  display_name = "SEC EDGAR Analyst Sensitive Data De-identification Template"
  description  = "Redacts detected PII and credential spans with [INFO_TYPE] placeholders before prompts reach Gemini or enter session memory."

  deidentify_config {
    info_type_transformations {
      transformations {
        info_types {
          name = "US_SOCIAL_SECURITY_NUMBER"
        }
        info_types {
          name = "CREDIT_CARD_NUMBER"
        }
        info_types {
          name = "EMAIL_ADDRESS"
        }
        info_types {
          name = "PHONE_NUMBER"
        }
        info_types {
          name = "GCP_API_KEY"
        }
        info_types {
          name = "AUTH_TOKEN"
        }
        primitive_transformation {
          replace_with_info_type_config = true
        }
      }
    }
  }

  depends_on = [google_project_service.required_apis]
}

resource "google_model_armor_template" "sec_edgar_analyst_armor" {
  count = var.enable_model_armor ? 1 : 0

  project     = var.project_id
  location    = var.region
  template_id = var.model_armor_template_id
  labels      = var.labels

  filter_config {
    pi_and_jailbreak_filter_settings {
      filter_enforcement = "ENABLED"
      confidence_level   = "MEDIUM_AND_ABOVE"
    }

    malicious_uri_filter_settings {
      filter_enforcement = "ENABLED"
    }

    rai_settings {
      rai_filters {
        filter_type      = "DANGEROUS"
        confidence_level = "MEDIUM_AND_ABOVE"
      }
      rai_filters {
        filter_type      = "HARASSMENT"
        confidence_level = "MEDIUM_AND_ABOVE"
      }
      rai_filters {
        filter_type      = "HATE_SPEECH"
        confidence_level = "MEDIUM_AND_ABOVE"
      }
      rai_filters {
        filter_type      = "SEXUALLY_EXPLICIT"
        confidence_level = "MEDIUM_AND_ABOVE"
      }
    }

    sdp_settings {
      advanced_config {
        inspect_template    = google_data_loss_prevention_inspect_template.sec_edgar_sdp_inspect[0].id
        deidentify_template = google_data_loss_prevention_deidentify_template.sec_edgar_sdp_deidentify[0].id
      }
    }
  }

  template_metadata {
    log_sanitize_operations = true
    log_template_operations = true
  }

  depends_on = [
    google_project_service.required_apis,
    google_data_loss_prevention_inspect_template.sec_edgar_sdp_inspect,
    google_data_loss_prevention_deidentify_template.sec_edgar_sdp_deidentify,
  ]
}
