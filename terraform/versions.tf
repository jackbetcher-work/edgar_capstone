terraform {
  # >= 1.7 required for `for_each` inside `import` blocks (cicd_and_secrets.tf).
  required_version = ">= 1.7.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 6.0"
    }
    time = {
      source  = "hashicorp/time"
      version = "~> 0.12"
    }
  }

  # Remote state stored in the dedicated versioned GCS bucket inside iron-radio-514217-q7.
  # Bootstrap the bucket once before `terraform init` via:
  #   ../.venv/bin/python ../scripts/bootstrap_tfstate_bucket.py
  backend "gcs" {
    bucket = "iron-radio-514217-q7-tfstate"
    prefix = "edgar-capstone/bigquery"
  }
}
