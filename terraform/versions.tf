terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
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
