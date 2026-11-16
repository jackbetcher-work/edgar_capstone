# ------------------------------------------------------------------------------
# Curated SEC 10-K Corpus & Vector Manifests GCS Bucket
# ------------------------------------------------------------------------------

locals {
  sec_corpus_bucket_name = (
    var.create_sec_corpus_bucket
    ? google_storage_bucket.sec_corpus[0].name
    : var.sec_corpus_bucket
  )
}

resource "google_storage_bucket" "sec_corpus" {
  count = var.create_sec_corpus_bucket ? 1 : 0

  project                     = var.project_id
  name                        = var.sec_corpus_bucket
  location                    = var.region
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = !var.deletion_protection
  labels                      = var.labels

  versioning {
    enabled = var.sec_corpus_bucket_versioning
  }

  soft_delete_policy {
    retention_duration_seconds = 604800
  }

  lifecycle_rule {
    action {
      type = "AbortIncompleteMultipartUpload"
    }
    condition {
      age = 7
    }
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      num_newer_versions = 5
      with_state         = "ARCHIVED"
    }
  }

  depends_on = [google_project_service.required_apis]
}
