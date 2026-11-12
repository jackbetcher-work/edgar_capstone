#!/usr/bin/env python3
"""Bootstrap the versioned GCS bucket used for Terraform remote state.

Terraform's `gcs` backend requires the state bucket to exist before `terraform init`.
This idempotent script creates `gs://iron-radio-514217-q7-tfstate` in `us-central1`
with object versioning, uniform bucket-level access, enforced public access prevention,
and a lifecycle policy retaining recent noncurrent state versions.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import google.auth
from google.auth.transport.requests import Request
from google.cloud import storage
import httpx

DEFAULT_PROJECT_ID = "iron-radio-514217-q7"
DEFAULT_TFSTATE_BUCKET = "iron-radio-514217-q7-tfstate"
DEFAULT_LOCATION = "us-central1"


def ensure_bootstrap_apis(project_id: str = DEFAULT_PROJECT_ID) -> None:
    """Enable Cloud Resource Manager & Service Usage APIs required by Terraform."""
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(Request())
    headers = {"Authorization": f"Bearer {creds.token}"}
    url = f"https://serviceusage.googleapis.com/v1/projects/{project_id}/services:batchEnable"
    services = [
        "serviceusage.googleapis.com",
        "cloudresourcemanager.googleapis.com",
        "iam.googleapis.com",
    ]
    logging.info(
        "Ensuring prerequisite Terraform APIs (%s) are enabled on '%s'...",
        ", ".join(services),
        project_id,
    )
    resp = httpx.post(url, headers=headers, json={"serviceIds": services}, timeout=60)
    resp.raise_for_status()
    op_name = resp.json().get("name", "")
    if op_name and not resp.json().get("done", False):
        op_url = f"https://serviceusage.googleapis.com/v1/{op_name}"
        for _ in range(30):
            time.sleep(2)
            poll = httpx.get(op_url, headers=headers, timeout=30)
            poll.raise_for_status()
            if poll.json().get("done"):
                break
    logging.info("Prerequisite Terraform APIs enabled on '%s'.", project_id)


def ensure_tfstate_bucket(
    *,
    project_id: str = DEFAULT_PROJECT_ID,
    bucket_name: str = DEFAULT_TFSTATE_BUCKET,
    location: str = DEFAULT_LOCATION,
    keep_newer_versions: int = 10,
) -> storage.Bucket:
    """Create or update the GCS Terraform state bucket with security & versioning best practices."""
    client = storage.Client(project=project_id)
    bucket = client.lookup_bucket(bucket_name)

    if bucket is None:
        logging.info(
            "Creating Terraform state bucket gs://%s in project '%s' (%s)...",
            bucket_name,
            project_id,
            location,
        )
        new_bucket = client.bucket(bucket_name)
        new_bucket.iam_configuration.uniform_bucket_level_access_enabled = True
        new_bucket.iam_configuration.public_access_prevention = "enforced"
        new_bucket.versioning_enabled = True
        new_bucket.labels = {
            "project": "sec-edgar-analyst",
            "purpose": "terraform-state",
            "managed_by": "bootstrap-script",
        }
        new_bucket.add_lifecycle_delete_rule(
            number_of_newer_versions=keep_newer_versions,
            is_live=False,
        )
        bucket = client.create_bucket(new_bucket, project=project_id, location=location)
        logging.info("Created versioned state bucket gs://%s", bucket_name)
    else:
        logging.info(
            "Verified existing Terraform state bucket gs://%s (location=%s)",
            bucket_name,
            bucket.location,
        )
        needs_patch = False
        if not bucket.versioning_enabled:
            bucket.versioning_enabled = True
            needs_patch = True
        if not bucket.iam_configuration.uniform_bucket_level_access_enabled:
            bucket.iam_configuration.uniform_bucket_level_access_enabled = True
            needs_patch = True
        if bucket.iam_configuration.public_access_prevention != "enforced":
            bucket.iam_configuration.public_access_prevention = "enforced"
            needs_patch = True
        if needs_patch:
            bucket.patch()
            logging.info("Updated security/versioning settings on gs://%s", bucket_name)

    return bucket


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap the GCS bucket for Terraform remote state.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--project-id",
        default=os.environ.get("EDGAR_GCP_PROJECT_ID", DEFAULT_PROJECT_ID),
        help="Google Cloud project ID.",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("EDGAR_TFSTATE_BUCKET", DEFAULT_TFSTATE_BUCKET),
        help="GCS bucket name for Terraform state.",
    )
    parser.add_argument(
        "--location",
        default=os.environ.get("EDGAR_GCP_REGION", DEFAULT_LOCATION),
        help="GCS region for the state bucket.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args(argv)
    ensure_bootstrap_apis(project_id=args.project_id)
    ensure_tfstate_bucket(
        project_id=args.project_id,
        bucket_name=args.bucket,
        location=args.location,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
