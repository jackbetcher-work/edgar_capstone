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

import httpx
from google.cloud import storage

try:
    from scripts._cli import add_gcp_args, configure_script_logging
except ImportError:  # pragma: no cover - direct script execution without editable install
    from _cli import (  # type: ignore[import-not-found,no-redef]
        add_gcp_args,
        configure_script_logging,
    )

from edgar_capstone.gcp.defaults import (
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
)
from edgar_capstone.gcp.lro import OperationError, poll_operation
from edgar_capstone.gcp.rest_client import GcpRestClient, GcpRestError

DEFAULT_TFSTATE_BUCKET = f"{DEFAULT_PROJECT_ID}-tfstate"


def ensure_bootstrap_apis(
    project_id: str = DEFAULT_PROJECT_ID,
    *,
    timeout_seconds: float = 120.0,
) -> None:
    """Enable Cloud Resource Manager & Service Usage APIs required by Terraform."""
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
    with GcpRestClient(default_timeout=60.0) as rest:
        payload = rest._request_json("POST", url, json_body={"serviceIds": services})
        if not isinstance(payload, dict):
            payload = {}
        if payload.get("error"):
            raise OperationError(
                f"Service Usage batchEnable failed on '{project_id}': {payload['error']}"
            )
        op_name = str(payload.get("name") or "")
        if op_name and not payload.get("done", False):
            op_url = f"https://serviceusage.googleapis.com/v1/{op_name}"
            final = poll_operation(
                lambda: rest._raw_request("GET", op_url, timeout=30.0),
                lambda data: bool(data.get("done")),
                label=f"serviceusage batchEnable ({project_id})",
                timeout_seconds=timeout_seconds,
                initial_interval_seconds=2.0,
                max_interval_seconds=10.0,
            )
            if final.get("error"):
                raise OperationError(
                    f"Service Usage batchEnable operation {op_name} failed on '{project_id}': "
                    f"{final['error']}"
                )
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
    add_gcp_args(
        parser,
        include_project=True,
        include_bucket=True,
        include_location=True,
        project_default=os.environ.get("EDGAR_GCP_PROJECT_ID") or DEFAULT_PROJECT_ID,
        bucket_default=os.environ.get("EDGAR_TFSTATE_BUCKET") or DEFAULT_TFSTATE_BUCKET,
        location_default=os.environ.get("EDGAR_GCP_REGION") or DEFAULT_LOCATION,
        project_help="Google Cloud project ID.",
        bucket_help="GCS bucket name for Terraform state.",
        location_help="GCS region for the state bucket.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_script_logging(fmt="%(asctime)s | %(levelname)-7s | %(message)s")
    args = parse_args(argv)
    try:
        ensure_bootstrap_apis(project_id=args.project_id)
        ensure_tfstate_bucket(
            project_id=args.project_id,
            bucket_name=args.bucket,
            location=args.location,
        )
    except (OperationError, GcpRestError, httpx.HTTPError, RuntimeError):
        logging.exception("Failed to bootstrap Terraform state bucket.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
