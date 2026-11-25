"""Unit tests for the GCS Terraform remote state bucket bootstrap script."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from google.api_core import exceptions as gcp_exceptions

from scripts import bootstrap_tfstate_bucket as bootstrap_mod
from scripts.bootstrap_tfstate_bucket import (
    ensure_bootstrap_apis,
    ensure_tfstate_bucket,
    main,
    parse_args,
)


def test_bootstrap_creates_versioned_state_bucket() -> None:
    mock_client = MagicMock()
    mock_client.lookup_bucket.return_value = None

    mock_bucket = MagicMock()
    mock_bucket.iam_configuration = SimpleNamespace(
        uniform_bucket_level_access_enabled=False,
        public_access_prevention="inherited",
    )
    mock_bucket.versioning_enabled = False
    mock_client.bucket.return_value = mock_bucket
    mock_client.create_bucket.return_value = mock_bucket

    with patch("scripts.bootstrap_tfstate_bucket.storage.Client", return_value=mock_client):
        ensure_tfstate_bucket(
            project_id="iron-radio-514217-q7",
            bucket_name="iron-radio-514217-q7-tfstate",
            location="us-central1",
        )

    assert mock_bucket.versioning_enabled is True
    assert mock_bucket.iam_configuration.uniform_bucket_level_access_enabled is True
    assert mock_bucket.iam_configuration.public_access_prevention == "enforced"
    mock_bucket.add_lifecycle_delete_rule.assert_called_once()
    mock_client.create_bucket.assert_called_once_with(
        mock_bucket,
        project="iron-radio-514217-q7",
        location="us-central1",
    )


def test_bootstrap_patches_existing_bucket_only_when_hardening_missing() -> None:
    mock_client = MagicMock()
    existing = MagicMock()
    existing.location = "US-CENTRAL1"
    existing.versioning_enabled = False
    existing.iam_configuration = SimpleNamespace(
        uniform_bucket_level_access_enabled=False,
        public_access_prevention="inherited",
    )
    mock_client.lookup_bucket.return_value = existing

    with patch("scripts.bootstrap_tfstate_bucket.storage.Client", return_value=mock_client):
        result = ensure_tfstate_bucket(
            project_id="iron-radio-514217-q7",
            bucket_name="iron-radio-514217-q7-tfstate",
            location="us-central1",
        )

    assert result is existing
    assert existing.versioning_enabled is True
    assert existing.iam_configuration.uniform_bucket_level_access_enabled is True
    assert existing.iam_configuration.public_access_prevention == "enforced"
    existing.patch.assert_called_once()
    mock_client.create_bucket.assert_not_called()

    # Already-hardened bucket must NOT call `.patch()`
    existing.patch.reset_mock()
    with patch("scripts.bootstrap_tfstate_bucket.storage.Client", return_value=mock_client):
        ensure_tfstate_bucket(
            project_id="iron-radio-514217-q7",
            bucket_name="iron-radio-514217-q7-tfstate",
            location="us-central1",
        )
    existing.patch.assert_not_called()


def test_ensure_bootstrap_apis_uses_service_usage_sdk_and_handles_errors() -> None:
    mock_op = MagicMock()
    mock_op.result.return_value = SimpleNamespace(services=[])
    mock_su = MagicMock()
    mock_su.batch_enable_services.return_value = mock_op

    ensure_bootstrap_apis(
        project_id="test-proj",
        timeout_seconds=10.0,
        service_usage_client=mock_su,
    )
    mock_su.batch_enable_services.assert_called_once()
    req = mock_su.batch_enable_services.call_args.kwargs["request"]
    assert req.parent == "projects/test-proj"
    assert list(req.service_ids) == [
        "serviceusage.googleapis.com",
        "cloudresourcemanager.googleapis.com",
        "iam.googleapis.com",
    ]
    mock_op.result.assert_called_once_with(timeout=10.0)

    # SDK permission error propagates cleanly
    mock_op.result.side_effect = gcp_exceptions.PermissionDenied("denied")
    with pytest.raises(gcp_exceptions.PermissionDenied, match="denied"):
        ensure_bootstrap_apis(project_id="test-proj", service_usage_client=mock_su)


def test_bootstrap_cli_defaults_and_main_error_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("EDGAR_GCP_PROJECT_ID", "EDGAR_TFSTATE_BUCKET", "EDGAR_GCP_REGION"):
        monkeypatch.delenv(key, raising=False)
    args = parse_args([])
    assert args.project_id == "iron-radio-514217-q7"
    assert args.bucket == "iron-radio-514217-q7-tfstate"
    assert args.location == "us-central1"

    monkeypatch.setattr(
        bootstrap_mod,
        "ensure_bootstrap_apis",
        lambda **_kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert main(["--project-id", "test-proj"]) == 1
