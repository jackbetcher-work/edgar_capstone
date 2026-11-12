"""Unit tests for the GCS Terraform remote state bucket bootstrap script."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from scripts.bootstrap_tfstate_bucket import ensure_tfstate_bucket, parse_args


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


def test_bootstrap_cli_defaults() -> None:
    args = parse_args([])
    assert args.project_id == "iron-radio-514217-q7"
    assert args.bucket == "iron-radio-514217-q7-tfstate"
    assert args.location == "us-central1"
