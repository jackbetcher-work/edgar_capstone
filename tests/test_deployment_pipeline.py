"""Automated unit tests for the GitHub push CI/CD deployment pipeline and Secret Manager integration."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import httpx
import pytest
import yaml

from edgar_capstone.security import (
    MANAGED_SECRETS,
    RedactedSecret,
    SecretManagerClient,
    scan_repository_for_exposed_secrets,
    scan_text_for_exposed_secrets,
)
from edgar_capstone.server import app
from scripts.verify_no_secrets import main as verify_no_secrets_main

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_github_actions_pipeline_stages_and_triggers() -> None:
    workflow_path = REPO_ROOT / ".github" / "workflows" / "deploy.yml"
    assert workflow_path.exists(), "GitHub Actions deployment workflow must exist"

    raw_text = workflow_path.read_text(encoding="utf-8")
    workflow = yaml.safe_load(raw_text)

    # PyYAML 1.1 parses unquoted `on:` as boolean True
    triggers = workflow.get("on") or workflow.get(True)
    assert triggers is not None, "Workflow must define an `on` trigger block"
    assert "push" in triggers, "Workflow must trigger on each push to GitHub"

    # Least-privilege OIDC token permissions for keyless Workload Identity Federation
    permissions = workflow.get("permissions", {})
    assert permissions.get("id-token") == "write"
    assert permissions.get("contents") == "read"

    # Must be composed of the test, build, and deploy stages
    jobs = workflow.get("jobs", {})
    assert list(jobs.keys()) == ["test", "build", "deploy"]

    # Stage 1: test
    test_job = jobs["test"]
    test_steps_str = json.dumps(test_job["steps"])
    assert "verify_no_secrets.py" in test_steps_str
    assert "pytest" in test_steps_str
    assert "terraform fmt -check" in test_steps_str

    # Stage 2: build (depends on test, uses keyless WIF + Secret Manager)
    build_job = jobs["build"]
    assert build_job.get("needs") == "test"
    build_uses = [step.get("uses", "") for step in build_job["steps"]]
    assert any("google-github-actions/auth@" in u for u in build_uses)
    assert any("google-github-actions/get-secretmanager-secrets@" in u for u in build_uses)
    build_steps_str = json.dumps(build_job["steps"])
    assert "docker build" in build_steps_str
    assert "docker push" in build_steps_str

    # Stage 3: deploy (depends on build, uses keyless WIF + Secret Manager + Cloud Run)
    deploy_job = jobs["deploy"]
    assert deploy_job.get("needs") == "build"
    deploy_uses = [step.get("uses", "") for step in deploy_job["steps"]]
    assert any("google-github-actions/auth@" in u for u in deploy_uses)
    assert any("google-github-actions/get-secretmanager-secrets@" in u for u in deploy_uses)
    assert any("google-github-actions/deploy-cloudrun@" in u for u in deploy_uses)

    deploy_cloudrun_step = next(
        step
        for step in deploy_job["steps"]
        if "google-github-actions/deploy-cloudrun@" in step.get("uses", "")
    )
    secrets_mount = deploy_cloudrun_step["with"]["secrets"]
    for spec in MANAGED_SECRETS:
        assert spec.cloud_run_mount_spec() in secrets_mount


def test_cloudbuild_pipeline_stages_and_secret_manager() -> None:
    cb_path = REPO_ROOT / "cloudbuild.yaml"
    assert cb_path.exists(), "cloudbuild.yaml must exist"

    cb = yaml.safe_load(cb_path.read_text(encoding="utf-8"))
    steps = cb.get("steps", [])
    step_ids = [s.get("id") for s in steps]
    assert step_ids == ["test", "build", "deploy"]

    build_step = next(s for s in steps if s["id"] == "build")
    assert build_step["waitFor"] == ["test"]
    assert "SEC_EDGAR_USER_AGENT" in build_step["secretEnv"]

    deploy_step = next(s for s in steps if s["id"] == "deploy")
    assert deploy_step["waitFor"] == ["build"]
    assert set(deploy_step["secretEnv"]) == {
        "SEC_EDGAR_USER_AGENT",
        "GEMINI_API_KEY",
        "INTERNAL_IAP_AUDIENCE",
    }
    deploy_cmd = " ".join(deploy_step["args"])
    assert "--set-secrets=" in deploy_cmd

    secret_mgr_entries = cb.get("availableSecrets", {}).get("secretManager", [])
    envs_bound = {entry["env"] for entry in secret_mgr_entries}
    assert envs_bound == {
        "SEC_EDGAR_USER_AGENT",
        "GEMINI_API_KEY",
        "INTERNAL_IAP_AUDIENCE",
    }


def test_no_secrets_or_keys_exposed_in_repository() -> None:
    findings = scan_repository_for_exposed_secrets(REPO_ROOT)
    assert findings == [], f"Unexpected exposed secrets found: {findings}"

    exit_code = verify_no_secrets_main(["--repo-root", str(REPO_ROOT)])
    assert exit_code == 0


def test_secret_scanner_detects_synthetic_leaks() -> None:
    # Build synthetic leaked strings dynamically so this test file itself stays clean
    fake_google_key = "AIza" + "SyD9_synthetic_test_key_0123456789abc"
    fake_gh_pat = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
    fake_pem = "-----BEGIN " + "RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...\n"
    fake_hardcoded_assign = 'GEMINI_API_KEY: "super_secret_production_credential_9999"'

    leaked_blob = "\n".join(
        [
            fake_google_key,
            fake_gh_pat,
            fake_pem,
            fake_hardcoded_assign,
        ]
    )
    findings = scan_text_for_exposed_secrets(leaked_blob, source_name="synthetic.yml")
    rule_ids = {f.rule_id for f in findings}
    assert "google_api_key" in rule_ids
    assert "github_pat_token" in rule_ids
    assert "private_key_block" in rule_ids
    assert "hardcoded_secret_assignment" in rule_ids


def test_secret_manager_client_retrieval_and_redaction(monkeypatch: pytest.MonkeyPatch) -> None:
    for spec in MANAGED_SECRETS:
        monkeypatch.delenv(spec.env_var, raising=False)

    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        raw_secret = "synthetic-secret-payload-from-gcp"
        encoded = base64.b64encode(raw_secret.encode("utf-8")).decode("ascii")
        return httpx.Response(
            200,
            json={
                "name": str(request.url),
                "payload": {"data": encoded},
            },
        )

    mock_creds = MagicMock()
    mock_creds.valid = True
    mock_creds.token = "short-lived-wif-oauth-token"

    client = SecretManagerClient(
        project_id="iron-radio-514217-q7",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        credentials=mock_creds,
    )

    secret = client.get_secret("sec-edgar-gemini-api-key")
    assert isinstance(secret, RedactedSecret)
    assert str(secret) == "***REDACTED***"
    assert "synthetic-secret-payload-from-gcp" not in repr(secret)
    assert secret.get_secret_value() == "synthetic-secret-payload-from-gcp"
    assert len(requested_urls) == 1
    assert (
        "projects/iron-radio-514217-q7/secrets/sec-edgar-gemini-api-key/versions/latest:access"
        in requested_urls[0]
    )

    # Second call hits in-memory TTL cache without making another network call
    cached_secret = client.get_secret("sec-edgar-gemini-api-key")
    assert cached_secret == secret
    assert len(requested_urls) == 1

    # Load all managed secrets
    all_loaded = client.load_managed_secrets()
    assert set(all_loaded.keys()) == {
        "SEC_EDGAR_USER_AGENT",
        "GEMINI_API_KEY",
        "INTERNAL_IAP_AUDIENCE",
    }


def test_dockerfile_and_terraform_secret_manager_definitions() -> None:
    dockerfile_text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.11-slim AS builder" in dockerfile_text
    assert "USER appuser" in dockerfile_text
    assert "edgar_capstone.server:app" in dockerfile_text

    cicd_tf = (REPO_ROOT / "terraform" / "cicd_and_secrets.tf").read_text(encoding="utf-8")
    assert 'resource "google_secret_manager_secret" "managed_secrets"' in cicd_tf
    assert 'resource "google_secret_manager_secret_iam_member" "runtime_secret_accessor"' in cicd_tf
    assert 'resource "google_secret_manager_secret_iam_member" "cicd_secret_accessor"' in cicd_tf
    assert 'resource "google_iam_workload_identity_pool" "github_pool"' in cicd_tf
    assert 'resource "google_iam_workload_identity_pool_provider" "github_provider"' in cicd_tf
    assert 'resource "google_cloud_run_v2_service" "sec_edgar_analyst"' in cicd_tf
    assert "secret_key_ref" in cicd_tf


def test_fastapi_cloud_run_service_endpoints() -> None:
    client = TestClient(app)

    health_resp = client.get("/healthz")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "ok"
    assert health_resp.json()["root_agent"] == "RootOrchestrator"

    secrets_resp = client.get("/api/v1/secrets/status")
    assert secrets_resp.status_code == 200
    secrets_body = secrets_resp.json()
    assert secrets_body["secret_manager_enabled"] is True
    assert all(s["value"] == "***REDACTED***" for s in secrets_body["secrets"])

    var_resp = client.post(
        "/api/v1/variances",
        json={"tickers": "NVDA,AAPL", "start_year": 2024, "end_year": 2026},
    )
    assert var_resp.status_code == 200
    assert var_resp.json()["company_count"] == 2

    intent_resp = client.post(
        "/api/v1/intent",
        json={"query": "Compare NVIDIA and Apple's R&D spend from 2024-2026"},
    )
    assert intent_resp.status_code == 200
    assert intent_resp.json()["intent"] == "multi_company_peer_comparison"
