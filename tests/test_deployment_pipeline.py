"""Automated unit tests for the GitHub push CI/CD deployment pipeline and Secret Manager integration."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from edgar_capstone.security import (
    MANAGED_SECRETS,
    RedactedSecret,
    SecretManagerClient,
    issue_local_dev_workspace_sso_token,
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

    # Must be composed of the test, build, eval, and deploy stages
    jobs = workflow.get("jobs", {})
    assert list(jobs.keys()) == ["test", "build", "eval", "deploy"]

    # Stage 1: test (backend + frontend lint, unit tests, and production bundle build)
    test_job = jobs["test"]
    test_uses = [step.get("uses", "") for step in test_job["steps"]]
    assert any("actions/setup-node@" in u for u in test_uses)
    test_steps_str = json.dumps(test_job["steps"])
    assert "verify_no_secrets.py" in test_steps_str
    assert "pytest" in test_steps_str
    assert "--cov-fail-under=80" in test_steps_str

    # pytest reads live GCP data, so keyless WIF auth must run before it with the live env.
    test_step_names = [step.get("name", "") for step in test_job["steps"]]
    auth_idx = next(
        i
        for i, step in enumerate(test_job["steps"])
        if "google-github-actions/auth@" in step.get("uses", "")
    )
    pytest_idx = next(
        i for i, step in enumerate(test_job["steps"]) if "pytest" in step.get("run", "")
    )
    assert auth_idx < pytest_idx, test_step_names
    auth_with = test_job["steps"][auth_idx]["with"]
    assert auth_with["workload_identity_provider"] == "${{ vars.GCP_WORKLOAD_IDENTITY_PROVIDER }}"
    assert auth_with["service_account"] == "${{ vars.GCP_CICD_SERVICE_ACCOUNT }}"
    pytest_env = test_job["steps"][pytest_idx]["env"]
    assert workflow["env"]["GCP_PROJECT_ID"] == "iron-radio-514217-q7"
    assert pytest_env["GOOGLE_CLOUD_PROJECT"] == "${{ env.GCP_PROJECT_ID }}"
    assert pytest_env["GOOGLE_CLOUD_LOCATION"] == "${{ env.GCP_REGION }}"
    assert pytest_env["GOOGLE_GENAI_USE_VERTEXAI"] == "true"
    assert pytest_env["EDGAR_GEMINI_LOCATION"] == "global"
    assert pytest_env["EDGAR_TELEMETRY_BACKEND"] == "local"
    assert pytest_env["EDGAR_FEEDBACK_BACKEND"] == "local"
    assert "terraform fmt -check" in test_steps_str
    assert "terraform validate" in test_steps_str
    assert "ruff check" in test_steps_str
    assert "ruff format --check" in test_steps_str
    assert "npm --prefix frontend ci" in test_steps_str
    assert "npm --prefix frontend run lint" in test_steps_str
    assert "npm --prefix frontend test" in test_steps_str
    assert "npm --prefix frontend run build" in test_steps_str
    assert "test -f frontend/dist/index.html" in test_steps_str

    # Stage 2: build (depends on test, uses keyless WIF + Secret Manager + frontend container check)
    build_job = jobs["build"]
    assert build_job.get("needs") == "test"
    # Cloud Build is the primary deploy path; GitHub Actions deploys only when opted in
    for gated_job in (jobs["build"], jobs["eval"], jobs["deploy"]):
        assert "refs/heads/main" in gated_job["if"]
        assert "vars.DEPLOY_WITH_GITHUB_ACTIONS == 'true'" in gated_job["if"]
    build_uses = [step.get("uses", "") for step in build_job["steps"]]
    assert any("google-github-actions/auth@" in u for u in build_uses)
    assert any("google-github-actions/get-secretmanager-secrets@" in u for u in build_uses)
    build_steps_str = json.dumps(build_job["steps"])
    assert "docker build" in build_steps_str
    assert "/app/frontend/dist/index.html" in build_steps_str
    assert "-m scripts.container_smoke_test --in-process --live-data" in build_steps_str
    assert "docker push" in build_steps_str
    build_image_step = next(s for s in build_job["steps"] if s.get("id") == "build_image")
    build_run = build_image_step["run"]
    assert '--user "$(id -u):$(id -g)"' in build_run
    assert "-e HOME=/tmp" in build_run
    assert (
        'test -n "${GOOGLE_APPLICATION_CREDENTIALS}" && test -f "${GOOGLE_APPLICATION_CREDENTIALS}"'
        in build_run
    )
    assert '-v "${GOOGLE_APPLICATION_CREDENTIALS}:/tmp/adc.json:ro"' in build_run
    assert "-e GOOGLE_APPLICATION_CREDENTIALS=/tmp/adc.json" in build_run
    assert "ADC_DIR" not in build_run

    # Stage 2.5: eval (depends on build, runs live golden eval regression gates before deploy)
    eval_job = jobs["eval"]
    assert eval_job.get("needs") == "build"
    eval_steps_str = json.dumps(eval_job["steps"])
    assert "scripts/run_evals.py --write-bigquery" in eval_steps_str

    # Stage 3: deploy (depends on build and eval, uses keyless WIF + Secret Manager + Cloud Run)
    deploy_job = jobs["deploy"]
    assert deploy_job.get("needs") == ["build", "eval"]
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

    deploy_env_vars = deploy_cloudrun_step["with"]["env_vars"]
    assert "GOOGLE_GENAI_USE_VERTEXAI=true" in deploy_env_vars
    assert "EDGAR_GEMINI_LOCATION=global" in deploy_env_vars
    assert "SDP_INSPECT_TEMPLATE_ID=sec-edgar-sdp-inspect" in deploy_env_vars
    assert "SDP_DEIDENTIFY_TEMPLATE_ID=sec-edgar-sdp-deidentify" in deploy_env_vars
    for removed in ("EDGAR_EXECUTION_MODE", "EDGAR_DATA_BACKEND"):
        assert removed not in raw_text

    deploy_steps_str = json.dumps(deploy_job["steps"])
    assert "${SERVICE_URL}/health" in deploy_steps_str
    assert "${SERVICE_URL}/healthz" not in deploy_steps_str

    # Cloud Run's ingress / IAP make run.app unreachable from GitHub runners, so health
    # is gated in-platform: a GET /health startup probe plus a revision/traffic check.
    assert "--startup-probe=httpGet.path=/health," in deploy_cloudrun_step["with"]["flags"]
    verify_step = next(s for s in deploy_job["steps"] if s.get("id") == "verify_revision")
    assert "latestReadyRevisionName" in verify_step["run"]
    assert "iapEnabled" in verify_step["run"]
    for step in deploy_job["steps"]:
        if "curl" in step.get("run", "") and step.get("id") != "verify_revision":
            assert step.get("if") == "steps.verify_revision.outputs.public_http == 'true'"


def test_cloudbuild_pipeline_stages_and_secret_manager() -> None:
    cb_path = REPO_ROOT / "cloudbuild.yaml"
    assert cb_path.exists(), "cloudbuild.yaml must exist"

    cb = yaml.safe_load(cb_path.read_text(encoding="utf-8"))
    steps = cb.get("steps", [])
    step_ids = [s.get("id") for s in steps]
    gate_ids = ["lint-python", "lint-terraform", "frontend", "test"]
    assert step_ids == [*gate_ids, "build", "eval", "deploy"]

    # Quality gates start immediately and run in parallel
    steps_by_id = {s["id"]: s for s in steps}
    for gate_id in gate_ids:
        assert steps_by_id[gate_id]["waitFor"] == ["-"]

    lint_py_cmd = " ".join(steps_by_id["lint-python"]["args"])
    assert "ruff check" in lint_py_cmd
    assert "ruff format --check" in lint_py_cmd
    lint_tf_cmd = " ".join(steps_by_id["lint-terraform"]["args"])
    assert "terraform fmt -check -recursive" in lint_tf_cmd
    assert "terraform init -backend=false" in lint_tf_cmd
    assert "terraform validate" in lint_tf_cmd
    frontend_cmd = " ".join(steps_by_id["frontend"]["args"])
    assert "npm --prefix frontend run lint" in frontend_cmd
    assert "npm --prefix frontend test" in frontend_cmd
    test_cmd = " ".join(steps_by_id["test"]["args"])
    assert "verify_no_secrets.py" in test_cmd
    assert "pytest" in test_cmd
    assert "--cov-fail-under=80" in test_cmd
    # pytest reads live GCP with the build SA's ADC; telemetry/feedback stay in memory.
    test_env = set(steps_by_id["test"]["env"])
    assert {
        "EDGAR_TEST_PROJECT_ID=${PROJECT_ID}",
        "GCP_PROJECT_ID=${PROJECT_ID}",
        "GOOGLE_CLOUD_PROJECT=${PROJECT_ID}",
        "GOOGLE_CLOUD_LOCATION=${_REGION}",
        "GOOGLE_GENAI_USE_VERTEXAI=true",
        "EDGAR_GEMINI_LOCATION=${_GEMINI_LOCATION}",
        "EDGAR_TELEMETRY_BACKEND=local",
        "EDGAR_FEEDBACK_BACKEND=local",
    } <= test_env

    # Pinned ruff version in CI matches the dev dependency pin
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'"ruff=={cb["substitutions"]["_RUFF_VERSION"]}"' in pyproject_text

    build_step = steps_by_id["build"]
    assert build_step["waitFor"] == gate_ids
    assert "SEC_EDGAR_USER_AGENT" in build_step["secretEnv"]
    build_cmd = " ".join(build_step["args"])
    assert "DOCKER_BUILDKIT=1" in build_cmd
    assert "/app/frontend/dist/index.html" in build_cmd
    assert "--network=cloudbuild" in build_cmd
    assert "-m scripts.container_smoke_test --in-process --live-data" in build_cmd

    eval_step = steps_by_id["eval"]
    assert eval_step["waitFor"] == ["build"]
    eval_cmd = " ".join(eval_step["args"])
    assert "scripts/run_evals.py --write-bigquery" in eval_cmd
    assert eval_cmd.index('"${_PIPELINE_MODE}" != "deploy"') < eval_cmd.index("run_evals.py")

    deploy_step = steps_by_id["deploy"]
    assert deploy_step["waitFor"] == ["build", "eval"]
    assert set(deploy_step["secretEnv"]) == {
        "SEC_EDGAR_USER_AGENT",
        "GEMINI_API_KEY",
        "INTERNAL_IAP_AUDIENCE",
    }
    deploy_cmd = " ".join(deploy_step["args"])
    assert "--set-secrets=" in deploy_cmd
    assert "GOOGLE_GENAI_USE_VERTEXAI=true" in deploy_cmd
    assert "EDGAR_GEMINI_LOCATION=${_GEMINI_LOCATION}" in deploy_cmd
    assert "SDP_INSPECT_TEMPLATE_ID=sec-edgar-sdp-inspect" in deploy_cmd
    assert "SDP_DEIDENTIFY_TEMPLATE_ID=sec-edgar-sdp-deidentify" in deploy_cmd
    for removed in ("EDGAR_EXECUTION_MODE", "EDGAR_DATA_BACKEND", "--expect-corpus"):
        assert removed not in cb_path.read_text(encoding="utf-8")

    # PR (verify) builds must never push images or deploy
    assert cb["substitutions"]["_PIPELINE_MODE"] == "deploy"
    assert "images" not in cb, "top-level images: would push even in verify mode"
    assert build_cmd.index('"${_PIPELINE_MODE}" != "deploy"') < build_cmd.index("docker push")
    assert deploy_cmd.index('"${_PIPELINE_MODE}" != "deploy"') < deploy_cmd.index(
        "gcloud run deploy"
    )

    secret_mgr_entries = cb.get("availableSecrets", {}).get("secretManager", [])
    envs_bound = {entry["env"] for entry in secret_mgr_entries}
    assert envs_bound == {
        "SEC_EDGAR_USER_AGENT",
        "GEMINI_API_KEY",
        "INTERNAL_IAP_AUDIENCE",
    }


def test_cloudbuild_triggers_managed_in_terraform() -> None:
    cb_tf = (REPO_ROOT / "terraform" / "cloudbuild.tf").read_text(encoding="utf-8")
    assert 'resource "google_cloudbuildv2_connection" "github"' in cb_tf
    assert 'resource "google_cloudbuildv2_repository" "edgar_capstone"' in cb_tf
    assert 'resource "google_cloudbuild_trigger" "main_deploy"' in cb_tf
    assert 'resource "google_cloudbuild_trigger" "pull_request_verify"' in cb_tf
    assert 'filename        = "cloudbuild.yaml"' in cb_tf
    assert '_PIPELINE_MODE = "deploy"' in cb_tf
    assert '_PIPELINE_MODE = "verify"' in cb_tf
    assert "COMMENTS_ENABLED_FOR_EXTERNAL_CONTRIBUTORS_ONLY" in cb_tf

    # Dedicated least-privilege build identity (no project-wide owner/editor/admin roles)
    assert 'resource "google_service_account" "cloudbuild_runner"' in cb_tf
    assert "service_account = google_service_account.cloudbuild_runner[0].id" in cb_tf
    assert "roles/run.developer" in cb_tf
    assert "roles/run.admin" not in cb_tf
    assert "roles/editor" not in cb_tf
    assert "roles/owner" not in cb_tf
    assert 'resource "google_artifact_registry_repository_iam_member" "cloudbuild_repo_writer"' in (
        cb_tf
    )

    # GitHub token is referenced by Secret Manager version name only, never read into state
    assert 'data "google_secret_manager_secret_version"' not in cb_tf
    assert "oauth_token_secret_version = local.cloudbuild_oauth_token_secret_version" in cb_tf

    variables_tf = (REPO_ROOT / "terraform" / "variables.tf").read_text(encoding="utf-8")
    for var_name in (
        "enable_cloudbuild_trigger",
        "enable_cloudbuild_pr_trigger",
        "create_cloudbuild_github_connection",
        "cloudbuild_github_app_installation_id",
        "cloudbuild_github_oauth_token_secret_id",
        "cloudbuild_trigger_branch_regex",
        "cloudbuild_service_account_id",
    ):
        assert f'variable "{var_name}"' in variables_tf


def test_ruff_lint_and_format_gate_configured() -> None:
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.ruff]" in pyproject_text
    assert "[tool.ruff.lint]" in pyproject_text
    assert "[tool.ruff.format]" in pyproject_text

    makefile_text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "\nlint:" in makefile_text
    assert "\nformat:" in makefile_text
    assert "ruff format --check" in makefile_text


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
    assert "FROM node:22-slim AS frontend-builder" in dockerfile_text
    assert "npm ci --no-audit --no-fund" in dockerfile_text
    assert "npm run build" in dockerfile_text
    assert "COPY --from=frontend-builder" in dockerfile_text
    assert "FROM python:3.11-slim AS builder" in dockerfile_text
    assert "USER appuser" in dockerfile_text
    assert "edgar_capstone.server:app" in dockerfile_text
    # No SEC corpus or build-time corpus check is baked into the image.
    for removed in ("curated_dataset", "corpus_hydration", "EDGAR_CORPUS_HYDRATION", "data/"):
        assert removed not in dockerfile_text, removed

    cicd_tf = (REPO_ROOT / "terraform" / "cicd_and_secrets.tf").read_text(encoding="utf-8")
    assert 'resource "google_secret_manager_secret" "managed_secrets"' in cicd_tf
    assert 'resource "google_secret_manager_secret_iam_member" "runtime_secret_accessor"' in cicd_tf
    assert 'resource "google_secret_manager_secret_iam_member" "cicd_secret_accessor"' in cicd_tf
    assert 'resource "google_iam_workload_identity_pool" "github_pool"' in cicd_tf
    assert 'resource "google_iam_workload_identity_pool_provider" "github_provider"' in cicd_tf
    assert 'resource "google_cloud_run_v2_service" "sec_edgar_analyst"' in cicd_tf
    assert "image = var.cloud_run_initial_image" in cicd_tf
    assert "template[0].containers[0].image" in cicd_tf
    assert "secret_key_ref" in cicd_tf
    assert "EDGAR_EXECUTION_MODE" not in cicd_tf
    assert "EDGAR_DATA_BACKEND" not in cicd_tf

    iap_tf = (REPO_ROOT / "terraform" / "iap_and_sso.tf").read_text(encoding="utf-8")
    assert 'resource "google_compute_backend_service" "sec_edgar_iap_backend"' in iap_tf
    assert "roles/iap.httpsResourceAccessor" in iap_tf
    assert "roles/run.invoker" in iap_tf


def test_cloud_run_direct_iap_internal_sharing_terraform() -> None:
    """Default access: IAP on the Cloud Run service itself, shared with an org domain/group."""
    tf_dir = REPO_ROOT / "terraform"
    cicd_tf = (tf_dir / "cicd_and_secrets.tf").read_text(encoding="utf-8")
    iap_tf = (tf_dir / "iap_and_sso.tf").read_text(encoding="utf-8")
    lb_tf = (tf_dir / "load_balancer.tf").read_text(encoding="utf-8")
    vars_tf = (tf_dir / "variables.tf").read_text(encoding="utf-8")
    outputs_tf = (tf_dir / "outputs.tf").read_text(encoding="utf-8")

    assert 'default     = "iap_direct"' in vars_tf
    assert 'contains(["iap_direct", "load_balancer"], var.cloud_run_access_mode)' in vars_tf

    # Cloud Run: IAP on the service; ingress opens only when IAP is on, else LB-only.
    assert "provider = google-beta" in cicd_tf
    assert "iap_enabled         = local.cloud_run_direct_iap" in cicd_tf
    assert (
        'ingress             = local.cloud_run_direct_iap ? "INGRESS_TRAFFIC_ALL" : '
        '"INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"'
    ) in cicd_tf
    assert 'name  = "IAP_ASSERTION_DEFAULT_ROLE"' in cicd_tf

    # IAP audience matches the Cloud Run IAP JWT `aud` format in direct mode.
    assert "/locations/${var.region}/services/${var.cloud_run_service_name}" in iap_tf
    assert 'resource "google_iap_web_cloud_run_service_iam_member"' in iap_tf
    assert "local.cloud_run_direct_iap ? local.effective_iap_group_principals" in iap_tf

    # No public LB unless load_balancer mode is chosen; when it is, it carries a TLS policy.
    assert "enable_https_lb = var.enable_https_load_balancer && local.cloud_run_lb_iap" in lb_tf
    assert 'profile         = "RESTRICTED"' in lb_tf
    assert "ssl_policy       = google_compute_ssl_policy.sec_edgar_tls[0].id" in lb_tf

    # Sharing is by group or whole domain, never by individual user.
    assert 'startswith(lower(trimspace(p)), "domain:")' in vars_tf
    assert "google_cloud_run_v2_service.sec_edgar_analyst[0].uri" in outputs_tf

    # CI deploys must keep the IAP role in the env vars they pass explicitly.
    cloudbuild = (REPO_ROOT / "cloudbuild.yaml").read_text(encoding="utf-8")
    deploy_yml = (REPO_ROOT / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")
    assert "IAP_ASSERTION_DEFAULT_ROLE=${_IAP_DEFAULT_ROLE}" in cloudbuild
    assert "IAP_ASSERTION_DEFAULT_ROLE=analyst" in deploy_yml


def test_vpc_service_controls_perimeter_terraform() -> None:
    tf_dir = REPO_ROOT / "terraform"
    vpcsc_tf = (tf_dir / "vpc_service_controls.tf").read_text(encoding="utf-8")
    vars_tf = (tf_dir / "variables.tf").read_text(encoding="utf-8")
    apis_tf = (tf_dir / "apis.tf").read_text(encoding="utf-8")

    assert '"accesscontextmanager.googleapis.com"' in apis_tf
    assert 'resource "google_access_context_manager_service_perimeter" "sec_edgar"' in vpcsc_tf
    assert 'resource "google_access_context_manager_access_policy" "sec_edgar_scoped"' in vpcsc_tf
    assert "scopes = [local.vpc_sc_project_resource]" in vpcsc_tf
    assert 'perimeter_type = "PERIMETER_TYPE_REGULAR"' in vpcsc_tf

    # Dry run first: the config goes to `spec` unless enforcement is switched on.
    assert "use_explicit_dry_run_spec = !var.vpc_sc_enforce" in vpcsc_tf
    assert 'dynamic "spec"' in vpcsc_tf and 'dynamic "status"' in vpcsc_tf
    enforce_block = vars_tf.split('variable "vpc_sc_enforce"', 1)[1].split("}", 1)[0]
    assert "default     = false" in enforce_block

    # Every service named by the reference architecture is restricted.
    restricted_block = vars_tf.split('variable "vpc_sc_restricted_services"', 1)[1].split("]", 1)[0]
    for svc in (
        "aiplatform",
        "artifactregistry",
        "bigquery",
        "discoveryengine",
        "dlp",
        "firestore",
        "modelarmor",
        "run",
        "secretmanager",
        "storage",
    ):
        assert f'"{svc}.googleapis.com"' in restricted_block

    # Ingress: identity-based for operators/CI/runtime/log sink; ANY_IDENTITY only for
    # Cloud Run invocation (IAP gates users), as VPC-SC can't see IAM on invocations.
    assert "google_service_account.cicd_runner.email" in vpcsc_tf
    assert "local.runtime_service_account_email" in vpcsc_tf
    assert "writer_identity" in vpcsc_tf
    assert vpcsc_tf.count('identity_type = "ANY_IDENTITY"') == 2
    assert vpcsc_tf.count('service_name = "run.googleapis.com"') == 2
    assert "precondition" in vpcsc_tf


def test_frontend_lockfile_uses_public_npm_registry() -> None:
    pkg_json = json.loads((REPO_ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
    lock_json = json.loads(
        (REPO_ROOT / "frontend" / "package-lock.json").read_text(encoding="utf-8")
    )
    packages = lock_json.get("packages", {})
    assert "" in packages
    assert packages[""].get("dependencies") == pkg_json.get("dependencies")
    assert packages[""].get("devDependencies") == pkg_json.get("devDependencies")

    resolved_urls = [
        meta["resolved"] for key, meta in packages.items() if key != "" and "resolved" in meta
    ]
    assert len(resolved_urls) > 0
    bad_urls = [u for u in resolved_urls if not u.startswith("https://registry.npmjs.org/")]
    assert bad_urls == [], f"Lockfile contains non-public npm registry URLs: {bad_urls[:5]}"


def test_fastapi_cloud_run_service_endpoints() -> None:
    from edgar_capstone.agents.intent import resolve_turn_intent
    from edgar_capstone.agents.tools import calculate_period_variances

    client = TestClient(app)

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    health_body = health_resp.json()
    assert health_body["status"] == "ok"
    assert health_body["root_agent"] == "FinancialAnalystAgent"
    assert health_body["active_agents"] == ["FinancialAnalystAgent", "NarrativeEvaluatorAgent"]
    assert health_body["sub_agents"] == ["FinancialAnalystAgent", "NarrativeEvaluatorAgent"]
    assert health_body["iap_jwt_auth_enabled"] is True

    # Pruned legacy healthz probe returns 404
    assert client.get("/healthz").status_code == 404

    # Internal APIs require IAP / Workspace Okta JWT Bearer authentication
    unauth_resp = client.get("/api/v1/secrets/status")
    assert unauth_resp.status_code == 401
    assert "Bearer" in unauth_resp.headers.get("WWW-Authenticate", "")

    bearer_jwt, _ = issue_local_dev_workspace_sso_token()
    auth_headers = {"Authorization": f"Bearer {bearer_jwt}"}

    secrets_resp = client.get("/api/v1/secrets/status", headers=auth_headers)
    assert secrets_resp.status_code == 200
    secrets_body = secrets_resp.json()
    assert secrets_body["secret_manager_enabled"] is True
    assert all(s["value"] == "***REDACTED***" for s in secrets_body["secrets"])

    # Retained filing catalog and section viewer endpoints
    companies_resp = client.get("/api/v1/companies", headers=auth_headers)
    assert companies_resp.status_code == 200
    companies_body = companies_resp.json()
    assert companies_body["status"] == "success"
    assert companies_body["company_count"] >= 15

    section_resp = client.get(
        "/api/v1/section",
        headers=auth_headers,
        params={"ticker": "NVDA", "fiscal_year": 2026, "section_id": "item_7", "max_chars": 2000},
    )
    assert section_resp.status_code == 200
    section_body = section_resp.json()
    assert section_body["status"] == "success"
    assert section_body["ticker"] == "NVDA"
    assert section_body["fiscal_year"] == 2026

    # Pruned compute routes are removed from the API (GET -> 404; POST -> 404/405 via SPA catch-all)
    assert client.get("/api/v1/variances", headers=auth_headers).status_code == 404
    assert client.get("/api/v1/intent", headers=auth_headers).status_code == 404
    var_resp = client.post(
        "/api/v1/variances",
        headers=auth_headers,
        json={"tickers": "NVDA,AAPL", "start_year": 2024, "end_year": 2026},
    )
    assert var_resp.status_code in {404, 405}

    intent_resp = client.post(
        "/api/v1/intent",
        headers=auth_headers,
        json={"query": "Compare NVIDIA and Apple's R&D spend from 2024-2026"},
    )
    assert intent_resp.status_code in {404, 405}

    schema_paths = app.openapi()["paths"]
    assert "/api/v1/variances" not in schema_paths
    assert "/api/v1/intent" not in schema_paths

    # Underlying Python tools remain directly callable and verified
    direct_var = calculate_period_variances(tickers="NVDA,AAPL", start_year=2024, end_year=2026)
    assert direct_var["status"] == "success"
    assert direct_var["company_count"] == 2

    direct_intent, source = resolve_turn_intent(
        agent_intent=None,
        scope={"tickers": ["NVDA", "AAPL"], "gics_sector": "ALL"},
        tool_calls=[{"name": "calculate_period_variances"}],
    )
    assert (direct_intent, source) == ("multi_company_peer_comparison", "tool_inference")


def test_adk_tools_read_live_gcp_data_in_a_clean_checkout() -> None:
    """Every ADK data tool answers from live GCP; no local corpus files are needed."""
    from edgar_capstone.agents.tools import (
        _SESSION_10K_CONTEXT_CACHE,
        calculate_period_variances,
        fetch_financial_metrics,
        load_cached_10k_section,
        query_looker_financial_explore,
        search_10k_chunks_rag,
    )

    _SESSION_10K_CONTEXT_CACHE.clear()

    metrics_res = fetch_financial_metrics(
        tickers="NVDA,AAPL",
        start_year=2024,
        end_year=2026,
        gics_sector="Information Technology",
    )
    assert metrics_res["status"] == "success"
    assert metrics_res["record_count"] == 6

    var_res = calculate_period_variances(
        tickers="NVDA,AAPL",
        start_year=2024,
        end_year=2026,
    )
    assert var_res["status"] == "success"
    assert var_res["company_count"] == 2

    health_res = query_looker_financial_explore(
        explore_name="chunk_health",
        tickers="ALL",
        gics_sector="ALL",
        start_year=2020,
        end_year=2026,
    )
    assert health_res["status"] == "success"
    assert health_res["health_status"] == "HEALTHY"
    assert health_res["total_chunks"] > 10000

    rag_res = search_10k_chunks_rag(
        query="H20 export control licensing China data center revenue",
        tickers="NVDA",
        gics_sector="Information Technology",
        section_id="item_1a",
        start_year=2025,
        end_year=2026,
        top_k=5,
    )
    assert rag_res["status"] == "success"
    assert rag_res["citation_count"] == 5
    assert rag_res["citations"][0]["ticker"] == "NVDA"
    assert rag_res["citations"][0]["section_id"] == "item_1a"

    sec_res = load_cached_10k_section(
        ticker="NVDA",
        fiscal_year=2026,
        section_id="item_7",
        max_chars=2500,
    )
    assert sec_res["status"] == "success"
    assert sec_res["token_estimate"] > 50


def test_ci_identities_have_least_privilege_live_read_access_terraform() -> None:
    tf_dir = REPO_ROOT / "terraform"
    ci_tf = (tf_dir / "ci_live_test_access.tf").read_text(encoding="utf-8")
    evals_tf = (tf_dir / "evals.tf").read_text(encoding="utf-8")
    vpcsc_tf = (tf_dir / "vpc_service_controls.tf").read_text(encoding="utf-8")

    # Both CI identities: GitHub Actions WIF SA and (when enabled) the Cloud Build SA.
    principals = ci_tf.split("ci_live_test_principals = merge(", 1)[1].split("\n  )", 1)[0]
    assert "google_service_account.cicd_runner.email" in principals
    assert "google_service_account.cloudbuild_runner[0].email" in principals
    assert "eval_ci_principals = local.ci_live_test_principals" in evals_tf

    expected = {
        "ci_live_test_bq_job_user": ("google_project_iam_member", "roles/bigquery.jobUser"),
        "ci_live_test_dataset_viewer": (
            "google_bigquery_dataset_iam_member",
            "roles/bigquery.dataViewer",
        ),
        "ci_live_test_corpus_viewer": (
            "google_storage_bucket_iam_member",
            "roles/storage.objectViewer",
        ),
        "ci_live_test_discoveryengine_viewer": (
            "google_project_iam_member",
            "roles/discoveryengine.viewer",
        ),
        "ci_live_test_vertex_user": ("google_project_iam_member", "roles/aiplatform.user"),
        "ci_live_test_modelarmor_user": ("google_project_iam_member", "roles/modelarmor.user"),
    }
    for name, (resource_type, role) in expected.items():
        header = f'resource "{resource_type}" "{name}" {{'
        assert header in ci_tf, header
        body = ci_tf.split(header, 1)[1].split("\n}", 1)[0]
        assert re.search(rf'\brole\s+=\s+"{re.escape(role)}"', body), (name, body)
        assert "for_each = local.ci_live_test_principals" in body
    assert "dataset_id = google_bigquery_dataset.sec_edgar.dataset_id" in ci_tf
    assert "bucket = local.sec_corpus_bucket_name" in ci_tf

    # Read-only: no admin/editor/write roles for the test identities.
    for forbidden in ("admin", "editor", "owner", "dataEditor", "objectAdmin", "objectCreator"):
        assert forbidden not in ci_tf.split("locals {", 1)[1], forbidden

    # Both CI identities are trusted VPC-SC ingress identities.
    assert '["serviceAccount:${google_service_account.cicd_runner.email}"]' in vpcsc_tf
    assert "google_service_account.cloudbuild_runner[0].email" in vpcsc_tf


def test_vector_search_private_service_connect_terraform() -> None:
    tf_dir = REPO_ROOT / "terraform"
    network_tf = (tf_dir / "network.tf").read_text(encoding="utf-8")
    vs_tf = (tf_dir / "vector_search.tf").read_text(encoding="utf-8")
    run_tf = (tf_dir / "cicd_and_secrets.tf").read_text(encoding="utf-8")
    vars_tf = (tf_dir / "variables.tf").read_text(encoding="utf-8")

    # PSC is the default; the endpoint has no public domain in that mode
    access_block = vars_tf.split('variable "vector_search_endpoint_access"', 1)[1].split("}", 1)[0]
    assert 'default     = "private_service_connect"' in access_block
    assert "public_endpoint_enabled = !local.vector_search_psc" in vs_tf
    assert "enable_private_service_connect = true" in vs_tf
    assert "project_allowlist              = [var.project_id]" in vs_tf

    # Consumer side: private subnet, internal IP, and a PSC forwarding rule to the
    # deployed index's service attachment
    assert 'resource "google_compute_network" "sec_edgar"' in network_tf
    assert "auto_create_subnetworks = false" in network_tf
    assert "private_ip_google_access = true" in network_tf
    assert 'address_type = "INTERNAL"' in network_tf
    assert "private_endpoints[0].service_attachment" in network_tf
    assert 'load_balancing_scheme = ""' in network_tf

    # Cloud Run reaches the PSC IP through Direct VPC egress and is told the address
    assert 'egress = "PRIVATE_RANGES_ONLY"' in run_tf
    assert "network_interfaces {" in run_tf
    assert '"VERTEX_VECTOR_SEARCH_PSC_ADDRESS"' in run_tf
    assert "google_compute_address.vector_search_psc[0].address" in run_tf

    # Cloud Build merges env vars so the Terraform-managed PSC address survives deploys
    cloudbuild = (REPO_ROOT / "cloudbuild.yaml").read_text(encoding="utf-8")
    assert "--update-env-vars=" in cloudbuild
    assert "--set-env-vars=" not in cloudbuild
