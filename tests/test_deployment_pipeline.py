"""Automated unit tests for the GitHub push CI/CD deployment pipeline, Terraform IaC, and secret scanner."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from edgar_capstone.security import (
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
    assert not (REPO_ROOT / "cloudbuild.yaml").exists(), "Legacy duplicate cloudbuild.yaml removed"

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
    assert "terraform fmt -check" in test_steps_str
    assert "terraform validate" in test_steps_str
    assert "ruff check" in test_steps_str
    assert "ruff format --check" in test_steps_str
    assert "npm --prefix frontend run lint" in test_steps_str
    assert "npm --prefix frontend test" in test_steps_str
    assert "npm --prefix frontend run build" in test_steps_str

    # Keyless WIF auth runs before live pytest
    auth_idx = next(
        i
        for i, step in enumerate(test_job["steps"])
        if "google-github-actions/auth@" in step.get("uses", "")
    )
    pytest_idx = next(
        i for i, step in enumerate(test_job["steps"]) if "pytest" in step.get("run", "")
    )
    assert auth_idx < pytest_idx

    # Stage 2: build (main pushes only, keyless WIF, container smoke test, push to GAR)
    build_job = jobs["build"]
    assert build_job.get("needs") == "test"
    for gated_job in (jobs["build"], jobs["eval"], jobs["deploy"]):
        assert "refs/heads/main" in gated_job["if"]
    build_steps_str = json.dumps(build_job["steps"])
    assert "docker build" in build_steps_str
    assert "/app/frontend/dist/index.html" in build_steps_str
    assert "-m scripts.container_smoke_test --in-process --live-data" in build_steps_str
    assert "docker push" in build_steps_str

    # Stage 2.5: optional eval gate (default skip)
    eval_input = triggers["workflow_dispatch"]["inputs"]["eval_mode"]
    assert eval_input["default"] == "skip"
    assert set(eval_input["options"]) == {"skip", "advisory", "blocking"}
    eval_job = jobs["eval"]
    assert eval_job.get("needs") == "build"
    assert "scripts/run_evals.py --write-bigquery" in json.dumps(eval_job["steps"])

    # Stage 3: deploy (image-only canary rollout + automated rollback; Terraform owns env vars)
    deploy_job = jobs["deploy"]
    assert deploy_job.get("needs") == ["build", "eval"]
    deploy_step = next(s for s in deploy_job["steps"] if s.get("id") == "deploy_cloud_run")
    assert "gcloud run services update" in deploy_step["run"]
    assert "--image" in deploy_step["run"]
    assert "--no-traffic" in deploy_step["run"]
    assert "--tag canary" in deploy_step["run"]
    assert "gcloud run services update-traffic" in deploy_step["run"]
    assert "--startup-probe" in deploy_step["run"]
    assert "httpGet.path=/health," in deploy_step["run"]
    verify_step = next(s for s in deploy_job["steps"] if s.get("id") == "verify_revision")
    assert "latestReadyRevisionName" in verify_step["run"]
    assert "iapEnabled" in verify_step["run"]
    assert "--to-revisions=" in verify_step["run"]
    rollback_step = next(s for s in deploy_job["steps"] if s.get("id") == "rollback_cloud_run")
    assert "failure()" in rollback_step["if"]
    assert "gcloud run services update-traffic" in rollback_step["run"]
    assert "${PREV_REV}=100" in rollback_step["run"]


def test_no_secrets_or_keys_exposed_in_repository() -> None:
    """Single full-repo scan via `verify_no_secrets` CLI entrypoint (avoiding duplicate walk)."""
    findings = scan_repository_for_exposed_secrets(REPO_ROOT)
    assert findings == [], f"Unexpected exposed secrets found: {findings}"


def test_verify_no_secrets_cli_passes_on_clean_dir_and_flags_leak(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "clean.py").write_text("X = 'no secrets here'\n", encoding="utf-8")
    assert verify_no_secrets_main(["--repo-root", str(tmp_path)]) == 0
    fake_key = "AIza" + "SyD9_synthetic_test_key_0123456789abc"
    (tmp_path / "src" / "leak.py").write_text(f"KEY = '{fake_key}'\n", encoding="utf-8")
    assert verify_no_secrets_main(["--repo-root", str(tmp_path)]) == 1


def test_secret_scanner_detects_synthetic_leaks() -> None:
    fake_google_key = "AIza" + "SyD9_synthetic_test_key_0123456789abc"
    fake_gh_pat = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
    fake_pem = "-----BEGIN " + "RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...\n"
    fake_hardcoded_assign = 'GEMINI_API_KEY: "super_secret_production_credential_9999"'

    leaked_blob = "\n".join([fake_google_key, fake_gh_pat, fake_pem, fake_hardcoded_assign])
    findings = scan_text_for_exposed_secrets(leaked_blob, source_name="synthetic.yml")
    rule_ids = {f.rule_id for f in findings}
    assert "google_api_key" in rule_ids
    assert "github_pat_token" in rule_ids
    assert "private_key_block" in rule_ids
    assert "hardcoded_secret_assignment" in rule_ids


def test_dockerfile_and_streamlined_terraform_definitions() -> None:
    dockerfile_text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"FROM\s+node:\S+\s+AS\s+frontend-builder", dockerfile_text)
    assert "npm ci" in dockerfile_text and "npm run build" in dockerfile_text
    assert "COPY --from=frontend-builder" in dockerfile_text
    assert re.search(r"FROM\s+python:\S+\s+AS\s+builder", dockerfile_text)
    assert "USER appuser" in dockerfile_text
    assert "edgar_capstone.server:app" in dockerfile_text
    assert "data/" not in dockerfile_text

    tf_dir = REPO_ROOT / "terraform"
    # Dead IaC modules removed
    for removed_tf in (
        "cloudbuild.tf",
        "network.tf",
        "load_balancer.tf",
        "vpc_service_controls.tf",
    ):
        assert not (tf_dir / removed_tf).exists(), f"{removed_tf} should be removed"

    cicd_tf = (tf_dir / "cicd_and_secrets.tf").read_text(encoding="utf-8")
    assert 'resource "google_iam_workload_identity_pool" "github_pool"' in cicd_tf
    assert 'resource "google_iam_workload_identity_pool_provider" "github_provider"' in cicd_tf
    assert 'resource "google_cloud_run_v2_service" "sec_edgar_analyst"' in cicd_tf
    assert re.search(r"image\s*=\s*var\.cloud_run_initial_image", cicd_tf)
    assert "roles/run.developer" in cicd_tf
    assert "roles/run.admin" not in cicd_tf
    assert "GOOGLE_GENAI_USE_VERTEXAI" in cicd_tf
    assert "INTERNAL_IAP_AUDIENCE" in cicd_tf
    assert "VERTEX_VECTOR_SEARCH" not in cicd_tf
    assert "GEMINI_API_KEY" not in cicd_tf
    assert "SEC_EDGAR_USER_AGENT" not in cicd_tf

    # Direct Cloud Run IAP only
    iap_tf = (tf_dir / "iap_and_sso.tf").read_text(encoding="utf-8")
    assert re.search(r"provider\s*=\s*google-beta", cicd_tf)
    assert re.search(r"iap_enabled\s*=\s*true", cicd_tf)
    assert "/locations/${var.region}/services/${var.cloud_run_service_name}" in iap_tf
    assert 'resource "google_iap_web_cloud_run_service_iam_member"' in iap_tf
    assert "roles/iap.httpsResourceAccessor" in iap_tf
    assert "roles/run.invoker" in iap_tf

    # Discovery Engine kept; Matching Engine removed
    vs_tf = (tf_dir / "vector_search.tf").read_text(encoding="utf-8")
    assert 'resource "google_discovery_engine_data_store" "sec_10k_corpus"' in vs_tf
    assert 'resource "google_discovery_engine_search_engine" "sec_10k_hybrid_search"' in vs_tf
    assert "google_vertex_ai_index" not in vs_tf


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


def test_fastapi_health_and_pruned_routes_offline(
    fake_gcp_adapters: object,
) -> None:
    """Verify `/health` and confirm legacy/unneeded routes are absent."""
    client = TestClient(app)

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    health_body = health_resp.json()
    assert health_body["status"] == "ok"
    assert health_body["root_agent"] == "FinancialAnalystAgent"
    assert health_body["active_agents"] == ["FinancialAnalystAgent", "NarrativeEvaluatorAgent"]
    assert health_body["iap_jwt_auth_enabled"] is True
    assert client.get("/healthz").status_code == 404

    bearer_jwt, _ = issue_local_dev_workspace_sso_token()
    auth_headers = {"Authorization": f"Bearer {bearer_jwt}"}

    schema_paths = app.openapi()["paths"]
    for removed in ("/api/v1/variances", "/api/v1/intent", "/api/v1/secrets/status"):
        assert removed not in schema_paths
        assert client.get(removed, headers=auth_headers).status_code == 404


def test_ci_identities_have_least_privilege_live_read_access_terraform() -> None:
    tf_dir = REPO_ROOT / "terraform"
    ci_tf = (tf_dir / "ci_live_test_access.tf").read_text(encoding="utf-8")
    evals_tf = (tf_dir / "evals.tf").read_text(encoding="utf-8")

    assert "google_service_account.cicd_runner.email" in ci_tf
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

    for forbidden in ("admin", "editor", "owner", "dataEditor", "objectAdmin", "objectCreator"):
        assert forbidden not in ci_tf.split("locals {", 1)[1], forbidden
