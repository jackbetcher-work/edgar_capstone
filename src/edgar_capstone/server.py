"""FastAPI Cloud Run service entrypoint for the SEC EDGAR Natural Language Analyst."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import time
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from edgar_capstone.agents.chat_engine import (
    list_curated_companies,
    run_conversational_rag_turn,
)
from edgar_capstone.agents.root_orchestrator import (
    RootOrchestrator,
    classify_query_intent,
)
from edgar_capstone.agents.tools import (
    _get_dataset_dir,
    _load_all_profiles_cached,
    calculate_period_variances,
    fetch_financial_metrics,
    load_cached_10k_section,
    query_looker_financial_explore,
    search_10k_chunks_rag,
)
from edgar_capstone.security.auth import (
    GOOGLE_IAP_ISSUER,
    AuthenticatedUser,
    AuthenticationError,
    AuthorizationError,
    IAPJwtVerifier,
    get_auth_verifier,
    get_security_audit_events,
    issue_local_dev_workspace_sso_token,
    record_security_audit_event,
    require_authenticated_user,
    require_roles,
    revoke_bearer_token,
)
from edgar_capstone.security.secrets import (
    DEFAULT_PROJECT_ID,
    MANAGED_SECRETS,
    SecretManagerClient,
)

logger = logging.getLogger(__name__)

_FEEDBACK_AUDIT_LOG: list[dict[str, Any]] = []

app = FastAPI(
    title="SEC EDGAR Natural Language Analyst API",
    version="0.2.0",
    description=(
        "Cloud Run FastAPI backend for deterministic SEC 10-K financial variance "
        "calculations, metadata-filtered Hybrid RAG search, and ADK RootOrchestrator routing. "
        "Internal APIs are protected via Identity-Aware Proxy (IAP) and Google Workspace "
        "Okta/SSO JWT Bearer authentication."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type", "X-Goog-IAP-JWT-Assertion"],
    expose_headers=["WWW-Authenticate"],
)


class VarianceRequest(BaseModel):
    tickers: str = Field(..., description="Comma-separated tickers, e.g. 'NVDA,AAPL'")
    start_year: int = Field(2024, ge=2015, le=2035)
    end_year: int = Field(2026, ge=2015, le=2035)


class RagSearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    tickers: str = Field("ALL")
    gics_sector: str = Field("ALL")
    section_id: str = Field("ALL")
    start_year: int = Field(2024, ge=2015, le=2035)
    end_year: int = Field(2026, ge=2015, le=2035)
    top_k: int = Field(5, ge=1, le=25)


class IntentClassifyRequest(BaseModel):
    query: str = Field(..., min_length=2)
    tickers: list[str] | None = None


class SsoTokenExchangeRequest(BaseModel):
    id_token: str = Field(..., min_length=16, description="OIDC / IAP signed JWT bearer token")


class DevWorkspaceSsoLoginRequest(BaseModel):
    email: str = Field(
        default="analyst@fde-workspace.corp",
        min_length=5,
        max_length=254,
        description="Google Workspace corporate email federated via Okta SSO",
    )
    name: str = Field(
        default="Senior Financial Analyst",
        min_length=1,
        max_length=120,
    )
    groups: list[str] = Field(
        default_factory=lambda: ["sec-edgar-analysts"],
        description="Cloud Identity / Okta SSO group memberships",
    )


@app.get("/health")
@app.get("/healthz", include_in_schema=False)
def health() -> dict[str, Any]:
    """Cloud Run external health check and internal probe endpoint."""
    return {
        "status": "ok",
        "service": os.environ.get("K_SERVICE", "sec-edgar-analyst"),
        "project_id": os.environ.get("GCP_PROJECT_ID", DEFAULT_PROJECT_ID),
        "root_agent": RootOrchestrator.name,
        "sub_agents": [sa.name for sa in RootOrchestrator.sub_agents],
        "iap_jwt_auth_enabled": True,
    }


# ---------------------------------------------------------------------------
# Identity-Aware Proxy (IAP) & Google Workspace Okta/SSO Authentication APIs
# ---------------------------------------------------------------------------


@app.get("/api/v1/auth/config")
def auth_discovery_config(
    verifier: IAPJwtVerifier = Depends(get_auth_verifier),
) -> dict[str, Any]:
    """Return public SSO & IAP authentication metadata (never exposes secret values)."""
    settings = verifier.settings
    return {
        "status": "ok",
        "auth_required": True,
        "access_pattern": "iap_jwt_bearer",
        "identity_provider": {
            "name": "Google Workspace (Okta SSO)",
            "protocol": "OIDC / SAML-Federated Workspace + IAP",
            "google_issuer": "https://accounts.google.com",
            "iap_issuer": GOOGLE_IAP_ISSUER,
            "okta_issuer": settings.okta_issuer_uri,
            "okta_client_id": settings.okta_client_id,
            "google_client_id": settings.google_oauth_client_id,
            "authorization_endpoint": settings.authorization_endpoint(),
            "allowed_workspace_domains": list(settings.allowed_workspace_domains),
            "required_analyst_groups": list(settings.required_analyst_groups),
            "local_sso_dev_broker_enabled": not settings.strict_production_mode,
        },
        "iap": {
            "enabled": True,
            "strict_production_mode": settings.strict_production_mode,
            "bearer_header": "Authorization",
            "assertion_header": "X-Goog-IAP-JWT-Assertion",
            "audience_configured": bool(settings.audiences),
        },
    }


@app.post("/api/v1/auth/sso/exchange")
def exchange_sso_id_token(
    req: SsoTokenExchangeRequest,
    verifier: IAPJwtVerifier = Depends(get_auth_verifier),
) -> dict[str, Any]:
    """Validate a Google Workspace / Okta SSO / IAP OIDC ID token for API Bearer access."""
    cleaned_jwt = req.id_token.strip()
    if cleaned_jwt.lower().startswith("bearer "):
        cleaned_jwt = cleaned_jwt[7:].strip()

    try:
        user = verifier.verify_jwt(cleaned_jwt)
    except AuthenticationError as exc:
        record_security_audit_event(
            event_type="sso_token_exchange_failed",
            user=None,
            resource="/api/v1/auth/sso/exchange",
            status_outcome="denied_401",
            detail=exc.message,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "status": "unauthorized",
                "error": exc.error_code,
                "message": exc.message,
            },
            headers={"WWW-Authenticate": 'Bearer realm="sec-edgar-iap", error="invalid_token"'},
        ) from exc
    except AuthorizationError as exc:
        record_security_audit_event(
            event_type="sso_token_exchange_forbidden",
            user=None,
            resource="/api/v1/auth/sso/exchange",
            status_outcome="denied_403",
            detail=exc.message,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "status": "forbidden",
                "error": "insufficient_permissions",
                "message": exc.message,
            },
        ) from exc

    record_security_audit_event(
        event_type="sso_token_exchange",
        user=user,
        resource="/api/v1/auth/sso/exchange",
        status_outcome="allowed",
    )
    return {
        "status": "authenticated",
        "token_type": "Bearer",
        "access_token": cleaned_jwt,
        "expires_at": user.expires_at,
        "user": user.to_public_dict(),
    }


@app.post("/api/v1/auth/sso/dev-token")
def mint_local_dev_sso_token(
    req: DevWorkspaceSsoLoginRequest,
    verifier: IAPJwtVerifier = Depends(get_auth_verifier),
) -> dict[str, Any]:
    """Issue a signed ES256 Workspace/Okta JWT for local development (disabled in Cloud Run)."""
    try:
        signed_jwt, user = issue_local_dev_workspace_sso_token(
            email=req.email,
            name=req.name,
            groups=req.groups,
            verifier=verifier,
        )
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"status": "unauthorized", "error": exc.error_code, "message": exc.message},
            headers={"WWW-Authenticate": 'Bearer realm="sec-edgar-iap"'},
        ) from exc
    except AuthorizationError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "status": "forbidden",
                "error": "local_sso_broker_forbidden",
                "message": exc.message,
            },
        ) from exc

    record_security_audit_event(
        event_type="sso_workspace_okta_login",
        user=user,
        resource="/api/v1/auth/sso/dev-token",
        status_outcome="allowed",
    )
    return {
        "status": "authenticated",
        "token_type": "Bearer",
        "access_token": signed_jwt,
        "expires_at": user.expires_at,
        "user": user.to_public_dict(),
    }


@app.get("/api/v1/auth/me")
def get_authenticated_session(
    user: AuthenticatedUser = Depends(require_authenticated_user),
) -> dict[str, Any]:
    """Return the verified Google Workspace / Okta SSO & IAP principal for the active Bearer JWT."""
    return {
        "status": "authenticated",
        "user": user.to_public_dict(),
    }


@app.post("/api/v1/auth/logout")
def logout_authenticated_session(
    request: Request,
    user: AuthenticatedUser = Depends(require_authenticated_user),
) -> dict[str, Any]:
    """Revoke the active Bearer JWT and record a session logout audit event."""
    raw_token = getattr(request.state, "raw_bearer_token", "")
    if raw_token:
        revoke_bearer_token(raw_token, expires_at=user.expires_at)
    record_security_audit_event(
        event_type="sso_logout",
        user=user,
        resource="/api/v1/auth/logout",
        status_outcome="revoked",
    )
    return {
        "status": "logged_out",
        "revoked_sub": user.sub,
        "user_email": user.email,
    }


@app.get("/api/v1/auth/audit")
def list_security_audit_trail(
    limit: int = Query(50, ge=1, le=500),
    user: AuthenticatedUser = Depends(require_roles("analyst", "admin")),
) -> dict[str, Any]:
    """Retrieve structured authentication and 10-K report access audit logs."""
    events = get_security_audit_events(limit=limit)
    return {
        "status": "ok",
        "requested_by": user.email,
        "event_count": len(events),
        "events": events,
    }


# ---------------------------------------------------------------------------
# IAP JWT Bearer Protected Internal SEC EDGAR Analyst APIs
# ---------------------------------------------------------------------------


@app.get("/api/v1/secrets/status")
def secrets_status(
    user: AuthenticatedUser = Depends(require_authenticated_user),
) -> dict[str, Any]:
    """Report Secret Manager binding metadata without ever exposing secret values."""
    record_security_audit_event(
        event_type="secrets_metadata_inspected",
        user=user,
        resource="/api/v1/secrets/status",
    )
    client = SecretManagerClient()
    bindings = []
    for spec in MANAGED_SECRETS:
        is_mounted_in_env = bool(os.environ.get(spec.env_var))
        bindings.append(
            {
                "env_var": spec.env_var,
                "secret_id": spec.secret_id,
                "resource_name": spec.resource_name(client.project_id),
                "cloud_run_mount": spec.cloud_run_mount_spec(),
                "required": spec.required,
                "mounted_in_env": is_mounted_in_env,
                "value": "***REDACTED***",
            }
        )
    return {
        "status": "ok",
        "project_id": client.project_id,
        "secret_manager_enabled": True,
        "secrets": bindings,
    }


@app.post("/api/v1/variances")
def compute_variances(
    req: VarianceRequest,
    user: AuthenticatedUser = Depends(require_authenticated_user),
) -> dict[str, Any]:
    """Compute deterministic period-over-period financial variances."""
    record_security_audit_event(
        event_type="variance_calculation",
        user=user,
        resource="/api/v1/variances",
        tickers=req.tickers,
        detail=f"FY{req.start_year}-FY{req.end_year}",
    )
    res = calculate_period_variances(
        tickers=req.tickers,
        start_year=req.start_year,
        end_year=req.end_year,
    )
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("message", "Invalid request"))
    return res


@app.post("/api/v1/search")
def rag_search(
    req: RagSearchRequest,
    user: AuthenticatedUser = Depends(require_authenticated_user),
) -> dict[str, Any]:
    """Run metadata-filtered Hybrid RAG search over SEC 10-K filing chunks."""
    record_security_audit_event(
        event_type="rag_chunk_search",
        user=user,
        resource="/api/v1/search",
        tickers=req.tickers,
        detail=f"section={req.section_id} FY{req.start_year}-FY{req.end_year}",
    )
    return search_10k_chunks_rag(
        query=req.query,
        tickers=req.tickers,
        gics_sector=req.gics_sector,
        section_id=req.section_id,
        start_year=req.start_year,
        end_year=req.end_year,
        top_k=req.top_k,
    )


@app.post("/api/v1/intent")
def classify_intent(
    req: IntentClassifyRequest,
    user: AuthenticatedUser = Depends(require_authenticated_user),
) -> dict[str, Any]:
    """Deterministically classify an analyst query into a RootOrchestrator routing intent."""
    record_security_audit_event(
        event_type="intent_classification",
        user=user,
        resource="/api/v1/intent",
        tickers=req.tickers,
    )
    intent = classify_query_intent(req.query, tickers=req.tickers)
    return {
        "status": "success",
        "query": req.query,
        "intent": intent,
        "orchestrator": RootOrchestrator.name,
    }


@app.get("/api/v1/metrics")
def get_metrics(
    tickers: str = "NVDA,AAPL",
    start_year: int = 2024,
    end_year: int = 2026,
    gics_sector: str = "ALL",
    user: AuthenticatedUser = Depends(require_authenticated_user),
) -> dict[str, Any]:
    """Retrieve structured SEC 10-K financial metrics."""
    record_security_audit_event(
        event_type="financial_metrics_read",
        user=user,
        resource="/api/v1/metrics",
        tickers=tickers,
        detail=f"FY{start_year}-FY{end_year}",
    )
    return fetch_financial_metrics(
        tickers=tickers,
        start_year=start_year,
        end_year=end_year,
        gics_sector=gics_sector,
    )


@app.get("/api/v1/explore/{explore_name}")
def get_looker_explore(
    explore_name: str,
    tickers: str = "ALL",
    gics_sector: str = "ALL",
    start_year: int = 2024,
    end_year: int = 2026,
    user: AuthenticatedUser = Depends(require_authenticated_user),
) -> dict[str, Any]:
    """Query Looker BI semantic explores (yoy_variances, sector_benchmarks, peer_comparison, chunk_health)."""
    record_security_audit_event(
        event_type="looker_explore_query",
        user=user,
        resource=f"/api/v1/explore/{explore_name}",
        tickers=tickers,
    )
    return query_looker_financial_explore(
        explore_name=explore_name,
        tickers=tickers,
        gics_sector=gics_sector,
        start_year=start_year,
        end_year=end_year,
    )


class ChatQueryRequest(BaseModel):
    query: str = Field(..., min_length=2)
    tickers: str = Field("ALL")
    gics_sector: str = Field("ALL")
    section_id: str = Field("ALL")
    start_year: int = Field(2024, ge=2015, le=2035)
    end_year: int = Field(2026, ge=2015, le=2035)
    top_k: int = Field(8, ge=1, le=25)


class SectionLoadRequest(BaseModel):
    ticker: str = Field(..., min_length=1)
    fiscal_year: int = Field(..., ge=2015, le=2035)
    section_id: Literal["item_1a", "item_7"] = Field("item_7")
    max_chars: int = Field(32000, ge=500, le=50000)


class FeedbackRequest(BaseModel):
    message_id: str = Field(..., min_length=1)
    rating: Literal["up", "down"]
    comment: str = Field(default="", max_length=1000)
    query: str = Field(default="", max_length=1000)


@app.get("/api/v1/companies")
def get_companies(
    user: AuthenticatedUser = Depends(require_authenticated_user),
) -> dict[str, Any]:
    """List all curated S&P 500 issuers, GICS sectors, and available fiscal years."""
    record_security_audit_event(
        event_type="companies_catalog_read",
        user=user,
        resource="/api/v1/companies",
    )
    return list_curated_companies()


@app.post("/api/v1/chat")
def conversational_rag_chat(
    req: ChatQueryRequest,
    user: AuthenticatedUser = Depends(require_authenticated_user),
) -> dict[str, Any]:
    """Execute a source-grounded conversational Hybrid RAG query with ReAct trace & fact-check verification."""
    turn_result = run_conversational_rag_turn(
        query=req.query,
        tickers=req.tickers,
        gics_sector=req.gics_sector,
        section_id=req.section_id,
        start_year=req.start_year,
        end_year=req.end_year,
        top_k=req.top_k,
    )
    resolved_tickers = turn_result.get("resolved_scope", {}).get("tickers") or req.tickers
    record_security_audit_event(
        event_type="conversational_rag_chat",
        user=user,
        resource="/api/v1/chat",
        tickers=resolved_tickers,
        status_outcome=str(turn_result.get("status", "success")),
        detail=f"intent={turn_result.get('intent', 'unknown')}",
    )
    turn_result["authenticated_principal"] = {
        "email": user.email,
        "identity_provider": user.identity_provider,
        "iap_assertion_verified": user.iap_assertion_verified,
    }
    return turn_result


def _enrich_section_response(
    ticker: str,
    fiscal_year: int,
    section_id: str,
    max_chars: int,
) -> dict[str, Any]:
    res = load_cached_10k_section(
        ticker=ticker,
        fiscal_year=fiscal_year,
        section_id=section_id,
        max_chars=max_chars,
    )
    if res.get("status") == "error":
        raise HTTPException(status_code=404, detail=res.get("message", "Section not found"))

    dataset_dir = _get_dataset_dir()
    profiles = _load_all_profiles_cached(str(dataset_dir))
    norm_t = ticker.strip().upper().replace(".", "-")
    matched_prof = next(
        (
            p
            for p in profiles
            if p.ticker.upper().replace(".", "-") == norm_t and p.fiscal_year == fiscal_year
        ),
        None,
    )
    if matched_prof is not None:
        res["cik"] = matched_prof.cik
        res["accession_number"] = matched_prof.accession_number
        res["filing_date"] = matched_prof.filing_date
        res["period_of_report"] = matched_prof.period_of_report
        res["gics_sector"] = matched_prof.gics_sector
        res["gics_sub_industry"] = matched_prof.gics_sub_industry
    return res


@app.get("/api/v1/section")
def get_cached_section(
    ticker: str = "NVDA",
    fiscal_year: int = 2026,
    section_id: Literal["item_1a", "item_7"] = "item_7",
    max_chars: int = 32000,
    user: AuthenticatedUser = Depends(require_authenticated_user),
) -> dict[str, Any]:
    """Load a cached 10-K filing section (Item 1A Risk Factors or Item 7 MD&A) for the split-pane source viewer."""
    record_security_audit_event(
        event_type="filing_10k_section_read",
        user=user,
        resource="/api/v1/section",
        tickers=[ticker],
        detail=f"FY{fiscal_year} {section_id}",
    )
    return _enrich_section_response(
        ticker=ticker,
        fiscal_year=fiscal_year,
        section_id=section_id,
        max_chars=max_chars,
    )


@app.post("/api/v1/section")
def post_cached_section(
    req: SectionLoadRequest,
    user: AuthenticatedUser = Depends(require_authenticated_user),
) -> dict[str, Any]:
    """Load a cached 10-K filing section via POST payload."""
    record_security_audit_event(
        event_type="filing_10k_section_read",
        user=user,
        resource="/api/v1/section",
        tickers=[req.ticker],
        detail=f"FY{req.fiscal_year} {req.section_id}",
    )
    return _enrich_section_response(
        ticker=req.ticker,
        fiscal_year=req.fiscal_year,
        section_id=req.section_id,
        max_chars=req.max_chars,
    )


@app.post("/api/v1/feedback")
def submit_analyst_feedback(
    req: FeedbackRequest,
    user: AuthenticatedUser = Depends(require_authenticated_user),
) -> dict[str, Any]:
    """Record analyst thumbs-up/down audit feedback and optional comment."""
    entry = {
        "message_id": req.message_id,
        "rating": req.rating,
        "comment": req.comment.strip(),
        "query": req.query.strip(),
        "user_email": user.email,
        "identity_provider": user.identity_provider,
        "recorded_at_epoch": round(time.time(), 3),
    }
    _FEEDBACK_AUDIT_LOG.append(entry)
    record_security_audit_event(
        event_type="analyst_feedback_submitted",
        user=user,
        resource="/api/v1/feedback",
        detail=f"message_id={req.message_id} rating={req.rating}",
    )
    logger.info(
        "Recorded analyst feedback message_id=%s rating=%s user=%s",
        req.message_id,
        req.rating,
        user.email,
    )
    return {
        "status": "success",
        "recorded": entry,
        "total_feedback_count": len(_FEEDBACK_AUDIT_LOG),
    }


def _resolve_frontend_dist_dir() -> Path | None:
    candidates: list[Path] = []
    env_override = os.environ.get("FRONTEND_DIST_DIR", "").strip()
    if env_override:
        candidates.append(Path(env_override))
    candidates.append(Path.cwd() / "frontend" / "dist")
    candidates.append(Path(__file__).resolve().parents[2] / "frontend" / "dist")
    for directory in candidates:
        if (directory / "index.html").is_file():
            return directory.resolve()
    return None


@app.get("/", include_in_schema=False)
@app.get("/{full_path:path}", include_in_schema=False)
def serve_terminal_ui(full_path: str = "") -> FileResponse:
    """Serve the compiled SEC EDGAR Terminal UI identically in local preview and Cloud Run."""
    normalized = full_path.strip("/")
    if normalized == "api" or normalized.startswith("api/"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")

    dist_dir = _resolve_frontend_dist_dir()
    if dist_dir is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend bundle not found. Run 'make build' to compile frontend/dist.",
        )

    if normalized:
        asset_path = (dist_dir / normalized).resolve()
        if asset_path.is_relative_to(dist_dir) and asset_path.is_file():
            return FileResponse(asset_path)

    return FileResponse(dist_dir / "index.html")

