"""Pydantic request schemas with strict scope validation for the SEC EDGAR API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ChatQueryRequest(BaseModel):
    """Conversational Hybrid RAG analyst query payload with fiscal-year validation."""

    tickers: str = Field(
        default="ALL",
        description="Comma-separated SEC ticker symbols or 'ALL'.",
    )
    gics_sector: str = Field(
        default="ALL",
        description="GICS sector filter or 'ALL'.",
    )
    section_id: str = Field(
        default="ALL",
        description="10-K section filter ('item_1a', 'item_7', or 'ALL').",
    )
    start_year: int = Field(
        default=2024,
        ge=2015,
        le=2035,
        description="Inclusive starting fiscal year.",
    )
    end_year: int = Field(
        default=2026,
        ge=2015,
        le=2035,
        description="Inclusive ending fiscal year.",
    )
    query: str = Field(
        ..., min_length=2, description="Natural-language financial analyst question."
    )
    top_k: int = Field(
        default=8,
        ge=1,
        le=25,
        description="Maximum number of Hybrid RAG filing chunks to retrieve.",
    )
    session_id: str | None = Field(
        default=None,
        max_length=128,
        description="Optional persistent user-scoped conversation session ID.",
    )

    @model_validator(mode="after")
    def _validate_year_range(self) -> ChatQueryRequest:
        if self.start_year > self.end_year:
            raise ValueError(
                f"start_year ({self.start_year}) must be less than or equal to "
                f"end_year ({self.end_year})."
            )
        return self


class SsoTokenExchangeRequest(BaseModel):
    """OIDC / Google IAP signed JWT exchange payload."""

    id_token: str = Field(
        ...,
        min_length=16,
        description="OIDC / IAP signed JWT bearer token.",
    )


class DevWorkspaceSsoLoginRequest(BaseModel):
    """Local development Workspace/Okta SSO broker request payload (disabled in Cloud Run)."""

    email: str = Field(
        default="analyst@fde-workspace.corp",
        min_length=5,
        max_length=254,
        description="Google Workspace corporate email federated via Okta SSO.",
    )
    name: str = Field(
        default="Senior Financial Analyst",
        min_length=1,
        max_length=120,
        description="Display name for the simulated analyst principal.",
    )
    groups: list[str] = Field(
        default_factory=lambda: ["sec-edgar-analysts"],
        description="Cloud Identity / Okta SSO group memberships.",
    )


class FeedbackRequest(BaseModel):
    """Analyst thumbs-up/down audit feedback persisted to Firestore."""

    message_id: str = Field(..., min_length=1, description="Assistant message identifier.")
    rating: Literal["up", "down"] = Field(..., description="Analyst rating ('up' or 'down').")
    comment: str = Field(default="", max_length=1000, description="Optional audit note.")
    query: str = Field(default="", max_length=1000, description="Original analyst query text.")
    session_id: str | None = Field(default=None, max_length=128)
    tickers: list[str] = Field(default_factory=list)
    intent: str = Field(default="", max_length=128)


__all__ = [
    "ChatQueryRequest",
    "DevWorkspaceSsoLoginRequest",
    "FeedbackRequest",
    "SsoTokenExchangeRequest",
]
