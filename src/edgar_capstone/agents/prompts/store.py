"""Local Git-versioned prompt loader for SEC EDGAR ADK agents and Model Armor policy."""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from edgar_capstone.gcp.paths import resolve_asset_dir

logger = logging.getLogger(__name__)

_MUSTACHE_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")
_VALID_ID_RE = re.compile(r"^[a-zA-Z0-9_.\-]+$")


@dataclass(frozen=True)
class PromptArtifact:
    """Loaded prompt template and metadata from the local `prompts/` directory."""

    prompt_id: str
    version: str
    release_tag: str
    template: str
    agent_name: str = ""
    model: str = ""
    description: str = ""
    variables: dict[str, str] = field(default_factory=dict)
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    source_backend: str = "local"

    def render(self, variables: dict[str, Any] | None = None) -> str:
        """Render `{{var}}` placeholders using artifact defaults overridden by `variables`."""
        merged: dict[str, str] = {k: str(v) for k, v in self.variables.items()}
        if variables:
            for k, v in variables.items():
                if v is not None:
                    merged[str(k)] = str(v)
        return _MUSTACHE_VAR_RE.sub(
            lambda m: merged.get(m.group(1), m.group(0)),
            self.template,
        ).strip()


def _runtime_model_variables() -> dict[str, str]:
    from edgar_capstone.agents.runtime.model_config import resolve_model_for_task

    return {
        "analyst_model": resolve_model_for_task("financial_reasoning"),
        "flash_model": resolve_model_for_task("evaluation"),
    }


class VersionedPromptStore:
    """Loads Git-versioned YAML prompt artifacts from the local `prompts/` directory."""

    def __init__(self, *, prompts_dir: Path | str | None = None) -> None:
        self.prompts_dir = (
            Path(prompts_dir).resolve()
            if prompts_dir is not None
            else resolve_asset_dir("prompts").resolve()
        )
        self._lock = threading.Lock()
        self._registry_cache: tuple[int, dict[str, Any]] | None = None
        self._artifact_cache: dict[tuple[str, str], PromptArtifact] = {}

    def _load_registry(self) -> dict[str, Any]:
        registry_path = self.prompts_dir / "registry.yaml"
        if not registry_path.is_file():
            raise FileNotFoundError(f"Prompt registry not found at {registry_path}")
        mtime_ns = registry_path.stat().st_mtime_ns
        with self._lock:
            if self._registry_cache is not None and self._registry_cache[0] == mtime_ns:
                return self._registry_cache[1]
        parsed = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
        if not isinstance(parsed, dict):
            raise ValueError(f"Invalid prompt registry format in {registry_path}")
        with self._lock:
            self._registry_cache = (mtime_ns, parsed)
        return parsed

    def _resolve_version_entry(
        self, prompt_id: str, version: str | None
    ) -> tuple[str, dict[str, Any]]:
        if not _VALID_ID_RE.match(prompt_id):
            raise ValueError(f"Invalid prompt_id: {prompt_id!r}")
        registry = self._load_registry()
        prompts_map = registry.get("prompts") or {}
        prompt_entry = prompts_map.get(prompt_id)
        if not isinstance(prompt_entry, dict):
            raise KeyError(f"Prompt '{prompt_id}' not found in registry.")
        versions = prompt_entry.get("versions") or {}
        req = (version or "").strip() or str(prompt_entry.get("latest") or "v1")
        if req == "latest":
            req = str(prompt_entry.get("latest") or "v1")
        ver_entry = versions.get(req)
        if not isinstance(ver_entry, dict):
            raise KeyError(f"Version '{req}' for prompt '{prompt_id}' not found in registry.")
        return req, ver_entry

    def load_prompt_artifact(
        self,
        prompt_id: str,
        *,
        version: str | None = None,
    ) -> PromptArtifact:
        """Load and cache a prompt YAML artifact from `prompts/`."""
        resolved_tag, ver_entry = self._resolve_version_entry(prompt_id, version)
        cache_key = (prompt_id, resolved_tag)
        with self._lock:
            cached = self._artifact_cache.get(cache_key)
        if cached is not None:
            return cached

        rel_path = str(ver_entry.get("path") or "").strip()
        full_path = (self.prompts_dir / rel_path).resolve()
        if not full_path.is_relative_to(self.prompts_dir) or not full_path.is_file():
            raise FileNotFoundError(f"Prompt file '{rel_path}' not found under {self.prompts_dir}")

        raw = yaml.safe_load(full_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Prompt file '{full_path}' must contain a YAML mapping.")

        artifact = PromptArtifact(
            prompt_id=str(raw.get("prompt_id") or prompt_id),
            version=str(raw.get("version") or ver_entry.get("semver") or resolved_tag),
            release_tag=str(raw.get("release_tag") or resolved_tag),
            template=str(raw.get("template") or ""),
            agent_name=str(raw.get("agent_name") or ""),
            model=str(raw.get("model") or ""),
            description=str(raw.get("description") or "").strip(),
            variables={str(k): str(v) for k, v in (raw.get("variables") or {}).items()},
            raw_metadata=dict(raw),
            source_backend="local",
        )
        with self._lock:
            self._artifact_cache[cache_key] = artifact
        return artifact

    def load_prompt(
        self,
        prompt_id: str,
        *,
        version: str | None = None,
        variables: dict[str, Any] | None = None,
    ) -> str:
        """Load and render a prompt template with active model variables."""
        artifact = self.load_prompt_artifact(prompt_id, version=version)
        merged_vars = _runtime_model_variables()
        if variables:
            merged_vars.update({k: str(v) for k, v in variables.items() if v is not None})
        return artifact.render(merged_vars)

    def to_status_metadata(self) -> dict[str, Any]:
        """Return status metadata describing available local prompts."""
        registry = self._load_registry()
        prompts_map = registry.get("prompts") or {}
        return {
            "backend": "local",
            "prompts_dir": str(self.prompts_dir),
            "prompts": {
                pid: str(meta.get("latest") or "v1")
                for pid, meta in sorted(prompts_map.items())
                if isinstance(meta, dict)
            },
        }


_STORE_LOCK = threading.Lock()
_GLOBAL_PROMPT_STORE: VersionedPromptStore | None = None


def get_prompt_store() -> VersionedPromptStore:
    """Return the process-wide `VersionedPromptStore` singleton."""
    global _GLOBAL_PROMPT_STORE
    with _STORE_LOCK:
        if _GLOBAL_PROMPT_STORE is None:
            _GLOBAL_PROMPT_STORE = VersionedPromptStore()
        return _GLOBAL_PROMPT_STORE


def configure_prompt_store(
    store: VersionedPromptStore | None = None, **kwargs: Any
) -> VersionedPromptStore:
    """Configure or replace the process-wide `VersionedPromptStore` instance."""
    global _GLOBAL_PROMPT_STORE
    with _STORE_LOCK:
        _GLOBAL_PROMPT_STORE = store if store is not None else VersionedPromptStore(**kwargs)
        return _GLOBAL_PROMPT_STORE


def reset_prompt_store() -> None:
    """Reset the process-wide `VersionedPromptStore` singleton."""
    global _GLOBAL_PROMPT_STORE
    with _STORE_LOCK:
        _GLOBAL_PROMPT_STORE = None


def load_prompt(
    prompt_id: str,
    *,
    version: str | None = None,
    variables: dict[str, Any] | None = None,
) -> str:
    """Load a rendered prompt string via the active local store."""
    return get_prompt_store().load_prompt(prompt_id, version=version, variables=variables)


def load_prompt_artifact(
    prompt_id: str,
    *,
    version: str | None = None,
) -> PromptArtifact:
    """Load a `PromptArtifact` via the active local store."""
    return get_prompt_store().load_prompt_artifact(prompt_id, version=version)


def make_adk_instruction_provider(
    prompt_id: str,
    *,
    version: str | None = None,
    variables: dict[str, Any] | None = None,
) -> Callable[[Any], str]:
    """Return an ADK `InstructionProvider` resolving the local prompt dynamically per run."""

    def _provide_instruction(readonly_context: Any) -> str:
        eff_version = version
        state = getattr(readonly_context, "state", None)
        if isinstance(state, dict):
            overrides = state.get("prompt_versions")
            if isinstance(overrides, dict) and prompt_id in overrides:
                eff_version = str(overrides[prompt_id])
        return get_prompt_store().load_prompt(
            prompt_id,
            version=eff_version,
            variables=variables,
        )

    return _provide_instruction


def get_prompt_store_status_metadata() -> dict[str, Any]:
    """Return status metadata describing the local prompt store."""
    return get_prompt_store().to_status_metadata()


__all__ = [
    "PromptArtifact",
    "VersionedPromptStore",
    "configure_prompt_store",
    "get_prompt_store",
    "get_prompt_store_status_metadata",
    "load_prompt",
    "load_prompt_artifact",
    "make_adk_instruction_provider",
    "reset_prompt_store",
]
