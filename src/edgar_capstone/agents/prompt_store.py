"""Versioned Prompt Store & Dynamic ADK Instruction Loader (`prompts/`, GCS, and Vertex AI).

Centralizes all ADK agent instructions and safety policy specifications outside Python
code in a version-controlled, integrity-hashed prompt store supporting three backends:

1. **`local`** (`prompts/` directory with `registry.yaml` and `<version>/<prompt_id>.yaml`).
2. **`gcs`** (`gs://<bucket>/<prefix>/<version>/<prompt_id>.yaml` with optional GCS object
   generation pinning via Google Cloud Storage REST API).
3. **`vertex`** (Google Cloud Vertex AI Prompt Management REST endpoint
   `projects/{project}/locations/{location}/datasets/{prompt_resource_id}` with version tags).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import httpx
import yaml
from google.auth.exceptions import DefaultCredentialsError

from edgar_capstone.agents.resilience import (
    RateLimitExceededError,
    ToolExecutionError,
    UpstreamTimeoutError,
    retry_transient_errors,
)
from edgar_capstone.common.cache import BoundedTTLCache
from edgar_capstone.common.domain import sha256_text
from edgar_capstone.gcp.defaults import (
    DEFAULT_PROJECT_ID,
    resolve_gcp_location,
    resolve_gcp_project_id,
)
from edgar_capstone.gcp.paths import resolve_asset_dir
from edgar_capstone.gcp.rest_client import GcpRestClient, GcpRestError, parse_retry_after

logger = logging.getLogger(__name__)

PromptBackendMode = Literal["auto", "local", "gcs", "vertex"]
ResolvedPromptBackend = Literal["local", "gcs", "vertex"]

DEFAULT_PROMPT_VERSION = "v1"
_GCS_READ_SCOPE = "https://www.googleapis.com/auth/devstorage.read_only"
_REMOTE_FETCH_ERRORS = (
    httpx.HTTPError,
    GcpRestError,
    ToolExecutionError,
    DefaultCredentialsError,
    OSError,
    KeyError,
    ValueError,
)


def _resolve_default_prompts_dir() -> Path:
    """Locate the repository or container `prompts/` directory."""
    env_dir = os.environ.get("EDGAR_PROMPTS_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return resolve_asset_dir("prompts", env_var="EDGAR_PROMPTS_DIR")


def _resolve_latest_from_registry(
    prompt_id: str,
    registry: dict[str, Any] | None,
) -> str | None:
    """Resolve the concrete version tag for `prompt_id` when `'latest'` is requested."""
    if not isinstance(registry, dict) or not registry:
        return None
    prompts_map = registry.get("prompts")
    entry = prompts_map.get(prompt_id) if isinstance(prompts_map, dict) else None
    if isinstance(entry, dict) and entry.get("latest"):
        return str(entry["latest"]).strip() or None
    default_ver = registry.get("default_version")
    if default_ver:
        return str(default_ver).strip() or None
    return None


def _render_mustache_variables(template: str, variables: dict[str, Any]) -> str:
    """Render `{{var_name}}` placeholders without disturbing single `{...}` LookML/JSON braces."""
    rendered = template
    for key, val in variables.items():
        placeholder_tight = "{{" + str(key) + "}}"
        placeholder_spaced = "{{ " + str(key) + " }}"
        replacement = str(val)
        if placeholder_tight in rendered:
            rendered = rendered.replace(placeholder_tight, replacement)
        if placeholder_spaced in rendered:
            rendered = rendered.replace(placeholder_spaced, replacement)
    return rendered


@dataclass(frozen=True)
class PromptArtifact:
    """Immutable versioned prompt artifact loaded from `prompts/`, GCS, or Vertex AI."""

    prompt_id: str
    version: str
    release_tag: str
    agent_name: str
    model: str
    template: str
    rendered_text: str
    sha256: str
    source_backend: ResolvedPromptBackend
    source_uri: str
    description: str = ""
    variables: dict[str, Any] = field(default_factory=dict)
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def to_public_metadata(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "version": self.version,
            "release_tag": self.release_tag,
            "agent_name": self.agent_name,
            "model": self.model,
            "sha256": self.sha256,
            "source_backend": self.source_backend,
            "source_uri": self.source_uri,
        }


def _verify_sha256(
    rendered: str,
    *,
    prompt_id: str,
    requested_version: str,
    expected_sha256: str | None,
) -> str:
    digest = sha256_text(rendered)
    if expected_sha256 and digest != expected_sha256.strip().lower():
        raise ValueError(
            f"Prompt integrity check failed for '{prompt_id}@{requested_version}': "
            f"expected sha256={expected_sha256}, actual={digest}"
        )
    return digest


def _extract_template_str(parsed: dict[str, Any]) -> str:
    """Extract the template text from a YAML prompt spec or Vertex AI prompt resource."""
    if isinstance(parsed.get("template"), str):
        return parsed["template"]
    if isinstance(parsed.get("promptData"), dict) or isinstance(parsed.get("prompt_data"), dict):
        pdata = parsed.get("promptData") or parsed.get("prompt_data") or {}
        sys_inst = pdata.get("systemInstruction") or pdata.get("system_instruction") or {}
        parts = sys_inst.get("parts") or []
        if parts:
            return "\n".join(
                str(p.get("text", "")) for p in parts if isinstance(p, dict) and p.get("text")
            )
        contents = pdata.get("contents") or []
        chunks = [
            str(p["text"])
            for c in contents
            if isinstance(c, dict)
            for p in (c.get("parts") or [])
            if isinstance(p, dict) and p.get("text")
        ]
        return "\n".join(chunks)
    if isinstance(parsed.get("systemInstruction"), dict):
        parts = parsed["systemInstruction"].get("parts") or []
        return "\n".join(
            str(p.get("text", "")) for p in parts if isinstance(p, dict) and p.get("text")
        )
    return ""


def _parse_prompt_document(
    raw_text: str,
    *,
    prompt_id: str,
    requested_version: str,
    source_backend: ResolvedPromptBackend,
    source_uri: str,
    override_variables: dict[str, Any] | None = None,
    expected_sha256: str | None = None,
) -> PromptArtifact:
    """Parse a YAML, JSON, or plain Markdown prompt artifact and compute its SHA-256 digest."""
    parsed: dict[str, Any] | None = None
    stripped = raw_text.lstrip()
    if stripped.startswith("{"):
        loaded_json = json.loads(raw_text)
        if isinstance(loaded_json, dict):
            parsed = loaded_json
    else:
        try:
            loaded_yaml = yaml.safe_load(raw_text)
            if isinstance(loaded_yaml, dict) and (
                "template" in loaded_yaml
                or "prompt_id" in loaded_yaml
                or "policy" in loaded_yaml
                or "systemInstruction" in loaded_yaml
                or "promptData" in loaded_yaml
            ):
                parsed = loaded_yaml
        except yaml.YAMLError:
            parsed = None

    if parsed is None:
        merged_vars = dict(override_variables or {})
        rendered = _render_mustache_variables(raw_text, merged_vars)
        digest = _verify_sha256(
            rendered,
            prompt_id=prompt_id,
            requested_version=requested_version,
            expected_sha256=expected_sha256,
        )
        return PromptArtifact(
            prompt_id=prompt_id,
            version=requested_version,
            release_tag=requested_version,
            agent_name=prompt_id,
            model="",
            template=raw_text,
            rendered_text=rendered,
            sha256=digest,
            source_backend=source_backend,
            source_uri=source_uri,
            variables=merged_vars,
        )

    template_str = _extract_template_str(parsed)
    default_vars = dict(parsed.get("variables") or {})
    if override_variables:
        default_vars.update(override_variables)

    rendered = _render_mustache_variables(template_str, default_vars)
    digest = _verify_sha256(
        rendered,
        prompt_id=prompt_id,
        requested_version=requested_version,
        expected_sha256=expected_sha256,
    )

    version_str = str(
        parsed.get("version")
        or parsed.get("versionId")
        or parsed.get("semver")
        or requested_version
    )
    release_tag = str(parsed.get("release_tag") or requested_version)
    agent_name = str(parsed.get("agent_name") or parsed.get("displayName") or prompt_id)
    model_name = str(parsed.get("model") or (parsed.get("promptData") or {}).get("model") or "")
    description = str(parsed.get("description") or "").strip()
    extra_meta: dict[str, Any] = {}
    if isinstance(parsed.get("policy"), dict):
        extra_meta["policy"] = parsed["policy"]

    return PromptArtifact(
        prompt_id=str(parsed.get("prompt_id") or prompt_id),
        version=version_str,
        release_tag=release_tag,
        agent_name=agent_name,
        model=model_name,
        template=template_str,
        rendered_text=rendered,
        sha256=digest,
        source_backend=source_backend,
        source_uri=source_uri,
        description=description,
        variables=default_vars,
        raw_metadata=extra_meta,
    )


def _raise_for_prompt_http_status(resp: httpx.Response, service_label: str) -> None:
    if resp.status_code < 400:
        return
    snippet = resp.text[:300]
    if resp.status_code == 429:
        raise RateLimitExceededError(
            f"{service_label} HTTP 429 rate limited: {snippet}",
            retry_after_seconds=parse_retry_after(resp),
        )
    if resp.status_code in (408, 500, 502, 503, 504):
        raise UpstreamTimeoutError(f"{service_label} transient HTTP {resp.status_code}: {snippet}")
    raise ToolExecutionError(f"{service_label} HTTP {resp.status_code}: {snippet}")


class LocalPromptStoreBackend:
    """Loads versioned prompt specs from the local `prompts/` directory with mtime-cached registry."""

    def __init__(self, prompts_dir: Path | str | None = None) -> None:
        self.prompts_dir = (
            Path(prompts_dir).expanduser().resolve()
            if prompts_dir is not None
            else _resolve_default_prompts_dir()
        )
        self._registry_lock = threading.Lock()
        self._registry_mtime_ns: int | None = None
        self._registry_cache: dict[str, Any] = {}

    def load_registry(self) -> dict[str, Any]:
        """Load `prompts/registry.yaml`, caching the parsed dict keyed by file `mtime_ns`."""
        reg_path = self.prompts_dir / "registry.yaml"
        try:
            stat = reg_path.stat()
        except OSError:
            with self._registry_lock:
                self._registry_mtime_ns = None
                self._registry_cache = {}
            return {}

        with self._registry_lock:
            if self._registry_mtime_ns == stat.st_mtime_ns:
                return self._registry_cache
            loaded = yaml.safe_load(reg_path.read_text(encoding="utf-8"))
            self._registry_cache = loaded if isinstance(loaded, dict) else {}
            self._registry_mtime_ns = stat.st_mtime_ns
            return self._registry_cache

    def resolve_relative_path(self, prompt_id: str, version: str) -> tuple[str, str]:
        registry = self.load_registry()
        prompts_map = registry.get("prompts") if isinstance(registry.get("prompts"), dict) else {}
        entry = prompts_map.get(prompt_id) if isinstance(prompts_map, dict) else None
        eff_version = version.strip() if version and version.strip() else DEFAULT_PROMPT_VERSION
        if eff_version == "latest":
            eff_version = (
                _resolve_latest_from_registry(prompt_id, registry) or DEFAULT_PROMPT_VERSION
            )

        if isinstance(entry, dict) and isinstance(entry.get("versions"), dict):
            ver_meta = entry["versions"].get(eff_version)
            if isinstance(ver_meta, dict) and ver_meta.get("path"):
                return eff_version, str(ver_meta["path"])

        # Conventional fallback paths: <version>/<prompt_id>.yaml or .md
        for ext in (".yaml", ".yml", ".md"):
            rel = f"{eff_version}/{prompt_id}{ext}"
            if (self.prompts_dir / rel).is_file():
                return eff_version, rel
        return eff_version, f"{eff_version}/{prompt_id}.yaml"

    def fetch(self, prompt_id: str, version: str) -> tuple[str, str, str]:
        eff_version, rel_path = self.resolve_relative_path(prompt_id, version)
        full_path = (self.prompts_dir / rel_path).resolve()
        # Prevent path traversal outside prompts_dir
        prompts_root = self.prompts_dir.resolve()
        if prompts_root not in full_path.parents and full_path != prompts_root:
            raise ValueError(f"Invalid prompt path outside prompts root: {rel_path!r}")
        if not full_path.is_file():
            raise FileNotFoundError(
                f"Prompt '{prompt_id}' (version='{eff_version}') not found at {full_path}"
            )
        return full_path.read_text(encoding="utf-8"), f"file://{full_path}", eff_version


class GcsPromptStoreBackend(GcpRestClient):
    """Loads versioned prompt specs from Google Cloud Storage (`gs://<bucket>/<prefix>`)."""

    def __init__(
        self,
        gcs_base_uri: str,
        *,
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
        timeout_seconds: float = 10.0,
        registry_loader: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            http_client=http_client,
            credentials=credentials,
            default_timeout=timeout_seconds,
            scopes=(_GCS_READ_SCOPE,),
        )
        cleaned = gcs_base_uri.strip()
        if cleaned.startswith("gs://"):
            cleaned = cleaned[5:]
        parts = cleaned.split("/", 1)
        if not parts[0]:
            raise ValueError(f"Invalid GCS prompt store URI: {gcs_base_uri!r}")
        self.bucket_name = parts[0]
        self.prefix = parts[1].strip("/") if len(parts) > 1 else "prompts"
        self.timeout_seconds = timeout_seconds
        self._registry_loader = registry_loader

    @retry_transient_errors(
        operation_name="gcs_prompt_store_fetch", max_retries=3, base_delay_seconds=0.1
    )
    def _download_object_text(self, object_name: str, *, generation: str | None = None) -> str:
        encoded_obj = quote(object_name, safe="")
        url = f"https://storage.googleapis.com/storage/v1/b/{self.bucket_name}/o/{encoded_obj}?alt=media"
        if generation:
            url += f"&generation={quote(str(generation), safe='')}"
        try:
            resp = self._raw_request("GET", url, timeout=self.timeout_seconds)
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError(
                f"GCS Prompt Store timeout for {object_name}: {exc}"
            ) from exc
        _raise_for_prompt_http_status(resp, "GCS Prompt Store")
        return resp.text

    def _load_remote_registry(self) -> dict[str, Any]:
        reg_obj = f"{self.prefix}/registry.yaml" if self.prefix else "registry.yaml"
        try:
            raw = self._download_object_text(reg_obj)
            loaded = yaml.safe_load(raw)
            return loaded if isinstance(loaded, dict) else {}
        except (*_REMOTE_FETCH_ERRORS, yaml.YAMLError) as exc:
            logger.debug("Remote GCS prompt registry unavailable (%s): %s", reg_obj, exc)
            return {}

    def _resolve_latest_version(self, prompt_id: str, registry: dict[str, Any] | None) -> str:
        resolved = _resolve_latest_from_registry(prompt_id, registry)
        if resolved:
            return resolved
        if self._registry_loader is not None:
            resolved = _resolve_latest_from_registry(prompt_id, self._registry_loader())
            if resolved:
                return resolved
        remote_reg = self._load_remote_registry()
        return _resolve_latest_from_registry(prompt_id, remote_reg) or DEFAULT_PROMPT_VERSION

    def fetch(
        self,
        prompt_id: str,
        version: str,
        *,
        registry: dict[str, Any] | None = None,
    ) -> tuple[str, str, str]:
        eff_version = version.strip() if version and version.strip() else DEFAULT_PROMPT_VERSION
        generation: str | None = None
        if "#" in eff_version:
            eff_version, generation = eff_version.split("#", 1)
        if eff_version == "latest":
            eff_version = self._resolve_latest_version(prompt_id, registry)

        rel_obj = f"{eff_version}/{prompt_id}.yaml"
        object_name = f"{self.prefix}/{rel_obj}" if self.prefix else rel_obj
        raw_text = self._download_object_text(object_name, generation=generation)
        uri = f"gs://{self.bucket_name}/{object_name}" + (f"#{generation}" if generation else "")
        return raw_text, uri, eff_version


class VertexPromptStoreBackend(GcpRestClient):
    """Loads versioned prompts from Google Cloud Vertex AI Prompt Management REST API."""

    def __init__(
        self,
        *,
        project_id: str | None = None,
        location: str | None = None,
        prompt_resource_map: dict[str, str] | None = None,
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
        timeout_seconds: float = 10.0,
        registry_loader: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            http_client=http_client,
            credentials=credentials,
            default_timeout=timeout_seconds,
        )
        self.project_id = resolve_gcp_project_id(project_id, default=DEFAULT_PROJECT_ID)
        self.location = resolve_gcp_location(location, allow_global=False)
        self.prompt_resource_map = dict(prompt_resource_map or {})
        self.timeout_seconds = timeout_seconds
        self._registry_loader = registry_loader

    def _resolve_resource_id(self, prompt_id: str, version: str) -> str:
        keyed = f"{prompt_id}:{version}"
        if keyed in self.prompt_resource_map:
            return self.prompt_resource_map[keyed]
        if prompt_id in self.prompt_resource_map:
            return self.prompt_resource_map[prompt_id]
        return prompt_id

    @retry_transient_errors(
        operation_name="vertex_prompt_store_fetch", max_retries=3, base_delay_seconds=0.1
    )
    def fetch(
        self,
        prompt_id: str,
        version: str,
        *,
        registry: dict[str, Any] | None = None,
    ) -> tuple[str, str, str]:
        eff_version = version.strip() if version and version.strip() else DEFAULT_PROMPT_VERSION
        if eff_version == "latest":
            resolved = _resolve_latest_from_registry(prompt_id, registry)
            if not resolved and self._registry_loader is not None:
                resolved = _resolve_latest_from_registry(prompt_id, self._registry_loader())
            eff_version = resolved or DEFAULT_PROMPT_VERSION

        resource_ref = self._resolve_resource_id(prompt_id, eff_version)
        if resource_ref.startswith("projects/"):
            resource_path = resource_ref
        else:
            resource_path = (
                f"projects/{self.project_id}/locations/{self.location}/datasets/{resource_ref}"
            )
        url = f"https://{self.location}-aiplatform.googleapis.com/v1beta1/{resource_path}"
        if eff_version and "datasetVersions" not in resource_path:
            url_with_ver = f"{url}?version={quote(eff_version, safe='')}"
        else:
            url_with_ver = url

        try:
            resp = self._raw_request(
                "GET",
                url_with_ver,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError(
                f"Vertex AI Prompt Store timeout for {resource_path}: {exc}"
            ) from exc
        _raise_for_prompt_http_status(resp, "Vertex AI Prompt Store")
        uri = f"vertexai://{resource_path}@{eff_version}"
        return resp.text, uri, eff_version


class VersionedPromptStore:
    """Thread-safe versioned prompt loader supporting `local` (`prompts/`), `gcs`, and `vertex`."""

    def __init__(
        self,
        *,
        backend_mode: PromptBackendMode | None = None,
        default_version: str | None = None,
        prompts_dir: Path | str | None = None,
        gcs_uri: str | None = None,
        vertex_prompt_map: dict[str, str] | None = None,
        project_id: str | None = None,
        location: str | None = None,
        cache_ttl_seconds: float = 300.0,
        fallback_to_local: bool | None = None,
        http_client: httpx.Client | None = None,
        credentials: Any | None = None,
    ) -> None:
        raw_mode = (
            (backend_mode or os.environ.get("EDGAR_PROMPT_STORE_BACKEND", "auto")).strip().lower()
        )
        if raw_mode not in ("auto", "local", "gcs", "vertex"):
            raise ValueError(
                f"Unsupported EDGAR_PROMPT_STORE_BACKEND={raw_mode!r}; "
                "expected 'auto', 'local', 'gcs', or 'vertex'."
            )
        self.backend_mode: PromptBackendMode = raw_mode  # type: ignore[assignment]
        self.default_version = (
            default_version or os.environ.get("EDGAR_PROMPT_VERSION", DEFAULT_PROMPT_VERSION)
        ).strip() or DEFAULT_PROMPT_VERSION
        self.cache_ttl_seconds = cache_ttl_seconds
        self._lock = threading.Lock()
        self._cache: BoundedTTLCache[str, PromptArtifact] = BoundedTTLCache(
            max_entries=256,
            ttl_seconds=cache_ttl_seconds,
        )
        self._loaded_versions: dict[str, dict[str, Any]] = {}

        self.local_backend = LocalPromptStoreBackend(prompts_dir=prompts_dir)

        eff_gcs_uri = (gcs_uri or os.environ.get("EDGAR_PROMPT_GCS_URI", "")).strip()
        self.gcs_backend: GcsPromptStoreBackend | None = (
            GcsPromptStoreBackend(
                eff_gcs_uri,
                http_client=http_client,
                credentials=credentials,
                registry_loader=self.local_backend.load_registry,
            )
            if eff_gcs_uri
            else None
        )

        eff_vertex_map = dict(vertex_prompt_map or {})
        raw_env_map = os.environ.get("EDGAR_VERTEX_PROMPT_MAP", "").strip()
        if raw_env_map and not eff_vertex_map:
            loaded_map = json.loads(raw_env_map)
            if isinstance(loaded_map, dict):
                eff_vertex_map = {str(k): str(v) for k, v in loaded_map.items()}

        self.vertex_backend: VertexPromptStoreBackend | None = (
            VertexPromptStoreBackend(
                project_id=project_id,
                location=location,
                prompt_resource_map=eff_vertex_map,
                http_client=http_client,
                credentials=credentials,
                registry_loader=self.local_backend.load_registry,
            )
            if (eff_vertex_map or self.backend_mode == "vertex")
            else None
        )

        if fallback_to_local is None:
            self.fallback_to_local = self.backend_mode == "auto"
        else:
            self.fallback_to_local = fallback_to_local

    def close(self) -> None:
        """Close pooled HTTP transports held by remote prompt store backends."""
        if self.gcs_backend is not None:
            self.gcs_backend.close()
        if self.vertex_backend is not None:
            self.vertex_backend.close()

    def resolve_active_backend(self) -> ResolvedPromptBackend:
        if self.backend_mode == "local":
            return "local"
        if self.backend_mode == "gcs":
            if self.gcs_backend is None:
                raise RuntimeError(
                    "EDGAR_PROMPT_STORE_BACKEND='gcs' requires EDGAR_PROMPT_GCS_URI (e.g. gs://bucket/prompts)."
                )
            return "gcs"
        if self.backend_mode == "vertex":
            return "vertex"
        # "auto" mode: prefer GCS or Vertex when explicitly configured, else local `prompts/`
        if self.gcs_backend is not None:
            return "gcs"
        if self.vertex_backend is not None:
            return "vertex"
        return "local"

    def load_prompt_artifact(
        self,
        prompt_id: str,
        *,
        version: str | None = None,
        variables: dict[str, Any] | None = None,
        expected_sha256: str | None = None,
    ) -> PromptArtifact:
        """Load, render, hash, and cache a versioned `PromptArtifact`."""
        eff_version = (version or "").strip() or self.default_version
        active_backend = self.resolve_active_backend()
        vars_key = json.dumps(variables or {}, sort_keys=True, default=str)
        cache_key = f"{active_backend}:{prompt_id}:{eff_version}:{vars_key}:{expected_sha256 or ''}"

        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        used_backend: ResolvedPromptBackend = active_backend
        local_registry = (
            self.local_backend.load_registry() if eff_version.split("#", 1)[0] == "latest" else None
        )
        try:
            if active_backend == "gcs" and self.gcs_backend is not None:
                raw_text, uri, resolved_ver = self.gcs_backend.fetch(
                    prompt_id, eff_version, registry=local_registry
                )
            elif active_backend == "vertex" and self.vertex_backend is not None:
                raw_text, uri, resolved_ver = self.vertex_backend.fetch(
                    prompt_id, eff_version, registry=local_registry
                )
            else:
                used_backend = "local"
                raw_text, uri, resolved_ver = self.local_backend.fetch(prompt_id, eff_version)
        except _REMOTE_FETCH_ERRORS as exc:
            if used_backend != "local" and self.fallback_to_local:
                logger.warning(
                    "Remote prompt store (%s) failed for '%s@%s' (%s); falling back to local prompts/.",
                    used_backend,
                    prompt_id,
                    eff_version,
                    exc,
                )
                used_backend = "local"
                raw_text, uri, resolved_ver = self.local_backend.fetch(prompt_id, eff_version)
            else:
                raise

        artifact = _parse_prompt_document(
            raw_text,
            prompt_id=prompt_id,
            requested_version=resolved_ver,
            source_backend=used_backend,
            source_uri=uri,
            override_variables=variables,
            expected_sha256=expected_sha256,
        )

        self._cache[cache_key] = artifact
        with self._lock:
            self._loaded_versions[prompt_id] = artifact.to_public_metadata()

        return artifact

    def load_prompt(
        self,
        prompt_id: str,
        *,
        version: str | None = None,
        variables: dict[str, Any] | None = None,
        expected_sha256: str | None = None,
    ) -> str:
        """Return the rendered prompt string for `prompt_id`."""
        return self.load_prompt_artifact(
            prompt_id,
            version=version,
            variables=variables,
            expected_sha256=expected_sha256,
        ).rendered_text

    def list_registered_prompts(self) -> list[dict[str, Any]]:
        """List all registered prompt specifications from `prompts/registry.yaml`."""
        reg = self.local_backend.load_registry()
        prompts_map = reg.get("prompts") if isinstance(reg.get("prompts"), dict) else {}
        items: list[dict[str, Any]] = []
        for pid, meta in sorted(prompts_map.items()):
            if isinstance(meta, dict):
                versions = sorted((meta.get("versions") or {}).keys())
                items.append(
                    {
                        "prompt_id": pid,
                        "agent_name": meta.get("agent_name", pid),
                        "model": meta.get("model", ""),
                        "latest": meta.get("latest", self.default_version),
                        "available_versions": versions,
                    }
                )
        return items

    def to_status_metadata(self) -> dict[str, Any]:
        with self._lock:
            loaded_snapshot = dict(self._loaded_versions)
        return {
            "requested_backend": self.backend_mode,
            "active_backend": self.resolve_active_backend(),
            "default_version": self.default_version,
            "local_prompts_dir_exists": self.local_backend.prompts_dir.is_dir(),
            "gcs_configured": self.gcs_backend is not None,
            "vertex_configured": self.vertex_backend is not None,
            "registered_prompts_count": len(self.list_registered_prompts()),
            "loaded_prompts": loaded_snapshot,
        }


_PROMPT_STORE_LOCK = threading.Lock()
_GLOBAL_PROMPT_STORE: VersionedPromptStore | None = None


def get_prompt_store() -> VersionedPromptStore:
    """Return the process-wide `VersionedPromptStore` singleton."""
    global _GLOBAL_PROMPT_STORE
    with _PROMPT_STORE_LOCK:
        if _GLOBAL_PROMPT_STORE is None:
            _GLOBAL_PROMPT_STORE = VersionedPromptStore()
        return _GLOBAL_PROMPT_STORE


def configure_prompt_store(
    store: VersionedPromptStore | None = None, **kwargs: Any
) -> VersionedPromptStore:
    """Configure or replace the process-wide `VersionedPromptStore` instance."""
    global _GLOBAL_PROMPT_STORE
    previous: VersionedPromptStore | None = None
    with _PROMPT_STORE_LOCK:
        previous = _GLOBAL_PROMPT_STORE
        _GLOBAL_PROMPT_STORE = store if store is not None else VersionedPromptStore(**kwargs)
        current = _GLOBAL_PROMPT_STORE
    if previous is not None and previous is not current:
        previous.close()
    return current


def reset_prompt_store() -> None:
    """Reset the process-wide `VersionedPromptStore` so env changes take effect cleanly."""
    global _GLOBAL_PROMPT_STORE
    with _PROMPT_STORE_LOCK:
        previous = _GLOBAL_PROMPT_STORE
        _GLOBAL_PROMPT_STORE = None
    if previous is not None:
        previous.close()


def load_agent_prompt(
    prompt_id: str,
    *,
    version: str | None = None,
    variables: dict[str, Any] | None = None,
    expected_sha256: str | None = None,
) -> str:
    """Load a rendered versioned prompt string via the active `VersionedPromptStore`."""
    return get_prompt_store().load_prompt(
        prompt_id,
        version=version,
        variables=variables,
        expected_sha256=expected_sha256,
    )


def load_agent_prompt_artifact(
    prompt_id: str,
    *,
    version: str | None = None,
    variables: dict[str, Any] | None = None,
    expected_sha256: str | None = None,
) -> PromptArtifact:
    """Load the full `PromptArtifact` (including version and SHA-256 digest) for `prompt_id`."""
    return get_prompt_store().load_prompt_artifact(
        prompt_id,
        version=version,
        variables=variables,
        expected_sha256=expected_sha256,
    )


def make_adk_instruction_provider(
    prompt_id: str,
    *,
    version: str | None = None,
    variables: dict[str, Any] | None = None,
) -> Callable[[Any], str]:
    """Build a dynamic ADK `InstructionProvider` resolving `prompt_id` at turn execution time.

    Honors per-session version overrides via `readonly_context.state['prompt_version']`
    (or `readonly_context.state['prompt_versions'][prompt_id]`) when present.
    """

    def _instruction_provider(readonly_context: Any) -> str:
        # Render the models actually configured for this deployment (EDGAR_ANALYST_MODEL /
        # EDGAR_FLASH_MODEL) instead of the defaults recorded in the prompt file.
        from edgar_capstone.agents.context_memory import resolve_model_for_task

        reasoning_model = resolve_model_for_task("financial_reasoning")
        runtime_variables: dict[str, Any] = {
            "analyst_model": reasoning_model,
            "orchestrator_model": reasoning_model,
            "flash_model": resolve_model_for_task("evaluation"),
            **(variables or {}),
        }
        state = getattr(readonly_context, "state", None)
        eff_version = version
        if state is not None and hasattr(state, "get"):
            per_prompt = state.get("prompt_versions")
            if isinstance(per_prompt, dict) and per_prompt.get(prompt_id):
                eff_version = str(per_prompt[prompt_id])
            elif state.get("prompt_version"):
                eff_version = str(state["prompt_version"])
        return get_prompt_store().load_prompt(
            prompt_id,
            version=eff_version,
            variables=runtime_variables,
        )

    _instruction_provider.__name__ = f"load_{prompt_id}_instruction"
    return _instruction_provider


def get_prompt_store_status_metadata() -> dict[str, Any]:
    """Return secret-redacted metadata describing the active `VersionedPromptStore`."""
    return get_prompt_store().to_status_metadata()
