"""Pipeline secret-leak scanning utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Secret-Exposure & Hardcoded Credential Scanner for CI/CD Pipelines
# ---------------------------------------------------------------------------

_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_key_block",
        re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    ),
    (
        "gcp_service_account_json_key",
        re.compile(r'"type"\s*:\s*"service_account"[\s\S]{0,400}"private_key_id"\s*:'),
    ),
    (
        "google_api_key",
        re.compile(r"\bAIza[0-9A-Za-z\-_]{30,45}\b"),
    ),
    (
        "google_oauth_access_token",
        re.compile(r"\bya29\.[0-9A-Za-z\-_]{20,}\b"),
    ),
    (
        "github_pat_token",
        re.compile(
            r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b|\bgithub_pat_[A-Za-z0-9_]{22,}\b"
        ),
    ),
    (
        "aws_access_key_id",
        re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    ),
    (
        "slack_token",
        re.compile(r"\bxox[baprs]-[0-9A-Za-z\-]{10,}\b"),
    ),
    (
        "hardcoded_secret_assignment",
        re.compile(
            r"(?i)\b(?:[a-z0-9_]*_)?(?:api[_-]?key|client[_-]?secret|private[_-]?key|access[_-]?token|auth[_-]?token|password|passwd)"
            r"\s*[:=]\s*['\"](?!\$\{|\$[A-Z_]|projects/|sec-edgar-|\*\*\*REDACTED\*\*\*|YOUR_|REPLACE_|example|dummy|test|fake|none|null)[A-Za-z0-9_\-\./+=]{16,}['\"]"
        ),
    ),
)

_MULTILINE_RULE_IDS: frozenset[str] = frozenset(
    {"private_key_block", "gcp_service_account_json_key"}
)
_MULTILINE_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (rule_id, pat) for rule_id, pat in _SECRET_PATTERNS if rule_id in _MULTILINE_RULE_IDS
)
_LINE_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (rule_id, pat) for rule_id, pat in _SECRET_PATTERNS if rule_id not in _MULTILINE_RULE_IDS
)

_SKIPPED_DIR_PARTS: frozenset[str] = frozenset({".terraform", "__pycache__", ".git", ".venv"})

_REPO_SCAN_GLOBS: tuple[str, ...] = (
    ".github/workflows/*.yml",
    ".github/workflows/*.yaml",
    "cloudbuild.yaml",
    "Dockerfile",
    "terraform/**/*.tf",
    "terraform/**/*.tfvars.example",
    "prompts/**/*.yaml",
    "prompts/**/*.yml",
    "prompts/**/*.md",
    "src/**/*.py",
    "scripts/**/*.py",
)


@dataclass(frozen=True)
class SecretFinding:
    """Represents a potential exposed secret or credential found during scanning."""

    source: str
    rule_id: str
    line_number: int
    snippet_redacted: str


def scan_text_for_exposed_secrets(
    content: str,
    *,
    source_name: str = "<memory>",
) -> list[SecretFinding]:
    """Scan text content for hardcoded keys, tokens, or credentials."""
    findings: list[SecretFinding] = []

    # Check multi-line patterns (e.g. JSON service account keys or PEM blocks)
    for rule_id, pattern in _MULTILINE_SECRET_PATTERNS:
        for match in pattern.finditer(content):
            line_no = content[: match.start()].count("\n") + 1
            findings.append(
                SecretFinding(
                    source=source_name,
                    rule_id=rule_id,
                    line_number=line_no,
                    snippet_redacted=f"[{rule_id} detected on line {line_no}]",
                )
            )

    # Check line-by-line patterns
    for idx, line in enumerate(content.splitlines(), start=1):
        for rule_id, pattern in _LINE_SECRET_PATTERNS:
            if pattern.search(line):
                findings.append(
                    SecretFinding(
                        source=source_name,
                        rule_id=rule_id,
                        line_number=idx,
                        snippet_redacted=f"[{rule_id} detected on line {idx}]",
                    )
                )

    return findings


def scan_repository_for_exposed_secrets(repo_root: Path) -> list[SecretFinding]:
    """Scan pipeline, infrastructure, and source files in the repository for exposed secrets."""
    resolved_root = repo_root.resolve()
    seen_paths: set[Path] = set()
    scan_targets: list[Path] = []

    for pattern in _REPO_SCAN_GLOBS:
        for candidate in sorted(resolved_root.glob(pattern)):
            if not candidate.is_file():
                continue
            rel_parts = candidate.relative_to(resolved_root).parts
            if any(part in _SKIPPED_DIR_PARTS for part in rel_parts):
                continue
            resolved = candidate.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            scan_targets.append(candidate)

    all_findings: list[SecretFinding] = []
    for path in scan_targets:
        rel = str(path.relative_to(resolved_root))
        text = path.read_text(encoding="utf-8", errors="replace")
        all_findings.extend(scan_text_for_exposed_secrets(text, source_name=rel))

    return all_findings
