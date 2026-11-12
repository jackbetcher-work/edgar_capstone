#!/usr/bin/env python3
"""CI/CD pipeline gate verifying zero hardcoded secrets or keys in the repository."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from edgar_capstone.security.secrets import (  # noqa: E402
    scan_repository_for_exposed_secrets,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify no secrets or private keys are exposed in pipeline or source files."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the repository root directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    findings = scan_repository_for_exposed_secrets(repo_root)
    if findings:
        for finding in findings:
            print(
                f"ERROR: Exposed secret detected in {finding.source}:{finding.line_number} "
                f"(rule={finding.rule_id})",
                file=sys.stderr,
            )
        return 1

    print("Secret hygiene check passed: 0 hardcoded secrets or keys found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
