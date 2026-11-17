"""Streaming JSONL and JSON file I/O utilities."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def iter_jsonl(
    path: Path,
    *,
    skip_corrupt: bool = False,
) -> Iterator[dict[str, Any]]:
    """Stream JSON objects line-by-line from a ``.jsonl`` file without loading it all into RAM."""
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                if skip_corrupt:
                    logger.warning("Skipping corrupt JSONL line %d in %s: %s", line_no, path, exc)
                    continue
                raise
            if isinstance(parsed, dict):
                yield parsed


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write an iterable of dict records to a JSONL file, returning the number of rows written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, default=str))
            fh.write("\n")
            count += 1
    return count


def write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Write a JSON document creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=indent, default=str), encoding="utf-8")
