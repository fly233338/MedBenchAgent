"""Stable helpers made available to generated preparation adapters."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def stable_id(*parts: object, prefix: str = "") -> str:
    payload = "\x1f".join(json.dumps(p, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for p in parts)
    return prefix + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def write_jsonl(path: str | Path, rows) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return str(target)


def write_provenance(path, *, source_id, locator, fields, rule):
    return write_jsonl(path, [{"source_id": source_id, "locator": locator, "fields": fields, "rule": rule}])


def write_transform(path, *, image_id, operations):
    return write_jsonl(path, [{"image_id": image_id, "operations": operations}])
