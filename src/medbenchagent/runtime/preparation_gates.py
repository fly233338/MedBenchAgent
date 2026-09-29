"""Deterministic validation of stage proposals."""
from __future__ import annotations

import json
from pathlib import Path


def read_proposal(path: str | Path, *, turn_id: str, stage: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != "dataset-agent-proposal.v1":
        raise ValueError("Unsupported preparation proposal schema")
    if value.get("turn_id") != turn_id or value.get("stage") != stage:
        raise ValueError("Proposal does not belong to the active turn/stage")
    if value.get("status") not in {"submitted", "needs_input", "blocked", "failed", "cancelled", "budget_exhausted"}:
        raise ValueError("Invalid preparation proposal status")
    return value


def required_paths(proposal: dict, workspace: Path) -> list[Path]:
    paths = proposal.get("artifact_paths", {})
    return [workspace / path for path in paths.values() if isinstance(path, str)]
