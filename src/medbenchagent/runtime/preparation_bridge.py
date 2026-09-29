"""Adapt preparation service snapshots into the existing workflow context."""
from __future__ import annotations

from pathlib import Path


def initialize_preparation_context(context: dict, *, sources, workspace: str | Path, goal: str = "") -> dict:
    updated = dict(context)
    updated["preparation_workspace"] = str(Path(workspace).resolve())
    updated["preparation_sources"] = list(sources)
    updated["preparation_goal"] = goal
    updated["preparation_status"] = "running"
    return updated


def apply_preparation_snapshot(context: dict, snapshot: dict) -> dict:
    updated = dict(context)
    updated["preparation_status"] = snapshot.get("status", "")
    updated["preparation_session_ref"] = snapshot.get("workspace")
    if snapshot.get("registered_dataset_ref"):
        updated["registered_dataset_ref"] = snapshot["registered_dataset_ref"]
    if snapshot.get("questions"):
        updated["issues"] = snapshot["questions"]
    return updated
