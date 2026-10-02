"""The single registered Blueprint used for post-tree capability reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_blueprint(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Blueprint does not exist: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("name") != "Blueprint":
        raise ValueError("Blueprint must have name=Blueprint")
    dimensions = value.get("capability_dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        raise ValueError("Blueprint requires capability_dimensions")
    ids = [row.get("id") for row in dimensions]
    if any(not isinstance(identifier, str) or not identifier for identifier in ids):
        raise ValueError("Blueprint capability dimensions require nonempty IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("Blueprint capability dimension IDs must be unique")
    concepts = value.get("task_concepts", [])
    if not isinstance(concepts, list):
        raise ValueError("Blueprint task_concepts must be a list")
    return value


def blueprint_dimension_ids(blueprint: dict[str, Any]) -> set[str]:
    return {row["id"] for row in blueprint.get("capability_dimensions", [])}
