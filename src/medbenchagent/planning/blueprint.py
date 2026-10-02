"""Validation for the registered horizontal Capability Category registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medbenchagent.planning.task_space import CapabilityCategory


def load_blueprint(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Blueprint does not exist: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("name") != "Blueprint":
        raise ValueError("Blueprint must have name=Blueprint")
    categories = value.get("capability_categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError("Blueprint requires capability_categories")
    ids = [row.get("id") for row in categories]
    expected = {category.value for category in CapabilityCategory}
    if set(ids) != expected or len(ids) != len(set(ids)):
        raise ValueError("Blueprint must register exactly the six Capability Categories")
    for row in categories:
        if not isinstance(row.get("description"), str) or not row["description"].strip():
            raise ValueError("Capability Category descriptions are required")
        if not isinstance(row.get("examples"), list):
            raise ValueError("Capability Category examples must be a list")
    operations = value.get("operations", [])
    if not isinstance(operations, list):
        raise ValueError("Blueprint operations must be a list")
    return value


def capability_category_ids(blueprint: dict[str, Any]) -> set[str]:
    return {row["id"] for row in blueprint.get("capability_categories", [])}


def blueprint_dimension_ids(blueprint: dict[str, Any]) -> set[str]:
    """Deprecated alias for callers that still use the old term."""
    return capability_category_ids(blueprint)
