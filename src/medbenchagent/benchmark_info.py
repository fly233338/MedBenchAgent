"""Benchmark brief extraction and deterministic validation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from medbenchagent.llm import complete_json

REQUIRED_FIELDS = ("disease", "modality", "description", "language", "sample_count")


def parse_sample_count(value: object) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value > 0 and value.is_integer():
        return int(value)
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([km]?)\s*", value.lower())
    if not match:
        return None
    number = float(match.group(1))
    multiplier = {"": 1, "k": 1_000, "m": 1_000_000}[match.group(2)]
    count = int(number * multiplier)
    return count if count > 0 else None


def missing_fields(info: dict[str, Any]) -> list[str]:
    missing = [field for field in REQUIRED_FIELDS if not info.get(field)]
    if "sample_count" not in missing and parse_sample_count(info["sample_count"]) is None:
        missing.append("sample_count")
    return missing


def merge_info(existing: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = {**existing}
    for field in REQUIRED_FIELDS:
        value = update.get(field)
        if value not in (None, ""):
            merged[field] = value
    parsed_count = parse_sample_count(merged.get("sample_count"))
    if parsed_count is not None:
        merged["sample_count"] = parsed_count
    return merged


def extract_benchmark_info(requirement: str, existing: dict[str, Any] | None = None) -> dict[str, Any]:
    """Use DeepSeek to extract only the five fields defined in ``blueprint.json``."""
    blueprint_path = Path(__file__).parents[2] / "blueprint.json"
    blueprint = blueprint_path.read_text(encoding="utf-8")
    result = complete_json(
        "Extract benchmark_info faithfully. Return JSON with exactly disease, modality, "
        "description, language, sample_count. Use null for unavailable values. "
        "sample_count may be a number or a k/m scale string.",
        f"Blueprint:\n{blueprint}\n\nCurrent values: {existing or {}}\n\nUser text:\n{requirement}",
    )
    return merge_info(existing or {}, result)
