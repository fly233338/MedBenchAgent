"""Benchmark brief extraction and deterministic validation."""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Any

from medbenchagent.runtime.llm import complete_json
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.settings import get_settings

REQUIRED_FIELDS = ("disease", "modality", "description", "language", "sample_count")

EXTRACTION_PROMPT = """Extract a medical VLM benchmark request into one JSON object.
Return exactly these five keys and no wrapper object:
{
  "disease": "English disease name or null",
  "modality": "English imaging modality or null",
  "description": "One complete English sentence describing only the benchmark aspects explicitly stated by the user. Mention the benchmark objective, disease, modality, or task focus only when that aspect is explicitly provided. Omit any unspecified aspect instead of inferring or inventing it. Never add task types, clinical goals, diagnoses, evaluation criteria, or dataset properties based on common medical benchmark conventions; never return only keywords or a phrase.",
  "language": "en, zh, or null",
  "sample_count": "positive integer or null"
}

The user may write Chinese, English, or both. Infer the requested benchmark language,
not the language of the user's message. Translate disease and modality to standard
English medical terms. Convert expressions such as "20题", "10k", and "2.5k" to a
positive integer. Use null only when the user did not provide or imply a value.
Every detail in the description must be traceable to the user's wording. If the user
did not mention a task focus, do not add one.
Return valid JSON only."""


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
    """Use DeepSeek to fill only the five benchmark-info fields."""
    result = complete_json(
        EXTRACTION_PROMPT,
        f"Current values: {json.dumps(existing or {})}\n\nUser requirement:\n{requirement}",
    )
    return merge_info(existing or {}, result)


def write_working_blueprint(
    benchmark_info: dict[str, Any], output_path: str | Path, template_path: Path | None = None
) -> str:
    """Copy the immutable blueprint template and replace only its benchmark-info section."""
    blueprint = json.loads((template_path or get_settings().paths.blueprint_template).read_text(encoding="utf-8"))
    blueprint["benchmark_info"] = benchmark_info
    destination = Path(output_path)
    if destination.suffix != ".json":
        destination = destination / "blueprint.json"
    atomic_write_json(destination, blueprint)
    return str(destination)
