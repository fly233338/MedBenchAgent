"""Benchmark brief extraction and deterministic validation."""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Any

from medbenchagent.llm import complete_json
from medbenchagent.run_artifacts import atomic_write_json

REQUIRED_FIELDS = ("disease", "modality", "description", "language", "sample_count")
TEMPLATE_BLUEPRINT_PATH = Path(__file__).parents[2] / "blueprint.json"

EXTRACTION_PROMPT = """Extract a medical VLM benchmark request into one JSON object.
Return exactly these five keys and no wrapper object:
{
  "disease": "English disease name or null",
  "modality": "English imaging modality or null",
  "description": "One complete English sentence describing the benchmark objective, disease, modality, and requested task focus; never return only keywords or a phrase.",
  "language": "en, zh, or null",
  "sample_count": "positive integer or null"
}

The user may write Chinese, English, or both. Infer the requested benchmark language,
not the language of the user's message. Translate disease and modality to standard
English medical terms. Convert expressions such as "20题", "10k", and "2.5k" to a
positive integer. Use null only when the user did not provide or imply a value.
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


def extract_benchmark_field(field: str, user_answer: str) -> object:
    """Normalize one explicitly supplied BenchmarkInfo value without revisiting other fields."""
    if field not in REQUIRED_FIELDS:
        raise ValueError(f"Unsupported BenchmarkInfo field: {field}")
    result = complete_json(
        f"""Extract only the {field!r} value for a medical VLM benchmark.
Return exactly one JSON object with one key: {{"{field}": value}}.
The user may answer in Chinese or English. Translate disease and modality to standard
English terms. For language return only en or zh. For sample_count return a positive
integer, converting expressions such as 20题, 10k, and 2.5k. Do not return any other
BenchmarkInfo fields. Return valid JSON only.""",
        f"User answer for {field}:\n{user_answer}",
    )
    return result.get(field)


def write_working_blueprint(
    benchmark_info: dict[str, Any], output_path: str | Path, template_path: Path = TEMPLATE_BLUEPRINT_PATH
) -> str:
    """Copy the immutable blueprint template and replace only its benchmark-info section."""
    blueprint = json.loads(template_path.read_text(encoding="utf-8"))
    blueprint["benchmark_info"] = benchmark_info
    destination = Path(output_path)
    if destination.suffix != ".json":
        destination = destination / "blueprint.json"
    atomic_write_json(destination, blueprint)
    return str(destination)
