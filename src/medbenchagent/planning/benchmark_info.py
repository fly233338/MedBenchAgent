"""Benchmark brief extraction and deterministic validation."""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Any

from medbenchagent.runtime.llm import complete_json
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.settings import get_settings

REQUIRED_FIELDS = ("clinical_scope", "language", "sample_count")
OPTIONAL_FIELDS = ("diseases", "modalities", "evaluation_focus")

EXTRACTION_PROMPT = """Extract a medical VLM benchmark request into one JSON object.
Return these keys and no wrapper object:
{
  "clinical_scope": "English description of only the benchmark aspects explicitly stated by the user, or null. A broad scope or explicitly requested dataset population is valid; do not demand a single disease.",
  "diseases": ["Explicitly requested disease names in English; empty if unspecified"],
  "modalities": ["Explicitly requested imaging modalities in English; empty if unspecified"],
  "evaluation_focus": "Only explicitly requested task priorities and exclusions, or null",
  "language": "en, zh, or null",
  "sample_count": "positive integer or null",
  "sample_scope": [{"id":"scope-1", "kind":"modality or population", "description":"Explicit sample inclusion/exclusion requirement", "quote":"Exact original user wording"}],
  "max_total_tasks_per_image": "Explicit positive integer total across all task types, or null"
}

The user may write Chinese, English, or both. Infer the requested benchmark language,
not the language of the user's message. Translate disease and modality to standard
English medical terms. Convert expressions such as "20题", "10k", and "2.5k" to a
positive integer. Use null only when the user did not provide or imply a value.
If the user explicitly supplied an invalid sample count, preserve that value (or its original text),
never replace it with null. Later user corrections override earlier requests.
Omit any unspecified aspect instead of inferring or inventing it.
Every detail must be traceable to the user's wording. If the user
did not mention a task focus, do not add one.
sample_scope must cover EVERY explicit modality and population restriction. Distinguish
testing disease knowledge from selecting only cases with that disease. The former is
evaluation_focus, not a positive-case filter. Do not invent restrictions. Use an empty
list when none were requested. Quote the user's original language, without translation.
Return valid JSON only."""


def parse_sample_count(value: object) -> int | None:
    if type(value) is int and value > 0:
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
    count = number * multiplier
    return int(count) if count > 0 and count.is_integer() else None


def missing_fields(info: dict[str, Any]) -> list[str]:
    missing = [field for field in REQUIRED_FIELDS if not info.get(field)]
    if "sample_count" not in missing and parse_sample_count(info["sample_count"]) is None:
        missing.append("sample_count")
    for field in ("clinical_scope", "language"):
        if field not in missing and (not isinstance(info[field], str) or not info[field].strip()):
            missing.append(field)
    return missing


def merge_info(existing: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = {k: v for k, v in existing.items() if k in REQUIRED_FIELDS + OPTIONAL_FIELDS}
    for field in REQUIRED_FIELDS + OPTIONAL_FIELDS:
        value = update.get(field)
        if field in ("diseases", "modalities") and value is not None:
            if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value):
                raise ValueError(f"{field} must be a list of nonempty strings")
            value = list(dict.fromkeys(v.strip() for v in value))
        if field in ("clinical_scope", "language", "evaluation_focus") and value is not None:
            if not isinstance(value, str):
                raise ValueError(f"{field} must be text")
            value = value.strip()
        if field == "sample_count" and value is not None:
            merged[field] = value
        elif value not in (None, ""):
            merged[field] = value
        elif field == "evaluation_focus" and field in update:
            merged[field] = None
    for field in ("diseases", "modalities"):
        merged.setdefault(field, [])
    merged.setdefault("evaluation_focus", None)
    parsed_count = parse_sample_count(merged.get("sample_count"))
    if parsed_count is not None:
        merged["sample_count"] = parsed_count
    return merged


def extract_benchmark_info(requirement: str, existing: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract only the explicitly requested benchmark-level intent."""
    result = complete_json(
        EXTRACTION_PROMPT,
        f"Current values: {json.dumps(existing or {})}\n\nUser requirement:\n{requirement}",
    )
    info = merge_info(existing or {}, result)
    if "sample_scope" in result:
        from medbenchagent.planning.scope import ScopeRequirement
        scopes = [ScopeRequirement.model_validate(r).model_dump() for r in result["sample_scope"]]
        if len({r["id"] for r in scopes}) != len(scopes) or any(r["quote"] not in requirement for r in scopes):
            raise ValueError("Sample scope requires unique IDs and exact user quotations")
        if info["modalities"] and not any(r["kind"] == "modality" for r in scopes):
            raise ValueError("Explicit modalities require executable sample scope requirements")
        info["sample_scope"] = scopes
    if "max_total_tasks_per_image" in result:
        limit = result["max_total_tasks_per_image"]
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("Total tasks per image must be a positive integer")
        info["max_total_tasks_per_image"] = limit
    return info


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
