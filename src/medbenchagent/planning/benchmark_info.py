"""Benchmark brief extraction and deterministic validation."""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from medbenchagent.planning.scope import ScopeRequirement
from medbenchagent.runtime.tool_contracts import AllocationPreference

from medbenchagent.runtime.llm import complete_json
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.settings import get_settings

REQUIRED_FIELDS = ("clinical_scope", "language", "sample_count")
OPTIONAL_FIELDS = ("diseases", "modalities", "evaluation_focus", "dataset_mentions", "intent_notes")


class ExtractedBrief(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    clinical_scope: str | None = None
    diseases: list[str] = Field(default_factory=list)
    modalities: list[str] = Field(default_factory=list)
    evaluation_focus: str | None = None
    dataset_mentions: list[str] = Field(default_factory=list)
    intent_notes: list[str] = Field(default_factory=list)
    language: str | None = None
    # Explicit invalid values must survive extraction so the user can correct them.
    sample_count: Any = None
    sample_scope: list[ScopeRequirement]
    allocation_preferences: list[AllocationPreference] = Field(default_factory=list)
    evidence: list[dict[str, Any]]
    max_total_tasks_per_image: int | None = Field(default=None, gt=0)

EXTRACTION_PROMPT = """Act as the benchmark-intake analyst. Extract the user's intended medical VLM benchmark
from the complete request and clarification history into one JSON object. Preserve the user's intent even when it
is broad, comparative, multi-dataset, or expressed with unfamiliar task wording. Do not force an intent into a disease
or modality when the user did not say that.

Return these keys and no wrapper object:
{
  "clinical_scope": "English description of only the benchmark aspects explicitly stated by the user, or null. A broad scope or explicitly requested dataset population is valid; do not demand a single disease.",
  "diseases": ["Explicitly requested disease names in English; empty if unspecified"],
  "modalities": ["Explicitly requested imaging modalities in English; empty if unspecified"],
  "evaluation_focus": "Only explicitly requested task priorities and exclusions, or null",
  "dataset_mentions": ["Dataset names or IDs explicitly named by the user; empty if unspecified"],
  "intent_notes": ["Short English summaries of explicit constraints that do not fit the fixed fields"],
  "language": "en, zh, or null",
  "sample_count": "positive integer or null",
  "sample_scope": [{"id":"scope-1", "kind":"modality or population", "description":"Explicit sample inclusion/exclusion requirement", "quote":"Exact original user wording", "modalities": ["Standard modality names covered by this scope; empty for population"], "operator":"in or not_in", "logic":"any for alternatives, all only for explicit simultaneous multi-sequence requirements"}],
  "allocation_preferences": [{"id":"preference-1", "description":"Only an explicit weighting or exclusion preference", "quote":"Exact user wording"}],
  "evidence": [{"field":"clinical_scope", "quote":"Exact original user wording", "input_index":0}],
  "max_total_tasks_per_image": "Explicit positive integer total across all task types, or null"
}

The user may write Chinese, English, or both. Infer the requested benchmark language,
not the language of the user's message. Translate disease and modality to standard
English medical terms. Convert expressions such as "20题", "10k", and "2.5k" to a
positive integer. Use null only when the user did not provide or imply a value.
If the user explicitly supplied an invalid sample count, preserve that value (or its original text),
never replace it with null. Later user corrections override earlier requests.
Omit any unspecified aspect instead of inferring or inventing it.
Every nonempty explicit field must have at least one evidence entry. Evidence quotes must be copied verbatim from
the numbered user inputs; never quote an assistant question. `input_index` is zero-based and must identify the input
containing the quote. Evidence is a source pointer, not a normalized-value equality check. If a value is inferred from
a clarification or configuration default, do not fabricate a quote for it.
sample_scope must cover EVERY explicit modality and population restriction. Distinguish
testing disease knowledge from selecting only cases with that disease. The former is
evaluation_focus, not a positive-case filter. Do not invent restrictions. Use an empty
list when none were requested. Quote the user's original language, without translation. Keep `dataset_mentions` as
user-facing names; do not resolve them to catalog IDs in this step.
Every explicit nonempty field, including invalid explicit values and max_total_tasks_per_image, requires evidence with
its field and exact user quote.
Unspecified values require no evidence. Do not quote assistant clarification questions as user evidence.
Always return sample_scope, even when empty. Group alternatives such as CT or MRI in ONE
modality scope listing both modalities; resolve them with one in condition, not intersecting filters.
Use at most one modality scope per operator: put all inclusion alternatives in one in scope,
all exclusions in one not_in scope. Do not turn alternatives into AND conditions.
allocation_preferences must be empty unless the user explicitly prioritizes/excludes a concept,
dataset or task for allocation. A clinical evaluation topic alone is NOT a weighting preference.
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
    if "language" not in missing and info["language"] not in {"en", "zh"}:
        missing.append("language")
    return missing


def merge_info(existing: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = {k: v for k, v in existing.items() if k in REQUIRED_FIELDS + OPTIONAL_FIELDS}
    for field in REQUIRED_FIELDS + OPTIONAL_FIELDS:
        value = update.get(field)
        if field in ("diseases", "modalities", "dataset_mentions", "intent_notes") and value is not None:
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
    for field in ("diseases", "modalities", "dataset_mentions", "intent_notes"):
        merged.setdefault(field, [])
    merged.setdefault("evaluation_focus", None)
    parsed_count = parse_sample_count(merged.get("sample_count"))
    if parsed_count is not None:
        merged["sample_count"] = parsed_count
    return merged


def extract_benchmark_info(requirement: str, existing: dict[str, Any] | None = None,
                           *, user_inputs: list[str] | None = None) -> dict[str, Any]:
    """Extract only the explicitly requested benchmark-level intent."""
    user_inputs = user_inputs if user_inputs is not None else [requirement]
    result = complete_json(
        EXTRACTION_PROMPT,
        f"Current values: {json.dumps(existing or {})}\n\nUser requirement:\n{requirement}"
        f"\n\nQuote evidence only from these numbered user inputs: {json.dumps({i: text for i, text in enumerate(user_inputs)}, ensure_ascii=False)}",
        ExtractedBrief,
        lambda parsed: _validate_extraction(parsed.model_dump(exclude_unset=True), existing, user_inputs),
    )
    return _validate_extraction(result, existing, user_inputs)


def _validate_extraction(result, existing, user_inputs):
    info = merge_info(existing or {}, result)
    if not isinstance(result.get("sample_scope"), list):
        raise ValueError("Extraction must include sample_scope as a list, including explicit modalities")
    if "sample_scope" in result:
        from medbenchagent.planning.scope import ScopeRequirement
        scopes = [ScopeRequirement.model_validate(r).model_dump() for r in result["sample_scope"]]
        if len({r["id"] for r in scopes}) != len(scopes) or any(not any(r["quote"] in text for text in user_inputs) for r in scopes):
            raise ValueError("Sample scope requires unique IDs and exact user quotations")
        for operator in ("in", "not_in"):
            if sum(r["kind"] == "modality" and r["operator"] == operator for r in scopes) > 1:
                raise ValueError("Combine modality alternatives into one scope per operator; do not intersect alternatives")
        included = {m for r in scopes if r["kind"] == "modality" and r["operator"] == "in" for m in r["modalities"]}
        excluded = {m for r in scopes if r["kind"] == "modality" and r["operator"] == "not_in" for m in r["modalities"]}
        if included and not included - excluded:
            raise ValueError("Modality inclusion and exclusion conflict; clarify the requirement")
        covered = {m for r in scopes for m in r["modalities"]}
        if covered != set(info["modalities"]):
            raise ValueError("Sample scope must cover exactly all explicit modalities")
        info["sample_scope"] = scopes
    if "max_total_tasks_per_image" in result:
        limit = result["max_total_tasks_per_image"]
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("Total tasks per image must be a positive integer")
        info["max_total_tasks_per_image"] = limit
    evidence = result.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("Extraction requires evidence for explicit values")
    expected = [(field, info.get(field)) for field in (*REQUIRED_FIELDS, *OPTIONAL_FIELDS, "max_total_tasks_per_image")
                if info.get(field) not in (None, "", [])]
    allowed_fields = set(REQUIRED_FIELDS + OPTIONAL_FIELDS +
                         ("sample_scope", "allocation_preferences", "max_total_tasks_per_image"))
    for item in evidence:
        if not isinstance(item, dict) or not isinstance(item.get("field"), str):
            raise ValueError("Extraction evidence must identify a field")
        if item["field"] not in allowed_fields:
            raise ValueError("Extraction evidence references an unknown field")
        quote = item.get("quote")
        index = item.get("input_index")
        if not isinstance(quote, str) or not quote.strip():
            raise ValueError("Extraction evidence requires a nonempty exact quote")
        if index is not None and (type(index) is not int or index < 0 or index >= len(user_inputs)):
            raise ValueError("Extraction evidence input_index is out of range")
        haystack = user_inputs[index] if index is not None else "\n".join(user_inputs)
        if quote not in haystack:
            raise ValueError("Extraction evidence quote must occur in the referenced user input")
    explicit_fields = {field for field, value in expected if field != "sample_count"}
    if result.get("sample_count") is not None:
        explicit_fields.add("sample_count")
    covered = {item["field"] for item in evidence if isinstance(item, dict)}
    if missing := sorted(explicit_fields - covered):
        raise ValueError("Extraction evidence is missing explicit fields: " + ", ".join(missing))
    preferences = result.get("allocation_preferences", [])
    if not isinstance(preferences, list):
        raise ValueError("allocation_preferences must be a list")
    from medbenchagent.runtime.tool_contracts import AllocationPreference
    preferences = [AllocationPreference.model_validate(p).model_dump() for p in preferences]
    if len({p["id"] for p in preferences}) != len(preferences) or any(not any(p["quote"] in text for text in user_inputs) for p in preferences):
        raise ValueError("Allocation preferences require unique IDs and exact user quotations")
    info["allocation_preferences"] = preferences
    info["evidence"] = evidence
    return info


def write_working_blueprint(
    benchmark_info: dict[str, Any], output_path: str | Path, template_path: Path | None = None
) -> str:
    """Write a v2 benchmark brief; legacy concepts remain only as an input snapshot."""
    template = template_path or get_settings().paths.blueprint_template
    blueprint = json.loads(template.read_text(encoding="utf-8")) if template.is_file() else {}
    blueprint = {"schema_version": "benchmark-brief.v2", "benchmark_info": benchmark_info,
                 "concepts": blueprint.get("concepts", {})}
    destination = Path(output_path)
    if destination.suffix != ".json":
        destination = destination / "blueprint.json"
    atomic_write_json(destination, blueprint)
    return str(destination)
