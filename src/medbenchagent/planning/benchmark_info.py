"""Benchmark brief extraction and deterministic validation."""

from __future__ import annotations

import re
import json
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from medbenchagent.planning.scope import ScopeRequirement
from medbenchagent.workflow.contracts import AllocationPreference

from medbenchagent.runtime.llm import complete_json

REQUIRED_FIELDS = ("language", "sample_count")
OPTIONAL_FIELDS = ("clinical_scope", "diseases", "modalities", "evaluation_focus", "dataset_mentions", "intent_notes")


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
    sample_scope: list[ScopeRequirement] = Field(default_factory=list)
    allocation_preferences: list[AllocationPreference] = Field(default_factory=list)
    evidence: list[dict[str, Any]]
    max_total_tasks_per_image: int | None = Field(default=None, gt=0)

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
    for field in ("language",):
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
                           *, user_inputs: list[str] | None = None,
                           prompt: str) -> dict[str, Any]:
    """Extract only the explicitly requested benchmark-level intent."""
    user_inputs = user_inputs if user_inputs is not None else [requirement]
    result = complete_json(
        prompt,
        f"Current values: {json.dumps(existing or {})}\n\nUser requirement:\n{requirement}"
        f"\n\nQuote evidence only from these numbered user inputs: {json.dumps({i: text for i, text in enumerate(user_inputs)}, ensure_ascii=False)}",
        ExtractedBrief,
        lambda parsed: _validate_extraction(parsed.model_dump(exclude_unset=True), existing, user_inputs),
    )
    return _validate_extraction(result, existing, user_inputs)


def _validate_extraction(result, existing, user_inputs):
    info = merge_info(existing or {}, result)
    raw_scopes = result.get("sample_scope", [])
    if info["modalities"] and not raw_scopes:
        raw_scopes = _derive_modality_scope(info["modalities"], user_inputs)
    if not isinstance(raw_scopes, list):
        raise ValueError("Extraction must include sample_scope as a list, including explicit modalities")
    if raw_scopes or "sample_scope" in result:
        from medbenchagent.planning.scope import ScopeRequirement
        scopes = [ScopeRequirement.model_validate(r).model_dump() for r in raw_scopes]
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
    evidence_fields = {item.get("field") for item in evidence if isinstance(item, dict)}
    if info.get("intent_notes") and "intent_notes" not in evidence_fields:
        info["intent_notes"] = []
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
    from medbenchagent.workflow.contracts import AllocationPreference
    preferences = [AllocationPreference.model_validate(p).model_dump() for p in preferences]
    if len({p["id"] for p in preferences}) != len(preferences) or any(not any(p["quote"] in text for text in user_inputs) for p in preferences):
        raise ValueError("Allocation preferences require unique IDs and exact user quotations")
    info["allocation_preferences"] = preferences
    info["evidence"] = evidence
    return info


def _derive_modality_scope(modalities: list[str], user_inputs: list[str]) -> list[dict[str, Any]]:
    """Build a scope only when every normalized modality has a source quote."""
    aliases = {
        "ultrasound": ("ultrasound", "sonography", "breast-us", "us"),
        "computed tomography": ("computed tomography", "ct"),
        "magnetic resonance imaging": ("magnetic resonance imaging", "mri"),
        "x-ray": ("x-ray", "xray", "radiograph"),
    }
    text = "\n".join(user_inputs)
    quotes = []
    for modality in modalities:
        candidates = aliases.get(modality.casefold(), (modality,))
        match = next((candidate for candidate in candidates if candidate.casefold() in text.casefold()), None)
        if match is None:
            return []
        start = text.casefold().find(match.casefold())
        quotes.append(text[start:start + len(match)])
    return [{"id": "scope-modality", "kind": "modality", "description": "Explicit modality selection",
             "quote": " or ".join(quotes), "modalities": modalities, "operator": "in", "logic": "any"}]
