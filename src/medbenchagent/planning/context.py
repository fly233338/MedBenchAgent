"""Explicit, bounded evidence packets; required evidence is never silently discarded."""
from __future__ import annotations
import json
from medbenchagent.runtime.llm import RequestContextBudgetError


def complete_planning_json(*args, **kwargs):
    """Use the same final-request budget for all direct planning/review calls."""
    from medbenchagent.runtime.llm import complete_json
    from medbenchagent.settings import get_settings
    return complete_json(*args, request_char_budget=get_settings().planning.context_chars, **kwargs)


class EvidenceContextBudgetError(RequestContextBudgetError):
    def __init__(self, message):
        ValueError.__init__(self, message)
        self.diagnostics = {"budget": "planning.context_chars", "reason": "required_evidence"}


def bounded_evidence(value, *, char_budget=64000):
    result = dict(value)
    required = set(result.get("required_evidence_names", []))
    catalog = list(result.get("evidence_catalog", []))
    catalog.sort(key=lambda row: row.get("name") not in required)
    mandatory = [row for row in catalog if row.get("name") in required]
    optional = [row for row in catalog if row.get("name") not in required]
    result["evidence_catalog"] = mandatory + optional[:max(0, 30 - len(mandatory))]
    result["representative_cases"] = list(result.get("representative_cases", []))[:6]
    result["existing_leaf_tasks"] = list(result.get("existing_leaf_tasks", []))[:20]
    omitted = {row.get("name") for row in catalog} - {row.get("name") for row in result["evidence_catalog"]}
    result["omissions"] = {"evidence_names": sorted(omitted), "sample_count": 0, "leaf_count": 0,
                            "read_tools": ["read_evidence_catalog", "read_field_values", "read_annotation_semantics", "read_leaf_tasks"]}
    result["omissions"]["sample_count"] = max(0, len(value.get("representative_cases", [])) - 6)
    result["omissions"]["leaf_count"] = max(0, len(value.get("existing_leaf_tasks", [])) - 20)
    def size():
        return len(json.dumps(result, ensure_ascii=False))
    while size() > char_budget:
        candidates = [row for row in result["evidence_catalog"] if row.get("name") not in required]
        if candidates:
            row = candidates[-1]
            result["evidence_catalog"].remove(row)
            result["omissions"]["evidence_names"].append(row.get("name"))
        elif result["representative_cases"]:
            result["representative_cases"].pop()
            result["omissions"]["sample_count"] += 1
        elif result["existing_leaf_tasks"]:
            result["existing_leaf_tasks"].pop()
            result["omissions"]["leaf_count"] += 1
        else:
            raise EvidenceContextBudgetError("Required planning evidence exceeds the context budget; narrow the evidence query or increase planning.context_chars")
    return result
