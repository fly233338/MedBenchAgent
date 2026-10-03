"""Resumable, evidence-staged Task Planning Agent."""

from __future__ import annotations

import json
import difflib
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from medbenchagent.data.task_facts import build_dataset_facts, case_facts
from medbenchagent.planning.node_builder import build_task_scenarios
from medbenchagent.planning.access import DatasetAccessReceipt
from medbenchagent.planning.resolver import NotApplicable, ResolverSpec, resolve
from medbenchagent.planning.bir import BIRStore, RegistrySnapshot
from medbenchagent.planning.task_discovery import leaf_identity, validate_candidate_leaf
from medbenchagent.planning.task_space import (
    AXIS_VALUES,
    AxisAnalysis,
    AxisDecisionSet,
    AxisValueDecision,
    CapabilityCategory,
    EvidenceBasis,
    EvidenceSelector,
    LeafTask,
    ScenarioReview,
    TaskAxis,
    TaskPlanState,
    TaskParadigm,
    OutputSpec,
)
from medbenchagent.runtime.dependencies import publish, require
from medbenchagent.runtime.harness import compile_agent, result
from medbenchagent.runtime.llm import complete_json
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.runtime.tool_contracts import NoArguments


class LeafTaskValidationError(ValueError):
    """A repairable Leaf failure with model-visible structured diagnostics."""

    def __init__(self, message: str, diagnostics: dict[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics


class AxisArguments(NoArguments):
    axis: TaskAxis = Field(description="Axis being assessed; use the next axis returned by planning context.")
    assessments: list[AxisValueDecision] | None = Field(
        default=None,
        description=(
            "One assessment for every allowed value, using only value, supported, "
            "evidence_basis[{reference,detail}], and reason."
        ),
    )

    @field_validator("axis", mode="before")
    @classmethod
    def parse_axis(cls, value):
        return TaskAxis(value)


class AxisAnalysisArguments(NoArguments):
    analysis: dict[str, Any]


class ScenarioArguments(NoArguments):
    scenario_id: str | None = None


class CandidateScenarioReviewArguments(NoArguments):
    scenario_id: str = Field(min_length=1)


class ScenarioScreeningArguments(NoArguments):
    scenario_id: str = Field(min_length=1)
    decision: Literal["keep", "exclude"] | None = None
    reason: str | None = None


class ScenarioScreeningResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    decision: Literal["keep", "exclude"]
    reason: str = Field(min_length=1)


class GlobalScenarioReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    results: list[ScenarioScreeningResult] = Field(min_length=1)


class LeafTaskDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    leaf_task_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    question_templates: list[str] = Field(min_length=1)
    options: list[str] | None = Field(
        default=None,
        description="Complete Closed VQA answer vocabulary; required for VQA label tasks.",
    )
    input: list[EvidenceSelector] = Field(min_length=1)
    target: str = Field(min_length=1)
    reference_rule: str = Field(min_length=1)
    resolver: ResolverSpec
    output_spec: OutputSpec
    task_paradigm: TaskParadigm
    capability_category: CapabilityCategory
    evidence: list[EvidenceBasis] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_template(cls, value):
        value = dict(value)
        if not value.get("question_templates"):
            legacy_question = value.get("question_template", value.get("question"))
            if legacy_question:
                value["question_templates"] = [legacy_question]
        value.pop("question_template", None)
        value.pop("question", None)
        return value

    @model_validator(mode="after")
    def validate_templates(self):
        if any(not template.strip() for template in self.question_templates):
            raise ValueError("question_templates must be non-empty")
        if len(set(self.question_templates)) != len(self.question_templates):
            raise ValueError("question_templates must be unique")
        return self


class LeafTaskArguments(NoArguments):
    leaf: LeafTaskDraft


class ScenarioReviewArguments(NoArguments):
    review: ScenarioReview | None = None


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _access(context: dict[str, Any]) -> dict[str, Any]:
    access = context.get("dataset_access_manifest")
    if access:
        return access
    if context.get("bir_path") and context.get("dataset_scope"):
        store = BIRStore(context["bir_path"])
        relative = f"datasets/{context['dataset_scope']}/access.json"
        path = store.root / Path(*relative.split("/"))
        if path.is_file():
            value = store.load(relative)
            return value.get("source_access", value)
    return {}


def _bir(context: dict[str, Any]) -> BIRStore | None:
    path = context.get("bir_path")
    return BIRStore(path) if path else None


def _registry(context: dict[str, Any]) -> RegistrySnapshot | None:
    store = _bir(context)
    return store.registry() if store else None


def _profile(context: dict[str, Any]) -> dict[str, Any]:
    return _read(_access(context)["profile_path"])


def _plan_path(context: dict[str, Any]) -> Path:
    if context.get("bir_path") and context.get("dataset_scope"):
        return Path(context["bir_path"]) / "datasets" / context["dataset_scope"] / "planning" / "task_plan.json"
    return Path(context.get("task_plan_path") or context.get("task_space_plan_path")
                or Path(context["operation_directory"]) / "task_plan.json")


def _evidence_dir(context: dict[str, Any]) -> Path:
    if context.get("bir_path") and context.get("dataset_scope"):
        return Path(context["bir_path"]) / "datasets" / context["dataset_scope"] / "evidence"
    return Path(context["operation_directory"]) / "evidence"


def _configured_template_count(context: dict[str, Any]) -> int | None:
    value = context.get("question_template_count")
    if value is None:
        return None
    return int(value)


def _catalog_map(value: Any) -> dict[str, dict[str, Any]]:
    if isinstance(value, dict):
        return {str(key): dict(item) for key, item in value.items() if isinstance(item, dict)}
    if isinstance(value, list):
        return {
            str(item["name"]): dict(item)
            for item in value
            if isinstance(item, dict) and item.get("name")
        }
    return {}


def _mapping_path(context: dict[str, Any]) -> Path | None:
    access = _access(context)
    explicit = access.get("mapping_path")
    if explicit:
        return Path(explicit)
    cases_path = access.get("cases_path")
    if cases_path:
        candidate = Path(cases_path).resolve().parent / "mapping.json"
        if candidate.is_file():
            return candidate
    return None


def _field_meanings(context: dict[str, Any]) -> dict[str, str]:
    path = _mapping_path(context)
    if not path:
        return {}
    try:
        value = _read(path)
    except (OSError, ValueError):
        return {}
    meanings: dict[str, str] = {}

    def add(name: Any, item: Any) -> None:
        if not isinstance(name, str) or not name.strip():
            return
        if isinstance(item, str) and item.strip():
            meanings.setdefault(name, item.strip())
            return
        if not isinstance(item, dict):
            return
        for key in ("meaning", "description", "interpretation", "reason"):
            text = item.get(key)
            if isinstance(text, str) and text.strip():
                meanings.setdefault(name, text.strip())
                return

    if isinstance(value, dict):
        for key in ("fields", "field_meanings", "mapping"):
            section = value.get(key)
            if isinstance(section, dict):
                for name, item in section.items():
                    add(name, item)
            elif isinstance(section, list):
                for item in section:
                    if isinstance(item, dict):
                        add(item.get("field") or item.get("name") or item.get("path") or item.get("target"), item)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                add(item.get("field") or item.get("name") or item.get("path") or item.get("target"), item)
    return meanings


def _compact_scalar(value: Any, *, limit: int = 180) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "…"
    if isinstance(value, list):
        return [_compact_scalar(item, limit=limit) for item in value[:8]]
    if isinstance(value, dict):
        return {key: _compact_scalar(item, limit=limit) for key, item in list(value.items())[:12]}
    return value


def _name_suggestions(name: str, available: list[str]) -> list[str]:
    if not name:
        return available[:20]
    folded = name.casefold()
    contains = [item for item in available if folded in item.casefold() or item.casefold() in folded]
    close = difflib.get_close_matches(name, available, n=5, cutoff=0.25)
    return list(dict.fromkeys([*contains[:5], *close]))[:8]


def _case_evidence_names(context: dict[str, Any]) -> set[str]:
    try:
        from medbenchagent.data.query import DatasetQuery
        query = DatasetQuery(_access(context))
        if hasattr(query, "iter_cases"):
            cases = []
            for case in query.iter_cases():
                cases.append(case)
                if len(cases) >= 64:
                    break
        else:
            cases = list(query.sample(offset=0, limit=64).get("values", []))
        return {
            item.get("name")
            for case in cases
            for item in case.get("evidence", [])
            if item.get("name")
        }
    except Exception:
        return set()


def _asset_rows(case: dict[str, Any], assets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in case.get("evidence", []):
        if item.get("type") != "asset" or item.get("value") not in assets:
            continue
        asset = assets[item["value"]]
        if asset.get("kind") != "image":
            continue
        name = str(item.get("name", "")).casefold()
        if name.startswith(("mask", "segmentation", "annotation", "roi")) or any(
            token in name for token in ("_mask", "_seg", "mask_", "seg_")
        ):
            continue
        rows.append(item)
    return rows


def _compact_case(case: dict[str, Any], assets: dict[str, dict[str, Any]], *, names: set[str] | None = None) -> dict[str, Any]:
    primary = _asset_rows(case, assets)
    primary_ids = {item.get("id") for item in primary}
    rows = []
    for item in case.get("evidence", []):
        name = item.get("name")
        if item.get("type") == "asset" and item in primary:
            rows.append(item)
            continue
        if names and name not in names:
            continue
        refs = set(item.get("ref", []))
        if item.get("type") in {"value", "text", "region"} and (refs & primary_ids or not primary_ids):
            rows.append({**item, "value": _compact_scalar(item.get("value"))})
    return {"evidence": rows[:20]}


def _compact_catalog_entry(entry: dict[str, Any], meaning: str | None = None) -> dict[str, Any]:
    item = {key: entry[key] for key in ("name", "cases", "types", "units", "value_count") if key in entry}
    if entry.get("observed_values"):
        item["observed_values"] = [
            {**value, "value": _compact_scalar(value.get("value"))}
            for value in entry["observed_values"][:8]
            if isinstance(value, dict)
        ]
    if meaning:
        item["meaning"] = meaning
    return item


def _scenario_context_payload(context: dict[str, Any], plan: TaskPlanState, scenario: Any, raw: dict[str, Any]) -> dict[str, Any]:
    from medbenchagent.data.index import DatasetIndex

    facts = _facts(context, plan)
    catalog = _catalog_map(facts.get("evidence_catalog", {}))
    meanings = _field_meanings(context)
    raw_entries = {item.get("name") for item in raw.get("evidence", []) if item.get("name")}
    input_names = {
        item.get("name")
        for case in raw.get("representative_cases", [])
        for item in _asset_rows(case, {row.get("asset_id"): row for row in DatasetIndex(_access(context)).assets()})
        if item.get("name")
    }
    ordered_names = [name for name in input_names if name in catalog]
    ordered_names.extend(name for name in raw_entries if name in catalog and name not in ordered_names)
    ordered_names.extend(name for name in catalog if name not in ordered_names)
    evidence = [_compact_catalog_entry(catalog[name], meanings.get(name) or catalog[name].get("meaning")) for name in ordered_names[:30]]
    assets = {row.get("asset_id"): row for row in DatasetIndex(_access(context)).assets()}
    candidate_names = {item.get("name") for item in evidence if item.get("name")}
    samples = [_compact_case(case, assets, names=candidate_names) for case in raw.get("representative_cases", [])[:3]]
    samples = [case for case in samples if case.get("evidence")]
    return {
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "input_structure": scenario.input_structure,
            "operation": scenario.operation,
            "clinical_target": scenario.clinical_target,
            "capability_category": scenario.capability_category.value,
            "feasible_case_count": scenario.feasible_case_count,
            "status": scenario.status,
            "leaf_task_ids": scenario.leaf_task_ids,
        },
        "planning_evidence": evidence,
        "representative_cases": samples,
        "capacity_hint": raw.get("capacity_hint"),
    }


def _leaf_diagnostics(context: dict[str, Any], plan: TaskPlanState, raw_leaf: Any, *, errors: list[dict[str, Any]], repair_hint: str) -> dict[str, Any]:
    if hasattr(raw_leaf, "model_dump"):
        attempted = raw_leaf.model_dump(mode="json")
    else:
        attempted = raw_leaf
    facts = _facts(context, plan)
    catalog = _catalog_map(facts.get("evidence_catalog", {}))
    return {
        "attempted_leaf": attempted,
        "errors": errors,
        "relevant_evidence": [_compact_catalog_entry(item, item.get("meaning")) for item in list(catalog.values())[:20]],
        "repair_hint": repair_hint,
    }


def _raise_leaf_failure(context: dict[str, Any], plan: TaskPlanState, raw_leaf: Any, code: str, message: str, *, path: str = "", actual: Any = None, expected: Any = None, repair_hint: str = "Use the reported evidence and repair this Leaf before resubmitting.") -> None:
    error = {"code": code, "message": message}
    if path:
        error["path"] = path
    if actual is not None:
        error["actual"] = actual
    if expected is not None:
        error["expected"] = expected
    raise LeafTaskValidationError(message, _leaf_diagnostics(context, plan, raw_leaf, errors=[error], repair_hint=repair_hint))


def _benchmark_brief(context: dict[str, Any]) -> dict[str, Any]:
    """Expose only benchmark intent needed by Task Planning model calls."""
    info = context.get("benchmark_info") or {}
    brief = {
        "clinical_scope": info.get("clinical_scope"),
        "diseases": info.get("diseases", []),
        "modalities": info.get("modalities", []),
        "evaluation_focus": info.get("evaluation_focus"),
        "language": info.get("language"),
        "sample_scope": context.get("sample_scope", info.get("sample_scope", [])),
        "intent_notes": info.get("intent_notes", []),
        "input_policy": context.get("benchmark_input_policy", "image-only"),
    }
    return {
        key: value for key, value in brief.items()
        if value not in (None, "", [], {})
    }


def _load_plan(context: dict[str, Any]) -> TaskPlanState:
    path = _plan_path(context)
    if path.is_file():
        return TaskPlanState.model_validate(_read(path))
    return TaskPlanState(
        dataset={"dataset_id": context.get("dataset_scope", context.get("selected_dataset_id", "dataset"))}
    )


def _write_json(path: Path, value: Any) -> str:
    atomic_write_json(path, value)
    return str(path)


def _write_evidence_artifacts(context: dict[str, Any], facts: dict[str, Any]) -> dict[str, str]:
    directory = _evidence_dir(context)
    directory.mkdir(parents=True, exist_ok=True)
    access = _access(context)
    mapping_path = _mapping_path(context)
    receipt = DatasetAccessReceipt.model_validate({
        "schema_version": "dataset-access",
        "dataset_id": context.get("dataset_scope", context.get("selected_dataset_id")),
        "revision": access.get("revision"),
        "cases_path": access.get("cases_path"),
        "asset_manifest_path": access.get("asset_manifest_path"),
        "profile_path": access.get("profile_path"),
        "duckdb_path": access.get("duckdb_path"),
        "annotation_digest": access.get("annotation_digest"),
        "asset_manifest_digest": access.get("asset_manifest_digest"),
        "profile_digest": access.get("profile_digest"),
        "duckdb_digest": access.get("duckdb_digest"),
        "case_count": facts.get("case_count", 0),
        "patient_count": facts.get("patient_count", 0),
        "patient_identity": facts.get("patient_identity", "unknown"),
        "mapping_path": str(mapping_path) if mapping_path else None,
    }).model_dump(mode="json")
    overview = {
        "schema_version": "dataset-overview",
        "dataset_id": receipt["dataset_id"],
        "revision": receipt["revision"],
        "case_count": facts.get("case_count", 0),
        "patient_count": facts.get("patient_count", 0),
        "patient_identity": facts.get("patient_identity", "unknown"),
        "assets": facts.get("assets", {}),
        "relations": facts.get("relations", {}),
        "evidence_counts": facts.get("evidence", {}),
    }
    meanings = _field_meanings(context)
    catalog_entries = []
    for entry in facts.get("evidence_catalog", {}).values():
        item = dict(entry)
        if item.get("name") in meanings:
            item["meaning"] = meanings[item["name"]]
        catalog_entries.append(item)
    catalog = {
        "schema_version": "evidence-catalog",
        "dataset_id": receipt["dataset_id"],
        "revision": receipt["revision"],
        "entries": catalog_entries,
        "names": facts.get("evidence_names", []),
        "annotation_mapping": facts.get("annotation_mapping", []),
    }
    relations = {
        "schema_version": "relation-summary",
        "dataset_id": receipt["dataset_id"],
        "revision": receipt["revision"],
        "relations": facts.get("relations", {}),
        "assets": facts.get("assets", {}),
    }
    samples_path = directory / "case_samples.json"
    sample = facts.get("representative_case")
    samples = {"schema_version": "case-samples", "cases": [sample] if sample else []}
    manifest = {
        "schema_version": "planning-evidence",
        "dataset_id": receipt["dataset_id"],
        "revision": receipt["revision"],
        "artifacts": {},
    }
    bir = _bir(context)
    if bir:
        dataset_id = receipt["dataset_id"]
        root = Path(bir.root) / "datasets" / dataset_id
        canonical = {
            f"datasets/{dataset_id}/access.json": receipt,
            f"datasets/{dataset_id}/evidence/overview.json": overview,
            f"datasets/{dataset_id}/evidence/catalog.json": catalog,
            f"datasets/{dataset_id}/evidence/relations.json": relations,
            f"datasets/{dataset_id}/evidence/case_samples.json": samples,
        }
        manifest["artifacts"] = {
            key.split("/")[-1].replace(".json", ""): str(root / Path(*key.split("/")[2:]))
            for key in canonical if key.endswith(".json") and "manifest" not in key
        }
        canonical[f"datasets/{dataset_id}/evidence/manifest.json"] = manifest
        commit = bir.transaction(canonical, run_id=context.get("run_id", "unknown"), actor="task_planning")
        context.update(bir_revision=commit.revision, bir_manifest_digest=commit.manifest_digest)
        # Materialize the compatibility evidence view beside the Agent state.
        derived = {
            "access": bir.materialize(f"datasets/{dataset_id}/access.json", directory.parent / "access.json"),
            "overview": bir.materialize(f"datasets/{dataset_id}/evidence/overview.json", directory / "overview.json"),
            "catalog": bir.materialize(f"datasets/{dataset_id}/evidence/catalog.json", directory / "catalog.json"),
            "relations": bir.materialize(f"datasets/{dataset_id}/evidence/relations.json", directory / "relations.json"),
            "case_samples": bir.materialize(f"datasets/{dataset_id}/evidence/case_samples.json", samples_path),
            "manifest": bir.materialize(f"datasets/{dataset_id}/evidence/manifest.json", directory / "manifest.json"),
        }
        paths = {key: str(bir.root / Path(*relative.split("/"))) for key, relative in {
            "access": f"datasets/{dataset_id}/access.json",
            "overview": f"datasets/{dataset_id}/evidence/overview.json",
            "catalog": f"datasets/{dataset_id}/evidence/catalog.json",
            "relations": f"datasets/{dataset_id}/evidence/relations.json",
            "case_samples": f"datasets/{dataset_id}/evidence/case_samples.json",
            "manifest": f"datasets/{dataset_id}/evidence/manifest.json",
        }.items()}
        paths["derived"] = derived
    else:
        paths = {
            "access": _write_json(directory.parent / "access.json", receipt),
            "overview": _write_json(directory / "overview.json", overview),
            "catalog": _write_json(directory / "catalog.json", catalog),
            "relations": _write_json(directory / "relations.json", relations),
            "case_samples": _write_json(samples_path, samples),
        }
        manifest["artifacts"] = paths
        paths["manifest"] = _write_json(directory / "manifest.json", manifest)
    return paths


def _write_artifacts(context: dict[str, Any], plan: TaskPlanState) -> dict[str, Any]:
    directory = Path(context["operation_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    plan_path = _plan_path(context)
    plan_value = plan.model_dump(mode="json")
    axis_path = directory / "axis_decisions.json"
    scenarios_path = directory / "task_scenarios.json"
    exclusions_path = directory / "scenario_exclusions.json"
    leaves_path = directory / "leaf_tasks.json"
    axis_value = {key: [item.model_dump(mode="json") for item in value] for key, value in plan.axis_decisions.items()}
    scenarios_value = {
        key: {
            "scenario_id": value.scenario_id,
            "input_structure": value.input_structure,
            "operation": value.operation,
            "clinical_target": value.clinical_target,
            "capability_category": value.capability_category.value,
            "feasible_case_count": value.feasible_case_count,
            "status": value.status,
            "leaf_task_ids": value.leaf_task_ids,
        }
        for key, value in plan.scenarios.items()
        if not value.exclusion_reason
    }
    exclusions_value = {
        key: {
            "scenario_id": value.scenario_id,
            "input_structure": value.input_structure,
            "operation": value.operation,
            "clinical_target": value.clinical_target,
            "reason": value.exclusion_reason,
        }
        for key, value in plan.scenarios.items()
        if value.exclusion_reason
    }
    leaves_value = {key: value.model_dump(mode="json") for key, value in plan.leaf_tasks.items()}
    reviews_value = {key: value.model_dump(mode="json") for key, value in plan.reviews.items()}
    bir = _bir(context)
    if bir:
        dataset_id = context.get("dataset_scope", context.get("selected_dataset_id", "dataset"))
        prefix = f"datasets/{dataset_id}/planning/"
        changes = {
            prefix + "task_plan.json": plan_value,
            prefix + "axis_decisions.json": axis_value,
            prefix + "scenarios.json": scenarios_value,
            prefix + "scenario_exclusions.json": exclusions_value,
            prefix + "leaf_tasks.json": leaves_value,
            prefix + "reviews.json": reviews_value,
        }
        changed = {}
        for relative, value in changes.items():
            path = bir.root / Path(*_safe_parts(relative))
            if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != value:
                changed[relative] = value
        if changed:
            commit = bir.transaction(changed, run_id=context.get("run_id", "unknown"), actor="task_planning",
                                     dependencies={"registry": bir.registry().digest, "access": _access(context).get("revision", "")})
            context.update(bir_revision=commit.revision, bir_manifest_digest=commit.manifest_digest)
        bir.materialize(prefix + "task_plan.json", plan_path)
    else:
        _write_json(plan_path, plan_value)
    _write_json(axis_path, axis_value)
    _write_json(scenarios_path, scenarios_value)
    _write_json(exclusions_path, exclusions_value)
    _write_json(leaves_path, leaves_value)
    legacy_view = plan.model_dump(mode="json")
    legacy_view.update({
        "pending_node_ids": plan.pending_scenario_ids,
        "current_node_id": plan.current_scenario_id,
        "nodes": {key: value.model_dump(mode="json") for key, value in plan.scenarios.items()},
        "leaves": {key: value.model_dump(mode="json") for key, value in plan.leaf_tasks.items()},
    })
    return {
        "task_plan_path": str(plan_path),
        "task_space_plan_path": str(plan_path),
        "task_plan": plan.model_dump(mode="json"),
        "task_space_plan": legacy_view,
        "axis_decisions_path": str(axis_path),
        "task_scenarios_path": str(scenarios_path),
        "scenario_exclusions_path": str(exclusions_path),
        "leaf_tasks_path": str(leaves_path),
        "reviews_path": str((Path(context.get("bir_path")) / "datasets" / context.get("dataset_scope", "dataset") / "planning" / "reviews.json") if context.get("bir_path") else directory / "reviews.json"),
        "task_nodes_path": str(scenarios_path),
        "task_scenarios": scenarios_value,
        "leaf_tasks": {key: value.model_dump(mode="json") for key, value in plan.leaf_tasks.items()},
        "bir_summary_ref": ({"revision": context.get("bir_revision"), "dataset_id": context.get("dataset_scope"),
                              "task_plan": str(plan_path)} if context.get("bir_path") else None),
        # Transitional in-process views; new persisted artifacts use the names above.
        "task_nodes": {key: value.model_dump(mode="json") for key, value in plan.scenarios.items()},
        "task_leaves": {key: value.model_dump(mode="json") for key, value in plan.leaf_tasks.items()},
    }


def _safe_parts(relative: str) -> tuple[str, ...]:
    return tuple(part for part in relative.replace("\\", "/").split("/") if part)


def _assert_derived_artifact_consistency(context: dict[str, Any], plan: TaskPlanState) -> None:
    """Ensure published plan views are exact projections of one canonical plan."""

    directory = Path(context["operation_directory"])
    scenarios_path = directory / "task_scenarios.json"
    leaves_path = directory / "leaf_tasks.json"
    if not scenarios_path.is_file() or not leaves_path.is_file():
        raise ValueError("Task Plan derived artifacts are missing")
    expected_scenarios = {
        key: {
            "scenario_id": value.scenario_id,
            "input_structure": value.input_structure,
            "operation": value.operation,
            "clinical_target": value.clinical_target,
            "capability_category": value.capability_category.value,
            "feasible_case_count": value.feasible_case_count,
            "status": value.status,
            "leaf_task_ids": value.leaf_task_ids,
        }
        for key, value in plan.scenarios.items()
        if not value.exclusion_reason
    }
    expected_leaves = {
        key: value.model_dump(mode="json") for key, value in plan.leaf_tasks.items()
    }
    if _read(scenarios_path) != expected_scenarios:
        raise ValueError("task_scenarios.json is inconsistent with task_plan.json")
    if _read(leaves_path) != expected_leaves:
        raise ValueError("leaf_tasks.json is inconsistent with task_plan.json")


_write_derived_artifacts = _write_artifacts


def _initialize_planning_context(context: dict[str, Any]) -> dict[str, Any]:
    context = dict(context)
    if _plan_path(context).is_file():
        plan = _load_plan(context)
        context.update(_write_artifacts(context, plan))
        return context
    access = _access(context)
    if access.get("schema_version") != "annotation" or not access.get("cases_path"):
        plan = TaskPlanState(
            dataset={"dataset_id": context.get("dataset_scope", "dataset")},
            phase="blocked",
            review={"reason": "Dataset Access manifest is unavailable"},
        )
        context.update(_write_artifacts(context, plan))
        return context

    from medbenchagent.data.integrity import verify_prepared_indexes
    from medbenchagent.data.index import DatasetIndex

    try:
        verify_prepared_indexes(access)
        query = DatasetIndex(access)
        cases = list(query.iter_cases())
        assets = {row["asset_id"]: row for row in query.assets()}
        facts = build_dataset_facts(cases, assets, profile=_profile(context), revision=access.get("revision"))
    except (OSError, ValueError, KeyError) as exc:
        plan = TaskPlanState(
            dataset={"dataset_id": context.get("dataset_scope", "dataset")},
            phase="blocked",
            review={"reason": f"Dataset Access verification failed: {exc}"},
        )
        context.update(_write_artifacts(context, plan))
        return context

    evidence_paths = _write_evidence_artifacts(context, facts)
    plan = TaskPlanState(
        dataset={"dataset_id": context.get("dataset_scope", "dataset")},
        access_path=evidence_paths["access"],
        evidence_manifest_path=evidence_paths["manifest"],
        phase="axes" if facts.get("case_count", 0) and facts.get("evidence_names") else "blocked",
        review={} if facts.get("case_count", 0) and facts.get("evidence_names") else {"reason": "Dataset has no Case or Evidence"},
    )
    context.update(_write_artifacts(context, plan))
    context["dataset_facts"] = facts
    context["planning_context"] = {
        "evidence_manifest_path": evidence_paths["manifest"],
        "overview_path": evidence_paths["overview"],
        "catalog_path": evidence_paths["catalog"],
        "relations_path": evidence_paths["relations"],
    }
    return context


def _axis_order() -> tuple[TaskAxis, ...]:
    return (TaskAxis.INPUT_STRUCTURE, TaskAxis.OPERATION, TaskAxis.CLINICAL_TARGET)


def _facts(context: dict[str, Any], plan: TaskPlanState) -> dict[str, Any]:
    facts = context.get("dataset_facts")
    if facts:
        return facts
    legacy_facts = getattr(plan, "dataset_facts", None)
    if legacy_facts:
        return legacy_facts
    manifest = plan.evidence_manifest_path
    if manifest:
        paths = _read(manifest).get("artifacts", {})
        overview = _read(paths["overview"])
        catalog_doc = _read(paths["catalog"])
        entries = catalog_doc.get("entries", [])
        catalog = _catalog_map(entries)
        return overview | {
            "evidence_catalog": catalog,
            "evidence_names": catalog_doc.get("names", sorted(catalog)),
            "annotation_mapping": catalog_doc.get("annotation_mapping", []),
            "evidence": overview.get("evidence_counts", {}),
        }
    try:
        profile = _profile(context)
    except (KeyError, OSError, ValueError):
        return {}
    evidence_catalog = {row["name"]: row for row in profile.get("evidence", []) if row.get("name")}
    evidence = {}
    for row in evidence_catalog.values():
        for kind, count in row.get("types", {}).items():
            evidence[kind] = evidence.get(kind, 0) + count
        evidence["numeric"] = evidence.get("numeric", 0) + row.get("value_types", {}).get("number", 0)
    assets = profile.get("assets", {})
    return {
        "schema_version": "dataset-facts",
        "revision": _access(context).get("revision"),
        "case_count": profile.get("case_count", 0),
        "patient_count": profile.get("patient_count", 0),
        "patient_identity": "present" if profile.get("patient_count", 0) else "absent",
        "evidence_catalog": evidence_catalog,
        "evidence_names": list(evidence_catalog),
        "evidence": evidence,
        "assets": {
            "image_assets": assets.get("kinds", {}).get("image", 0),
            "primary_image_assets": assets.get("kinds", {}).get("image", 0),
            "volume_assets": assets.get("kinds", {}).get("volume", 0),
            "multi_image_cases": 0,
            "related_image_cases": 0,
        },
        "relations": {
            "paired_text": 0,
            "paired_study": 0,
            "comparable": 0,
            "related_image_groups": 0,
        },
    }


def _build_scenarios(context: dict[str, Any], plan: TaskPlanState) -> TaskPlanState:
    facts = _facts(context, plan)
    node_context = {
        "evidence": facts.get("evidence", {}),
        "assets": facts.get("assets", {}),
        "relations": facts.get("relations", {}),
        "evidence_names": facts.get("evidence_names", []),
        "annotation_mapping": facts.get("annotation_mapping", []),
    }
    access = _access(context)
    if access.get("schema_version") == "annotation" and access.get("cases_path"):
        from medbenchagent.data.index import DatasetIndex
        query = DatasetIndex(access)
        assets = {row["asset_id"]: row for row in query.assets()}
        node_context["case_facts"] = [case_facts(case, assets) for case in query.iter_cases()]
    analysis = AxisAnalysis(
        input_structure=plan.axis_decisions[TaskAxis.INPUT_STRUCTURE.value],
        operation=plan.axis_decisions[TaskAxis.OPERATION.value],
        clinical_target=plan.axis_decisions[TaskAxis.CLINICAL_TARGET.value],
        dataset_revision=str(facts.get("revision") or access.get("revision") or "1"),
    )
    scenarios = build_task_scenarios(analysis, node_context, registry=_registry(context))
    active = sorted(key for key, value in scenarios.items() if not value.exclusion_reason)
    if not active:
        raise ValueError("Axis analysis produced no executable Task Scenario")
    plan.scenarios = scenarios
    plan.phase = "scenario_review"
    # Keep the compatibility pending list populated for run summaries; the
    # screening-specific list controls which Scenarios may be screened.
    plan.pending_scenario_ids = active.copy()
    plan.screening_pending_scenario_ids = []
    plan.scenario_review_pending_ids = active.copy()
    plan.scenario_review_completed_ids = []
    plan.scenario_review_excluded_ids = []
    plan.completed_scenario_ids = []
    plan.current_scenario_id = active[0] if active else None
    return plan


def screen_scenario(context: dict[str, Any], *, scenario_id: str, decision: str | None = None, reason: str | None = None) -> dict[str, Any]:
    plan = _load_plan(context)
    if plan.phase not in {"screening", "scenario_review"}:
        raise ValueError("Scenario screening is not available in the current phase")
    if scenario_id not in (plan.screening_pending_scenario_ids or plan.scenario_review_pending_ids):
        raise ValueError("Scenario is not pending screening")
    if decision is None and reason is None:
        registry = _registry(context)
        scenario = plan.scenarios[scenario_id]
        operation = registry.operation_definition(scenario.operation) if registry else {
            "description": scenario.operation, "answer_options": "forbidden"
        }
        facts = _facts(context, plan)
        catalog = list(facts.get("evidence_catalog", {}).values())
        payload = {
            "scenario": scenario.model_dump(mode="json"),
            "evidence_catalog": catalog,
            "operation_description": operation["description"],
            "answer_options": operation["answer_options"],
            "benchmark_brief": _benchmark_brief(context),
        }
        screening = complete_json(
            registry.prompt("scenario_screening") if registry else "Return only JSON with decision keep or exclude and a concise reason.",
            json.dumps(payload, ensure_ascii=False),
            ScenarioScreeningResult,
        )
        decision, reason = screening["decision"], screening["reason"]
    if decision not in {"keep", "exclude"} or not reason or not reason.strip():
        raise ValueError("Scenario screening requires decision keep/exclude and a non-empty reason")
    scenario = plan.scenarios[scenario_id]
    queue = plan.screening_pending_scenario_ids or plan.scenario_review_pending_ids
    remaining = [item for item in queue if item != scenario_id]
    if decision == "exclude":
        plan.scenarios[scenario_id] = scenario.model_copy(update={"status": "excluded", "exclusion_reason": reason.strip()})
        plan.screening_pending_scenario_ids = []
        plan.scenario_review_pending_ids = remaining
        plan.pending_scenario_ids = [item for item in plan.pending_scenario_ids if item != scenario_id]
        plan.current_scenario_id = remaining[0] if remaining else None
        if not remaining:
            plan.phase = "complete"
    else:
        plan.phase = "leaves"
        plan.current_scenario_id = scenario_id
        plan.pending_scenario_ids = [scenario_id]
        plan.screening_pending_scenario_ids = []
        plan.scenario_review_pending_ids = remaining
        plan.scenarios[scenario_id] = scenario.model_copy(update={"status": "active"})
    plan.revision += 1
    updates = _write_artifacts(context, plan)
    return result("success", f"Scenario {scenario_id} screening decision: {decision}", **updates)


def review_candidate_scenarios(context: dict[str, Any], *, scenario_id: str) -> dict[str, Any]:
    """Review one candidate Scenario; keep it unchanged or exclude it before Leaf design."""
    plan = _load_plan(context)
    if plan.phase not in {"scenario_review", "screening"}:
        raise ValueError("Scenario Review is not available in the current phase")
    pending = plan.scenario_review_pending_ids or plan.screening_pending_scenario_ids
    if not pending:
        raise ValueError("No candidate Scenarios are pending Scenario Review")
    if scenario_id != pending[0]:
        raise ValueError(
            f"Scenario Review must process the queue head {pending[0]}, received {scenario_id}"
        )
    scenario = plan.scenarios[scenario_id]
    registry = _registry(context)
    facts = _facts(context, plan)
    guidance = {}
    if registry:
        guidance = {
            operation: {
                "description": registry.operation_definition(operation)["description"],
                "answer_options": registry.operation_definition(operation)["answer_options"],
            }
            for operation in [scenario.operation]
        }
    catalog = []
    for entry in list(facts.get("evidence_catalog", {}).values())[:80]:
        item = {key: entry[key] for key in ("name", "types", "cases", "value_count") if key in entry}
        if entry.get("value_count", 0) <= 20 and entry.get("observed_values"):
            item["observed_values"] = entry["observed_values"][:20]
        catalog.append(item)
    payload = {
        "dataset": {"dataset_id": plan.dataset.get("dataset_id"), "case_count": facts.get("case_count", 0), "image_only_policy": True},
        "scenario": scenario.model_dump(mode="json"),
        "evidence_catalog": catalog,
        "operation_guidance": guidance,
        "benchmark_brief": _benchmark_brief(context),
    }
    prompt = registry.prompt("scenario_screening") if registry else "Review one Scenario and return keep or exclude with a non-empty reason."
    reviewed = ScenarioScreeningResult.model_validate(complete_json(prompt, json.dumps(payload, ensure_ascii=False), ScenarioScreeningResult))
    remaining = pending[1:]
    plan.scenario_review_pending_ids = remaining
    plan.scenario_review_completed_ids = [*plan.scenario_review_completed_ids, scenario.scenario_id]
    if reviewed.decision == "exclude":
        plan.scenarios[scenario.scenario_id] = scenario.model_copy(update={"status": "excluded", "exclusion_reason": reviewed.reason})
        plan.scenario_review_excluded_ids = [*plan.scenario_review_excluded_ids, scenario.scenario_id]
    if remaining:
        plan.phase = "scenario_review"
        plan.current_scenario_id = remaining[0]
        # Keep the canonical pending list aligned with the active workflow
        # queue.  During Scenario Review these are the candidates still
        # awaiting review; once the queue is exhausted the list is replaced
        # with the accepted Scenarios that still need Leaf design.
        plan.pending_scenario_ids = remaining.copy()
    else:
        accepted = [key for key, item in plan.scenarios.items() if not item.exclusion_reason]
        plan.pending_scenario_ids = accepted
        plan.current_scenario_id = accepted[0] if accepted else None
        plan.phase = "leaves" if accepted else "complete"
    plan.revision += 1
    updates = _write_artifacts(context, plan)
    return result("success", f"Scenario Review {scenario.scenario_id}: {reviewed.decision}", **updates)


def analyze_axis(context: dict[str, Any], *, axis: TaskAxis, assessments: list[AxisValueDecision] | None = None) -> dict[str, Any]:
    axis = TaskAxis(axis)
    plan = _load_plan(context)
    if plan.phase != "axes":
        raise ValueError(f"Axis analysis is not available in phase {plan.phase}")
    expected = _axis_order()[len(plan.axis_decisions)]
    if axis != expected:
        raise ValueError(f"Expected Axis {expected}, received {axis}")
    registry = _registry(context)
    allowed = set(registry.axis_values(axis.value) if registry else AXIS_VALUES[axis])
    if assessments is None:
        evidence = read_axis_evidence(context, axis=axis)["summary"]
        axis_definition = next(item for item in registry.axis_definitions() if item["id"] == axis.value) if registry else {"id": axis.value}
        parsed = complete_json(
            registry.prompt("axis_analysis") if registry else "Assess the supplied Axis using bounded evidence.",
            json.dumps({"axis": axis.value, "axis_definition": axis_definition, "allowed_values": sorted(allowed),
                        "bounded_evidence": json.loads(evidence), "dataset_policy": "image-only benchmark",
                        "benchmark_brief": _benchmark_brief(context)}, ensure_ascii=False),
            AxisDecisionSet,
        )
        assessments = parsed["values"]
    assessments = [AxisValueDecision.model_validate(value) for value in assessments]
    actual = [value.value for value in assessments]
    if set(actual) != allowed or len(actual) != len(set(actual)):
        raise ValueError(f"Axis analysis must cover registry exactly for {axis}")
    plan.axis_decisions[axis.value] = assessments
    plan.revision += 1
    if len(plan.axis_decisions) == len(_axis_order()):
        plan.next_axis = None
        plan = _build_scenarios(context, plan)
        summary = (
            f"Axis {axis.value} saved; built {len(plan.scenarios)} Task Scenarios "
            f"({len(plan.pending_scenario_ids)} executable)"
        )
    else:
        plan.next_axis = _axis_order()[len(plan.axis_decisions)]
        summary = (
            f"Axis {axis.value} saved; supported values: "
            + ", ".join(value.value for value in assessments if value.supported)
        )
    updates = _write_artifacts(context, plan)
    return result("success", summary, milestone={"kind": "axis", "axis": axis.value}, **updates)


def analyze_axes(context: dict[str, Any], *, analysis: dict[str, Any]) -> dict[str, Any]:
    """Import a complete AxisAnalysis for deterministic/in-process callers."""
    parsed = AxisAnalysis.model_validate(analysis)
    plan = _load_plan(context)
    plan.axis_decisions = {
        TaskAxis.INPUT_STRUCTURE.value: parsed.input_structure,
        TaskAxis.OPERATION.value: parsed.operation,
        TaskAxis.CLINICAL_TARGET.value: parsed.clinical_target,
    }
    plan.next_axis = None
    plan.phase = "axes"
    plan.revision += 1
    plan = _build_scenarios(context, plan)
    return result("success", "Axis decisions saved", **_write_artifacts(context, plan))


def build_nodes(context: dict[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper; new plans build Task Scenarios after Axis decisions."""
    plan = _load_plan(context)
    if not plan.axis_decisions:
        path = context.get("axis_analysis_path")
        if not path:
            raise ValueError("Axis decisions are required before building Scenarios")
        parsed = AxisAnalysis.model_validate(_read(path))
        plan.axis_decisions = {
            TaskAxis.INPUT_STRUCTURE.value: parsed.input_structure,
            TaskAxis.OPERATION.value: parsed.operation,
            TaskAxis.CLINICAL_TARGET.value: parsed.clinical_target,
        }
    plan.phase = "axes"
    plan = _build_scenarios(context, plan)
    return result("success", "Task Scenarios saved", **_write_artifacts(context, plan))


def read_planning_context(context: dict[str, Any]) -> dict[str, Any]:
    plan = _load_plan(context)
    return result("success", json.dumps({
        "schema_version": "planning-index",
        "phase": plan.phase,
        "next_axis": plan.next_axis,
        "evidence_manifest_path": plan.evidence_manifest_path,
        "axis_decisions": list(plan.axis_decisions),
        "scenario_count": len(plan.scenarios),
        "leaf_task_count": len(plan.leaf_tasks),
        "question_template_count": _configured_template_count(context),
    }, ensure_ascii=False))


def read_dataset_overview(context: dict[str, Any]) -> dict[str, Any]:
    plan = _load_plan(context)
    if not plan.evidence_manifest_path:
        raise ValueError("Planning evidence manifest is unavailable")
    paths = _read(plan.evidence_manifest_path)["artifacts"]
    overview = _read(paths["overview"])
    return result(
        "success",
        f"Access verified for {overview.get('dataset_id')} revision {overview.get('revision')} with {overview.get('case_count', 0)} Cases",
        milestone={"kind": "access", "dataset_id": overview.get("dataset_id"), "revision": overview.get("revision")},
        overview=json.dumps(overview, ensure_ascii=False),
    )


class EvidenceCatalogArguments(NoArguments):
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=50)
    names: list[str] = Field(default_factory=list)


class AxisEvidenceArguments(NoArguments):
    axis: TaskAxis

    @field_validator("axis", mode="before")
    @classmethod
    def parse_axis(cls, value):
        return TaskAxis(value)


def read_evidence_catalog(context: dict[str, Any], *, offset=0, limit=20, names=None) -> dict[str, Any]:
    """Read catalog entries, concrete values and compact matching Case samples."""
    from medbenchagent.data.index import DatasetIndex

    state = context.setdefault("planning_read_state", {
        "scenario_id": None,
        "scenario_context_read": False,
        "catalog_queries": [],
        "catalog_cache": {},
    })
    state.setdefault("catalog_cache", {})
    query = json.dumps({"offset": offset, "limit": limit, "names": sorted(names or [])}, sort_keys=True)
    plan = _load_plan(context)
    paths = _read(plan.evidence_manifest_path)["artifacts"]
    catalog_doc = _read(paths["catalog"])
    all_entries = catalog_doc.get("entries", [])
    by_name = _catalog_map(all_entries)
    available = sorted(by_name)
    requested = list(dict.fromkeys(names or []))
    unmatched = [name for name in requested if name not in by_name]
    matched_names = [name for name in requested if name in by_name]
    entries = [by_name[name] for name in matched_names] if requested else all_entries
    page = entries[offset:offset + limit]
    payload = {
        "revision": _read(paths["catalog"]).get("revision"),
        "query": {"offset": offset, "limit": limit, "names": requested},
        "entries": [_compact_catalog_entry(item, item.get("meaning")) for item in page],
        "total": len(entries),
        "next_offset": offset + limit if offset + limit < len(entries) else None,
        "unmatched_names": unmatched,
        "suggestions": {name: _name_suggestions(name, available) for name in unmatched},
    }
    if requested:
        query_index = DatasetIndex(_access(context))
        assets = {row.get("asset_id"): row for row in query_index.assets()}
        values = {}
        samples = {}
        for name in matched_names:
            value_page = query_index.values(name, limit=50)
            compact_values = []
            truncated = False
            for item in value_page.get("values", []):
                value = _compact_scalar(item.get("value"), limit=180)
                truncated = truncated or value != item.get("value")
                compact_values.append({"value": value, "count": item.get("count", 0)})
            values[name] = {
                "values": compact_values,
                "total": value_page.get("total", 0),
                "truncated": truncated,
            }
            cases = query_index.sample(evidence_name=name, limit=3).get("values", [])
            samples[name] = [_compact_case(case, assets, names={name}) for case in cases]
        payload["field_values"] = values
        payload["case_samples"] = samples
    cached = state["catalog_cache"].get(query)
    if cached is None:
        state["catalog_queries"] = [*state.get("catalog_queries", []), query]
        state["catalog_cache"][query] = payload
        cached = payload
    return result(
        "success",
        json.dumps(cached, ensure_ascii=False),
        planning_read_state=state,
        planning_evidence_cache={
            **(context.get("planning_evidence_cache") or {}),
            "catalog": list(state["catalog_cache"].values())[-8:],
        },
    )


def read_axis_evidence(context: dict[str, Any], *, axis: TaskAxis) -> dict[str, Any]:
    axis = TaskAxis(axis)
    plan = _load_plan(context)
    facts = _facts(context, plan)
    registry = _registry(context)
    axis_definition = next(
        (item for item in registry.axis_definitions() if item.get("id") == axis.value),
        {"id": axis.value, "description": ""},
    ) if registry else {"id": axis.value, "description": ""}
    value_definitions = {
        item.get("id"): item.get("description", "")
        for item in axis_definition.get("values", [])
        if item.get("id")
    }
    evidence = {
        "schema_version": "axis-evidence",
        "axis": axis.value,
        "revision": facts.get("revision"),
        "allowed_values": list((_registry(context).axis_values(axis.value) if _registry(context) else AXIS_VALUES[axis])),
        "axis_definition": {
            "id": axis_definition.get("id", axis.value),
            "description": axis_definition.get("description", ""),
        },
        "allowed_value_definitions": [
            {"id": value, "description": value_definitions.get(value, "")}
            for value in ((_registry(context).axis_values(axis.value) if _registry(context) else AXIS_VALUES[axis]))
        ],
        "dataset_overview": {
            "case_count": facts.get("case_count", 0),
            "patient_count": facts.get("patient_count", 0),
            "patient_identity": facts.get("patient_identity", "unknown"),
            "assets": facts.get("assets", {}),
            "relations": facts.get("relations", {}),
            "evidence": facts.get("evidence", {}),
        },
        "evidence_catalog": list(facts.get("evidence_catalog", {}).values())
        if isinstance(facts.get("evidence_catalog"), dict) else facts.get("evidence_catalog", []),
        "catalog_semantics": {
            "cases": "number of Cases containing an evidence name",
            "types": "counts of evidence entries by type for that name",
            "units": "counts of observed units for that name",
        },
        "reference_semantics": {
            "planning": "evidence_catalog:<name> and dataset_overview:* are registered planning references; do not pass them to read_evidence.",
            "inspection": "Only references returned by list_evidence are valid for read_evidence.",
        },
    }
    axis_dir = Path(context["operation_directory"]) / "axis_evidence"
    path = axis_dir / f"{axis.value}.json"
    _write_json(path, evidence)
    return result("success", json.dumps(evidence, ensure_ascii=False), evidence_path=str(path))


def read_scenario_context(context: dict[str, Any], *, scenario_id: str | None = None) -> dict[str, Any]:
    """Read the current Scenario once; the next Leaf action must be evidence lookup or save."""
    plan = _load_plan(context)
    scenario_id = scenario_id or plan.current_scenario_id or (
        plan.pending_scenario_ids[0] if plan.pending_scenario_ids else
        (plan.screening_pending_scenario_ids[0] if plan.screening_pending_scenario_ids else None)
    )
    if not scenario_id or scenario_id not in plan.scenarios:
        raise ValueError("No pending Task Scenario is available")
    if plan.current_scenario_id != scenario_id:
        scenario = plan.scenarios[scenario_id]
        plan.current_scenario_id = scenario_id
        plan.scenarios[scenario_id] = scenario.model_copy(update={"status": "active"})
        _write_artifacts(context, plan)
    from medbenchagent.data.index import DatasetIndex
    context_value = DatasetIndex(_access(context)).node_context(
        plan.scenarios[scenario_id], sample_limit=3, registry=_registry(context)
    )
    scenario = plan.scenarios[scenario_id]
    compact = _scenario_context_payload(context, plan, scenario, context_value)
    context_value = compact
    registry = _registry(context)
    if registry:
        operation = registry.operation_definition(scenario.operation)
        context_value["operation_guidance"] = {
            "description": operation["description"],
            "answer_options": operation["answer_options"],
        }
    context_value["plan_phase"] = plan.phase
    context_value["question_template_count"] = _configured_template_count(context)
    context_value["evidence_source"] = "planning_catalog"
    context_value["available_followup_tool"] = "read_evidence_catalog"
    state = context.setdefault("planning_read_state", {
        "scenario_id": scenario_id,
        "scenario_context_read": False,
        "catalog_queries": [],
        "catalog_cache": {},
    })
    if state.get("scenario_id") == scenario_id and state.get("scenario_context_read"):
        raise ValueError("This Scenario context was already read; choose save_leaf_task or a new evidence request")
    state.update({"scenario_id": scenario_id, "scenario_context_read": True, "catalog_queries": []})
    context_value["next_action"] = "design_leaf"
    context_value["instruction"] = "Use the listed evidence to design and save a Leaf Task. Do not reread this Scenario."
    state["catalog_cache"] = {}
    return result(
        "success",
        json.dumps(context_value, ensure_ascii=False),
        planning_read_state=state,
        planning_evidence_cache={
            "scenario_id": scenario_id,
            "planning_evidence": context_value.get("planning_evidence", []),
            "representative_cases": context_value.get("representative_cases", []),
        },
    )


def save_leaf_task(context: dict[str, Any], *, leaf: LeafTaskDraft) -> dict[str, Any]:
    """Terminal Leaf-design action after the current Scenario and required evidence are available."""
    plan = _load_plan(context)
    raw_leaf = leaf
    if plan.phase != "leaves" or not plan.current_scenario_id:
        raise ValueError("An active Task Scenario is required before saving a Leaf Task")
    try:
        draft = leaf if isinstance(leaf, LeafTaskDraft) else LeafTaskDraft.model_validate(leaf)
        parsed = LeafTask.model_validate(draft.model_dump(mode="json"))
    except ValueError as exc:
        _raise_leaf_failure(context, plan, raw_leaf, "leaf_schema", str(exc), repair_hint="Preserve the submitted Leaf and correct only the reported schema fields.")
    if parsed.scenario_id != plan.current_scenario_id:
        _raise_leaf_failure(context, plan, raw_leaf, "scenario_mismatch", "Leaf Task must belong to the active Scenario", path="scenario_id", actual=parsed.scenario_id, expected=plan.current_scenario_id)
    expected_templates = _configured_template_count(context)
    if expected_templates is not None and len(parsed.question_templates) != expected_templates:
        _raise_leaf_failure(
            context, plan, raw_leaf, "template_count",
            f"Leaf Task requires exactly {expected_templates} question_templates; received {len(parsed.question_templates)}",
            path="question_templates", actual=len(parsed.question_templates), expected=expected_templates,
        )
    scenario = plan.scenarios[parsed.scenario_id]
    if parsed.capability_category != scenario.capability_category:
        _raise_leaf_failure(context, plan, raw_leaf, "capability_category", "Leaf Task Category must match its Scenario", path="capability_category", actual=parsed.capability_category.value, expected=scenario.capability_category.value)
    evidence_names = set(_facts(context, plan).get("evidence_names", []))
    facts = _facts(context, plan)
    try:
        spec = ResolverSpec.parse(parsed.resolver)
    except ValueError as exc:
        _raise_leaf_failure(context, plan, raw_leaf, "resolver_schema", str(exc), repair_hint="Keep the same Leaf and repair the Resolver steps or result_step reported above.")
    catalog = _catalog_map(facts.get("evidence_catalog", {}))
    evidence_names.update(_case_evidence_names(context))
    available_names = sorted(set(catalog) | evidence_names)
    for index, selector in enumerate(parsed.input):
        if selector.name and selector.name not in evidence_names:
            _raise_leaf_failure(
                context, plan, raw_leaf, "missing_input_evidence",
                f"Input evidence name {selector.name!r} does not exist for this dataset.",
                path=f"input[{index}].name", actual=selector.name,
                expected=_name_suggestions(selector.name, available_names),
                repair_hint="Replace the input selector with one of the reported real evidence names and keep resolver.inputs identical.",
            )
    if parsed.input != spec.inputs:
        _raise_leaf_failure(
            context, plan, raw_leaf, "input_resolver_mismatch",
            "Leaf.input and Resolver.inputs differ.",
            actual={"input": [item.model_dump(mode="json") for item in parsed.input], "resolver_inputs": [item.model_dump(mode="json") for item in spec.inputs]},
            repair_hint="Make Resolver.inputs exactly equal to Leaf.input; keep reference-only evidence in resolver steps.",
        )
    try:
        validate_candidate_leaf(parsed, plan.scenarios, resolver_context={
            "evidence_names": evidence_names,
            "evidence_catalog": catalog,
            "registry": _registry(context),
        })
    except ValueError as exc:
        _raise_leaf_failure(context, plan, raw_leaf, "leaf_contract", str(exc), repair_hint="Use the current evidence catalog and change only the rejected selector or contract field.")
    identity = leaf_identity(parsed)
    if any(key != parsed.leaf_id and leaf_identity(value) == identity for key, value in plan.leaf_tasks.items()):
        _raise_leaf_failure(context, plan, raw_leaf, "duplicate_leaf", "Duplicate Leaf Task semantics", repair_hint="Change the target or resolver semantics while preserving the current Scenario evidence.")
    from medbenchagent.data.query import DatasetQuery
    try:
        query = DatasetQuery(_access(context))
        if hasattr(query, "iter_cases"):
            cases = list(query.iter_cases())
        else:
            page = query.sample(offset=0, limit=512)
            cases = list(page.get("values", []))
    except Exception as exc:
        _raise_leaf_failure(context, plan, raw_leaf, "dataset_query", str(exc), repair_hint="Use the available planning evidence and retry this Leaf after the dataset query is available.")
    resolution_reasons = Counter()
    resolved_answers = set()
    operation_mode = (_registry(context).answer_options_mode(scenario.operation)
                      if _registry(context) else ("required" if scenario.operation in {"recognize", "compare", "align"} else "forbidden"))
    for case in cases:
        resolved = resolve(case, ResolverSpec.parse(parsed.resolver))
        if isinstance(resolved, NotApplicable):
            resolution_reasons[resolved.reason] += 1
        elif operation_mode == "required" and parsed.task_paradigm == TaskParadigm.VQA and parsed.output_spec == OutputSpec.LABEL:
            value = resolved.reference.get("value", resolved.reference) if isinstance(resolved.reference, dict) else resolved.reference
            if isinstance(value, (str, int, float, bool)):
                resolved_answers.add(str(value))
    eligible = len(cases) - sum(resolution_reasons.values())
    if eligible <= 0:
        common = ", ".join(f"{reason}={count}" for reason, count in resolution_reasons.most_common())
        _raise_leaf_failure(
            context, plan, raw_leaf, "resolver_zero_capacity",
            f"Resolver resolved 0 of {len(cases)} Cases. Common reasons: {common or 'unknown'}",
            actual=dict(resolution_reasons),
            repair_hint="Use the relevant evidence names, types and relations reported above, then resubmit this same Leaf.",
        )
    if operation_mode == "required" and parsed.task_paradigm == TaskParadigm.VQA and parsed.output_spec == OutputSpec.LABEL:
        missing_options = resolved_answers - {option for option in (parsed.options or [])}
        if missing_options:
            _raise_leaf_failure(
                context, plan, raw_leaf, "options_incomplete",
                f"Closed VQA options are missing resolved answers: {sorted(missing_options)}",
                actual={"provided": parsed.options or [], "resolved": sorted(resolved_answers), "missing": sorted(missing_options)},
                repair_hint="Add every resolved answer to options without inventing new categories.",
            )
    parsed = parsed.model_copy(update={"eligible_count": eligible})
    plan.leaf_tasks[parsed.leaf_id] = parsed
    plan.revision += 1
    scenario = plan.scenarios[parsed.scenario_id]
    plan.scenarios[parsed.scenario_id] = scenario.model_copy(
        update={
            "leaf_task_ids": [*scenario.leaf_task_ids, parsed.leaf_id],
            "status": "review_pending",
        }
    )
    context.pop("planning_read_state", None)
    updates = _write_artifacts(context, plan)
    return result(
        "success",
        f"Leaf Task {parsed.leaf_id} saved ({eligible} eligible Cases)",
        milestone={"kind": "leaf", "leaf_task_id": parsed.leaf_id, "capacity": eligible},
        planning_read_state={},
        **updates,
    )


def save_task_leaf(context: dict[str, Any], *, leaf: dict[str, Any]) -> dict[str, Any]:
    """Compatibility entry point for callers using the old tool name."""
    plan = _load_plan(context)
    if plan.phase != "leaves":
        plan.phase = "leaves"
    if not plan.current_scenario_id:
        current = getattr(plan, "current_node_id", None) or (plan.pending_scenario_ids[0] if plan.pending_scenario_ids else None)
        if current:
            plan.current_scenario_id = current
            _write_artifacts(context, plan)
    parsed = LeafTask.model_validate(leaf)
    normalized = {
        "leaf_task_id": parsed.leaf_id,
        "name": parsed.name,
        "scenario_id": parsed.scenario_id,
        "question_templates": parsed.question_templates,
        "options": parsed.options,
        "input": [selector.model_dump(mode="json") for selector in parsed.input],
        "target": parsed.target,
        "reference_rule": parsed.reference_rule,
        "resolver": parsed.resolver,
        "output_spec": parsed.output_spec.value,
        "task_paradigm": parsed.task_paradigm.value,
        "capability_category": parsed.capability_category.value,
        "evidence": [basis.model_dump(mode="json") for basis in parsed.evidence],
    }
    return save_leaf_task(context, leaf=LeafTaskDraft.model_validate(normalized))


def _leaf_validation_summary(context: dict[str, Any], leaf: LeafTask) -> dict[str, Any]:
    """Return a bounded resolver summary for Scenario Review."""
    summary: dict[str, Any] = {
        "contract_validated": True,
        "eligible_count": leaf.eligible_count,
        "case_count": None,
        "successful_case_count": None,
        "failure_reasons": {},
        "answer_distribution": {},
        "successful_examples": [],
    }
    access = _access(context)
    if not access.get("cases_path") and not access.get("duckdb_path"):
        return summary
    try:
        from medbenchagent.data.query import DatasetQuery
        query = DatasetQuery(access)
        cases = list(query.iter_cases()) if hasattr(query, "iter_cases") else list(query.sample(offset=0, limit=512).get("values", []))
    except Exception as exc:  # pragma: no cover - diagnostics only
        summary["validation_error"] = str(exc)
        return summary
    failures = Counter()
    answers = Counter()
    examples = []
    for case in cases:
        resolved = resolve(case, ResolverSpec.parse(leaf.resolver))
        if isinstance(resolved, NotApplicable):
            failures[resolved.reason] += 1
            continue
        reference = resolved.reference
        value = reference.get("value", reference) if isinstance(reference, dict) else reference
        if isinstance(value, (str, int, float, bool)):
            answer = str(value)
            answers[answer] += 1
        else:
            answer = None
        if len(examples) < 3:
            examples.append({
                "case_id": case.get("case_id"),
                "reference": reference if isinstance(reference, (str, int, float, bool, list, dict)) else str(reference),
                "answer": answer,
            })
    summary.update({
        "case_count": len(cases),
        "successful_case_count": len(cases) - sum(failures.values()),
        "failure_reasons": dict(failures),
        "answer_distribution": dict(answers),
        "successful_examples": examples,
    })
    return summary


def review_current_scenario(context: dict[str, Any], *, review: ScenarioReview | None = None) -> dict[str, Any]:
    plan = _load_plan(context)
    if plan.phase != "leaves" or not plan.current_scenario_id:
        raise ValueError("An active Scenario is required before review")
    scenario = plan.scenarios[plan.current_scenario_id]
    if review is None:
        summaries = [
            {
                **{key: value for key, value in leaf.model_dump(mode="json").items()
                   if key in {"leaf_task_id", "name", "target", "input", "reference_rule", "resolver",
                              "evidence", "output_spec", "task_paradigm", "question_templates", "options",
                              "eligible_count"}},
                "validation": _leaf_validation_summary(context, leaf),
            }
            for leaf in (plan.leaf_tasks[item] for item in scenario.leaf_task_ids)
        ]
        registry = _registry(context)
        operation_guidance = {}
        if registry:
            operation = registry.operation_definition(scenario.operation)
            operation_guidance = {
                "description": operation["description"],
                "answer_options": operation["answer_options"],
            }
        prompt = registry.prompt("scenario_review") if registry else "Review the supplied Leaf summaries and return a complete ScenarioReview."
        review = ScenarioReview.model_validate(complete_json(
            prompt,
            json.dumps({
                "scenario": scenario.model_dump(mode="json"),
                "leaf_tasks": summaries,
                "operation_guidance": operation_guidance,
                "benchmark_brief": _benchmark_brief(context),
            }, ensure_ascii=False),
            ScenarioReview,
        ))
    elif not isinstance(review, ScenarioReview):
        review = ScenarioReview.model_validate(review)
    if review.scenario_id != plan.current_scenario_id:
        raise ValueError("Review must belong to the active Scenario")
    scenario = plan.scenarios[review.scenario_id]
    expected = set(scenario.leaf_task_ids)
    checked = set(review.checked_leaf_task_ids)
    discarded = {item.leaf_task_id for item in review.discarded_leaf_tasks}
    if checked != expected or not expected:
        raise ValueError("Scenario review must cover every saved Leaf Task")
    if not discarded <= expected:
        raise ValueError("Scenario review can discard only saved Leaf Tasks")
    remaining = expected - discarded
    evidence_names = set(_facts(context, plan).get("evidence_names", []))
    evidence_names.update(
        selector.name
        for leaf_id in remaining
        for selector in plan.leaf_tasks[leaf_id].input
        if selector.name and selector.evidence_type == "asset"
    )
    expected_templates = _configured_template_count(context)
    for leaf_id in remaining:
        leaf = plan.leaf_tasks[leaf_id]
        if expected_templates is not None and len(leaf.question_templates) != expected_templates:
            raise ValueError(
                f"Leaf Task {leaf_id} requires exactly {expected_templates} question_templates"
            )
        validate_candidate_leaf(
            leaf,
            plan.scenarios,
            resolver_context={"evidence_names": evidence_names, "registry": _registry(context)},
        )
    if review.status == "approved":
        if not remaining:
            raise ValueError("Cannot approve a Scenario after discarding every Leaf Task")
        if any((plan.leaf_tasks[key].eligible_count or 0) <= 0 for key in remaining):
            raise ValueError("Cannot approve a Scenario with a zero-capacity Leaf Task")
        status = "completed"
    else:
        status = "active"
    for leaf_id in discarded:
        plan.leaf_tasks.pop(leaf_id, None)
    normalized_review = review.model_copy(update={"checked_leaf_task_ids": sorted(remaining)})
    plan.scenarios[review.scenario_id] = scenario.model_copy(
        update={"status": status, "leaf_task_ids": sorted(remaining)}
    )
    plan.reviews[review.scenario_id] = normalized_review
    if review.status == "approved":
        plan.completed_scenario_ids = sorted(set(plan.completed_scenario_ids) | {review.scenario_id})
        plan.pending_scenario_ids = [key for key in plan.pending_scenario_ids if key != review.scenario_id]
        if plan.pending_scenario_ids:
            plan.current_scenario_id = plan.pending_scenario_ids[0]
        elif plan.screening_pending_scenario_ids:
            plan.phase = "screening"
            plan.current_scenario_id = plan.screening_pending_scenario_ids[0]
        else:
            plan.current_scenario_id = None
            plan.phase = "complete"
    else:
        plan.completed_scenario_ids = [
            key for key in plan.completed_scenario_ids if key != review.scenario_id
        ]
    plan.revision += 1
    updates = _write_artifacts(context, plan)
    next_pending = plan.pending_scenario_ids[0] if plan.pending_scenario_ids else None
    return result(
        "success",
        f"Scenario {review.scenario_id} review saved",
        milestone={
            "kind": "scenario_progress" if next_pending else "scenario_complete",
            "scenario_id": review.scenario_id,
            "next_scenario_id": next_pending,
            "status": review.status,
        },
        planning_incomplete=bool(next_pending),
        completed_scenario_id=review.scenario_id,
        next_pending_scenario_id=next_pending,
        **updates,
    )


def review_current_node(context: dict[str, Any], *, review: dict[str, Any]) -> dict[str, Any]:
    """Review the current Scenario directly."""
    parsed = dict(review)
    scenario_id = parsed.pop("scenario_id", parsed.pop("node_id", None))
    checked = parsed.pop("checked_leaf_task_ids", parsed.pop("checked_leaf_ids", []))
    discarded = parsed.pop("discarded_leaf_tasks", [])
    return review_current_scenario(
        context,
        review=ScenarioReview(
            scenario_id=scenario_id,
            checked_leaf_task_ids=checked,
            discarded_leaf_tasks=discarded,
            status=parsed.get("status", "rejected"),
            issues=parsed.get("issues", []),
        ),
    )


def classify_task_nodes(context: dict[str, Any], *, assignments: list[dict[str, Any]]) -> dict[str, Any]:
    """Compatibility no-op: Capability Category is derived from Operation."""
    plan = _load_plan(context)
    if plan.phase == "leaves" and not plan.pending_scenario_ids and not plan.current_scenario_id:
        plan.phase = "complete"
    return result("success", "Capability Categories are derived from Operation", **_write_artifacts(context, plan))


def finish_task_plan(context: dict[str, Any]) -> dict[str, Any]:
    require(context, "Dataset", "PlanningConfig", "TemplateConfig")
    plan = _load_plan(context)
    if plan.phase != "complete" or plan.pending_scenario_ids or plan.current_scenario_id:
        raise ValueError("Task Plan still has pending Scenarios")
    expected_templates = _configured_template_count(context)
    for scenario_id, scenario in plan.scenarios.items():
        if scenario.exclusion_reason:
            continue
        if not scenario.leaf_task_ids:
            raise ValueError(f"Scenario has no Leaf Task: {scenario_id}")
        review = plan.reviews.get(scenario_id)
        if review is None or review.status != "approved":
            raise ValueError(f"Scenario lacks an approved review: {scenario_id}")
        if expected_templates is not None:
            for leaf_id in scenario.leaf_task_ids:
                actual = len(plan.leaf_tasks[leaf_id].question_templates)
                if actual != expected_templates:
                    raise ValueError(
                        f"Leaf Task {leaf_id} has {actual} templates; "
                        f"expected {expected_templates}"
                    )
    updates = _write_artifacts(context, plan)
    _assert_derived_artifact_consistency(context, plan)
    versions = publish(
        context, "Plan", {"task_plan": updates["task_plan_path"]},
        ("Dataset", "PlanningConfig", "TemplateConfig"),
    ) if context.get("run_directory") else {}
    return result(
        "complete",
        "Task Plan published",
        working_blueprint_path=updates["task_plan_path"],
        milestone={"kind": "plan_complete"},
        planning_incomplete=False,
        planning_status="complete",
        dataset_task_plans={
            context.get("dataset_scope", context.get("selected_dataset_id", "dataset")): updates["task_plan_path"]
        },
        **updates,
        **versions,
    )


def _available_tools(context: dict[str, Any]) -> set[str]:
    plan = _load_plan(context)
    if plan.phase in {"access", "overview"}:
        return {"read_planning_context", "read_dataset_overview", "read_evidence_catalog"}
    if plan.phase == "axes":
        return {
            "read_planning_context", "read_dataset_overview", "read_evidence_catalog",
            "read_axis_evidence", "analyze_axis",
        }
    if plan.phase in {"screening", "scenario_review"}:
        return {"review_candidate_scenarios"}
    if plan.phase == "leaves":
        state = context.get("planning_read_state", {})
        if not state.get("scenario_context_read"):
            return {"read_scenario_context"}
        scenario = plan.scenarios.get(plan.current_scenario_id)
        if not scenario or not scenario.leaf_task_ids:
            return {"read_evidence_catalog", "save_leaf_task"}
        return {"save_leaf_task", "review_current_scenario"}
    if plan.phase == "complete":
        return {"finish"}
    if plan.phase == "blocked":
        return {"read_planning_context", "read_dataset_overview"}
    return set()


def _workflow_pending_scenario_ids(plan: TaskPlanState) -> list[str]:
    """Return the queue that represents unfinished Scenario work in this phase."""
    if plan.phase in {"screening", "scenario_review"}:
        return (
            plan.scenario_review_pending_ids
            or plan.screening_pending_scenario_ids
            or plan.pending_scenario_ids
        )
    return plan.pending_scenario_ids


def _planning_progress_key(context: dict[str, Any]) -> dict[str, Any]:
    """Identify the persisted planning state for action retries and receipts."""
    try:
        plan = _load_plan(context)
    except (OSError, TypeError, ValueError, KeyError):
        return {}
    return {
        "revision": plan.revision,
        "phase": plan.phase,
        "current_scenario_id": plan.current_scenario_id,
        "pending_scenario_count": len(_workflow_pending_scenario_ids(plan)),
        "scenario_review_pending_count": len(plan.scenario_review_pending_ids),
        "leaf_task_count": len(plan.leaf_tasks),
    }


finish_task_space = finish_task_plan


def _planning_instructions(context: dict[str, Any]) -> str:
    """Load the frozen task-planning prompt from the experiment BIR."""
    # The external prompt covers workflow operations; stage prompts carry semantic constraints.
    store = _bir(context)
    base = store.registry().prompt("task_planning") if store else (
        "Build one resumable annotation Task Plan. Follow the current phase, use bounded "
        "evidence, and return one valid ToolAction at a time."
    )
    plan = _load_plan(context)
    scenario = plan.scenarios.get(plan.current_scenario_id) if plan.current_scenario_id else None
    scenario_text = "none"
    if scenario:
        scenario_text = ", ".join((scenario.input_structure, scenario.operation, scenario.clinical_target))
    available = ", ".join(sorted(_available_tools(context)))
    evidence_cache = context.get("planning_evidence_cache") or {}
    cache_text = json.dumps(evidence_cache, ensure_ascii=False)
    if len(cache_text) > 5000:
        cache_text = cache_text[:5000] + "…"
    runtime = (
        "\n\n## Current Runtime Configuration\n"
        f"- Phase: {plan.phase}\n"
        f"- Next Axis: {plan.next_axis.value if plan.next_axis else 'none'}\n"
        f"- Pending Scenario count: {len(_workflow_pending_scenario_ids(plan))}\n"
        f"- Scenario Review queue remaining: {len(plan.scenario_review_pending_ids)}\n"
        f"- Required question template count: {_configured_template_count(context) or 'not specified'}\n"
        f"- Current scenario (input_structure / operation / clinical_target): {scenario_text}\n"
        f"- Saved Leaf Task count for the current Scenario: {len(scenario.leaf_task_ids) if scenario else 0}\n"
        f"- Benchmark brief: {json.dumps(_benchmark_brief(context), ensure_ascii=False)}\n"
        f"- Current Scenario evidence cache: {cache_text if cache_text != '{}' else 'none'}\n"
        f"- Available planning tools: {available or 'none'}\n"
        "Use these runtime values as the active contract for the next action.\n"
        "When phase=axes, read the current Axis evidence once and then call analyze_axis for every allowed value.\n"
        "When next_axis=clinical_target, analyze_axis is the required next state-changing action; do not repeat an unchanged evidence-catalog read.\n"
        "Axis completion is persisted only by analyze_axis.\n"
        "Read tools are bounded queries, not progress actions. Do not repeat an unchanged read."
    )
    if plan.phase in {"screening", "scenario_review"}:
        runtime += "\nReview exactly one pending Scenario with review_candidate_scenarios, passing its scenario_id. Do not design Leaf Tasks until all Scenario Reviews are complete."
    if plan.phase == "leaves" and store:
        base += "\n\n" + store.registry().prompt("leaf_task_design")
    return base + runtime

def compile_task_planning_agent(*, checkpointer=None):
    tools = (
        read_planning_context,
        read_dataset_overview,
        read_evidence_catalog,
        read_axis_evidence,
        analyze_axis,
        read_scenario_context,
        screen_scenario,
        review_candidate_scenarios,
        save_leaf_task,
        review_current_scenario,
    )
    return compile_agent(
        "TaskPlanningAgent",
        _planning_instructions,
        {tool.__name__: tool for tool in tools},
        lambda c: {
            "task_plan": str(_plan_path(c)),
            "phase": _load_plan(c).phase,
            "next_axis": _load_plan(c).next_axis,
            "current_scenario_id": _load_plan(c).current_scenario_id,
            "pending_scenario_count": len(_workflow_pending_scenario_ids(_load_plan(c))),
            "scenario_review_pending_count": len(_load_plan(c).scenario_review_pending_ids),
            "evidence_manifest_path": _load_plan(c).evidence_manifest_path,
        },
        finish_task_plan,
        args_schemas={
            "read_evidence_catalog": EvidenceCatalogArguments,
            "read_axis_evidence": AxisEvidenceArguments,
            "analyze_axis": AxisArguments,
            "read_scenario_context": ScenarioArguments,
            "screen_scenario": ScenarioScreeningArguments,
            "review_candidate_scenarios": CandidateScenarioReviewArguments,
            "save_leaf_task": LeafTaskArguments,
            "review_current_scenario": ScenarioReviewArguments,
        },
        available_tools=_available_tools,
        initialize_context=_initialize_planning_context,
        progress_key=_planning_progress_key,
        include_inspection_tools=False,
        checkpointer=checkpointer,
    )


def apply_template_feedback(context, feedback):
    return result("blocked", "Template review is part of Task Planning")

