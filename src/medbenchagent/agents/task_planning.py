"""Resumable, evidence-staged Task Planning Agent."""

from __future__ import annotations


import json
from medbenchagent.planning.task_discovery import _cross_scenario_leaf_summaries
import difflib
from medbenchagent.data.task_facts import _field_meanings, _documented_annotation_semantics, _field_semantics, _case_evidence_names
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from medbenchagent.data.task_facts import _asset_rows, build_dataset_facts, case_facts, compact_case, compact_catalog_entry, compact_scalar
from medbenchagent.planning.node_builder import build_task_scenarios
from medbenchagent.planning.resolver import ResolverSpec
from medbenchagent.planning.bir import BIRStore, RegistrySnapshot
from medbenchagent.planning.extensions import PlanningExtensionCatalog
from medbenchagent.planning.task_discovery import validate_candidate_leaf, validate_task_tree
from medbenchagent.planning.task_space import (
    AXIS_VALUES,
    AxisAnalysis,
    AxisDecisionSet,
    AxisValueDecision,
    CapabilityCategory,
    EvidenceBasis,
    EvidenceSelector,
    ScenarioReview,
    TaskAxis,
    TaskPlanState,
    TaskParadigm,
    OutputSpec,
    ValidatedClinicalTarget,
)
from medbenchagent.planning.bir import require
from medbenchagent.runtime.harness import result
from medbenchagent.workflow.agent_runtime import compile_agent
from medbenchagent.runtime.llm import complete_json
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.runtime.tool_contracts import NoArguments


class LeafTaskValidationError(ValueError):
    """A repairable Leaf failure with model-visible structured diagnostics."""

    def __init__(self, message: str, diagnostics: dict[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics


def _target_model(value: ValidatedClinicalTarget | dict[str, Any]) -> ValidatedClinicalTarget:
    return value if isinstance(value, ValidatedClinicalTarget) else ValidatedClinicalTarget.model_validate(value)


def _target_dump(value: ValidatedClinicalTarget | dict[str, Any]) -> dict[str, Any]:
    return _target_model(value).model_dump(mode="json")


class AxisArguments(NoArguments):
    axis: TaskAxis = Field(description="Axis being assessed; use the next axis returned by planning context.")

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


class ScenarioScreeningResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    decision: Literal["keep", "exclude"]
    reason: str = Field(min_length=1)
    supporting_fields: list[str] = Field(default_factory=list)
    validated_clinical_targets: list[ValidatedClinicalTarget] = Field(default_factory=list)
    evidence_basis: list[EvidenceBasis] = Field(min_length=1)
    exclusion_category: Literal["insufficient_evidence", "unsupported", "user_excluded"] | None = None

    @model_validator(mode="after")
    def validate_record(self):
        if len(set(self.supporting_fields)) != len(self.supporting_fields):
            raise ValueError("Scenario screening supporting_fields must be unique")
        target_ids = [item.target_id for item in self.validated_clinical_targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("Scenario screening target IDs must be unique")
        if not self.evidence_basis:
            raise ValueError("Scenario screening requires evidence_basis")
        if self.decision == "keep" and not self.validated_clinical_targets:
            raise ValueError("Keeping a Scenario requires at least one validated clinical target")
        if self.decision == "exclude" and self.exclusion_category is None:
            raise ValueError("Exclusion must distinguish insufficient evidence, unsupported tasks and user exclusions")
        return self


class LeafTaskDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    leaf_task_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    clinical_target_id: str = Field(min_length=1)
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

    @field_validator("output_spec", mode="before")
    @classmethod
    def parse_output_spec(cls, value):
        return OutputSpec(value)

    @field_validator("task_paradigm", mode="before")
    @classmethod
    def parse_task_paradigm(cls, value):
        return TaskParadigm(value)

    @field_validator("capability_category", mode="before")
    @classmethod
    def parse_capability_category(cls, value):
        return CapabilityCategory(value)

    @model_validator(mode="after")
    def validate_templates(self):
        if any(not template.strip() for template in self.question_templates):
            raise ValueError("question_templates must be non-empty")
        if any("{{" in template or "}}" in template for template in self.question_templates):
            raise ValueError("question_templates must be complete; unresolved placeholders are not supported")
        if len(set(self.question_templates)) != len(self.question_templates):
            raise ValueError("question_templates must be unique")
        return self


class LeafTaskArguments(NoArguments):
    leaf: LeafTaskDraft


class ScenarioReviewArguments(NoArguments):
    pass


class ScenarioExclusionArguments(NoArguments):
    scenario_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    evidence_basis: list[EvidenceBasis] = Field(min_length=1)
    exclusion_category: Literal["insufficient_evidence", "unsupported", "user_excluded"]


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _access(context: dict[str, Any]) -> dict[str, Any]:
    if context.get("bir_path") and context.get("dataset_scope"):
        store = BIRStore(context["bir_path"])
        value = store.load("dataset_bindings.json")["datasets"][context["dataset_scope"]]["access"]
        return value.get("source_access", value)
    return context.get("dataset_access_manifest", {})


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
    return Path(context.get("task_plan_path") or Path(context["operation_directory"]) / "task_plan.json")


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



def _planning_extensions(context: dict[str, Any]) -> PlanningExtensionCatalog | None:
    registry = _registry(context)
    return registry.planning_extensions() if registry else None


def _selected_extension_ids(context: dict[str, Any]) -> list[str]:
    plan = _load_plan(context)
    selection = plan.extension_selection or context.get("planning_extension_selection") or {}
    return list(selection.get("selected_extension_ids", []))


def _render_extensions(context: dict[str, Any], *, stage: str) -> str:
    catalog = _planning_extensions(context)
    if catalog is None:
        return "No dataset-specific planning extensions are selected for this stage."
    return catalog.render(_selected_extension_ids(context), stage=stage)


def _extension_insertions(context: dict[str, Any], *, stage: str) -> dict[str, str]:
    return {"{{DATASET_SPECIFIC_RULES}}": _render_extensions(context, stage=stage)}



_LEAF_CONTEXT_CHAR_BUDGET = 64000


def _planning_binding(context: dict[str, Any], plan: TaskPlanState, scenario_id: str | None) -> dict[str, Any]:
    access = _access(context)
    facts = _facts(context, plan)
    return {
        "dataset_id": access.get(
            "dataset_id",
            context.get("dataset_scope", context.get("selected_dataset_id", plan.dataset.get("dataset_id"))),
        ) or plan.dataset.get("dataset_id"),
        "revision": access.get("revision", facts.get("revision")),
        "scenario_id": scenario_id,
    }


def _bound_cache(context: dict[str, Any], plan: TaskPlanState, scenario_id: str | None) -> dict[str, Any]:
    binding = _planning_binding(context, plan, scenario_id)
    cache = context.get("planning_evidence_cache")
    if not isinstance(cache, dict) or cache.get("binding") != binding:
        cache = {
            "binding": binding,
            "scenario_id": scenario_id,
            "planning_evidence": [],
            "representative_cases": [],
            "catalog_fields": {},
        }
        context["planning_evidence_cache"] = cache
    cache.setdefault("planning_evidence", [])
    cache.setdefault("representative_cases", [])
    cache.setdefault("catalog_fields", {})
    cache["scenario_id"] = scenario_id
    return cache


def _bound_read_state(context: dict[str, Any], plan: TaskPlanState, scenario_id: str | None) -> dict[str, Any]:
    binding = _planning_binding(context, plan, scenario_id)
    state = context.get("planning_read_state")
    if not isinstance(state, dict) or state.get("binding") != binding:
        state = {
            "binding": binding,
            "scenario_id": scenario_id,
            "scenario_context_read": False,
            "catalog_queries": [],
            "catalog_cache": {},
        }
        context["planning_read_state"] = state
    state.setdefault("catalog_queries", [])
    state.setdefault("catalog_cache", {})
    return state


def _reset_planning_context(context: dict[str, Any], plan: TaskPlanState, scenario_id: str | None) -> None:
    binding = _planning_binding(context, plan, scenario_id)
    context["planning_read_state"] = {
        "binding": binding,
        "scenario_id": scenario_id,
        "scenario_context_read": False,
        "catalog_queries": [],
        "catalog_cache": {},
    }
    context["planning_evidence_cache"] = {
        "binding": binding,
        "scenario_id": scenario_id,
        "planning_evidence": [],
        "representative_cases": [],
        "catalog_fields": {},
    }


def _merge_catalog_payload(context: dict[str, Any], plan: TaskPlanState, payload: dict[str, Any]) -> dict[str, Any]:
    cache = _bound_cache(context, plan, payload.get("scenario_id") or plan.current_scenario_id)
    fields = cache.setdefault("catalog_fields", {})
    for entry in payload.get("entries", []):
        if isinstance(entry, dict) and entry.get("name"):
            fields[str(entry["name"])] = dict(entry)
    for name, value in (payload.get("field_values") or {}).items():
        item = fields.setdefault(name, {"name": name})
        previous = {json.dumps(row.get("value"), sort_keys=True, ensure_ascii=False): row
                    for row in item.get("field_values", []) if isinstance(row, dict)}
        for row in value.get("values", []):
            if isinstance(row, dict):
                previous[json.dumps(row.get("value"), sort_keys=True, ensure_ascii=False)] = row
        item["field_values"] = list(previous.values())
        item["field_value_total"] = value.get("total")
        item["field_values_truncated"] = value.get("truncated", False)
    for name, samples in (payload.get("case_samples") or {}).items():
        item = fields.setdefault(name, {"name": name})
        previous = {str(row.get("case_id")): row for row in item.get("case_samples", []) if isinstance(row, dict)}
        for row in samples:
            if isinstance(row, dict) and row.get("case_id"):
                previous[str(row["case_id"])] = row
        item["case_samples"] = list(previous.values())[:6]
    return cache


def _bounded_leaf_context(value: dict[str, Any]) -> dict[str, Any]:
    from medbenchagent.planning.context import bounded_evidence
    from medbenchagent.settings import get_settings
    return bounded_evidence(value, char_budget=get_settings().planning.context_chars)


def _name_suggestions(name: str, available: list[str]) -> list[str]:
    if not name:
        return available[:20]
    folded = name.casefold()
    contains = [item for item in available if folded in item.casefold() or item.casefold() in folded]
    close = difflib.get_close_matches(name, available, n=5, cutoff=0.25)
    return list(dict.fromkeys([*contains[:5], *close]))[:8]



def _scenario_context_payload(context: dict[str, Any], plan: TaskPlanState, scenario: Any, raw: dict[str, Any]) -> dict[str, Any]:
    from medbenchagent.data.index import DatasetIndex

    facts = _facts(context, plan)
    catalog = _catalog_map(facts.get("evidence_catalog", {}))
    meanings = _field_meanings(_access(context))
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
    required = {name for target in scenario.validated_clinical_targets for name in target.reference_fields}
    ordered_names = sorted(required & set(catalog)) + [name for name in ordered_names if name not in required]
    evidence = [compact_catalog_entry(catalog[name], meanings.get(name) or catalog[name].get("meaning")) for name in ordered_names]
    assets = {row.get("asset_id"): row for row in DatasetIndex(_access(context)).assets()}
    candidate_names = {item.get("name") for item in evidence if item.get("name")}
    samples = [
        compact_case(
            case,
            assets,
            names=candidate_names,
            reference_names=set(facts.get("reference_only_evidence_names", [])),
        )
        for case in raw.get("representative_cases", [])[:6]
    ]
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




def scenario_evidence_context(context, plan, scenario_id):
    """Fresh, source-backed evidence packet shared by design and independent review."""
    from medbenchagent.data.index import DatasetIndex
    local = dict(context)
    scoped = plan.model_copy(update={"current_scenario_id": scenario_id})
    _reset_planning_context(local, scoped, scenario_id)
    raw = DatasetIndex(_access(local)).node_context(scoped.scenarios[scenario_id], sample_limit=6, registry=_registry(local))
    compact = _scenario_context_payload(local, scoped, scoped.scenarios[scenario_id], raw)
    cache = _bound_cache(local, scoped, scenario_id)
    cache.update(planning_evidence=compact["planning_evidence"], representative_cases=compact["representative_cases"])
    return _leaf_context(local, scoped)


def _leaf_context(context: dict[str, Any], plan: TaskPlanState) -> dict[str, Any]:
    scenario = plan.scenarios.get(plan.current_scenario_id) if plan.current_scenario_id else None
    facts = _facts(context, plan)
    scenario_id = scenario.scenario_id if scenario else None
    cache = _bound_cache(context, plan, scenario_id)
    catalog = list(cache.get("planning_evidence") or [
        compact_catalog_entry(item, item.get("meaning"))
        for item in list(_catalog_map(facts.get("evidence_catalog", {})).values())
    ])
    for name, item in cache.get("catalog_fields", {}).items():
        if name not in {entry.get("name") for entry in catalog}:
            compact = compact_catalog_entry(item, item.get("meaning"))
            if item.get("field_values"):
                compact["field_values"] = item["field_values"]
            if item.get("case_samples"):
                compact["case_samples"] = item["case_samples"][:6]
            catalog.append(compact)
        else:
            for index, entry in enumerate(catalog):
                if entry.get("name") == name:
                    merged = dict(entry)
                    if item.get("field_values"):
                        merged["field_values"] = item["field_values"]
                    if item.get("case_samples"):
                        merged["case_samples"] = item["case_samples"][:6]
                    catalog[index] = merged
    representative = cache.get("representative_cases", [])[:6]
    try:
        from medbenchagent.data.index import DatasetIndex
        asset_rows = {row.get("asset_id"): row for row in DatasetIndex(_access(context)).assets()}
        representative = [
            compact_case(
                case,
                asset_rows,
                names={item.get("name") for item in case.get("evidence", []) if item.get("name")},
                reference_names=set(facts.get("reference_only_evidence_names", [])),
            )
            for case in representative
        ]
        representative = [case for case in representative if case.get("evidence")]
    except Exception:
        representative = []
    existing = []
    if scenario:
        for leaf_id in scenario.leaf_task_ids:
            leaf = plan.leaf_tasks.get(leaf_id)
            if leaf:
                existing.append({
                    key: value
                    for key, value in leaf.model_dump(mode="json").items()
                    if key in {"leaf_task_id", "name", "clinical_target_id", "target", "input", "reference_rule",
                               "resolver", "output_spec", "task_paradigm", "question_templates", "options"}
                })
    required = set(cache.get("catalog_fields", {}))
    if scenario:
        required.update(name for target in scenario.validated_clinical_targets for name in target.reference_fields)
        required.update(selector.name for leaf_id in scenario.leaf_task_ids for selector in plan.leaf_tasks[leaf_id].input if selector.name)
    by_name = _catalog_map(facts.get("evidence_catalog", {}))
    present = {entry.get("name") for entry in catalog}
    catalog.extend(compact_catalog_entry(by_name[name], by_name[name].get("meaning")) for name in sorted(required - present) if name in by_name)
    access = _access(context)
    return _bounded_leaf_context({
        "required_evidence_names": sorted(required),
        "dataset_binding": {
            "dataset_id": access.get("dataset_id", context.get("dataset_scope", context.get("selected_dataset_id"))),
            "revision": access.get("revision", facts.get("revision")),
        },
        "extension_selection": plan.extension_selection,
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "input_structure": scenario.input_structure,
            "operation": scenario.operation,
            "clinical_target": scenario.clinical_target,
            "capability_category": scenario.capability_category.value,
            "feasible_case_count": scenario.feasible_case_count,
            "status": scenario.status,
            "leaf_task_ids": scenario.leaf_task_ids,
            "screening_reason": scenario.screening_reason,
            "validated_clinical_targets": [_target_dump(item) for item in scenario.validated_clinical_targets],
        } if scenario else None,
        "evidence_catalog": catalog,
        "annotation_semantics": _documented_annotation_semantics(_access(context)),
        "field_semantics": _field_semantics(_access(context), [entry.get("name") for entry in catalog if entry.get("name")]),
        "relations": facts.get("relations", {}),
        "derived_capabilities": facts.get("derived_capabilities", {}),
        "mask_feature_coverage": facts.get("mask_feature_coverage", {}),
        "view_panel_capabilities": facts.get("view_panel_capabilities", {}),
        "reference_only_evidence_names": facts.get("reference_only_evidence_names", []),
        "representative_cases": representative,
        "coverage_and_limitations": {
            "case_count": facts.get("case_count"),
            "evidence_coverage": {
                key: value.get("cases")
                for key, value in _catalog_map(facts.get("evidence_catalog", {})).items()
                if isinstance(value, dict) and "cases" in value
            },
            "annotation_completeness": "unavailable unless documented by preparation evidence",
        },
        "existing_leaf_tasks": existing,
    })


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
        "relevant_evidence": [compact_catalog_entry(item, item.get("meaning")) for item in list(catalog.values())[:20]],
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
        "input_policy": context.get("benchmark_input_policy"),
    }
    return {
        key: value for key, value in brief.items()
        if value not in (None, "", [], {})
    }


def _load_plan(context: dict[str, Any]) -> TaskPlanState:
    bir = _bir(context)
    if bir:
        dataset = context.get("dataset_scope", context.get("selected_dataset_id", "dataset"))
        relative = f"datasets/{dataset}/planning/task_plan.json"
        if (bir.root / relative).is_file():
            from medbenchagent.planning.bir import BIRStaleError, business_dependencies
            with bir.lock:
                context["planning_read_digest"] = bir.digest(relative)
                try:
                    value = bir.load(relative)
                except BIRStaleError:
                    value = None
                commit = json.loads((bir.root / relative).read_text(encoding="utf-8"))["_commit"]
                expected = business_dependencies(context, "benchmark_info.json", "dataset_bindings.json", registry=True)
                if value is not None and commit["dependencies"] == expected:
                    return TaskPlanState.model_validate(value)
            return TaskPlanState(dataset={"dataset_id": dataset})
    path = _plan_path(context)
    if not bir and path.is_file():
        return TaskPlanState.model_validate(_read(path))
    return TaskPlanState(
        dataset={"dataset_id": context.get("dataset_scope", context.get("selected_dataset_id", "dataset"))}
    )


def _write_json(path: Path, value: Any) -> str:
    atomic_write_json(path, value)
    return str(path)


def _write_artifacts(context, plan):
    from medbenchagent.planning.bir import persist_task_plan
    return persist_task_plan(context, plan, plan_path=_plan_path(context), bir=_bir(context), access=_access(context))


def _initialize_planning_context(context: dict[str, Any]) -> dict[str, Any]:
    context = dict(context)
    if _plan_path(context).is_file():
        plan = _load_plan(context)
        if plan.phase != "access":
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

    bir = _bir(context)
    access_path = bir.root / "dataset_bindings.json" if bir else Path(context["operation_directory"]) / "access.json"
    if not bir:
        _write_json(access_path, access)
    plan = TaskPlanState(
        dataset={"dataset_id": context.get("dataset_scope", "dataset")},
        access_path=str(access_path),
        phase="axes" if facts.get("case_count", 0) and facts.get("evidence_names") else "blocked",
        extension_selection=context.get("planning_extension_selection", {}),
        review={} if facts.get("case_count", 0) and facts.get("evidence_names") else {"reason": "Dataset has no Case or Evidence"},
    )
    context.update(_write_artifacts(context, plan))
    context["dataset_facts"] = facts
    return context


def _axis_order() -> tuple[TaskAxis, ...]:
    return (TaskAxis.INPUT_STRUCTURE, TaskAxis.OPERATION, TaskAxis.CLINICAL_TARGET)


def _facts(context: dict[str, Any], plan: TaskPlanState) -> dict[str, Any]:
    facts = context.get("dataset_facts")
    access = _access(context)
    if facts is not None and facts.get("revision", access.get("revision")) == access.get("revision"):
        return facts
    from medbenchagent.data.index import DatasetIndex
    facts = DatasetIndex(access).facts()
    context["dataset_facts"] = facts
    return facts


def _build_scenarios(context: dict[str, Any], plan: TaskPlanState) -> TaskPlanState:
    facts = _facts(context, plan)
    node_context = {
        "evidence": facts.get("evidence", {}),
        "assets": facts.get("assets", {}),
        "relations": facts.get("relations", {}),
        "derived_capabilities": facts.get("derived_capabilities", {}),
        "mask_feature_coverage": facts.get("mask_feature_coverage", {}),
        "view_panel_capabilities": facts.get("view_panel_capabilities", {}),
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
    plan.scenarios = scenarios
    plan.phase = "screening" if active else "complete"
    plan.pending_scenario_ids = active.copy()
    plan.completed_scenario_ids = []
    plan.current_scenario_id = active[0] if active else None
    return plan


def _clinical_target_definition(context: dict[str, Any], target: str) -> dict[str, Any]:
    registry = _registry(context)
    if not registry:
        return {"id": target, "description": ""}
    axis = next((item for item in registry.axis_definitions() if item.get("id") == "clinical_target"), {})
    return next((item for item in axis.get("values", []) if item.get("id") == target), {"id": target})


def _screening_payload(context: dict[str, Any], plan: TaskPlanState, scenario) -> dict[str, Any]:
    registry = _registry(context)
    facts = _facts(context, plan)
    meanings = _field_meanings(_access(context))
    catalog = []
    for entry in _catalog_map(facts.get("evidence_catalog", {})).values():
        item = compact_catalog_entry(entry, meanings.get(entry.get("name")) or entry.get("meaning"))
        catalog.append(item)
    operation = registry.operation_definition(scenario.operation) if registry else {
        "description": scenario.operation, "answer_options": "forbidden"
    }
    return _bounded_leaf_context({
        "required_evidence_names": [item["name"] for item in catalog],
        "dataset": {
            "dataset_id": plan.dataset.get("dataset_id"),
            "revision": facts.get("revision"),
            "case_count": facts.get("case_count", 0),
        },
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "input_structure": scenario.input_structure,
            "operation": scenario.operation,
            "clinical_target": scenario.clinical_target,
            "capability_category": scenario.capability_category.value,
            "feasible_case_count": scenario.feasible_case_count,
        },
        "operation_definition": operation,
        "clinical_target_definition": _clinical_target_definition(context, scenario.clinical_target),
        "evidence_catalog": catalog,
        "allowed_reference_fields": [item["name"] for item in catalog if item.get("name")],
        "field_semantics": _field_semantics(_access(context), [item.get("name") for item in catalog if item.get("name")]),
        "relations": facts.get("relations", {}),
        "derived_capabilities": facts.get("derived_capabilities", {}),
        "annotation_semantics": _documented_annotation_semantics(_access(context)),
        "count_semantics": {
            "feasible_case_count": "Cases satisfying structural Scenario constraints; not a question or Leaf count.",
            "eligible_count": "SamplingUnits with valid model images and a deterministic answer.",
            "review": "Model review checks task meaning, wording, reference validity and template applicability.",
        },
        "benchmark_brief": _benchmark_brief(context),
    })


def _apply_screening_record(plan: TaskPlanState, scenario, reviewed: ScenarioScreeningResult) -> None:
    update = {
        "screening_decision": reviewed.decision,
        "screening_reason": reviewed.reason,
        "screening_supporting_fields": reviewed.supporting_fields,
        "validated_clinical_targets": reviewed.validated_clinical_targets,
    }
    if reviewed.decision == "exclude":
        update.update({
            "status": "excluded",
            "exclusion_reason": reviewed.reason,
            "exclusion_stage": "screening",
            "exclusion_evidence": reviewed.evidence_basis,
            "exclusion_category": reviewed.exclusion_category,
        })
    else:
        update.update({"status": "active", "exclusion_reason": None, "exclusion_stage": None, "exclusion_evidence": []})
    plan.scenarios[scenario.scenario_id] = scenario.model_copy(update=update)


def _validate_screening_targets(context: dict[str, Any], plan: TaskPlanState, scenario,
                                targets: list[ValidatedClinicalTarget]) -> None:
    names = set(_facts(context, plan).get("evidence_names", []))
    names.update(_catalog_map(_facts(context, plan).get("evidence_catalog", {})))
    seen = set()
    normalized = [_target_model(target) for target in targets]
    for target in normalized:
        if target.target_id in seen:
            raise ValueError(f"Duplicate validated clinical target ID: {target.target_id}")
        seen.add(target.target_id)
        if target.target_type != scenario.clinical_target:
            raise ValueError(
                f"Validated target {target.target_id} has type {target.target_type!r}; "
                f"Scenario requires {scenario.clinical_target!r}"
            )
        missing = [field for field in target.reference_fields if field not in names]
        if missing:
            raise ValueError(
                f"Validated target {target.target_id} references unknown evidence fields: {missing}"
            )


def review_candidate_scenarios(context, *, scenario_id, review=None):
    """Screen one candidate using its source evidence and optional user constraints."""
    plan = _load_plan(context)
    if plan.phase != "screening" or not plan.pending_scenario_ids:
        raise ValueError("No candidate Scenarios are pending screening")
    if scenario_id != plan.pending_scenario_ids[0]:
        raise ValueError(f"Scenario screening must process the queue head {plan.pending_scenario_ids[0]}")
    scenario = plan.scenarios[scenario_id]
    if review is None:
        review = complete_json(
            _stage_prompt(context, "scenario_screening"),
            json.dumps(_screening_payload(context, plan, scenario), ensure_ascii=False),
            ScenarioScreeningResult,
            validate=lambda value: _validate_screening_targets(context, plan, scenario, value.validated_clinical_targets))
    reviewed = ScenarioScreeningResult.model_validate(review)
    _validate_screening_targets(context, plan, scenario, reviewed.validated_clinical_targets)
    _apply_screening_record(plan, scenario, reviewed)
    remaining = plan.pending_scenario_ids[1:]
    if not remaining:
        remaining = [key for key, item in plan.scenarios.items()
                     if not item.exclusion_reason and item.status != "completed"]
        plan.phase = "leaves" if remaining else "complete"
    plan.pending_scenario_ids = remaining
    plan.current_scenario_id = remaining[0] if remaining else None
    plan.revision += 1
    _reset_planning_context(context, plan, plan.current_scenario_id)
    return result("success", f"Scenario Review {scenario_id}: {reviewed.decision}",
                  **_write_artifacts(context, plan))


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
            _stage_prompt(context, "axis_analysis") if registry else "Assess the supplied Axis using bounded evidence.",
            json.dumps({"axis": axis.value, "axis_definition": axis_definition, "allowed_values": sorted(allowed),
                        "bounded_evidence": json.loads(evidence),
                        "benchmark_brief": _benchmark_brief(context),
                        "resolver_policy": registry.resolver_policy if registry else {},
                        "task_contracts": registry.task_contracts if registry else {},
                        "input_policy": _benchmark_brief(context).get("input_policy") if isinstance(_benchmark_brief(context), dict) else None}, ensure_ascii=False),
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


def read_planning_context(context: dict[str, Any]) -> dict[str, Any]:
    plan = _load_plan(context)
    return result("success", json.dumps({
        "schema_version": "planning-index",
        "phase": plan.phase,
        "next_axis": plan.next_axis,
        "extension_selection": plan.extension_selection,
        "axis_decisions": list(plan.axis_decisions),
        "scenario_count": len(plan.scenarios),
        "leaf_task_count": len(plan.leaf_tasks),
        "question_template_count": _configured_template_count(context),
    }, ensure_ascii=False))


def read_dataset_overview(context: dict[str, Any]) -> dict[str, Any]:
    plan = _load_plan(context)
    facts = _facts(context, plan)
    overview = {key: facts.get(key) for key in ("revision", "case_count", "patient_count", "assets", "relations")}
    overview["dataset_id"] = plan.dataset.get("dataset_id")
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


class FieldValuesArguments(NoArguments):
    name: str = Field(min_length=1)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=50)
    text_offset: int = Field(default=0, ge=0)


class SemanticsArguments(NoArguments):
    pointer: list[str] = Field(default_factory=list)
    offset: int = Field(default=0, ge=0)


class LeafPageArguments(NoArguments):
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=5, ge=1, le=20)


def read_field_values(context, *, name, offset=0, limit=20, text_offset=0):
    """Read exact observed values with explicit pagination; excerpts are never answer labels."""
    from medbenchagent.data.index import DatasetIndex
    plan = _load_plan(context)
    if name not in _catalog_map(_facts(context, plan).get("evidence_catalog", {})):
        raise ValueError("Unknown evidence field")
    page = DatasetIndex(_access(context)).values(name, offset=offset, limit=limit)
    values = page["values"]
    while len(values) > 1 and len(json.dumps(values, ensure_ascii=False)) > 12000:
        values.pop()
    payload = {"field": name, "values": values, "total": page["total"],
               "next_offset": offset + len(values) if offset + len(values) < page["total"] else None,
               "revision": _access(context).get("revision")}
    encoded = json.dumps(values, ensure_ascii=False)
    if len(encoded) > 12000:
        payload.update(values=[], excerpt=encoded[text_offset:text_offset + 12000],
                       excerpt_is_label=False, value_index=offset,
                       next_text_offset=text_offset + 12000 if text_offset + 12000 < len(encoded) else None)
    else:
        _merge_catalog_payload(context, plan, {"entries": [{"name": name}],
            "field_values": {name: {"values": values, "total": page["total"], "truncated": payload["next_offset"] is not None}}})
    return result("success", "Observed value page loaded", data=payload,
                  planning_evidence_cache=context.get("planning_evidence_cache", {}))


def read_annotation_semantics(context, *, pointer=None, offset=0):
    """Read a JSON pointer into frozen annotation metadata, in explicit text pages."""
    from medbenchagent.data.task_facts import annotation_metadata_path
    path = annotation_metadata_path(_access(context))
    if not path:
        return result("success", "Annotation semantics are unavailable", data={"status": "unavailable"})
    value = json.loads(path.read_text(encoding="utf-8"))
    for key in pointer or []:
        try:
            value = value[int(key)] if isinstance(value, list) else value[key]
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise ValueError("Unknown annotation metadata pointer") from exc
    encoded = json.dumps(value, ensure_ascii=False)
    return result("success", "Frozen annotation semantics loaded", data={
        "source": str(path), "pointer": pointer or [], "text": encoded[offset:offset + 12000],
        "next_offset": offset + 12000 if offset + 12000 < len(encoded) else None,
        "revision": _access(context).get("revision")})


def read_leaf_tasks(context, *, offset=0, limit=5):
    """Read existing canonical Leaf Tasks without rebuilding them from observation summaries."""
    plan = _load_plan(context)
    leaves = list(plan.leaf_tasks.values())
    return result("success", "Canonical Leaf page loaded", data={
        "leaves": [leaf.model_dump(mode="json") for leaf in leaves[offset:offset + limit]],
        "total": len(leaves), "next_offset": offset + limit if offset + limit < len(leaves) else None})


class AxisEvidenceArguments(NoArguments):
    axis: TaskAxis

    @field_validator("axis", mode="before")
    @classmethod
    def parse_axis(cls, value):
        return TaskAxis(value)


def read_evidence_catalog(context: dict[str, Any], *, offset=0, limit=20, names=None) -> dict[str, Any]:
    """Read catalog entries, concrete values and compact matching Case samples."""
    from medbenchagent.data.index import DatasetIndex

    plan = _load_plan(context)
    state = _bound_read_state(context, plan, plan.current_scenario_id)
    query = json.dumps({"offset": offset, "limit": limit, "names": sorted(names or [])}, sort_keys=True)
    facts = _facts(context, plan)
    all_entries = list(_catalog_map(facts.get("evidence_catalog", {})).values())
    by_name = _catalog_map(all_entries)
    available = sorted(by_name)
    requested = list(dict.fromkeys(names or []))
    unmatched = [name for name in requested if name not in by_name]
    matched_names = [name for name in requested if name in by_name]
    entries = [by_name[name] for name in matched_names] if requested else all_entries
    page = entries[offset:offset + limit]
    payload = {
        "revision": facts.get("revision"),
        "query": {"offset": offset, "limit": limit, "names": requested},
        "entries": [compact_catalog_entry(item, item.get("meaning")) for item in page],
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
                value = compact_scalar(item.get("value"), limit=180)
                truncated = truncated or value != item.get("value")
                compact_values.append({"value": value, "count": item.get("count", 0)})
            values[name] = {
                "values": compact_values,
                "total": value_page.get("total", 0),
                "truncated": truncated,
            }
            cases = query_index.sample(evidence_name=name, limit=6, diverse=True).get("values", [])
            samples[name] = [compact_case(case, assets, names={name}) for case in cases]
        payload["field_values"] = values
        payload["case_samples"] = samples
    cached = state["catalog_cache"].get(query)
    if cached is None:
        state["catalog_queries"] = [*state.get("catalog_queries", []), query]
        state["catalog_cache"][query] = payload
        cached = payload
    cached["scenario_id"] = plan.current_scenario_id
    cache = _merge_catalog_payload(context, plan, cached)
    return result(
        "success",
        json.dumps(cached, ensure_ascii=False),
        planning_read_state=state,
        planning_evidence_cache=cache,
        leaf_context=_leaf_context(context, plan),
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
            "derived_capabilities": facts.get("derived_capabilities", {}),
            "evidence": facts.get("evidence", {}),
        },
        "evidence_catalog": list(facts.get("evidence_catalog", {}).values())
        if isinstance(facts.get("evidence_catalog"), dict) else facts.get("evidence_catalog", []),
        "field_semantics": _field_semantics(
            context,
            [item.get("name") for item in facts.get("evidence_catalog", {}).values()]
            if isinstance(facts.get("evidence_catalog"), dict) else [],
        ),
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
    evidence["required_evidence_names"] = [row["name"] for row in evidence["evidence_catalog"]]
    evidence = _bounded_leaf_context(evidence)
    axis_dir = Path(context["operation_directory"]) / "axis_evidence"
    path = axis_dir / f"{axis.value}.json"
    _write_json(path, evidence)
    return result("success", json.dumps(evidence, ensure_ascii=False), evidence_path=str(path))


def read_scenario_context(context: dict[str, Any], *, scenario_id: str | None = None) -> dict[str, Any]:
    """Read the current Scenario once; the next Leaf action must be evidence lookup or save."""
    plan = _load_plan(context)
    scenario_id = scenario_id or plan.current_scenario_id or (
        plan.pending_scenario_ids[0] if plan.pending_scenario_ids else None
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
    state = _bound_read_state(context, plan, scenario_id)
    if state.get("scenario_id") == scenario_id and state.get("scenario_context_read"):
        raise ValueError("This Scenario context was already read; choose save_leaf_task or a new evidence request")
    state.update({"scenario_id": scenario_id, "scenario_context_read": True})
    context_value["next_action"] = "design_leaf"
    context_value["instruction"] = "Use the listed evidence to design and save a Leaf Task. Do not reread this Scenario."
    cache = _bound_cache(context, plan, scenario_id)
    cache.update({
        "planning_evidence": context_value.get("planning_evidence", []),
        "representative_cases": context_value.get("representative_cases", []),
    })
    return result(
        "success",
        json.dumps(context_value, ensure_ascii=False),
        planning_read_state=state,
        leaf_context=_leaf_context(context, plan),
        planning_evidence_cache=cache,
    )


def save_leaf_task(context: dict[str, Any], *, leaf: LeafTaskDraft) -> dict[str, Any]:
    """Terminal Leaf-design action after the current Scenario and required evidence are available."""
    plan = _load_plan(context)
    raw_leaf = leaf
    if plan.phase != "leaves" or not plan.current_scenario_id:
        raise ValueError("An active Task Scenario is required before saving a Leaf Task")
    from medbenchagent.planning.task_discovery import validate_leaf_design, LeafDesignError
    try:
        parsed = validate_leaf_design(leaf, plan, facts=_facts(context, plan), access=_access(context),
            registry=_registry(context), expected_templates=_configured_template_count(context),
            case_names=_case_evidence_names(_access(context)))
    except LeafDesignError as exc:
        _raise_leaf_failure(context, plan, raw_leaf, exc.code, str(exc), **exc.details)
    scenario = plan.scenarios[parsed.scenario_id]
    from medbenchagent.data.index import DatasetIndex
    try:
        query = DatasetIndex(_access(context))
        if hasattr(query, "iter_cases"):
            cases = list(query.iter_cases())
        else:
            page = query.sample(offset=0, limit=512)
            cases = list(page.get("values", []))
    except Exception as exc:
        _raise_leaf_failure(context, plan, raw_leaf, "dataset_query", str(exc), repair_hint="Use the available planning evidence and retry this Leaf after the dataset query is available.")
    from medbenchagent.planning.task_discovery import summarize_leaf
    summary = summarize_leaf(_access(context), parsed, cases=cases, scenarios=plan.scenarios)
    resolution_reasons = Counter(summary["failure_reasons"])
    resolved_answers = set(summary["answer_distribution"])
    eligible = summary["eligible_count"]
    operation_mode = (_registry(context).answer_options_mode(scenario.operation)
                      if _registry(context) else ("required" if scenario.operation in {"recognize", "compare", "align"} else "forbidden"))
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
    existing = plan.leaf_tasks.get(parsed.leaf_task_id)
    if existing is not None and existing.scenario_id != parsed.scenario_id:
        _raise_leaf_failure(context, plan, raw_leaf, "leaf_id_conflict", "A Leaf ID cannot move between Scenarios", path="leaf_task_id", actual=parsed.leaf_task_id, expected=existing.scenario_id)
    plan.leaf_tasks[parsed.leaf_task_id] = parsed
    plan.revision += 1
    scenario = plan.scenarios[parsed.scenario_id]
    leaf_ids = list(scenario.leaf_task_ids)
    if parsed.leaf_task_id not in leaf_ids:
        leaf_ids.append(parsed.leaf_task_id)
    plan.scenarios[parsed.scenario_id] = scenario.model_copy(
        update={
            "leaf_task_ids": leaf_ids,
            "status": "review_pending",
        }
    )
    read_state = _bound_read_state(context, plan, plan.current_scenario_id)
    cache = _bound_cache(context, plan, plan.current_scenario_id)
    updates = _write_artifacts(context, plan)
    return result(
        "success",
        f"Leaf Task {parsed.leaf_task_id} saved ({eligible} eligible Cases)",
        milestone={"kind": "leaf", "leaf_task_id": parsed.leaf_task_id, "capacity": eligible},
        planning_read_state=read_state,
        planning_evidence_cache=cache,
        leaf_context=_leaf_context(context, plan),
        **updates,
    )


def _leaf_validation_summary(context, leaf):
    from medbenchagent.planning.task_discovery import summarize_leaf
    return summarize_leaf(_access(context), leaf, scenarios=_load_plan(context).scenarios)


def exclude_current_scenario(
    context: dict[str, Any], *, scenario_id: str, reason: str,
    evidence_basis: list[EvidenceBasis],
    exclusion_category: Literal["insufficient_evidence", "unsupported", "user_excluded"],
) -> dict[str, Any]:
    """Exclude a Leaf-design Scenario after its required evidence was read."""
    plan = _load_plan(context)
    if plan.phase != "leaves" or plan.current_scenario_id != scenario_id:
        raise ValueError("Only the active Leaf-design Scenario can be excluded")
    state = _bound_read_state(context, plan, scenario_id)
    if not state.get("scenario_context_read"):
        raise ValueError("Scenario context must be read before excluding the Scenario")
    scenario = plan.scenarios[scenario_id]
    if scenario.leaf_task_ids:
        raise ValueError("A Scenario can be excluded only after it has no saved Leaf Tasks")
    if not reason.strip() or not evidence_basis:
        raise ValueError("Scenario exclusion requires a non-empty reason and evidence_basis")
    plan.scenarios[scenario_id] = scenario.model_copy(update={
        "status": "excluded",
        "exclusion_reason": reason.strip(),
        "exclusion_stage": "leaf_review",
        "exclusion_category": exclusion_category,
        "exclusion_evidence": evidence_basis,
    })
    plan.pending_scenario_ids = [item for item in plan.pending_scenario_ids if item != scenario_id]
    plan.completed_scenario_ids = [item for item in plan.completed_scenario_ids if item != scenario_id]
    if plan.pending_scenario_ids:
        plan.phase = "leaves"
        plan.current_scenario_id = plan.pending_scenario_ids[0]
    else:
        plan.phase = "complete"
        plan.current_scenario_id = None
    plan.revision += 1
    _reset_planning_context(context, plan, plan.current_scenario_id)
    updates = _write_artifacts(context, plan)
    return result(
        "success",
        f"Scenario {scenario_id} excluded after Leaf exploration",
        milestone={
            "kind": "scenario_excluded",
            "scenario_id": scenario_id,
            "next_scenario_id": plan.current_scenario_id,
            "status": "excluded",
        },
        planning_incomplete=plan.phase not in {"complete", "blocked"},
        excluded_scenario_id=scenario_id,
        next_pending_scenario_id=plan.current_scenario_id,
        planning_read_state=context["planning_read_state"],
        planning_evidence_cache=context["planning_evidence_cache"],
        **updates,
    )



def evaluate_scenario(context, plan, scenario):
    summaries = [
        {
            **{key: value for key, value in leaf.model_dump(mode="json").items()
               if key in {"leaf_task_id", "name", "clinical_target_id", "target", "input", "reference_rule", "resolver",
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
    prompt = _stage_prompt(context, "scenario_review") if registry else "Review the supplied Leaf summaries and return a complete ScenarioReview."
    review = ScenarioReview.model_validate(complete_json(
        prompt,
        json.dumps({
            "source_evidence": scenario_evidence_context(context, plan, scenario.scenario_id),
            "scenario": {
                "scenario_id": scenario.scenario_id,
                "input_structure": scenario.input_structure,
                "operation": scenario.operation,
                "clinical_target": scenario.clinical_target,
                "validated_clinical_targets": [_target_dump(item) for item in scenario.validated_clinical_targets],
                "capability_category": scenario.capability_category.value,
                "feasible_case_count": scenario.feasible_case_count,
            },
            "leaf_tasks": summaries,
            "operation_guidance": operation_guidance,
            "clinical_target_definition": _clinical_target_definition(context, scenario.clinical_target),
            "cross_scenario_leaf_summaries": _cross_scenario_leaf_summaries(plan, scenario.scenario_id),
            "count_semantics": {
                "feasible_case_count": "Structural Scenario capacity, not generated question count.",
                "eligible_count": "Resolver-successful SamplingUnits for this Leaf.",
                "review": "Semantic validity of task, wording and reference contract.",
            },
            "benchmark_brief": _benchmark_brief(context),
        }, ensure_ascii=False),
        ScenarioReview,
    ))
    return review


def review_current_scenario(context: dict[str, Any], *, review: ScenarioReview | None = None) -> dict[str, Any]:
    plan = _load_plan(context)
    if plan.phase != "leaves" or not plan.current_scenario_id:
        raise ValueError("An active Scenario is required before review")
    scenario = plan.scenarios[plan.current_scenario_id]
    if review is None:
        review = evaluate_scenario(context, plan, scenario)
    elif not isinstance(review, ScenarioReview):
        review = ScenarioReview.model_validate(review)
    if review.scenario_id != plan.current_scenario_id:
        raise ValueError("Review must belong to the active Scenario")
    if review.source_concerns:
        raise LeafTaskValidationError("Source semantics require supervision before accepting these tasks", {
            "source_concerns": [item.model_dump(mode="json") for item in review.source_concerns]})
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
            resolver_context={
                "evidence_names": evidence_names,
                "reference_only_evidence_names": _facts(context, plan).get("reference_only_evidence_names", []),
                "mask_feature_coverage": _facts(context, plan).get("mask_feature_coverage", {}),
                "image_manifest_digest": _access(context).get("image_manifest_digest"),
                "sampling_manifest_digest": _access(context).get("sampling_manifest_digest"),
                "ordered_collection_available": bool(_access(context).get("sampling_manifest_digest")),
                "registry": _registry(context),
            },
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
        else:
            plan.current_scenario_id = None
            plan.phase = "complete"
    else:
        plan.completed_scenario_ids = [
            key for key in plan.completed_scenario_ids if key != review.scenario_id
        ]
    plan.revision += 1
    _reset_planning_context(context, plan, plan.current_scenario_id)
    updates = _write_artifacts(context, plan)
    next_pending = plan.pending_scenario_ids[0] if plan.pending_scenario_ids else None
    review_summary = {
        "status": review.status,
        "checked_leaf_task_ids": sorted(remaining),
        "discarded_leaf_task_ids": sorted(discarded),
        "issues": list(review.issues),
    }
    return result(
        "success",
        f"Scenario {review.scenario_id} review saved: {json.dumps(review_summary, ensure_ascii=False)}",
        milestone={
            "kind": "scenario_progress" if next_pending else "scenario_complete",
            "scenario_id": review.scenario_id,
            "next_scenario_id": next_pending,
            "status": review.status,
        },
        planning_incomplete=bool(next_pending),
        completed_scenario_id=review.scenario_id,
        next_pending_scenario_id=next_pending,
        planning_read_state=context["planning_read_state"],
        planning_evidence_cache=context["planning_evidence_cache"],
        scenario_review=review_summary,
        **updates,
    )


def finish_task_plan(context: dict[str, Any]) -> dict[str, Any]:
    require(context, "Dataset", "PlanningConfig", "TemplateConfig")
    plan = _load_plan(context)
    if (
        plan.phase != "complete"
        or plan.pending_scenario_ids
        or plan.current_scenario_id
    ):
        raise ValueError("Task Plan still has pending Scenarios")
    validate_task_tree(plan, require_reviews=True, registry=_registry(context))
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
    executable_scenarios = [item for item in plan.scenarios.values() if not item.exclusion_reason]
    executable_leaves = [
        leaf_id for item in executable_scenarios for leaf_id in item.leaf_task_ids
    ]
    excluded_count = sum(bool(item.exclusion_reason) for item in plan.scenarios.values())
    summary = "Task Plan published"
    if not executable_leaves:
        summary = "Task Plan published; no executable tasks remain because all Scenarios were excluded"
    return result(
        "complete",
        summary,
        milestone={"kind": "plan_complete"},
        planning_incomplete=False,
        planning_status="complete",
        executable_scenario_count=len(executable_scenarios),
        executable_leaf_count=len(executable_leaves),
        excluded_scenario_count=excluded_count,
        dataset_task_plans={
            context.get("dataset_scope", context.get("selected_dataset_id", "dataset")): updates["task_plan_path"]
        },
        **updates,
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
    if plan.phase == "screening":
        return {"review_candidate_scenarios"}
    if plan.phase == "leaves":
        state = _bound_read_state(context, plan, plan.current_scenario_id)
        if not state.get("scenario_context_read"):
            return {"read_scenario_context"}
        scenario = plan.scenarios.get(plan.current_scenario_id)
        if not scenario or not scenario.leaf_task_ids:
            return {"read_evidence_catalog", "read_field_values", "read_annotation_semantics", "read_leaf_tasks", "save_leaf_task", "exclude_current_scenario"}
        return {"read_evidence_catalog", "read_field_values", "read_annotation_semantics", "read_leaf_tasks", "save_leaf_task", "review_current_scenario"}
    if plan.phase == "complete":
        return {"finish"}
    if plan.phase == "blocked":
        return {"read_planning_context", "read_dataset_overview"}
    return set()




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
        "pending_scenario_count": len(plan.pending_scenario_ids),
        "scenario_review_pending_count": (len(plan.pending_scenario_ids) if plan.phase == "screening" else 0),
        "leaf_task_count": len(plan.leaf_tasks),
        "excluded_scenario_count": sum(bool(item.exclusion_reason) for item in plan.scenarios.values()),
        "executable_scenario_count": sum(not item.exclusion_reason for item in plan.scenarios.values()),
    }




def _planning_description(context: dict[str, Any]) -> dict[str, Any]:
    plan = _load_plan(context)
    description = {
        "task_plan": str(_plan_path(context)),
        "phase": plan.phase,
        "next_axis": plan.next_axis,
        "current_scenario_id": plan.current_scenario_id,
        "pending_scenario_count": len(plan.pending_scenario_ids),
        "scenario_review_pending_count": (len(plan.pending_scenario_ids) if plan.phase == "screening" else 0),
        "excluded_scenario_count": sum(bool(item.exclusion_reason) for item in plan.scenarios.values()),
        "executable_leaf_count": sum(
            len(item.leaf_task_ids) for item in plan.scenarios.values() if not item.exclusion_reason
        ),
        "benchmark_brief": _benchmark_brief(context),
        "feedback": plan.review.get("revision_request", context.get("planning_feedback", {})),
    }
    if plan.phase == "leaves":
        description["leaf_context"] = _leaf_context(context, plan)
    return description


def _render_leaf_prompt(context: dict[str, Any], plan: TaskPlanState, template: str) -> str:
    registry = _registry(context)
    scenario = plan.scenarios.get(plan.current_scenario_id) if plan.current_scenario_id else None
    operation = registry.operation_definition(scenario.operation) if registry and scenario else {}
    target = {}
    if registry and scenario:
        target_axis = next(
            (item for item in registry.axis_definitions() if item.get("id") == "clinical_target"),
            {},
        )
        target = next(
            (item for item in target_axis.get("values", []) if item.get("id") == scenario.clinical_target),
            {},
        )
    replacements = {
        "{{current_planning_state_and_scenario_json}}": "See the structured state and leaf_context in this request.",
        "{{question_template_count}}": str(_configured_template_count(context) or "not specified"),
        "{{benchmark_output_language}}": str(_benchmark_brief(context).get("language", "unavailable")),
        "{{benchmark_input_policy}}": str(context.get("benchmark_input_policy", "unavailable")),
        "{{current_operation_and_clinical_target_definitions_json}}": json.dumps(
            {"operation": operation, "clinical_target": target}, ensure_ascii=False
        ),
        "{{resolver_policy_and_task_contracts_json}}": json.dumps(
            {
                "resolver_policy": registry.resolver_policy if registry else {},
                "task_contracts": registry.task_contracts if registry else {},
            }, ensure_ascii=False
        ),
    }
    replacements.update(_extension_insertions(context, stage="leaf_task_design"))
    rendered = template
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def _stage_prompt(context: dict[str, Any], name: str) -> str:
    registry = _registry(context)
    template = registry.prompt(name) if registry else ""
    rendered = template
    for placeholder, value in _extension_insertions(context, stage=name).items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def _planning_instructions(context: dict[str, Any]) -> str:
    """Load the frozen task-planning prompt from the experiment BIR."""
    # The external prompt covers workflow operations; stage prompts carry semantic constraints.
    store = _bir(context)
    base = store.registry().prompt("task_planning") if store else (
        "Build one resumable annotation Task Plan. Follow the current phase, use bounded "
        "evidence, and return one valid ToolAction at a time."
    )
    plan = _load_plan(context)
    if plan.phase == "leaves" and store:
        leaf = _render_leaf_prompt(context, plan, store.registry().prompt("leaf_task_design", include_shared=False))
        return "\n\n".join((base, leaf))
    return base


def _planning_automatic_action(context):
    plan = _load_plan(context)
    if plan.phase == "axes" and plan.next_axis:
        return {"tool": "analyze_axis", "arguments": {"axis": plan.next_axis.value}, "reason": "Assess the next registered Axis"}
    if plan.phase == "screening" and plan.pending_scenario_ids:
        return {"tool": "review_candidate_scenarios", "arguments": {"scenario_id": plan.pending_scenario_ids[0]}, "reason": "Screen the next candidate against source evidence"}
    if plan.phase == "leaves" and not _bound_read_state(context, plan, plan.current_scenario_id).get("scenario_context_read"):
        return {"tool": "read_scenario_context", "arguments": {"scenario_id": plan.current_scenario_id}, "reason": "Load source evidence for this Scenario"}
    if plan.phase == "complete":
        return {"tool": "finish", "arguments": {}, "reason": "Validate the completed task plan"}
    return None


def _planning_model_context(context, payload):
    """Inject dynamic facts once, independently of generic executor state."""
    description = payload["state"]
    leaf_context = description.pop("leaf_context", None)
    payload["leaf_context"] = leaf_context
    # Read output lives in the scoped packet or explicit structured tool data.
    payload["recent_observations"] = [{key: row[key] for key in ("tool", "status") if key in row}
                                      for row in payload.get("recent_observations", [])]
    payload["last_result"] = {key: value for key, value in payload.get("last_result", {}).items()
                              if key in {"status", "diagnostics", "data"}}
    from medbenchagent.runtime.construction import record_event
    encoded = json.dumps(payload, ensure_ascii=False)
    record_event(context["construction_events_path"], "planning_context_metrics",
                 run_id=context.get("run_id"), agent="TaskPlanningAgent", dataset_id=context.get("dataset_scope"),
                 context_chars=len(encoded), context_token_estimate=(len(encoded.encode()) + 3) // 4)
    return payload


def compile_task_planning_agent(*, checkpointer=None):
    tools = (
        read_planning_context,
        read_dataset_overview,
        read_evidence_catalog,
        read_field_values, read_annotation_semantics, read_leaf_tasks,
        read_axis_evidence,
        analyze_axis,
        read_scenario_context,
        review_candidate_scenarios,
        save_leaf_task,
        review_current_scenario,
        exclude_current_scenario,
    )
    return compile_agent(
        "TaskPlanningAgent",
        _planning_instructions,
        {tool.__name__: tool for tool in tools},
        lambda c: {
            "task_plan": str(_plan_path(c)),
            **_planning_description(c),
        },
        finish_task_plan,
        args_schemas={
            "read_evidence_catalog": EvidenceCatalogArguments,
            "read_field_values": FieldValuesArguments, "read_annotation_semantics": SemanticsArguments,
            "read_leaf_tasks": LeafPageArguments,
            "read_axis_evidence": AxisEvidenceArguments,
            "analyze_axis": AxisArguments,
            "read_scenario_context": ScenarioArguments,
            "review_candidate_scenarios": CandidateScenarioReviewArguments,
            "save_leaf_task": LeafTaskArguments,
            "review_current_scenario": ScenarioReviewArguments,
            "exclude_current_scenario": ScenarioExclusionArguments,
        },
        available_tools=_available_tools,
        automatic_action=_planning_automatic_action,
        model_context=_planning_model_context,
        read_tools=frozenset({"read_planning_context", "read_dataset_overview", "read_evidence_catalog", "read_axis_evidence", "read_field_values", "read_annotation_semantics", "read_leaf_tasks"}),
        initialize_context=_initialize_planning_context,
        progress_key=_planning_progress_key,
        include_inspection_tools=False,
        checkpointer=checkpointer,
    )
