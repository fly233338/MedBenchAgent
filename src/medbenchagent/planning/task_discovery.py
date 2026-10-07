"""Deterministic Task Scenario and Leaf Task validation."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterable

from medbenchagent.planning.node_builder import build_task_scenarios
from medbenchagent.planning.resolver import (
    ArgmaxStep,
    MaskToBBoxStep,
    ReadMaskFeatureStep,
    ResolverSpec,
    validate_resolver_policy,
)
from medbenchagent.planning.task_space import (
    AxisAnalysis,
    LeafTask,
    TaskPlanState,
    TaskScenario,
    ValidatedClinicalTarget,
)


def validate_candidate_leaf(
    leaf: LeafTask,
    scenarios: dict[str, TaskScenario],
    *,
    resolver_context: dict[str, Any] | None = None,
    registry=None,
) -> LeafTask:
    scenario = scenarios.get(leaf.scenario_id)
    if scenario is None:
        raise ValueError("Leaf Task references an unknown Task Scenario")
    if scenario.exclusion_reason or scenario.status == "excluded":
        raise ValueError("Leaf Task cannot be created under an excluded Task Scenario")
    if leaf.capability_category != scenario.capability_category:
        raise ValueError("Leaf Task Category differs from its Task Scenario")
    target = next(
        (item if isinstance(item, ValidatedClinicalTarget) else ValidatedClinicalTarget.model_validate(item)
         for item in scenario.validated_clinical_targets
         if (item.target_id if isinstance(item, ValidatedClinicalTarget) else item.get("target_id")) == leaf.clinical_target_id),
        None,
    )
    if target is None:
        raise ValueError("Leaf Task references an unknown validated clinical target")
    if leaf.target.strip() != target.name.strip():
        raise ValueError("Leaf Task target does not match its validated clinical target")
    registry = registry or (resolver_context or {}).get("registry")
    operation_mode = registry.answer_options_mode(scenario.operation) if registry else (
        "required" if scenario.operation in {"recognize", "compare", "align"} else "forbidden"
    )
    if operation_mode == "required" and leaf.task_paradigm.value == "VQA" and leaf.output_spec.value == "label":
        if not leaf.options or len(leaf.options) < 2:
            raise ValueError("Closed VQA label tasks require at least two options")
    elif operation_mode == "forbidden" and leaf.options is not None:
        raise ValueError(f"Operation {scenario.operation} forbids answer options")
    spec = ResolverSpec.parse(leaf.resolver)
    if registry is not None:
        validate_resolver_policy(spec, registry.resolver_policy)
    if leaf.input != spec.inputs or leaf.output_spec != spec.output_spec:
        input_value = [selector.model_dump(mode="json") for selector in leaf.input]
        resolver_value = [selector.model_dump(mode="json") for selector in spec.inputs]
        raise ValueError(
            f"Leaf Task input and OutputSpec must exactly match Resolver; "
            f"input={input_value}, resolver_inputs={resolver_value}, "
            f"leaf_output_spec={leaf.output_spec.value}, resolver_output_spec={spec.output_spec.value}. "
            "Resolver.inputs must equal Leaf.input; keep reference-only evidence "
            "only in resolver steps such as reference_text."
        )
    contracts = (resolver_context or {}).get("registry").task_contracts if (resolver_context or {}).get("registry") else None
    if contracts:
        allowed = set(contracts.get("paradigm_output_specs", {}).get(leaf.task_paradigm.value, []))
        if leaf.output_spec.value not in allowed:
            raise ValueError("OutputSpec is incompatible with TaskParadigm")
        if leaf.task_paradigm.value == "Report Generation" and any(
            selector.evidence_type not in set(contracts.get("report_generation_input_types", ["asset"]))
            for selector in leaf.input
        ):
            raise ValueError("Report Generation accepts only Registry-approved input types")
    elif leaf.task_paradigm.value == "Report Generation" and any(
        selector.evidence_type != "asset" for selector in leaf.input
    ):
        raise ValueError("Report Generation accepts only assets as input")
    if resolver_context is not None and "evidence_names" in resolver_context:
        names = resolver_context.get("evidence_names", set())
        missing = [
            selector.role for selector in leaf.input
            if selector.name and selector.name not in names
        ]
        if missing:
            raise ValueError(f"Input Evidence roles cannot be resolved: {missing}")
        reference_only = set(resolver_context.get("reference_only_evidence_names", []))
        catalog = resolver_context.get("evidence_catalog", {})
        if isinstance(catalog, dict):
            for name, entry in catalog.items():
                if not isinstance(entry, dict):
                    continue
                if entry.get("input_visibility") == "reference_only" or entry.get("semantic_role") in {
                    "mask", "annotation", "label", "calibration", "reference"
                }:
                    reference_only.add(name)
        leaked = [
            selector.name for selector in leaf.input
            if selector.name and selector.name in reference_only
        ]
        if leaked:
            raise ValueError(f"Reference-only Evidence cannot be model inputs: {sorted(set(leaked))}")
        mask_steps = [step for step in spec.steps if isinstance(step, (MaskToBBoxStep, ReadMaskFeatureStep))]
        if mask_steps:
            mask_names = {
                step.selector.name
                for step in spec.steps
                if hasattr(step, "selector") and step.selector.name in reference_only
            }
            if not mask_names:
                raise ValueError("Mask-derived Resolver steps require reference-only mask Evidence")
            coverage = resolver_context.get("mask_feature_coverage", {})
            for step in mask_steps:
                if isinstance(step, MaskToBBoxStep):
                    feature = "bbox"
                else:
                    feature = step.feature
                if coverage and not coverage.get(feature, coverage.get(f"plane_{feature}", 0)):
                    raise ValueError(f"Resolver mask feature is unavailable: {feature}")
        if any(isinstance(step, ArgmaxStep) for step in spec.steps):
            if not resolver_context.get("ordered_collection_available") and not resolver_context.get("sampling_manifest_digest"):
                raise ValueError("argmax requires a frozen ordered sampling manifest")
        if scenario.input_structure in {"multi-view", "multi-image", "volume"}:
            if not any(
                resolver_context.get(key)
                for key in ("image_manifest_digest", "sampling_manifest_digest")
            ):
                raise ValueError("Multi-view, multi-image and volume tasks require a presentation or sampling manifest")
        if leaf.options is not None:
            catalog = resolver_context.get("evidence_catalog", {})
            derived = any(step.op in {"compare", "compute", "map_label"} for step in spec.steps)
            if catalog and not derived:
                names = {
                    step.selector.name
                    for step in spec.steps
                    if hasattr(step, "selector") and step.selector.name
                }
                candidates = set()
                for name in names:
                    entry = catalog.get(name, {})
                    from medbenchagent.planning.resolver import encode_closed_label
                    for item in entry.get("observed_values", []):
                        if isinstance(item, dict):
                            try:
                                candidates.add(encode_closed_label(item.get("value")))
                            except ValueError:
                                continue
                if candidates and not set(leaf.options).issubset(candidates):
                    raise ValueError("VQA options must come from Evidence Catalog observed_values")
    return leaf


def leaf_identity(leaf: LeafTask, scenario: TaskScenario | None = None):
    """Return a Scenario-independent semantic fingerprint for a Leaf.

    Scenario IDs are deliberately excluded so equivalent tasks cannot be
    counted twice when axis combinations put them in different Scenarios.
    Input and Resolver order remain significant for comparison direction and
    role binding semantics; options are normalized as a vocabulary.
    """
    spec = ResolverSpec.parse(leaf.resolver).model_dump(mode="json")
    return (
        scenario.clinical_target if scenario is not None else "",
        leaf.clinical_target_id,
        leaf.target.strip().casefold(),
        json.dumps([item.model_dump(mode="json") for item in leaf.input], sort_keys=True),
        leaf.reference_rule.strip().casefold() if leaf.reference_rule else "",
        json.dumps(spec, sort_keys=True),
        leaf.task_paradigm.value,
        leaf.output_spec.value,
        json.dumps(sorted((option.casefold() for option in (leaf.options or []))), ensure_ascii=False),
    )


def validate_task_tree(plan: TaskPlanState, *, require_reviews: bool = True, registry=None) -> TaskPlanState:
    if not plan.axis_decisions:
        raise ValueError("Task Plan requires completed Axis decisions")
    active = {key: value for key, value in plan.scenarios.items() if not value.exclusion_reason}
    for scenario_id, scenario in plan.scenarios.items():
        if scenario.exclusion_reason:
            if scenario.status != "excluded":
                raise ValueError(f"Excluded Scenario has invalid status: {scenario_id}")
            if scenario.leaf_task_ids:
                raise ValueError(f"Excluded Scenario still references Leaf Tasks: {scenario_id}")
    identities: set[tuple[str, str, str, str]] = set()
    leaf_ids: set[str] = set()
    for leaf_id, leaf in plan.leaf_tasks.items():
        validate_candidate_leaf(
            leaf,
            plan.scenarios,
            resolver_context={"registry": registry},
            registry=registry,
        )
        if leaf_id != leaf.leaf_task_id:
            raise ValueError(f"Leaf map key does not match leaf_task_id: {leaf_id}")
        identity = leaf_identity(leaf, plan.scenarios.get(leaf.scenario_id))
        if identity in identities:
            raise ValueError("Duplicate Leaf Task semantics")
        identities.add(identity)
        leaf_ids.add(leaf_id)
    for scenario_id, scenario in active.items():
        attached = [
            leaf_id for leaf_id in leaf_ids
            if plan.leaf_tasks[leaf_id].scenario_id == scenario_id
        ]
        if not attached:
            raise ValueError(f"Executable Task Scenario has no Leaf Task: {scenario_id}")
        if sorted(scenario.leaf_task_ids) != sorted(attached):
            raise ValueError(f"Scenario leaf_task_ids do not match leaves: {scenario_id}")
        if require_reviews:
            review = plan.reviews.get(scenario_id)
            if review is None or review.status != "approved" or set(review.checked_leaf_task_ids) != set(attached):
                raise ValueError(f"Scenario has no complete approval review: {scenario_id}")
        if any((plan.leaf_tasks[key].eligible_count or 0) <= 0 for key in attached):
            raise ValueError(f"Scenario contains a zero-capacity Leaf Task: {scenario_id}")
        if scenario.status != "completed":
            raise ValueError(f"Scenario is not completed: {scenario_id}")
    if set(plan.pending_scenario_ids) or plan.current_scenario_id:
        raise ValueError("Task Plan still has pending or active Scenarios")
    return plan


def compile_task_space_plan(
    *,
    dataset: dict[str, Any],
    axis_analysis: AxisAnalysis,
    node_context: dict[str, Any],
    leaves: Iterable[LeafTask] = (),
    revision: int = 0,
    registry=None,
) -> TaskPlanState:
    scenarios = build_task_scenarios(axis_analysis, node_context, registry=registry)
    accepted: dict[str, LeafTask] = {}
    identities: set[tuple[str, str, str, str]] = set()
    for leaf in leaves:
        validate_candidate_leaf(
            leaf,
            scenarios,
            resolver_context={"evidence_names": node_context.get("evidence_names", [])},
            registry=registry,
        )
        identity = leaf_identity(leaf, scenarios.get(leaf.scenario_id))
        if identity in identities:
            raise ValueError("Duplicate Leaf Task semantics")
        identities.add(identity)
        if leaf.leaf_task_id in accepted:
            raise ValueError(f"Duplicate Leaf Task identity: {leaf.leaf_task_id}")
        accepted[leaf.leaf_task_id] = leaf
        scenario = scenarios[leaf.scenario_id]
        scenarios[leaf.scenario_id] = scenario.model_copy(update={
            "leaf_task_ids": [*scenario.leaf_task_ids, leaf.leaf_task_id],
            "status": "completed" if leaf.eligible_count else scenario.status,
        })
    active = [key for key, value in scenarios.items() if not value.exclusion_reason]
    completed = [key for key in active if scenarios[key].leaf_task_ids]
    pending = [key for key in active if key not in completed]
    return TaskPlanState(
        dataset=dataset,
        axis_decisions={
            "input_structure": axis_analysis.input_structure,
            "operation": axis_analysis.operation,
            "clinical_target": axis_analysis.clinical_target,
        },
        scenarios=scenarios,
        leaf_tasks=accepted,
        phase="leaves",
        current_scenario_id=pending[0] if pending else None,
        pending_scenario_ids=pending,
        completed_scenario_ids=completed,
        revision=revision,
    )


def node_statistics(nodes: dict[str, TaskScenario]) -> dict[str, Any]:
    return {
        "total": len(nodes),
        "excluded": sum(bool(node.exclusion_reason) for node in nodes.values()),
        "with_leaves": sum(bool(node.leaf_task_ids) for node in nodes.values()),
        "by_capability_category": dict(Counter(node.capability_category for node in nodes.values())),
    }


def summarize_leaf(access, leaf, *, cases=None):
    """Capacity and review facts share one sampling/input resolution pass."""
    from medbenchagent.data.index import DatasetIndex
    from .sampling import sampling_units
    from .resolver import ResolverContext, resolve_task_input, NotApplicable, encode_closed_label
    if cases is None:
        if not access.get("cases_path") and not access.get("duckdb_path"):
            return {"contract_validated": True, "eligible_count": leaf.eligible_count, "failure_reasons": {},
                    "answer_distribution": {}, "successful_examples": []}
        cases = list(DatasetIndex(access).iter_cases())
    context = ResolverContext.from_access(access)
    units = sampling_units(cases, context.manifests.get("sampling"))
    failures, answers, examples = Counter(), Counter(), []
    for case, unit in units:
        resolved = resolve_task_input(case, leaf, unit, context)
        if isinstance(resolved, NotApplicable):
            failures[resolved.reason] += 1
            continue
        reference = resolved.reference
        value = reference.get("value", reference) if isinstance(reference, dict) else reference
        answer = encode_closed_label(value) if isinstance(value, (str, int, float, bool)) else None
        if answer is not None:
            answers[answer] += 1
        if len(examples) < 3:
            examples.append({"case_id": case["case_id"], "reference": reference, "answer": answer})
    eligible = len(units) - sum(failures.values())
    return {"contract_validated": True, "case_count": len(cases), "eligible_count": eligible,
            "successful_unit_count": eligible, "all_cases_eligible": not failures,
            "failure_reasons": dict(failures), "answer_distribution": dict(answers),
            "successful_examples": examples, "boundary_case_examples": examples}


class LeafDesignError(ValueError):
    def __init__(self, code, message, **details):
        super().__init__(message)
        self.code, self.details = code, details


def validate_leaf_design(leaf, plan, *, facts, access, registry, expected_templates, case_names):
    from difflib import get_close_matches
    def fail(code, message, **details):
        raise LeafDesignError(code, message, **details)
    try:
        parsed = LeafTask.model_validate(leaf.model_dump(mode="json") if hasattr(leaf, "model_dump") else leaf)
    except ValueError as exc:
        fail( "leaf_schema", str(exc), repair_hint="Preserve the submitted Leaf and correct only the reported schema fields.")
    if parsed.scenario_id != plan.current_scenario_id:
        fail( "scenario_mismatch", "Leaf Task must belong to the active Scenario", path="scenario_id", actual=parsed.scenario_id, expected=plan.current_scenario_id)
    if expected_templates is not None and len(parsed.question_templates) != expected_templates:
        fail( "template_count",
            f"Leaf Task requires exactly {expected_templates} question_templates; received {len(parsed.question_templates)}",
            path="question_templates", actual=len(parsed.question_templates), expected=expected_templates,
        )
    scenario = plan.scenarios[parsed.scenario_id]
    if parsed.capability_category != scenario.capability_category:
        fail( "capability_category", "Leaf Task Category must match its Scenario", path="capability_category", actual=parsed.capability_category.value, expected=scenario.capability_category.value)
    target = next(
        (ValidatedClinicalTarget.model_validate(item) for item in scenario.validated_clinical_targets
         if ValidatedClinicalTarget.model_validate(item).target_id == parsed.clinical_target_id),
        None,
    )
    if target is None:
        fail( "unknown_clinical_target_id",
            "Leaf Task must reference a validated clinical target from its Scenario",
            path="clinical_target_id", actual=parsed.clinical_target_id,
            expected=[ValidatedClinicalTarget.model_validate(item).target_id for item in scenario.validated_clinical_targets],
        )
    if parsed.target.strip() != target.name.strip():
        fail( "clinical_target_name_mismatch",
            "Leaf Task target must match the selected validated clinical target name",
            path="target", actual=parsed.target, expected=target.name,
        )
    evidence_names = set(facts.get("evidence_names", []))
    try:
        spec = ResolverSpec.parse(parsed.resolver)
    except ValueError as exc:
        fail( "resolver_schema", str(exc), repair_hint="Keep the same Leaf and repair the Resolver steps or result_step reported above.")
    catalog = facts.get("evidence_catalog", {})
    if not isinstance(catalog, dict):
        catalog = {row["name"]: row for row in catalog}
    evidence_names.update(case_names)
    available_names = sorted(set(catalog) | evidence_names)
    for index, selector in enumerate(parsed.input):
        if selector.name and selector.name not in evidence_names:
            fail( "missing_input_evidence",
                f"Input evidence name {selector.name!r} does not exist for this dataset.",
                path=f"input[{index}].name", actual=selector.name,
                expected=get_close_matches(selector.name, available_names, n=5, cutoff=0.3),
                repair_hint="Replace the input selector with one of the reported real evidence names and keep resolver.inputs identical.",
            )
    if parsed.input != spec.inputs:
        fail( "input_resolver_mismatch",
            "Leaf.input and Resolver.inputs differ.",
            actual={"input": [item.model_dump(mode="json") for item in parsed.input], "resolver_inputs": [item.model_dump(mode="json") for item in spec.inputs]},
            repair_hint="Make Resolver.inputs exactly equal to Leaf.input; keep reference-only evidence in resolver steps.",
        )
    try:
        validate_candidate_leaf(parsed, plan.scenarios, resolver_context={
            "evidence_names": evidence_names,
            "evidence_catalog": catalog,
            "reference_only_evidence_names": facts.get("reference_only_evidence_names", []),
            "mask_feature_coverage": facts.get("mask_feature_coverage", {}),
            "image_manifest_digest": access.get("image_manifest_digest"),
            "sampling_manifest_digest": access.get("sampling_manifest_digest"),
            "split_manifest_digest": access.get("split_manifest_digest"),
            "ordered_collection_available": bool(access.get("sampling_manifest_digest")),
            "registry": registry,
        })
    except ValueError as exc:
        fail( "leaf_contract", str(exc), repair_hint="Use the current evidence catalog and change only the rejected selector or contract field.")
    identity = leaf_identity(parsed, plan.scenarios.get(parsed.scenario_id))
    if any(
        key != parsed.leaf_task_id
        and leaf_identity(value, plan.scenarios.get(value.scenario_id)) == identity
        for key, value in plan.leaf_tasks.items()
    ):
        fail( "duplicate_leaf", "Duplicate Leaf Task semantics", repair_hint="Reuse or skip the existing Leaf Task when the semantics are already present; do not alter target or resolver semantics to bypass deduplication.")
    return parsed
