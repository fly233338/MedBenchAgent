from compilation_fixtures import planning_context
from compilation_fixtures import registry
from compilation_fixtures import screen
from image_fixtures import image_context
import pytest

from medbenchagent.planning.node_builder import build_task_scenarios
from medbenchagent.planning.resolver import ResolverSpec, resolve
from medbenchagent.planning.task_discovery import leaf_identity
from medbenchagent.planning.task_space import (
    AxisAnalysis,
    AxisValueDecision,
    AxisDecisionSet,
    EvidenceBasis,
    LeafTask,
    OutputSpec,
    TaskScenario,
)


def _analysis(**supported):
    def values(axis, names):
        allowed = registry().axis_values(axis)
        return [AxisValueDecision(value=value, supported=value in names,
                               evidence_basis=[EvidenceBasis(reference="case:c1", detail="observed")], reason="observed") for value in allowed]
    return AxisAnalysis(input_structure=values("input_structure", supported.get("input_structure", [])),
                        operation=values("operation", supported.get("operation", [])),
                        clinical_target=values("clinical_target", supported.get("clinical_target", [])),
                        dataset_revision="1")


def test_axis_registry_and_node_ids_are_deterministic():
    analysis = _analysis(input_structure=["single-image"], operation=["recognize"], clinical_target=["finding"])
    first = build_task_scenarios(analysis, {"evidence": {}}, registry=registry())
    second = build_task_scenarios(analysis, {"evidence": {}}, registry=registry())
    assert first.keys() == second.keys()
    assert len(first) == 1
    assert next(iter(first.values())).capability_category == "Recognition"


def test_axis_decision_set_accepts_json_axis_value_under_strict_contract():
    value = AxisDecisionSet.model_validate({
        "axis": "input_structure",
        "dataset_revision": "1",
        "values": [{
            "value": "single-image",
            "supported": True,
            "evidence_basis": [{"reference": "dataset_overview:assets", "detail": "observed"}],
            "reason": "The dataset contains one primary image per case.",
        }],
    })

    assert value.axis.value == "input_structure"


def test_node_constraints_preserve_empty_nodes_with_reason():
    analysis = _analysis(input_structure=["multi-view"], operation=["recognize"], clinical_target=["finding"])
    nodes = build_task_scenarios(analysis, {"assets": {"related_image_cases": 0}, "evidence": {}}, registry=registry())
    node = next(iter(nodes.values()))
    assert node.exclusion_reason == "multi-view requires related image assets"


def test_resolver_separates_input_asset_from_reference_text():
    case = {"case_id": "c1", "evidence": [
        {"id": "asset", "type": "asset", "name": "image", "value": "a1", "ref": []},
        {"id": "report", "type": "text", "name": "report", "value": "Clear.", "ref": ["asset"]},
    ]}
    spec = ResolverSpec(
        inputs=[{"role": "image", "name": "image", "evidence_type": "asset"}],
        steps=[
            {"op": "read", "selector": {"role": "image", "name": "image", "evidence_type": "asset"}},
            {"op": "reference_text", "selector": {"role": "reference", "name": "report", "evidence_type": "text"}},
        ], output_spec=OutputSpec.TEXT, result_step="step_1",
    )
    result = resolve(case, spec)
    assert result.reference == "Clear." or result.reference.get("value") == "Clear."
    assert result.input_evidence_ids == ["asset"]
    assert result.reference_evidence_ids == ["report"]


def test_resolver_map_label_is_explicit_and_unmapped_values_are_not_applicable():
    def spec(mapping):
        return ResolverSpec(
            inputs=[{"role": "image", "name": "image", "evidence_type": "asset"}],
            steps=[
                {"op": "read", "selector": {"role": "label", "name": "margin", "evidence_type": "value"}},
                {"op": "map_label", "input": "step_0", "mapping": mapping},
            ],
            output_spec=OutputSpec.LABEL,
            result_step="step_1",
        )

    case = {"case_id": "c1", "evidence": [
        {"id": "image", "type": "asset", "name": "image", "value": "a1", "ref": []},
        {"id": "margin", "type": "value", "name": "margin", "value": "not circumscribed - angular", "ref": []},
    ]}
    mapped = resolve(case, spec({"not circumscribed - angular": "not circumscribed"}))
    assert mapped.reference == "not circumscribed"
    assert mapped.reference_evidence_ids == ["margin"]
    unmapped = resolve(case, spec({"circumscribed": "circumscribed"}))
    assert unmapped.reason == "map_label_unmapped"


def test_leaf_identity_is_cross_scenario_but_preserves_distinct_targets():
    base = {
        "question_templates": ["What is the category?"],
        "input": [{"role": "image", "name": "image", "evidence_type": "asset"}],
        "reference_rule": "Use the documented category.",
        "resolver": {
            "inputs": [{"role": "image", "name": "image", "evidence_type": "asset"}],
            "steps": [{"op": "read", "selector": {"role": "label", "name": "label", "evidence_type": "value"}}],
            "output_spec": "label", "result_step": "step_0",
        },
        "output_spec": "label", "task_paradigm": "VQA", "capability_category": "Recognition",
        "options": ["a", "b"],
    }
    assessment = LeafTask.model_validate({"leaf_task_id": "a", "name": "a", "scenario_id": "scenario-a", "clinical_target_id": "assessment", "target": "assessment", **base})
    same_task = assessment.model_copy(update={"leaf_task_id": "b", "scenario_id": "scenario-b"})
    diagnosis = assessment.model_copy(update={"leaf_task_id": "c", "scenario_id": "scenario-c", "clinical_target_id": "diagnosis", "target": "diagnosis"})
    assert leaf_identity(assessment) == leaf_identity(same_task)
    assert leaf_identity(assessment) != leaf_identity(diagnosis)
    assessment_scenario = TaskScenario(
        scenario_id="scenario-a", input_structure="single-image", operation="recognize",
        clinical_target="assessment", capability_category="Recognition",
    )
    diagnosis_scenario = assessment_scenario.model_copy(update={
        "scenario_id": "scenario-b", "clinical_target": "diagnosis",
    })
    assert leaf_identity(assessment, assessment_scenario) != leaf_identity(same_task, diagnosis_scenario)


def test_resolver_accepts_named_region_and_value_inputs():
    case = {"case_id": "c1", "evidence": [
        {"id": "image", "type": "asset", "name": "image", "value": "a1", "ref": []},
        {"id": "region", "type": "region", "name": "lesion_region", "value": [0, 0, 1, 1], "ref": []},
        {"id": "size", "type": "value", "name": "pixel_size", "value": 2.0, "ref": []},
    ]}
    spec = ResolverSpec(
        inputs=[
            {"role": "image", "name": "image", "evidence_type": "asset"},
        ],
        steps=[
            {"op": "collect", "selector": {"role": "region", "name": "lesion_region", "evidence_type": "region"}},
            {"op": "compute", "inputs": ["step_0"], "operation": "count"},
        ],
        output_spec=OutputSpec.NUMBER,
            result_step="step_1",
    )
    result = resolve(case, spec)
    assert result.reference == 1

    # Structured value/region selectors are also valid when explicitly named as inputs.
    for evidence_type, name in (("region", "lesion_region"), ("value", "pixel_size")):
        ResolverSpec(
            inputs=[{"role": "item", "name": name, "evidence_type": evidence_type}],
            steps=[{"op": "collect", "selector": {"role": "item", "name": name, "evidence_type": evidence_type}}],
            output_spec=OutputSpec.NUMBER,
            result_step="step_0",
        )


def test_resolver_rejects_unnamed_or_relation_inputs():
    base = {
        "steps": [{"op": "collect", "selector": {"role": "x", "evidence_type": "region"}}],
        "output_spec": OutputSpec.NUMBER,
        "result_step": "step_0",
    }
    for selector in (
        {"role": "x", "evidence_type": "region"},
        {"role": "x", "name": "region", "evidence_type": "region", "relation": "image"},
    ):
        try:
            ResolverSpec(inputs=[selector], **base)
        except ValueError as exc:
            assert "explicit asset/text/value/region names" in str(exc)
        else:
            raise AssertionError("invalid Resolver input was accepted")


def test_resolver_reports_positional_result_step_ids():
    with pytest.raises(ValueError, match=r"result_step must be one of: step_0"):
        ResolverSpec(
            inputs=[{"role": "image", "name": "image", "evidence_type": "asset"}],
            steps=[{"op": "read", "selector": {"role": "image", "name": "image", "evidence_type": "asset"}}],
            output_spec=OutputSpec.TEXT,
            result_step="read",
        )


def test_task_paradigm_rejects_incompatible_output():
    from test_global_construction import _plan
    from medbenchagent.planning.task_discovery import validate_candidate_leaf
    plan = _plan("d", {"leaf": 1})
    original = plan.leaf_tasks["leaf"]
    leaf = LeafTask.model_validate({**original.model_dump(mode="json"), "task_paradigm": "Report Generation", "options": None})
    with pytest.raises(ValueError, match="incompatible"):
        validate_candidate_leaf(leaf, plan.scenarios, registry=registry())


def test_canonical_planner_axis_and_node_tools_persist_bounded_artifacts(tmp_path, monkeypatch):
    from medbenchagent.agents import task_planning

    context = {**planning_context(tmp_path),
        "dataset_scope": "d",
        "operation_directory": str(tmp_path),
    }
    context = task_planning._initialize_planning_context(context)
    analysis = _analysis(input_structure=["single-image"], operation=["recognize"], clinical_target=["finding"]).model_copy(update={"dataset_revision": "7"}).model_dump()
    for axis in ("input_structure", "operation", "clinical_target"):
        saved = task_planning.analyze_axis(context, axis=axis, assessments=analysis[axis])
    context.update(saved["updates"])
    built = saved
    assert built["updates"]["task_plan_path"]
    assert task_planning._load_plan(context).model_dump(mode="json")["scenarios"]


def test_save_task_leaf_compiles_resolver_and_counts_capacity(tmp_path, monkeypatch):
    from medbenchagent.agents import task_planning

    context = {**planning_context(tmp_path),"operation_directory": str(tmp_path), "dataset_scope": "d",
               }
    context = task_planning._initialize_planning_context(context)
    axis = _analysis(input_structure=["single-image"], operation=["describe"], clinical_target=["finding"]).model_copy(update={"dataset_revision": "7"}).model_dump()
    for name in ("input_structure", "operation", "clinical_target"):
        context.update(task_planning.analyze_axis(context, axis=name, assessments=axis[name])["updates"])
    node_id = next(key for key, value in task_planning._load_plan(context).model_dump(mode="json")["scenarios"].items() if not value["exclusion_reason"])
    screened = screen(
        context, scenario_id=node_id, decision="keep", reason="The report scenario is suitable.",
        validated_clinical_targets=[{"target_id": "finding_report", "name": "finding", "target_type": "finding", "subject": "case", "scope": "case", "reference_kind": "direct", "reference_fields": ["report"]}],
    )
    context.update(screened["updates"])

    class FakeQuery:
        def __init__(self, _access): pass
        def sample(self, **_kwargs):
            return {"values": [{"case_id": "c1", "evidence": [
                {"id": "asset", "type": "asset", "name": "image", "value": "a1", "ref": []},
                {"id": "report", "type": "text", "name": "report", "value": "Clear", "ref": ["asset"]},
            ]}]}
    monkeypatch.setattr("medbenchagent.data.index.DatasetIndex", FakeQuery)
    image_inputs = image_context(tmp_path / "images", FakeQuery(None).sample()["values"])
    monkeypatch.setattr("medbenchagent.planning.resolver.ResolverContext.from_access", lambda access: image_inputs)
    leaf = {"leaf_task_id": "leaf-1", "name": "report", "scenario_id": node_id, "clinical_target_id": "finding_report", "question_templates": ["Describe the finding."],
            "input": [{"role": "image", "name": "image", "evidence_type": "asset"}], "target": "finding",
            "reference_rule": "Read the paired report", "resolver": {"inputs": [{"role": "image", "name": "image", "evidence_type": "asset"}],
            "steps": [{"op": "read", "selector": {"role": "image", "name": "image", "evidence_type": "asset"}},
                      {"op": "reference_text", "selector": {"role": "reference", "name": "report", "evidence_type": "text"}}],
            "output_spec": "text", "result_step": "step_1"}, "output_spec": "text", "task_paradigm": "Report Generation",
            "capability_category": "Generation"}
    task_planning.save_leaf_task(
        context,
        leaf=task_planning.LeafTaskDraft.model_validate(leaf),
    )
    assert task_planning._load_plan(context).model_dump(mode="json")["leaf_tasks"]["leaf-1"]["eligible_count"] == 1
