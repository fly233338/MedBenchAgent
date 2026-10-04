from medbenchagent.planning.pipeline import run_pipeline
from medbenchagent.planning.task_space import (
    AxisAnalysis,
    AxisValueDecision,
    EvidenceBasis,
    TaskPlanState,
    TaskScenario,
)


def test_task_space_pipeline_reaches_audited_manifest():
    allowed = {"input_structure": ["single-image"], "operation": ["describe"], "clinical_target": ["finding"]}
    registries = {
        "input_structure": ["single-image", "multi-image", "multi-view", "volume", "image-text", "paired-study"],
        "operation": ["recognize", "localize", "quantify", "compare", "align", "describe"],
        "clinical_target": ["acquisition", "anatomy", "finding", "attribute", "assessment", "diagnosis", "management", "quality"],
    }
    def assessments(axis):
        return [AxisValueDecision(value=value, supported=value in allowed[axis], evidence_basis=[EvidenceBasis(reference="c", detail="observed")], reason="observed") for value in registries[axis]]
    analysis = AxisAnalysis(input_structure=assessments("input_structure"), operation=assessments("operation"), clinical_target=assessments("clinical_target"), dataset_revision="1")
    leaf = {"leaf_task_id": "l", "name": "report", "scenario_id": "node-87e180f32ee34e5d", "clinical_target_id": "finding", "question_templates": ["Describe."],
            "input": [{"role": "image", "name": "image", "evidence_type": "asset"}], "target": "finding", "reference_rule": "Read report",
            "output_spec": "text", "task_paradigm": "Report Generation", "capability_category": "Generation",
            "resolver": {"inputs": [{"role": "image", "name": "image", "evidence_type": "asset"}], "steps": [{"op": "read", "selector": {"role": "image", "name": "image", "evidence_type": "asset"}}, {"op": "reference_text", "selector": {"role": "reference", "name": "report", "evidence_type": "text"}}], "output_spec": "text", "result_step": "step_1"}}
    case = {"case_id": "c", "evidence": [{"id": "a", "type": "asset", "name": "image", "value": "asset", "ref": []}, {"id": "r", "type": "text", "name": "report", "value": "Clear", "ref": ["a"]}]}
    result = run_pipeline(dataset={"dataset_id": "d"}, axis_analysis=analysis, node_context={"assets": {"primary_image_assets": 1}, "evidence": {"image": {}, "report": {}, "text": 1}, "relations": {"related": 1}, "validated_clinical_targets": {"finding": [{"target_id": "finding", "name": "finding", "target_type": "finding", "subject": "case", "scope": "case", "reference_kind": "direct", "reference_fields": ["report"]}]}}, leaves=[leaf], cases=[case], requested_count=1)
    assert result["audit"]["accepted"] == 1
    assert result["manifest"]["schema_version"] == "benchmark-manifest"


def test_task_space_pipeline_uses_leaves_persisted_in_plan_when_omitted():
    leaf = {
        "leaf_task_id": "persisted-leaf", "name": "report", "scenario_id": "scenario-1", "clinical_target_id": "finding",
        "question_templates": ["Describe."],
        "input": [{"role": "image", "name": "image", "evidence_type": "asset"}],
        "target": "finding", "reference_rule": "Read report", "output_spec": "text",
        "task_paradigm": "Report Generation", "capability_category": "Generation",
        "resolver": {
            "inputs": [{"role": "image", "name": "image", "evidence_type": "asset"}],
            "steps": [
                {"op": "read", "selector": {"role": "image", "name": "image", "evidence_type": "asset"}},
                {"op": "reference_text", "selector": {"role": "reference", "name": "report", "evidence_type": "text"}},
            ],
            "output_spec": "text", "result_step": "step_1",
        },
    }
    scenario = TaskScenario(
        scenario_id="scenario-1", input_structure="single-image", operation="describe",
        clinical_target="finding", capability_category="Generation",
        validated_clinical_targets=[{"target_id": "finding", "name": "finding", "target_type": "finding", "subject": "case", "scope": "case", "reference_kind": "direct", "reference_fields": ["report"]}],
        leaf_task_ids=["persisted-leaf"], status="completed",
    )
    plan = TaskPlanState(
        dataset={"dataset_id": "d"}, scenarios={"scenario-1": scenario},
        leaf_tasks={"persisted-leaf": leaf}, phase="complete",
    )
    case = {"case_id": "c", "evidence": [
        {"id": "a", "type": "asset", "name": "image", "value": "asset", "ref": []},
        {"id": "r", "type": "text", "name": "report", "value": "Clear", "ref": ["a"]},
    ]}

    result = run_pipeline(dataset={"dataset_id": "d"}, plan=plan, cases=[case], requested_count=1)

    assert len(result["items"]) == 1
    assert result["items"][0]["task_id"] == "persisted-leaf"


def test_explicit_leaves_override_plan_leaves():
    analysis = AxisAnalysis(
        input_structure=[AxisValueDecision(value="single-image", supported=True, evidence_basis=[EvidenceBasis(reference="c", detail="observed")], reason="observed")],
        operation=[AxisValueDecision(value="describe", supported=True, evidence_basis=[EvidenceBasis(reference="c", detail="observed")], reason="observed")],
        clinical_target=[AxisValueDecision(value="finding", supported=True, evidence_basis=[EvidenceBasis(reference="c", detail="observed")], reason="observed")],
        dataset_revision="1",
    )
    leaf = {
        "leaf_task_id": "explicit", "name": "report", "scenario_id": "node-87e180f32ee34e5d", "clinical_target_id": "finding",
        "question_templates": ["Describe."],
        "input": [{"role": "image", "name": "image", "evidence_type": "asset"}],
        "target": "finding", "reference_rule": "Read report", "output_spec": "text",
        "task_paradigm": "Report Generation", "capability_category": "Generation",
        "resolver": {
            "inputs": [{"role": "image", "name": "image", "evidence_type": "asset"}],
            "steps": [
                {"op": "read", "selector": {"role": "image", "name": "image", "evidence_type": "asset"}},
                {"op": "reference_text", "selector": {"role": "reference", "name": "report", "evidence_type": "text"}},
            ],
            "output_spec": "text", "result_step": "step_1",
        },
    }
    case = {"case_id": "c", "evidence": [
        {"id": "a", "type": "asset", "name": "image", "value": "asset", "ref": []},
        {"id": "r", "type": "text", "name": "report", "value": "Clear", "ref": ["a"]},
    ]}

    result = run_pipeline(
        dataset={"dataset_id": "d"}, axis_analysis=analysis,
        node_context={"assets": {"primary_image_assets": 1}, "evidence": {"image": {}, "report": {}, "text": 1}, "relations": {"related": 1}, "validated_clinical_targets": {"finding": [{"target_id": "finding", "name": "finding", "target_type": "finding", "subject": "case", "scope": "case", "reference_kind": "direct", "reference_fields": ["report"]}]}},
        leaves=[leaf], cases=[case], requested_count=1,
    )

    assert set(result["leaves"]) == {"explicit"}

