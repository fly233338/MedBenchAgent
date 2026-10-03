from medbenchagent.planning.pipeline import run_pipeline
from medbenchagent.planning.task_space import AxisAnalysis, AxisAssessment, EvidenceBasis


def test_task_space_pipeline_reaches_audited_manifest():
    allowed = {"input_structure": ["single-image"], "operation": ["describe"], "clinical_target": ["finding"]}
    registries = {
        "input_structure": ["single-image", "multi-image", "multi-view", "volume", "image-text", "paired-study"],
        "operation": ["recognize", "localize", "quantify", "compare", "align", "describe"],
        "clinical_target": ["acquisition", "anatomy", "finding", "attribute", "measurement", "assessment", "diagnosis", "quality"],
    }
    def assessments(axis):
        return [AxisAssessment(value=value, supported=value in allowed[axis], evidence_basis=[EvidenceBasis(reference="c", detail="observed")], reason="observed") for value in registries[axis]]
    analysis = AxisAnalysis(input_structure=assessments("input_structure"), operation=assessments("operation"), clinical_target=assessments("clinical_target"), dataset_revision="1")
    leaf = {"leaf_id": "l", "name": "report", "node_id": "node-87e180f32ee34e5d", "question": "Describe.",
            "input": [{"role": "image", "name": "image", "evidence_type": "asset"}], "target": "finding", "answer": "Read report",
            "output_spec": "text", "task_paradigm": "Report Generation", "capability_category": "Generation",
            "resolver": {"inputs": [{"role": "image", "name": "image", "evidence_type": "asset"}], "steps": [{"op": "read", "selector": {"role": "image", "name": "image", "evidence_type": "asset"}}, {"op": "reference_text", "selector": {"role": "reference", "name": "report", "evidence_type": "text"}}], "output_spec": "text", "result_step": "step_1"}}
    case = {"case_id": "c", "evidence": [{"id": "a", "type": "asset", "name": "image", "value": "asset", "ref": []}, {"id": "r", "type": "text", "name": "report", "value": "Clear", "ref": ["a"]}]}
    result = run_pipeline(dataset={"dataset_id": "d"}, axis_analysis=analysis, node_context={"evidence": {"image": {}, "report": {}, "text": 1}, "relations": {"related": 1}}, leaves=[leaf], cases=[case], requested_count=1)
    assert result["audit"]["accepted"] == 1
    assert result["manifest"]["schema_version"] == "benchmark-manifest"

