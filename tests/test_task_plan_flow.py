from pathlib import Path

import pytest

from medbenchagent.agents import task_planning
from medbenchagent.planning.task_space import AXIS_VALUES, TaskAxis, TaskSpacePlan


def _assessments(axis, supported):
    return [
        {
            "value": value,
            "supported": value in supported,
            "evidence_basis": [{"reference": "profile:report", "detail": "observed"}],
            "reason": "observed",
        }
        for value in AXIS_VALUES[TaskAxis(axis)]
    ]


def _context(tmp_path):
    facts = {
        "schema_version": "dataset-facts",
        "revision": "7",
        "case_count": 1,
        "evidence_names": ["report"],
        "evidence": {"text": 1},
        "assets": {"image_assets": 1, "multi_image_cases": 0, "related_image_cases": 0, "volume_assets": 0},
        "relations": {"paired_text": 1, "paired": 1, "paired_study": 0, "related": 1},
    }
    plan = TaskSpacePlan(
        dataset={"dataset_id": "d"}, dataset_facts=facts,
        blueprint={"name": "Blueprint", "capability_dimensions": [{"id": "report_generation"}], "task_concepts": []},
        phase="axes",
    )
    path = Path(tmp_path) / "task_space_plan.json"
    path.write_text(plan.model_dump_json(), encoding="utf-8")
    return {
        "operation_directory": str(tmp_path),
        "task_space_plan_path": str(path),
        "dataset_access_manifest": {"schema_version": "annotation.v5", "revision": "7"},
    }


def test_axis_calls_are_strictly_ordered(tmp_path):
    context = _context(tmp_path)
    with pytest.raises(ValueError, match="Expected Axis"):
        task_planning.analyze_axis(context, axis="operation", assessments=_assessments("operation", ["describe"]))
    task_planning.analyze_axis(context, axis="input_structure", assessments=_assessments("input_structure", ["single-image"]))
    task_planning.analyze_axis(context, axis="operation", assessments=_assessments("operation", ["describe"]))
    result = task_planning.analyze_axis(context, axis="clinical_target", assessments=_assessments("clinical_target", ["finding"]))
    assert result["updates"]["task_space_plan"]["phase"] == "leaves"
    assert result["updates"]["task_space_plan"]["pending_node_ids"]


def test_axis_evidence_exposes_catalog_counts_and_semantics(tmp_path):
    context = _context(tmp_path)
    plan = task_planning._load_plan(context)
    plan.dataset_facts["evidence_catalog"] = {
        "report": {"name": "report", "cases": 1, "types": {"text": 1}, "units": {}}
    }
    task_planning._write_derived_artifacts(context, plan)
    result = task_planning.read_axis_evidence(context, axis="input_structure")
    payload = __import__("json").loads(result["summary"])
    assert payload["evidence_catalog"][0]["cases"] == 1
    assert payload["evidence_catalog"][0]["types"] == {"text": 1}
    assert payload["catalog_semantics"]["cases"]


def test_node_review_and_capability_assignment_gate_completion(tmp_path, monkeypatch):
    context = _context(tmp_path)
    for axis, values in (("input_structure", ["single-image"]), ("operation", ["describe"]), ("clinical_target", ["finding"])):
        task_planning.analyze_axis(context, axis=axis, assessments=_assessments(axis, values))
        context.update(task_planning._write_derived_artifacts(context, task_planning._load_plan(context)))

    class Query:
        def __init__(self, _access):
            pass

        def iter_cases(self):
            return iter([{"case_id": "c1", "evidence": [
                {"id": "asset", "type": "asset", "name": "image", "value": "a1", "ref": []},
                {"id": "report", "type": "text", "name": "report", "value": "Clear", "ref": ["asset"]},
            ]}])

    monkeypatch.setattr("medbenchagent.data.query.DatasetQuery", Query)
    plan = task_planning._load_plan(context)
    node_id = plan.pending_node_ids[0]
    context.update(task_planning.read_scenario_context(context, scenario_id=node_id)["updates"])
    leaf = {
        "leaf_id": "leaf-1", "name": "report", "node_id": node_id,
        "question_template": "Describe the finding.",
        "input": [{"role": "image", "name": "image", "evidence_type": "asset"}],
        "target": "finding", "reference_rule": "paired report",
        "resolver": {"inputs": [{"role": "image", "name": "image", "evidence_type": "asset"}],
                     "steps": [{"op": "read", "selector": {"role": "image", "name": "image", "evidence_type": "asset"}},
                               {"op": "reference_text", "selector": {"role": "reference", "name": "report", "evidence_type": "text"}}],
                     "output_spec": "text", "result_step": "step_1"},
        "output_spec": "text", "task_paradigm": "Report Generation", "capability_category": "Generation",
    }
    saved = task_planning.save_leaf_task(context, leaf={
        "leaf_task_id": leaf["leaf_id"], "name": leaf["name"],
        "scenario_id": leaf["node_id"], "question_template": leaf["question_template"],
        "input": leaf["input"], "target": leaf["target"],
        "reference_rule": leaf["reference_rule"], "resolver": leaf["resolver"],
        "output_spec": leaf["output_spec"], "task_paradigm": leaf["task_paradigm"],
        "capability_category": leaf["capability_category"],
    })
    context.update(saved["updates"])
    with pytest.raises(ValueError, match="checked_leaf_ids|cover every"):
        task_planning.review_current_node(context, review={"node_id": node_id, "checked_leaf_ids": [], "status": "approved"})
    reviewed = task_planning.review_current_scenario(context, review={"scenario_id": node_id, "checked_leaf_task_ids": ["leaf-1"], "status": "approved"})
    context.update(reviewed["updates"])
    assigned = task_planning.classify_task_nodes(context, assignments=[{"node_id": node_id, "capability_dimension_ids": ["report_generation"]}])
    context.update(assigned["updates"])
    monkeypatch.setattr(task_planning, "require", lambda *_args: None)
    finished = task_planning.finish_task_space(context)
    assert finished["status"] == "complete"
