import json
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
    axis_path = Path(context["operation_directory"]) / "axis_decisions.json"
    assert set(json.loads(axis_path.read_text(encoding="utf-8"))) == {
        "input_structure", "operation", "clinical_target"
    }
    scenarios = json.loads((Path(context["operation_directory"]) / "task_scenarios.json").read_text(encoding="utf-8"))
    assert scenarios
    assert all("annotation_mapping" not in value and "evidence_stats" not in value for value in scenarios.values())


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
    assert payload["axis_definition"]["id"] == "input_structure"
    assert isinstance(payload["allowed_value_definitions"], list)
    assert payload["allowed_values"] == [item["id"] for item in payload["allowed_value_definitions"]]


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
    plan.current_scenario_id = node_id
    task_planning._write_derived_artifacts(context, plan)
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
    assert reviewed["updates"]["milestone"]["kind"] == "scenario_complete"
    assert reviewed["updates"]["planning_incomplete"] is False
    assigned = task_planning.classify_task_nodes(context, assignments=[{"node_id": node_id, "capability_dimension_ids": ["report_generation"]}])
    context.update(assigned["updates"])
    monkeypatch.setattr(task_planning, "require", lambda *_args: None)
    finished = task_planning.finish_task_space(context)
    assert finished["status"] == "complete"
    assert finished["updates"]["planning_incomplete"] is False
    assert finished["updates"]["planning_status"] == "complete"


def test_scenario_review_can_discard_leaf_atomically(tmp_path, monkeypatch):
    context = _context(tmp_path)
    context["question_template_count"] = 3
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
    scenario_id = plan.pending_node_ids[0]
    plan.current_scenario_id = scenario_id
    task_planning._write_derived_artifacts(context, plan)

    def leaf(leaf_id, target):
        return {
            "leaf_task_id": leaf_id,
            "name": target,
            "scenario_id": scenario_id,
            "question_templates": [
                f"Describe {target}.",
                f"What describes {target}?",
                f"Identify {target}.",
            ],
            "input": [{"role": "image", "name": "image", "evidence_type": "asset"}],
            "target": target,
            "reference_rule": "paired report",
            "resolver": {
                "inputs": [{"role": "image", "name": "image", "evidence_type": "asset"}],
                "steps": [
                    {"op": "read", "selector": {"role": "image", "name": "image", "evidence_type": "asset"}},
                    {"op": "reference_text", "selector": {"role": "reference", "name": "report", "evidence_type": "text"}},
                ],
                "output_spec": "text",
                "result_step": "step_1",
            },
            "output_spec": "text",
            "task_paradigm": "Report Generation",
            "capability_category": "Generation",
        }

    for value in (leaf("good", "finding"), leaf("bad", "context")):
        saved = task_planning.save_leaf_task(context, leaf=value)
        context.update(saved["updates"])

    reviewed = task_planning.review_current_scenario(
        context,
        review={
            "scenario_id": scenario_id,
            "checked_leaf_task_ids": ["good", "bad"],
            "discarded_leaf_tasks": [{"leaf_task_id": "bad", "reason": "Not suitable for this visual task."}],
            "status": "approved",
        },
    )
    context.update(reviewed["updates"])
    plan = task_planning._load_plan(context)

    assert set(plan.leaf_tasks) == {"good"}
    assert plan.scenarios[scenario_id].leaf_task_ids == ["good"]
    assert plan.reviews[scenario_id].checked_leaf_task_ids == ["good"]
    assert plan.reviews[scenario_id].discarded_leaf_tasks[0].leaf_task_id == "bad"


def test_derived_plan_views_must_match_canonical_plan(tmp_path):
    context = _context(tmp_path)
    for axis, values in (("input_structure", ["single-image"]), ("operation", ["describe"]), ("clinical_target", ["finding"])):
        task_planning.analyze_axis(context, axis=axis, assessments=_assessments(axis, values))
    plan = task_planning._load_plan(context)
    task_planning._write_derived_artifacts(context, plan)
    task_planning._assert_derived_artifact_consistency(context, plan)

    path = Path(context["operation_directory"]) / "task_scenarios.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenario_id = next(iter(payload))
    payload[scenario_id]["status"] = "completed"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="task_scenarios.json is inconsistent"):
        task_planning._assert_derived_artifact_consistency(context, plan)
