from image_fixtures import image_context
import json
from pathlib import Path

import pytest

from medbenchagent.agents import task_planning
from medbenchagent.planning.bir import RegistrySnapshot
from medbenchagent.planning.task_space import AXIS_VALUES, CapabilityCategory, EvidenceBasis, LeafTask, TaskAxis, TaskPlanState, TaskScenario


ROOT = Path(__file__).resolve().parents[1]


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
        "evidence_names": ["image", "report"],
        "evidence": {"text": 1},
        "assets": {"image_assets": 1, "multi_image_cases": 0, "related_image_cases": 0, "volume_assets": 0},
        "relations": {"paired_text": 1, "paired": 1, "paired_study": 0, "related": 1},
    }
    plan = TaskPlanState(
        dataset={"dataset_id": "d"},
        blueprint={"name": "Blueprint", "capability_dimensions": [{"id": "report_generation"}], "task_concepts": []},
        phase="axes",
    )
    path = Path(tmp_path) / "task_plan.json"
    path.write_text(plan.model_dump_json(), encoding="utf-8")
    return {
        "operation_directory": str(tmp_path),
        "task_plan_path": str(path),
        "dataset_facts": facts,
        "dataset_access_manifest": {"schema_version": "annotation", "revision": "7"},
    }


def test_axis_calls_are_strictly_ordered(tmp_path):
    context = _context(tmp_path)
    with pytest.raises(ValueError, match="Expected Axis"):
        task_planning.analyze_axis(context, axis="operation", assessments=_assessments("operation", ["describe"]))
    task_planning.analyze_axis(context, axis="input_structure", assessments=_assessments("input_structure", ["single-image"]))
    task_planning.analyze_axis(context, axis="operation", assessments=_assessments("operation", ["describe"]))
    result = task_planning.analyze_axis(context, axis="clinical_target", assessments=_assessments("clinical_target", ["finding"]))
    assert result["updates"]["task_plan"]["phase"] == "scenario_review"
    assert result["updates"]["task_plan"]["scenario_review_pending_ids"]
    assert set(task_planning._load_plan(context).axis_decisions) == {
        "input_structure", "operation", "clinical_target"
    }
    scenarios = {key: value.model_dump(mode="json") for key, value in task_planning._load_plan(context).scenarios.items()}
    assert scenarios
    assert not (Path(context["operation_directory"]) / "task_scenarios.json").exists()


def test_axis_evidence_exposes_catalog_counts_and_semantics(tmp_path):
    context = _context(tmp_path)
    plan = task_planning._load_plan(context)
    context["dataset_facts"]["evidence_catalog"] = {
        "report": {"name": "report", "cases": 1, "types": {"text": 1}, "units": {}}
    }
    task_planning._write_artifacts(context, plan)
    result = task_planning.read_axis_evidence(context, axis="input_structure")
    payload = __import__("json").loads(result["summary"])
    assert payload["evidence_catalog"][0]["cases"] == 1
    assert payload["evidence_catalog"][0]["types"] == {"text": 1}
    assert payload["catalog_semantics"]["cases"]
    assert payload["axis_definition"]["id"] == "input_structure"
    assert isinstance(payload["allowed_value_definitions"], list)
    assert payload["allowed_values"] == [item["id"] for item in payload["allowed_value_definitions"]]


def test_candidate_scenario_review_requires_pending_queue_head(tmp_path):
    context = _context(tmp_path)
    for axis, values in (
        ("input_structure", ["single-image"]),
        ("operation", ["describe"]),
        ("clinical_target", ["finding"]),
    ):
        task_planning.analyze_axis(context, axis=axis, assessments=_assessments(axis, values))
    plan = task_planning._load_plan(context)
    pending = plan.scenario_review_pending_ids

    with pytest.raises(ValueError, match="queue head"):
        task_planning.review_candidate_scenarios(context, scenario_id="scenario-wrong")

    assert task_planning._load_plan(context).scenario_review_pending_ids == pending


def test_candidate_scenario_review_advances_the_explicit_scenario(tmp_path, monkeypatch):
    context = _context(tmp_path)
    for axis, values in (
        ("input_structure", ["single-image"]),
        ("operation", ["recognize", "describe"]),
        ("clinical_target", ["finding"]),
    ):
        task_planning.analyze_axis(context, axis=axis, assessments=_assessments(axis, values))
    scenario_id = task_planning._load_plan(context).scenario_review_pending_ids[0]
    captured = {}

    def fake_complete_json(prompt, user_prompt, schema, **kwargs):
        captured["payload"] = json.loads(user_prompt)
        return {"decision": "keep", "reason": "Suitable for review.", "validated_clinical_targets": [{
            "target_id": "finding_target", "name": "finding", "target_type": "finding",
            "subject": "case", "scope": "case", "reference_kind": "direct", "reference_fields": ["report"],
        }]}

    monkeypatch.setattr(task_planning, "complete_json", fake_complete_json)
    reviewed = task_planning.review_candidate_scenarios(context, scenario_id=scenario_id)
    plan = task_planning._load_plan(context)

    assert captured["payload"]["scenario"]["scenario_id"] == scenario_id
    assert "count_semantics" in captured["payload"]
    assert scenario_id not in plan.scenario_review_pending_ids
    assert plan.pending_scenario_ids
    if plan.phase == "scenario_review":
        assert plan.pending_scenario_ids == plan.scenario_review_pending_ids
    assert task_planning._planning_progress_key(context)["pending_scenario_count"] == len(plan.pending_scenario_ids)
    assert reviewed["summary"] == f"Scenario Review {scenario_id}: keep"
    assert plan.scenarios[scenario_id].screening_decision == "keep"
    assert plan.scenarios[scenario_id].validated_clinical_targets


def test_review_progress_uses_review_queue_when_legacy_pending_list_is_empty(tmp_path):
    context = _context(tmp_path)
    for axis, values in (
        ("input_structure", ["single-image"]),
        ("operation", ["recognize", "describe"]),
        ("clinical_target", ["finding"]),
    ):
        task_planning.analyze_axis(context, axis=axis, assessments=_assessments(axis, values))
    plan = task_planning._load_plan(context)
    plan.pending_scenario_ids = []
    task_planning._write_artifacts(context, plan)

    progress = task_planning._planning_progress_key(context)
    assert progress["pending_scenario_count"] == len(plan.scenario_review_pending_ids)
    assert progress["pending_scenario_count"] > 0


def test_node_review_and_capability_assignment_gate_completion(tmp_path, monkeypatch):
    context = _context(tmp_path)
    for axis, values in (("input_structure", ["single-image"]), ("operation", ["describe"]), ("clinical_target", ["finding"])):
        task_planning.analyze_axis(context, axis=axis, assessments=_assessments(axis, values))
        context.update(task_planning._write_artifacts(context, task_planning._load_plan(context)))

    class Query:
        def __init__(self, _access):
            pass

        def iter_cases(self):
            return iter([{"case_id": "c1", "evidence": [
                {"id": "asset", "type": "asset", "name": "image", "value": "a1", "ref": []},
                {"id": "report", "type": "text", "name": "report", "value": "Clear", "ref": ["asset"]},
            ]}])

    monkeypatch.setattr("medbenchagent.data.index.DatasetIndex", Query)
    image_inputs = image_context(tmp_path / "images", list(Query(None).iter_cases()))
    monkeypatch.setattr("medbenchagent.planning.resolver.ResolverContext.from_access", lambda access: image_inputs)
    plan = task_planning._load_plan(context)
    node_id = plan.scenario_review_pending_ids[0]
    screened = task_planning.screen_scenario(
        context, scenario_id=node_id, decision="keep", reason="The scenario is semantically suitable.",
            validated_clinical_targets=[
                {"target_id": "finding_target", "name": "finding", "target_type": "finding", "subject": "case", "scope": "case", "reference_kind": "direct", "reference_fields": ["report"]},
                {"target_id": "context_target", "name": "context", "target_type": "finding", "subject": "case", "scope": "case", "reference_kind": "direct", "reference_fields": ["report"]},
            ]
    )
    context.update(screened["updates"])
    plan = task_planning._load_plan(context)
    node_id = plan.pending_scenario_ids[0]
    plan.current_scenario_id = node_id
    task_planning._write_artifacts(context, plan)
    leaf = {
        "leaf_task_id": "leaf-1", "name": "report", "scenario_id": node_id,
        "question_templates": ["Describe the finding."],
        "input": [{"role": "image", "name": "image", "evidence_type": "asset"}],
        "clinical_target_id": "finding_target", "target": "finding", "reference_rule": "paired report",
        "resolver": {"inputs": [{"role": "image", "name": "image", "evidence_type": "asset"}],
                     "steps": [{"op": "read", "selector": {"role": "image", "name": "image", "evidence_type": "asset"}},
                               {"op": "reference_text", "selector": {"role": "reference", "name": "report", "evidence_type": "text"}}],
                     "output_spec": "text", "result_step": "step_1"},
        "output_spec": "text", "task_paradigm": "Report Generation", "capability_category": "Generation",
    }
    saved = task_planning.save_leaf_task(context, leaf={
        "leaf_task_id": leaf["leaf_task_id"], "name": leaf["name"],
        "scenario_id": leaf["scenario_id"], "question_templates": leaf["question_templates"],
            "input": leaf["input"], "clinical_target_id": leaf["clinical_target_id"], "target": leaf["target"],
        "reference_rule": leaf["reference_rule"], "resolver": leaf["resolver"],
        "output_spec": leaf["output_spec"], "task_paradigm": leaf["task_paradigm"],
        "capability_category": leaf["capability_category"],
    })
    context.update(saved["updates"])
    with pytest.raises(ValueError, match="checked_leaf_ids|cover every"):
        task_planning.review_current_scenario(context, review={"scenario_id": node_id, "checked_leaf_task_ids": [], "status": "approved"})
    reviewed = task_planning.review_current_scenario(context, review={"scenario_id": node_id, "checked_leaf_task_ids": ["leaf-1"], "status": "approved"})
    context.update(reviewed["updates"])
    assert reviewed["updates"]["milestone"]["kind"] == "scenario_complete"
    assert reviewed["updates"]["planning_incomplete"] is False
    monkeypatch.setattr(task_planning, "require", lambda *_args: None)
    finished = task_planning.finish_task_plan(context)
    assert finished["status"] == "complete"
    assert finished["updates"]["planning_incomplete"] is False
    assert finished["updates"]["planning_status"] == "complete"


def test_scenario_review_can_discard_leaf_atomically(tmp_path, monkeypatch):
    context = _context(tmp_path)
    context["question_template_count"] = 3
    for axis, values in (("input_structure", ["single-image"]), ("operation", ["describe"]), ("clinical_target", ["finding"])):
        task_planning.analyze_axis(context, axis=axis, assessments=_assessments(axis, values))
        context.update(task_planning._write_artifacts(context, task_planning._load_plan(context)))

    class Query:
        def __init__(self, _access):
            pass

        def iter_cases(self):
            return iter([{"case_id": "c1", "evidence": [
                {"id": "asset", "type": "asset", "name": "image", "value": "a1", "ref": []},
                {"id": "report", "type": "text", "name": "report", "value": "Clear", "ref": ["asset"]},
            ]}])

    monkeypatch.setattr("medbenchagent.data.index.DatasetIndex", Query)
    image_inputs = image_context(tmp_path / "images", list(Query(None).iter_cases()))
    monkeypatch.setattr("medbenchagent.planning.resolver.ResolverContext.from_access", lambda access: image_inputs)
    plan = task_planning._load_plan(context)
    scenario_id = plan.scenario_review_pending_ids[0]
    screened = task_planning.screen_scenario(
        context, scenario_id=scenario_id, decision="keep", reason="The scenario is semantically suitable.",
        validated_clinical_targets=[
            {"target_id": "finding_target", "name": "finding", "target_type": "finding", "subject": "case", "scope": "case", "reference_kind": "direct", "reference_fields": ["report"]},
            {"target_id": "context_target", "name": "context", "target_type": "finding", "subject": "case", "scope": "case", "reference_kind": "direct", "reference_fields": ["report"]},
        ]
    )
    context.update(screened["updates"])
    plan = task_planning._load_plan(context)
    plan.current_scenario_id = scenario_id
    task_planning._write_artifacts(context, plan)

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
                    "clinical_target_id": "finding_target" if target == "finding" else "context_target",
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


def test_planning_does_not_write_projection_files(tmp_path):
    context = _context(tmp_path)
    plan = task_planning._load_plan(context)
    task_planning._write_artifacts(context, plan)
    files = {p.name for p in Path(context["operation_directory"]).glob("*.json")}
    assert files == {"task_plan.json"}


def test_default_scenario_review_receives_brief_and_complete_leaf_contract(tmp_path, monkeypatch):
    context = _context(tmp_path)
    context["benchmark_info"] = {
        "clinical_scope": "breast imaging",
        "language": "en",
        "evaluation_focus": "recognition",
        "diseases": [],
        "modalities": ["ultrasound"],
        "intent_notes": [],
    }
    for axis, values in (
        ("input_structure", ["single-image"]),
        ("operation", ["describe"]),
        ("clinical_target", ["finding"]),
    ):
        task_planning.analyze_axis(context, axis=axis, assessments=_assessments(axis, values))
    plan = task_planning._load_plan(context)
    scenario_id = plan.scenario_review_pending_ids[0]
    plan.phase = "leaves"
    plan.current_scenario_id = scenario_id
    plan.pending_scenario_ids = [scenario_id]
    plan.scenarios[scenario_id] = plan.scenarios[scenario_id].model_copy(update={
        "validated_clinical_targets": [{"target_id": "finding_target", "name": "finding", "target_type": "finding", "subject": "case", "scope": "case", "reference_kind": "direct", "reference_fields": ["report"]}],
    })
    leaf = LeafTask.model_validate({
        "leaf_task_id": "leaf-review",
        "name": "finding-report",
        "scenario_id": scenario_id,
        "clinical_target_id": "finding_target",
        "question_templates": ["Describe the finding."],
        "input": [{"role": "image", "name": "image", "evidence_type": "asset"}],
        "target": "finding",
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
        "evidence": [{"reference": "profile:report", "detail": "paired report"}],
        "eligible_count": 1,
    })
    plan.leaf_tasks[leaf.leaf_task_id] = leaf
    plan.scenarios[scenario_id] = plan.scenarios[scenario_id].model_copy(
        update={"leaf_task_ids": [leaf.leaf_task_id], "status": "review_pending"}
    )
    task_planning._write_artifacts(context, plan)

    registry = RegistrySnapshot.from_directory(ROOT / "config" / "bir_template" / "registry")
    monkeypatch.setattr(task_planning, "_registry", lambda _context: registry)
    captured = {}

    def fake_complete_json(prompt, user_prompt, schema):
        captured["prompt"] = prompt
        captured["payload"] = json.loads(user_prompt)
        return {
            "scenario_id": scenario_id,
            "checked_leaf_task_ids": [leaf.leaf_task_id],
            "discarded_leaf_tasks": [],
            "status": "approved",
            "issues": [],
        }

    monkeypatch.setattr(task_planning, "complete_json", fake_complete_json)
    result = task_planning.review_current_scenario(context)

    assert result["status"] == "success"
    assert "{{PLANNING_RULES}}" not in captured["prompt"]
    assert "accepted target" in captured["prompt"]
    assert captured["payload"]["benchmark_brief"]["clinical_scope"] == "breast imaging"
    reviewed_leaf = captured["payload"]["leaf_tasks"][0]
    for field in ("input", "resolver", "reference_rule", "evidence", "question_templates", "options", "output_spec", "task_paradigm"):
        assert field in reviewed_leaf
    assert "validation" in reviewed_leaf


def test_leaf_scenario_can_be_excluded_without_leaf_and_all_excluded_can_finish(tmp_path, monkeypatch):
    context = _context(tmp_path)
    for axis, values in (
        ("input_structure", ["single-image"]),
        ("operation", ["describe"]),
        ("clinical_target", ["finding"]),
    ):
        task_planning.analyze_axis(context, axis=axis, assessments=_assessments(axis, values))
    plan = task_planning._load_plan(context)
    first_id = plan.scenario_review_pending_ids[0]
    second = TaskScenario(
        scenario_id="scenario-next",
        input_structure="single-image",
        operation="recognize",
        clinical_target="finding",
        capability_category=CapabilityCategory.RECOGNITION,
        status="active",
    )
    plan.phase = "leaves"
    plan.current_scenario_id = first_id
    plan.pending_scenario_ids = [first_id, second.scenario_id]
    plan.scenarios[second.scenario_id] = second
    task_planning._write_artifacts(context, plan)
    context["planning_read_state"] = {
        "binding": {"dataset_id": "d", "revision": "7", "scenario_id": first_id},
        "scenario_id": first_id,
        "scenario_context_read": True,
        "catalog_queries": [],
        "catalog_cache": {},
    }

    excluded = task_planning.exclude_current_scenario(
        context,
        scenario_id=first_id,
        reason="No supported Leaf direction remains after reading the Scenario evidence.",
        evidence_basis=[EvidenceBasis(reference="catalog:report", detail="The required scope is unavailable.")],
    )
    context.update(excluded["updates"])
    plan = task_planning._load_plan(context)
    assert plan.scenarios[first_id].status == "excluded"
    assert plan.current_scenario_id == second.scenario_id
    assert task_planning._load_plan(context).scenarios[first_id].exclusion_stage == "leaf_review"
    context["planning_read_state"]["scenario_context_read"] = True

    excluded = task_planning.exclude_current_scenario(
        context,
        scenario_id=second.scenario_id,
        reason="The remaining Scenario has no supported reference route.",
        evidence_basis=[EvidenceBasis(reference="catalog:report", detail="No deterministic reference.")],
    )
    context.update(excluded["updates"])
    monkeypatch.setattr(task_planning, "require", lambda *_args: None)
    finished = task_planning.finish_task_plan(context)
    assert finished["status"] == "complete"
    assert finished["updates"]["executable_leaf_count"] == 0
    assert "no executable tasks" in finished["summary"]


def test_review_discarding_all_leafs_allows_explicit_scenario_exclusion(tmp_path):
    context = _context(tmp_path)
    for axis, values in (
        ("input_structure", ["single-image"]),
        ("operation", ["describe"]),
        ("clinical_target", ["finding"]),
    ):
        task_planning.analyze_axis(context, axis=axis, assessments=_assessments(axis, values))
    plan = task_planning._load_plan(context)
    scenario_id = plan.scenario_review_pending_ids[0]
    plan.phase = "leaves"
    plan.current_scenario_id = scenario_id
    plan.pending_scenario_ids = [scenario_id]
    plan.scenarios[scenario_id] = plan.scenarios[scenario_id].model_copy(update={
        "validated_clinical_targets": [{"target_id": "finding_target", "name": "finding", "target_type": "finding", "subject": "case", "scope": "case", "reference_kind": "direct", "reference_fields": ["report"]}],
    })
    leaf = LeafTask.model_validate({
            "leaf_task_id": "discard-me", "name": "report", "scenario_id": scenario_id, "clinical_target_id": "finding_target",
        "question_templates": ["Describe the image."],
        "input": [{"role": "image", "name": "image", "evidence_type": "asset"}],
        "target": "finding", "reference_rule": "Use paired report.",
        "resolver": {
            "inputs": [{"role": "image", "name": "image", "evidence_type": "asset"}],
            "steps": [
                {"op": "read", "selector": {"role": "image", "name": "image", "evidence_type": "asset"}},
                {"op": "reference_text", "selector": {"role": "reference", "name": "report", "evidence_type": "text"}},
            ],
            "output_spec": "text", "result_step": "step_1",
        },
        "output_spec": "text", "task_paradigm": "Report Generation", "capability_category": "Generation",
        "eligible_count": 1,
    })
    plan.leaf_tasks[leaf.leaf_task_id] = leaf
    plan.scenarios[scenario_id] = plan.scenarios[scenario_id].model_copy(
        update={"leaf_task_ids": [leaf.leaf_task_id], "status": "review_pending"}
    )
    task_planning._write_artifacts(context, plan)
    reviewed = task_planning.review_current_scenario(context, review={
        "scenario_id": scenario_id,
        "checked_leaf_task_ids": [leaf.leaf_task_id],
        "discarded_leaf_tasks": [{"leaf_task_id": leaf.leaf_task_id, "reason": "No valid wording."}],
        "status": "rejected",
        "issues": ["The only candidate is invalid."],
    })
    context.update(reviewed["updates"])
    context["planning_read_state"]["scenario_context_read"] = True
    excluded = task_planning.exclude_current_scenario(
        context,
        scenario_id=scenario_id,
        reason="Review discarded every saved Leaf.",
        evidence_basis=[EvidenceBasis(reference="review:discard-me", detail="The only Leaf was rejected.")],
    )
    assert excluded["status"] == "success"
    assert task_planning._load_plan(context).scenarios[scenario_id].status == "excluded"

