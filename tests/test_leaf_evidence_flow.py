from compilation_fixtures import planning_context, write_plan
import json
from pathlib import Path

import pytest

from medbenchagent.agents import task_planning
from medbenchagent.planning.task_space import CapabilityCategory, TaskPlanState, TaskScenario
from medbenchagent.workflow.graph import planning_draft_advanced


def _plan(tmp_path, *, phase="leaves", facts=None):
    scenario = TaskScenario(
        scenario_id="scenario-1",
        input_structure="single-image",
        operation="recognize",
        clinical_target="finding",
        capability_category=CapabilityCategory.RECOGNITION,
        validated_clinical_targets=[{"target_id": "finding", "name": "finding", "target_type": "finding", "subject": "case", "scope": "case", "reference_kind": "direct", "reference_fields": ["ultrasound_bmode_image"]}],
        status="active",
    )
    plan = TaskPlanState(
        dataset={"dataset_id": "dataset"},
        phase=phase,
        scenarios={scenario.scenario_id: scenario},
        current_scenario_id=scenario.scenario_id,
        pending_scenario_ids=[scenario.scenario_id],
    )
    (Path(tmp_path) / "fixture_facts.json").write_text(json.dumps(facts or {}), encoding="utf-8")
    records = [{"case_id": "c1", "evidence": [
        {"id": "image", "name": "ultrasound_bmode_image", "type": "asset", "value": "image.png", "ref": []},
        {"id": "label", "name": "classification", "type": "text", "value": "benign", "ref": ["image"]}]}]
    context = planning_context(tmp_path, plan=plan, cases=records)
    path = Path(context["bir_path"]) / "task_plan.json"
    return plan, path


def test_scenario_context_prioritizes_input_and_related_evidence(tmp_path, monkeypatch):
    facts = {
        "evidence_names": ["ultrasound_bmode_image", "classification", "age"],
        "evidence_catalog": {
            "ultrasound_bmode_image": {"name": "ultrasound_bmode_image", "cases": 2, "types": {"asset": 2}},
            "classification": {
                "name": "classification", "cases": 2, "types": {"text": 2},
                "meaning": "Pathology status", "observed_values": [{"value": "benign", "count": 2}],
            },
            "age": {"name": "age", "cases": 2, "types": {"value": 2}},
        },
        "assets": {"primary_image_assets": 2},
        "evidence": {"asset": 2, "text": 2, "value": 2},
    }

    class FakeIndex:
        def __init__(self, _access):
            pass

        def assets(self):
            return [{"asset_id": "asset-1", "kind": "image"}]

        def node_context(self, _scenario, *, sample_limit=3, registry=None):
            return {
                "evidence": [
                    {"name": "ultrasound_bmode_image", "cases": 2, "types": {"asset": 2}},
                    {"name": "classification", "cases": 2, "types": {"text": 2}},
                    {"name": "age", "cases": 2, "types": {"value": 2}},
                ],
                "capacity_hint": 2,
                "representative_cases": [
                    {"case_id": "c1", "evidence": [
                        {"id": "image", "type": "asset", "name": "ultrasound_bmode_image", "value": "asset-1", "ref": []},
                        {"id": "label", "type": "text", "name": "classification", "value": "benign", "ref": ["image"]},
                        {"id": "age", "type": "value", "name": "age", "value": 57, "ref": []},
                    ]},
                ],
            }

    import medbenchagent.data.index as index
    monkeypatch.setattr(index, "DatasetIndex", FakeIndex)
    plan, path = _plan(tmp_path, facts=facts)
    context = {**planning_context(tmp_path),
        "operation_directory": str(tmp_path),
        "task_plan_path": str(path),
        "dataset_facts": json.loads((tmp_path / "fixture_facts.json").read_text(encoding="utf-8")),
        "planning_read_state": {},
    }

    output = task_planning.read_scenario_context(context)
    payload = json.loads(output["summary"])

    assert payload["planning_evidence"][0]["name"] == "ultrasound_bmode_image"
    assert any(item.get("meaning") == "Pathology status" for item in payload["planning_evidence"])
    assert payload["representative_cases"][0]["evidence"]
    assert all(item["name"] != "age" for item in payload["representative_cases"][0]["evidence"])
    assert output["updates"]["planning_evidence_cache"]["scenario_id"] == "scenario-1"
    assert output["updates"]["leaf_context"]["evidence_catalog"]
    assert "coverage_and_limitations" in output["updates"]["leaf_context"]
    assert output["updates"]["leaf_context"]["annotation_semantics"]["status"] == "unavailable"


def test_catalog_returns_text_values_samples_and_suggestions(tmp_path, monkeypatch):
    catalog = {
        "schema_version": "evidence-catalog",
        "revision": "r1",
        "entries": [
            {"name": "classification", "cases": 2, "types": {"text": 2}},
            {"name": "ultrasound_bmode_image", "cases": 2, "types": {"asset": 2}},
        ],
        "names": ["classification", "ultrasound_bmode_image"],
    }
    plan, path = _plan(tmp_path)

    class FakeIndex:
        def __init__(self, _access):
            pass

        def values(self, name, limit=50):
            if name == "classification":
                return {"values": [{"value": "benign", "count": 1}, {"value": "malignant", "count": 1}], "total": 2}
            return {"values": [], "total": 0}

        def assets(self):
            return [{"asset_id": "asset-1", "kind": "image"}]

        def sample(self, *, evidence_name=None, limit=3, **_kwargs):
            return {"values": [{"case_id": "c1", "evidence": [{"id": "label", "type": "text", "name": evidence_name, "value": "benign", "ref": []}]}]}

    import medbenchagent.data.index as index
    monkeypatch.setattr(index, "DatasetIndex", FakeIndex)
    context = {**planning_context(tmp_path),
        "operation_directory": str(tmp_path),
        "task_plan_path": str(path),
        "planning_read_state": {},
        "dataset_facts": {"evidence_catalog": {row["name"]: row for row in catalog["entries"]}},
    }

    first = task_planning.read_evidence_catalog(context, names=["classification", "clasification"])
    payload = json.loads(first["summary"])
    assert payload["field_values"]["classification"]["values"]
    assert payload["case_samples"]["classification"][0]["evidence"]
    assert payload["suggestions"]["clasification"][0] == "classification"
    leaf_context = first["updates"]["leaf_context"]
    cached_field = next(item for item in leaf_context["evidence_catalog"] if item["name"] == "classification")
    assert cached_field["field_values"][0]["value"] == "benign"

    second = task_planning.read_evidence_catalog(context, names=["classification", "clasification"])
    assert json.loads(second["summary"]) == payload


def test_bad_leaf_contains_repairable_diagnostics(tmp_path):
    facts = {
        "evidence_names": ["ultrasound_bmode_image"],
        "evidence_catalog": {"ultrasound_bmode_image": {"name": "ultrasound_bmode_image", "types": {"asset": 1}}},
    }
    _, path = _plan(tmp_path, facts=facts)
    context = {**planning_context(tmp_path),
        "operation_directory": str(tmp_path),
        "task_plan_path": str(path),
        "dataset_facts": json.loads((tmp_path / "fixture_facts.json").read_text(encoding="utf-8")),
    }
    leaf = {
        "leaf_task_id": "bad",
        "options": ["yes", "no"],
        "name": "finding",
        "scenario_id": "scenario-1",
        "clinical_target_id": "finding",
        "question_templates": ["What is shown?"],
        "input": [{"role": "image", "name": "image", "evidence_type": "asset"}],
        "target": "finding",
        "reference_rule": "classification",
        "resolver": {
            "inputs": [{"role": "image", "name": "image", "evidence_type": "asset"}],
            "steps": [{"op": "read", "selector": {"role": "image", "name": "image", "evidence_type": "asset"}}],
            "output_spec": "label",
            "result_step": "step_0",
        },
        "output_spec": "label",
        "task_paradigm": "VQA",
        "capability_category": "Recognition",
    }

    with pytest.raises(task_planning.LeafTaskValidationError) as exc_info:
        task_planning.save_leaf_task(context, leaf=leaf)
    diagnostics = exc_info.value.diagnostics
    assert diagnostics["attempted_leaf"]["leaf_task_id"] == "bad"
    assert diagnostics["errors"][0]["code"] == "missing_input_evidence"
    assert diagnostics["errors"][0]["expected"] == ["ultrasound_bmode_image"]


def test_leaf_context_drops_cache_when_scenario_binding_changes(tmp_path):
    plan, path = _plan(tmp_path, facts={
        "revision": "r1",
        "evidence_catalog": {"old": {"name": "old", "cases": 1, "types": {"value": 1}}},
        "evidence_names": ["old"],
    })
    second = plan.scenarios["scenario-1"].model_copy(update={"scenario_id": "scenario-2"})
    plan.scenarios["scenario-2"] = second
    plan.current_scenario_id = "scenario-2"
    plan.pending_scenario_ids = ["scenario-2"]
    path = write_plan(tmp_path, plan)
    context = {**planning_context(tmp_path),
        "operation_directory": str(tmp_path),
        "task_plan_path": str(path),
        "dataset_facts": json.loads((tmp_path / "fixture_facts.json").read_text(encoding="utf-8")),
        "planning_evidence_cache": {
            "binding": {"dataset_id": "dataset", "revision": "r1", "scenario_id": "scenario-1"},
            "scenario_id": "scenario-1",
            "planning_evidence": [],
            "representative_cases": [],
            "catalog_fields": {"stale": {"name": "stale", "field_values": [{"value": "old"}]}},
        },
    }
    leaf_context = task_planning._leaf_context(context, task_planning._load_plan(context))
    assert all(item.get("name") != "stale" for item in leaf_context["evidence_catalog"])


def test_planning_recovery_compares_pre_call_draft():
    previous = {"revision": 3, "phase": "leaves", "current_scenario_id": "scenario-1", "leaf_tasks": {}}
    current = {"revision": 4, "phase": "leaves", "current_scenario_id": "scenario-1", "leaf_tasks": {"leaf-1": {}}}
    assert planning_draft_advanced(previous, current)
    assert not planning_draft_advanced(previous, previous)
