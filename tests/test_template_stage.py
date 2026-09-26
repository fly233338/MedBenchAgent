import copy
import json
from pathlib import Path

import pytest

from medbenchagent.agents import task_planning as planning
from medbenchagent.agents.templates import prepare_templates
from medbenchagent.runtime.dependencies import publish, require
from medbenchagent.runtime.run_artifacts import atomic_write_json
from test_task_planning_agent import context, ready, call, save, complete


def template_context(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    c["dataset_scope"] = "test"
    for kind in ("Dataset", "DatasetCard", "Materials", "PlanningDelegation"):
        publish(c, kind, {}, ())
    publish(c, "TemplateConfig", {"count": 3}, ())
    ready(c, monkeypatch)
    call(c, planning.finish)
    return c


def model_responses(monkeypatch, verdicts, calls):
    def model(system, prompt, *contracts, **kwargs):
        calls.append(json.loads(prompt))
        if system.startswith("Review"):
            return {"verdict": next(verdicts), "reason": "Clarify specification"}
        return {"templates": ["Status?", "Which status?", "Identify status."]}
    monkeypatch.setattr("medbenchagent.generation.templates.complete_json", model)


def test_template_wording_repair_cache_and_specification_changes(tmp_path, monkeypatch):
    c = template_context(tmp_path, monkeypatch)
    calls = []
    model_responses(monkeypatch, iter(["revise_templates", "approved", "approved"]), calls)
    assert call(c, prepare_templates)["status"] == "success"
    assert len(calls) == 4
    require(c, "Templates")
    call(c, prepare_templates)
    assert len(calls) == 4
    # A new Plan version with the same narrow specification reuses its approved set.
    publish(c, "Plan", {"path": c["working_blueprint_path"], "revision": 2}, ())
    call(c, prepare_templates)
    assert len(calls) == 4
    # Separate leaf budget permits another specification only when explicitly configured.
    c["limits"]["max_reworks"] = 2
    path = Path(c["working_blueprint_path"])
    plan = json.loads(path.read_text(encoding="utf-8"))
    plan["tasks"]["opaque.identity"]["protocol"]["question_intent"] = "Clarified target"
    atomic_write_json(path, plan)
    publish(c, "Plan", {"path": str(path)}, ())
    call(c, prepare_templates)
    assert len(calls) == 6


def test_template_task_defect_reopens_planning_with_persistent_budget(tmp_path, monkeypatch):
    c = template_context(tmp_path, monkeypatch)
    calls = []
    model_responses(monkeypatch, iter(["revise_plan"]), calls)
    outcome = call(c, prepare_templates)
    feedback = outcome["updates"]["template_revision_request"]
    assert feedback["issues"][0]["leaf_id"] == "opaque.identity"
    assert feedback["issues"][0]["templates"] == ["Status?", "Which status?", "Identify status."]
    assert "templates_path" not in outcome["updates"]
    c["operation_directory"] = str(tmp_path / "feedback")
    first = planning.apply_template_feedback(c, feedback)
    c.update(first["updates"])
    replay = planning.apply_template_feedback(c, feedback)
    assert replay == first
    assert planning._tree(c)["reworks_used"] == 1
    assert planning._tree(c)["review"] == {}
    assert "finish" not in planning.available_tools(c)
    with pytest.raises(ValueError, match="stale"):
        require(c, "Plan")
    save(c)
    complete(c)
    with pytest.raises(ValueError, match="problem not fixed"):
        call(c, planning.review_tree)
    with pytest.raises(ValueError, match="problem not fixed"):
        call(c, planning.finish)
    assert planning._tree(c)["reworks_used"] == 1
    leaf = planning._tree(c)["tasks"]["opaque.identity"]
    call(c, planning.save_leaf, leaf_id="opaque.identity", concept_id="core", description=leaf["description"],
         protocol={**leaf["protocol"], "question_intent": "Classify the explicit status"}, evidence=[],
         mapping_grounding=leaf["mapping_grounding"])
    complete(c)
    call(c, planning.review_tree)
    call(c, planning.finish)
    c["operation_directory"] = str(tmp_path / "second-feedback")
    assert planning.apply_template_feedback(c, {**feedback, "plan_version": "next"})["status"] == "success"
    c["operation_directory"] = str(tmp_path / "new-feedback")
    assert planning.apply_template_feedback(c, {**feedback, "feedback_id": "new-input"})["status"] == "blocked"


def test_template_attempt_budget_survives_redelegation(tmp_path, monkeypatch):
    c = template_context(tmp_path, monkeypatch)
    calls = []
    model_responses(monkeypatch, iter(["revise_templates", "revise_templates"]), calls)
    assert call(c, prepare_templates)["status"] == "blocked"
    assert len(calls) == 4
    assert call(copy.deepcopy(c), prepare_templates)["status"] == "blocked"
    assert len(calls) == 4


def test_template_service_failure_reuses_pending_attempt(tmp_path, monkeypatch):
    from medbenchagent.runtime.llm import ModelUnavailable
    c = template_context(tmp_path, monkeypatch)
    c["limits"]["max_reworks"] = 0
    def fail(*args, **kwargs):
        raise ModelUnavailable("Service offline")
    monkeypatch.setattr("medbenchagent.agents.templates.create_templates", fail)
    with pytest.raises(ModelUnavailable):
        call(c, prepare_templates)
    monkeypatch.setattr("medbenchagent.agents.templates.create_templates",
                        lambda *a, **kw: ["Status?", "Which status?", "Identify status."])
    monkeypatch.setattr("medbenchagent.agents.templates.review_templates",
                        lambda *a, **kw: {"verdict": "approved", "reason": "Valid"})
    assert call(c, prepare_templates)["status"] == "success"
