import json

import pytest

from medbenchagent.runtime.clarifications import new_issue, inject_answers, resolve_clarification
from medbenchagent.runtime.dependencies import publish


@pytest.mark.parametrize("kind", ["semantic", "requirement_change"])
def test_resolution_closes_saved_origin_and_only_invalidates_changed_requirement(tmp_path, kind):
    context = {"run_directory": str(tmp_path), "operation_directory": str(tmp_path / "resolve"), "user_requirement": "Build"}
    publish(context, "Requirement", {"text": "Build"}, ())
    before = context["artifact_versions"]["Requirement"]
    issue = new_issue("TaskPlanningAgent", {"dataset_scope": "a"}, "field", "What does 0 mean?")
    key = issue["issue_id"]
    saved = tmp_path / "child.json"
    saved.write_text(json.dumps({"global_issue": issue, "issues": {key: issue}}), encoding="utf-8")
    context["child_states"] = {"a": {"planning": str(saved)}}
    context["issues"] = {key: {**issue, "status": "answered", "answer": "0 means benign"}}
    context["global_issue"] = context["issues"][key]
    context.update(resolve_clarification(context, issue_id=key, kind=kind)["updates"])
    child = json.loads(open(context["child_states"]["a"]["planning"], encoding="utf-8").read())
    assert child["global_issue"] is None and context["global_issue"] is None
    assert child["decisions"][key] == "0 means benign"
    assert inject_answers(context, {}, "TaskPlanningAgent", "b")["decisions"] == {}
    assert (context["artifact_versions"]["Requirement"] != before) == (kind == "requirement_change")
    if kind == "requirement_change":
        assert "What does 0 mean?" in context["user_requirement"]
    assert resolve_clarification(context, issue_id=key, kind=kind)["updates"] == {}
