
import pytest

from medbenchagent.runtime.clarifications import new_issue, inject_answers, resolve_clarification, record_answer
from medbenchagent.planning.bir import BIRStore
from medbenchagent.runtime import sessions


@pytest.mark.parametrize("kind", ["semantic", "requirement_change"])
def test_resolution_closes_saved_origin_and_only_invalidates_changed_requirement(tmp_path, kind):
    context = {"run_directory": str(tmp_path), "operation_directory": str(tmp_path / "resolve"), "user_requirement": "Build"}
    context["bir_path"] = sessions.create_experiment("pilot", root=tmp_path)["bir_path"]
    store = BIRStore(context["bir_path"])
    store.update("benchmark_info.json", {"text":"Build"}, run_id="r", actor="test")
    before = store.digest("benchmark_info.json")
    child_context = {**context, "dataset_scope": "a", "agent_role": "TaskPlanningAgent"}
    issue = new_issue("TaskPlanningAgent", child_context, "field", "What does 0 mean?")
    key = issue["issue_id"]
    saved = tmp_path / "child.json"
    from medbenchagent.runtime.contexts import save_context, load_context
    save_context(saved, {**child_context, "global_issue": issue, "issues": {key: issue}}, "planning", "a")
    context["child_states"] = {"a": {"planning": str(saved)}}
    context["issues"] = {key: record_answer(child_context, issue, "0 means benign")}
    context["global_issue"] = context["issues"][key]
    context.update(resolve_clarification(context, issue_id=key, kind=kind)["updates"])
    child = load_context(context["child_states"]["a"]["planning"], "planning", "a")
    assert child["global_issue"] is None and context["global_issue"] is None
    assert child["decisions"][key] == context["issues"][key]["answer_reference"]
    assert inject_answers(context, {}, "TaskPlanningAgent", "b")["decisions"] == {}
    assert (store.digest("benchmark_info.json") != before) == (kind == "requirement_change")
    if kind == "requirement_change":
        assert "What does 0 mean?" in context["user_requirement"]
    assert resolve_clarification(context, issue_id=key, kind=kind)["updates"] == {}


def test_explicit_stop_answer_marks_run_for_termination(tmp_path):
    context = {"run_directory": str(tmp_path), "operation_directory": str(tmp_path / "resolve"), "user_requirement": "Build"}
    issue = new_issue("SupervisorAgent", context, "generation_failure", "Generation failed. Continue or stop?")
    key = issue["issue_id"]
    context["issues"] = {key: record_answer(context, issue, "you can stop here")}
    context["global_issue"] = context["issues"][key]

    updates = resolve_clarification(context, issue_id=key, kind="requirement_change")["updates"]

    assert updates["stop_requested"] is True
