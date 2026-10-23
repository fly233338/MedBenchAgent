from types import SimpleNamespace
import pytest

from medbenchagent.compilation import run
from medbenchagent.compilation.assembly import assemble_compilation
from medbenchagent.compilation.contracts import GoalSpec, WorkScope
from medbenchagent.compilation.demo import create_demo_source
from medbenchagent.compilation.artifacts import resolve_artifact
from medbenchagent.planning.semantic import scoped_objects
from dataset_fixtures import create_synthetic_inputs
from native_agent_fixtures import ScriptedChatModel


@pytest.mark.parametrize("goal,raw,expected", [
    (GoalSpec(kind="profile"), True, {"Supervisor", "Coding", "data_processing"}),
    (GoalSpec(kind="plan", plan_level="scenarios"), True, {"Supervisor", "Coding", "Planner", "data_processing"}),
    (GoalSpec(kind="plan", plan_level="scenarios"), False, {"Supervisor", "Planner"}),
    (GoalSpec(kind="plan", plan_level="executable_leaves"), False, {"Supervisor", "Coding", "Planner", "data_processing"}),
    (GoalSpec(), True, {"Supervisor", "Coding", "Planner", "data_processing", "generation", "review"}),
    (GoalSpec(), False, {"Supervisor", "Coding", "Planner", "data_processing", "generation", "review"}),
])
def test_goal_loads_only_required_role_configuration(tmp_path, monkeypatch, goal, raw, expected):
    import medbenchagent.runtime.model_services as services
    import medbenchagent.runtime.native as native
    calls = []
    def resolve(kind, *_):
        assert kind in expected, f"Unnecessary model discovery: {kind}"
        calls.append(kind)
        return "offline"
    monkeypatch.setattr(services, "resolve_model_service", resolve)
    if "Coding" not in expected:
        monkeypatch.setattr(run, "load_coding_settings", lambda *_: pytest.fail("Loaded unused Coding config"))
    if "Planner" not in expected:
        monkeypatch.setattr(run, "load_planner_settings", lambda *_: pytest.fail("Loaded unused Planner config"))
    if raw:
        inputs = {"source_manifest": create_demo_source(tmp_path / "source")}
    else:
        ref = create_synthetic_inputs(tmp_path / "data", "data")[0]
        inputs = {"dataset_manifest": resolve_artifact(tmp_path / "data", ref)}
    options = run.create_run(tmp_path / "run", compilation_id="goal", goal="Check", goal_spec=goal, **inputs)
    assert set(calls) == expected
    assert (options.coding_skills is not None) == ("Coding" in expected)
    monkeypatch.setattr(native, "chat_model", lambda *_: ScriptedChatModel(responder=lambda *_: pytest.fail("Assembly invoked a model")))
    with assemble_compilation(options, tmp_path / "run") as (agents, _):
        assert set(agents) == ({name for name in expected if name.islower()}
                              | ({"task_planner"} if "Planner" in expected else set())
                              | ({"delivery"} if goal.kind == "benchmark" else set()))


def test_requirement_and_task_scope_do_not_expand_to_siblings():
    plan = SimpleNamespace(scenarios={"s": None},
        tasks={key: SimpleNamespace(scenario_id="s", requirement_ids=["r-" + key]) for key in ("a", "b")},
        requirements={"r-" + key: SimpleNamespace(affected_scenario_ids=["s"], affected_task_ids=[key]) for key in ("a", "b")})
    def selected(kind, key):
        return scoped_objects(plan, [WorkScope(artifact_id="plan", kind=kind, object_ids=[key])])
    assert selected("requirement", "r-a") == ({"s"}, set(), {"r-a"})
    assert selected("task", "a") == ({"s"}, {"a"}, {"r-a"})
    assert selected("scenario", "s") == ({"s"}, {"a", "b"}, {"r-a", "r-b"})


def test_profile_cannot_execute_taskdata_only_tool(tmp_path, monkeypatch):
    import medbenchagent.agents.data_processing.agent as dp
    from medbenchagent.runtime.native import OperationTools
    from medbenchagent.settings import load_work_agent_settings
    captured = {}
    monkeypatch.setattr(dp, "create_agent", lambda *args, **kwargs: captured.update(kwargs))
    dp.build_data_processing_spec(load_work_agent_settings("data_processing"), None,
                                 instructions={}, model=ScriptedChatModel(responder=lambda *_: pytest.fail("Unexpected model call")))
    middleware = next(item for item in captured["middleware"] if isinstance(item, OperationTools))
    context = SimpleNamespace(request=SimpleNamespace(operation="profile"))
    request = SimpleNamespace(runtime=SimpleNamespace(context=context),
                              tool_call={"name": "inspect_candidate_cases", "id": "one"})
    result = middleware.wrap_tool_call(request, lambda _: pytest.fail("Profile executed TaskData tool"))
    assert result.status == "error"
    context.request.operation = "prepare_task_data"
    assert middleware.wrap_tool_call(request, lambda _: "allowed") == "allowed"
