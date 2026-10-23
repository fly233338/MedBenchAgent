"""Real native Planner loop over small deterministic domain artifacts, with no network."""

import json
from dataclasses import replace
import pytest

from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AIMessage

from medbenchagent.agents.semantic_planner import build_planner_spec
from medbenchagent.compilation.artifacts import read_json_artifact, resolve_artifact
from medbenchagent.compilation.contracts import ArtifactRef, Delegate, Finish
from medbenchagent.compilation.demo import (
    create_synthetic_inputs,
    scripted_coding_spec,
    DERIVED_OUTPUT,
)
from medbenchagent.compilation.graph import build_compilation_graph
from medbenchagent.compilation.interfaces import CompilationContext
from medbenchagent.data.dataset import FileDatasetReader
from medbenchagent.planning.contracts import TaskPlan
from medbenchagent.planning.semantic import validate_plan
from medbenchagent.prompts import load_prompt
from medbenchagent.settings import load_planner_settings
from native_agent_fixtures import ScriptedChatModel, call, current_view


def planner_model():
    scenario_id = None
    first_requests = []
    dataset = None
    evidence = [{"case_id": key, "evidence_id": "value"} for key in ("case-a", "case-b")]
    inputs = [{"case_id": key, "evidence_id": "image"} for key in ("case-a", "case-b")]

    def respond(messages, index):
        nonlocal scenario_id, dataset
        if "Current authoritative view:\n" not in messages[0].content:
            return AIMessage(
                content="Preserve the immutable Task Plan and its data requirements; re-read current facts."
            )
        view = current_view(messages)
        request_id = view["request"]["request_id"]
        if request_id not in first_requests:
            first_requests.append(request_id)
            if len(first_requests) == 2:
                assert len(messages) > 3  # The Planner continued its own history after Coding.
                assert not any("PRIVATE_CODING_MESSAGE" in str(m.content) for m in messages)
        dataset = view["dataset"]
        revision = view["plan"]["revision"]

        def mutate(name, **kwargs):
            return call(name, {**kwargs, "base_revision": revision}, index)

        def task(feasible=False):
            return {
                "task_id": "double",
                "dataset": dataset,
                "scenario_id": scenario_id,
                "task_family": "closed_vqa",
                "objective": "Choose the doubled value label",
                "input_requirements": ["image"],
                "reference_requirements": ["derived doubled value"],
                "output_paradigm": "label",
                "evidence_refs": evidence,
                "input_evidence_refs": inputs,
                "reference_evidence_refs": evidence,
                "feasibility": "feasible" if feasible else "needs_data",
                "requirement_ids": ["double-value"],
                "revision": 2 if feasible else 1,
                "change_reason": "Accepted derived values" if feasible else "Initial design",
            }

        if len(first_requests) == 1:
            if revision == 0:
                return mutate(
                    "record_axis_assessment",
                    axis="input_structure",
                    assessments=[
                        {
                            "value": "single-image",
                            "supported": True,
                            "reason": "One input image per Case",
                            "evidence_refs": inputs,
                        }
                    ],
                )
            if revision in {1, 2}:
                axis, value = (
                    ("operation", "recognition")
                    if revision == 1
                    else ("clinical_target", "synthetic-number")
                )
                return mutate(
                    "record_axis_assessment",
                    axis=axis,
                    assessments=[
                        {
                            "value": value,
                            "supported": True,
                            "reason": "Source values",
                            "evidence_refs": evidence,
                        }
                    ],
                )
            if revision == 3:
                return mutate("build_candidate_scenarios")
            if revision == 4:
                if getattr(messages[-1], "name", None) == "inspect_task_plan":
                    scenario_id = json.loads(messages[-1].content)["items"][0]["id"]
                    return mutate(
                        "record_scenario_decision",
                        scenario_id=scenario_id,
                        keep=True,
                        reason="Evidence supports the task",
                        evidence_refs=evidence,
                    )
                return call("inspect_task_plan", {"section": "scenarios"}, index)
            if revision == 5:
                return mutate(
                    "propose_data_requirement",
                    requirement={
                        "requirement_id": "double-value",
                        "dataset": dataset,
                        "case_ids": ["case-a", "case-b"],
                        "purpose": "Compute doubled reference values",
                        "representation": DERIVED_OUTPUT,
                        "evidence_refs": evidence,
                        "acceptance_criteria": ["Values equal source multiplied by two"],
                        "affected_task_ids": ["double"],
                    },
                )
            if revision == 6:
                return mutate("save_semantic_task", task=task())
            if not view["submitted"]:
                return mutate("publish_task_plan")
            return call(
                "PlannerResponse",
                {
                    "outcome": "needs_input",
                    "summary": "Need derived values",
                    "artifact_ids": [
                        item["artifact_id"]
                        for item in view["submitted"]
                        if item["kind"] == "task_plan"
                    ],
                },
                index,
            )
        if revision == 7:
            ref = next(item for item in view["request"]["inputs"] if item["kind"] == "derived_view")
            if getattr(messages[-1], "name", None) != "inspect_input_artifact":
                return call("inspect_input_artifact", {"artifact_id": ref["artifact_id"]}, index)
            assert "doubled_value" in messages[-1].content
            return mutate(
                "accept_data_requirement",
                requirement_id="double-value",
                artifact_id=ref["artifact_id"],
                reason="Verified two times each source value",
                evidence_refs=evidence,
            )
        if revision == 8:
            return mutate("save_semantic_task", task=task(True))
        if not view["submitted"]:
            return mutate("publish_task_plan")
        return call(
            "PlannerResponse",
            {
                "outcome": "completed",
                "summary": "Published the revised plan",
                "artifact_ids": [
                    item["artifact_id"] for item in view["submitted"] if item["kind"] == "task_plan"
                ],
            },
            index,
        )

    return ScriptedChatModel(responder=respond)


def test_planner_data_requirement_revision_and_private_persistence(tmp_path):
    refs = create_synthetic_inputs(tmp_path, "planning")
    settings = load_planner_settings()
    model = planner_model()
    planner = build_planner_spec(settings, instructions=load_prompt("planner.md"), model=model)

    def decide(view):
        if len(view.history) == 3:
            return Finish(summary="Task Plan complete")
        agent = "coding" if len(view.history) == 1 else "task_planner"
        return Delegate(
            agent_id=agent,
            objective="PRIVATE_CODING_MESSAGE" if agent == "coding" else "Plan tasks",
            inputs=list(view.artifacts),
        )

    config = {"configurable": {"thread_id": "planning"}}
    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoint.sqlite")) as saver:
        graph = build_compilation_graph(
            {"task_planner": planner, "coding": scripted_coding_spec()}, checkpointer=saver
        )
        state = graph.invoke(
            {
                "compilation_id": "planning",
                "goal": "Plan tasks",
                "artifacts": [r.model_dump() for r in refs],
            },
            config,
            context=CompilationContext(tmp_path, decide, "Plan tasks"),
        )
        assert [row["result"]["outcome"] for row in state["history"]] == [
            "needs_input",
            "completed",
            "completed",
        ]
        assert "messages" not in state and "draft_path" not in state
        plans = [
            TaskPlan.model_validate(read_json_artifact(tmp_path, ArtifactRef.model_validate(ref)))
            for ref in state["artifacts"]
            if ref["kind"] == "task_plan"
        ]
        assert plans[0].tasks["double"].feasibility == "needs_data"
        assert (
            plans[1].tasks["double"].feasibility == "feasible"
            and plans[1].tasks["double"].revision == 2
        )
        assert plans[1].acceptances["double-value"].reason == "Verified two times each source value"
        reader = FileDatasetReader(resolve_artifact(tmp_path, refs[0]))
        validate_plan(plans[-1], reader, publish=True)
        assert (
            sum(1 for row in saver.list(config) if row.config["configurable"].get("checkpoint_ns"))
            > 2
        )


def test_missing_final_response_cannot_reuse_previous_success(tmp_path):
    refs = create_synthetic_inputs(tmp_path, "missing")
    settings = load_planner_settings()
    model = ScriptedChatModel(responder=lambda *_: AIMessage(content="I am done"))
    planner = build_planner_spec(settings, instructions=load_prompt("planner.md"), model=model)

    def decide(view):
        return (
            Finish(summary="Stopped")
            if view.last_result
            else Delegate(agent_id="task_planner", objective="Plan", inputs=list(view.artifacts))
        )

    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoint.sqlite")) as saver:
        graph = build_compilation_graph({"task_planner": planner}, checkpointer=saver)
        state = graph.invoke(
            {
                "compilation_id": "missing",
                "goal": "Plan",
                "artifacts": [r.model_dump() for r in refs],
            },
            {"configurable": {"thread_id": "missing"}},
            context=CompilationContext(tmp_path, decide, "Plan"),
        )
    assert state["result"]["outcome"] == "failed"
    assert "without a current structured result" in state["result"]["issues"][0]["message"]


def test_parallel_mutations_are_rejected_without_changing_draft(tmp_path):
    refs = create_synthetic_inputs(tmp_path, "parallel")

    def respond(messages, index):
        view = current_view(messages)
        if index == 1:
            args = {
                "axis": "operation",
                "base_revision": 0,
                "assessments": [
                    {
                        "value": "recognition",
                        "supported": True,
                        "reason": "test",
                        "evidence_refs": [{"case_id": "case-a", "evidence_id": "value"}],
                    }
                ],
            }
            one = call("record_axis_assessment", args, 1)
            return one.model_copy(
                update={
                    "tool_calls": [
                        *one.tool_calls,
                        *call("record_axis_assessment", args, 2).tool_calls,
                    ]
                }
            )
        assert view["plan"]["revision"] == 0
        assert all("one state-changing tool" in message.content for message in messages[-2:])
        return call(
            "PlannerResponse", {"outcome": "blocked", "summary": "Stopped safely"}, index + 2
        )

    settings = load_planner_settings()
    planner = build_planner_spec(
        settings, instructions=load_prompt("planner.md"), model=ScriptedChatModel(responder=respond)
    )

    def decide(view):
        return (
            Finish(summary="Stopped")
            if view.last_result
            else Delegate(agent_id="task_planner", objective="Plan", inputs=list(view.artifacts))
        )

    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoint.sqlite")) as saver:
        graph = build_compilation_graph({"task_planner": planner}, checkpointer=saver)
        result = graph.invoke(
            {
                "compilation_id": "parallel",
                "goal": "Plan",
                "artifacts": [r.model_dump() for r in refs],
            },
            {"configurable": {"thread_id": "parallel"}},
            context=CompilationContext(tmp_path, decide, "Plan"),
        )
    assert result["result"]["outcome"] == "blocked"


def empty_plan_model(*, missing_on_second=False):
    first_request = None

    def respond(messages, index):
        nonlocal first_request
        view = current_view(messages)
        first_request = first_request or view["request"]["request_id"]
        if missing_on_second and view["request"]["request_id"] != first_request:
            return AIMessage(content="No new result")
        revision = view["plan"]["revision"]
        if revision < 3:
            return call(
                "record_axis_assessment",
                {
                    "base_revision": revision,
                    "axis": ("input_structure", "operation", "clinical_target")[revision],
                    "assessments": [
                        {
                            "value": "not-supported",
                            "supported": False,
                            "reason": "Insufficient evidence",
                            "evidence_refs": [{"case_id": "case-a", "evidence_id": "value"}],
                        }
                    ],
                },
                index,
            )
        if not view["submitted"]:
            return call("publish_task_plan", {"base_revision": revision}, index)
        return call(
            "PlannerResponse",
            {
                "outcome": "completed",
                "summary": "Published unsupported-task evidence",
                "artifact_ids": [row["artifact_id"] for row in view["submitted"]],
            },
            index,
        )

    return ScriptedChatModel(responder=respond)


def test_completed_child_is_handed_off_after_reopening_checkpoint_and_stale_response_is_not_reused(
    tmp_path,
):
    refs = create_synthetic_inputs(tmp_path, "resume")
    settings = load_planner_settings()
    first_model = empty_plan_model()

    def decide(view):
        if len(view.history) == 2:
            return Finish(summary="Stopped")
        return Delegate(
            agent_id="task_planner", objective="Plan or inspect again", inputs=list(view.artifacts)
        )

    config = {"configurable": {"thread_id": "resume"}}
    context = CompilationContext(tmp_path, decide, "Plan")
    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoint.sqlite")) as saver:
        graph = build_compilation_graph(
            {
                "task_planner": build_planner_spec(
                    settings, instructions=load_prompt("planner.md"), model=first_model
                )
            },
            checkpointer=saver,
        )
        graph.invoke(
            {
                "compilation_id": "resume",
                "goal": "Plan",
                "artifacts": [r.model_dump() for r in refs],
            },
            config,
            context=context,
            interrupt_before=["collect"],
        )
        snapshot = graph.get_state(config)
        assert snapshot.next == ("collect",) and snapshot.values["result"]["outcome"] == "completed"
        assert not snapshot.values.get("history")
    files = {p: p.stat().st_mtime_ns for p in (tmp_path / "artifacts").rglob("*.json")}
    calls = []

    def no_final(messages, index):
        view = current_view(messages)
        calls.append(view["request"]["request_id"])
        assert view["submitted"] == []
        return AIMessage(content="No current final response")

    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoint.sqlite")) as saver:
        graph = build_compilation_graph(
            {
                "task_planner": build_planner_spec(
                    settings,
                    instructions=load_prompt("planner.md"),
                    model=ScriptedChatModel(responder=no_final),
                )
            },
            checkpointer=saver,
        )
        state = graph.invoke(None, config, context=context)
    assert [r["result"]["outcome"] for r in state["history"]] == ["completed", "failed"]
    assert calls == ["resume-000002"]
    assert all(path.stat().st_mtime_ns == mtime for path, mtime in files.items())


@pytest.mark.parametrize("limit", [1, 4])
def test_native_call_limit_returns_failure_not_success(tmp_path, limit):
    refs = create_synthetic_inputs(tmp_path, "limited")
    settings = load_planner_settings()
    settings = replace(settings, loop=replace(settings.loop, max_model_calls=limit))
    model = empty_plan_model()
    planner = build_planner_spec(settings, instructions=load_prompt("planner.md"), model=model)

    def decide(view):
        return (
            Finish(summary="Stopped")
            if view.last_result
            else Delegate(agent_id="task_planner", objective="Plan", inputs=list(view.artifacts))
        )

    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoint.sqlite")) as saver:
        graph = build_compilation_graph({"task_planner": planner}, checkpointer=saver)
        result = graph.invoke(
            {
                "compilation_id": "limited",
                "goal": "Plan",
                "artifacts": [r.model_dump() for r in refs],
            },
            {"configurable": {"thread_id": "limited"}},
            context=CompilationContext(tmp_path, decide, "Plan"),
        )
    assert (
        result["result"]["outcome"] == "failed"
        and result["result"]["issues"][0]["code"] == "agent_limit"
    )
    assert len(model.calls) == limit
    assert len(result["result"]["artifacts"]) == (1 if limit == 4 else 0)


def test_native_cli_freezes_real_input_planner_config_and_prompts(tmp_path, monkeypatch, capsys):
    import medbenchagent.compilation.__main__ as cli
    import medbenchagent.runtime.native as native
    import medbenchagent.runtime.model_services as services
    import medbenchagent.prompts as prompts
    import medbenchagent.settings as settings_module

    source_root = tmp_path / "source"
    reference = create_synthetic_inputs(source_root, "source")[0]
    source_manifest = resolve_artifact(source_root, reference)
    monkeypatch.setattr(settings_module, "load_project_env", lambda: None)
    for name in (*settings_module._ENVIRONMENT_OVERRIDES, "MEDBENCH_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        services, "resolve_model_service", lambda _role, _url, model, _credential: model
    )
    monkeypatch.setattr(
        cli,
        "create_synthetic_inputs",
        lambda *_args, **_kwargs: pytest.fail("Real input cannot create a demo"),
    )
    configuration = tmp_path / "model.toml"
    configuration.write_text('[llm]\nmodel="offline"\n[planner]\npage_size=7\n', encoding="utf-8")

    def supervise(messages, index):
        view = current_view(messages)
        assert "coding" not in view["agents"]
        if not view["history"]:
            decision = {
                "action": "delegate",
                "agent_id": "task_planner",
                "objective": "Plan",
                "input_artifact_ids": [row["artifact_id"] for row in view["available_artifacts"]],
            }
        elif not view["user_answers"]:
            decision = {"action": "ask_user", "question": "Inspect the published plan?"}
        else:
            decision = {"action": "finish", "summary": "Published plan available"}
        return call("SupervisorResponse", {"decision": decision}, index)

    supervisor_model, planner = ScriptedChatModel(responder=supervise), empty_plan_model()
    models = iter([supervisor_model, planner, supervisor_model, planner])
    monkeypatch.setattr(native, "chat_model", lambda *_: next(models))
    run_dir = tmp_path / "run"
    common = ["--directory", str(run_dir), "--compilation-id", "input"]
    cli.main(
        [
            "run",
            *common,
            "--supervisor",
            "llm",
            "--planner",
            "llm",
            "--dataset-manifest",
            str(source_manifest),
            "--config",
            str(configuration),
        ]
    )
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "waiting" and first["history"][0]["outcome"] == "completed"
    frozen = (run_dir / "run.json").read_text(encoding="utf-8")
    assert json.loads(frozen)["planner_settings"]["planner"]["page_size"] == 7
    assert json.loads(frozen)["prompts"]["planner"]["text"] == load_prompt("planner.md")
    source_manifest.write_text("source changed", encoding="utf-8")
    configuration.write_text("invalid config now", encoding="utf-8")
    monkeypatch.setattr(
        prompts, "load_prompt", lambda *_: pytest.fail("Resume cannot reload source instructions")
    )
    monkeypatch.setattr(
        services, "resolve_model_service", lambda *_: pytest.fail("Resume cannot rediscover models")
    )
    count = len(planner.calls)
    cli.main(["resume", *common, "--answer", "Done"])
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed" and len(planner.calls) == count
    assert (run_dir / "run.json").read_text(encoding="utf-8") == frozen
    dataset = next(
        ArtifactRef.model_validate(row) for row in result["artifacts"] if row["kind"] == "dataset"
    )
    assert (
        FileDatasetReader(resolve_artifact(run_dir, dataset)).get_case("case-a").case_id == "case-a"
    )
