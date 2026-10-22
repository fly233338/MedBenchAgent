"""Composition boundaries and frozen resources, using the existing fake SDK runner."""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from langgraph.checkpoint.sqlite import SqliteSaver
import pytest

from medbenchagent import prompts
from medbenchagent.compilation import assembly, coding, graph, interfaces, run
from medbenchagent.compilation.contracts import Delegate, Finish, WorkRequest, WorkResult
from test_compilation_coding import Crash, FakeRunner


def test_assembly_freezes_prompts_and_resumes_without_live_sources(tmp_path, monkeypatch):
    import medbenchagent.settings as settings
    import medbenchagent.runtime.model_services as services

    monkeypatch.setattr(settings, "load_project_env", lambda: None)
    monkeypatch.setattr(run, "load_project_env", lambda: None)
    for name in (*settings._ENVIRONMENT_OVERRIDES, "MEDBENCH_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(services, "resolve_model_service", lambda kind, *_: f"{kind.lower()}-model")
    options = run.create_run(tmp_path, compilation_id="freeze", goal="Preserve this original goal",
                             supervisor_mode="llm", coding_mode="openhands")
    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite")) as saver:
        saver.setup()
    request = WorkRequest(request_id="freeze-1", compilation_id="freeze", agent_id="coding",
                          workstream_id="ws-freeze", objective="Write a report")
    broken = FakeRunner(crash="sent")
    monkeypatch.setattr(coding, "run_local_worker", broken)
    extra = {"inspector": interfaces.AgentSpec("Independent test capability", lambda state: {})}
    with assembly.assemble_compilation(options, tmp_path, additional_agents=extra) as (agents, context):
        assert "inspector" in agents
        assert context.decide.instructions == options.prompts["supervisor"].text
        with pytest.raises(Crash):
            agents["coding"].node({"request": request.model_dump()}, SimpleNamespace(context=context))
    original = prompts.load_prompt
    monkeypatch.setattr(prompts, "load_prompt", lambda name: original(name) + "\nCHANGED SOURCE\n")
    fresh = run.create_run(tmp_path / "new", compilation_id="new", goal="New run",
                           supervisor_mode="llm", coding_mode="openhands")
    assert all(fresh.prompts[name].digest != options.prompts[name].digest for name in options.prompts)
    monkeypatch.setattr(prompts, "load_prompt", lambda _: pytest.fail("Resume must not read prompt resources"))
    monkeypatch.setattr(run, "load_coding_settings", lambda _: pytest.fail("Resume must not load live settings"))
    monkeypatch.setattr(run, "load_supervisor_settings", lambda _: pytest.fail("Resume must not load live settings"))
    monkeypatch.setattr(services, "resolve_model_service", lambda *_: pytest.fail("Resume must not discover models"))
    restored = run.resume_run(tmp_path, "freeze")
    assert restored == options
    runner = FakeRunner()
    monkeypatch.setattr(coding, "run_local_worker", runner)
    with assembly.assemble_compilation(restored, tmp_path) as (agents, context):
        result = agents["coding"].node({"request": request.model_dump()}, SimpleNamespace(context=context))["result"]
    assert result["outcome"] == "completed"
    assert runner.payloads[0]["prompt"] == broken.payloads[0]["prompt"]
    assert runner.payloads[0]["compilation_goal"] == options.goal
    assert "CHANGED SOURCE" not in runner.payloads[0]["prompt"]
    document = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    document["prompts"]["coding"]["text"] += "tampered"
    (tmp_path / "run.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="content digest"):
        run.resume_run(tmp_path, "freeze")


def test_agent_interfaces_do_not_depend_on_graph_or_sdk():
    root = Path(graph.__file__).parent
    assert all(not hasattr(graph, name) for name in (
        "AgentInput", "AgentOutput", "AgentSpec", "CompilationContext", "SupervisorView",
    ))
    for name in ("interfaces", "graph", "coding", "supervisor", "demo", "__main__"):
        tree = ast.parse((root / f"{name}.py").read_text(encoding="utf-8"))
        dependencies = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        if name in {"interfaces", "coding", "supervisor", "demo"}:
            assert "graph" not in dependencies and "medbenchagent.compilation.graph" not in dependencies
        if name in {"interfaces", "graph", "__main__"}:
            assert not {"coding", "coding_worker", "supervisor"}.intersection(dependencies)
        if name in {"interfaces", "graph"}:
            assert not any("openhands" in module for module in dependencies)
    assert not hasattr(__import__("medbenchagent.compilation.demo", fromlist=["demo_agents"]), "demo_agents")
    with pytest.raises((FileNotFoundError, ValueError)):
        prompts.load_prompt("missing.md")


def test_artifact_digest_detects_changes_and_cannot_be_reregistered_over_an_accepted_ref(tmp_path):
    from medbenchagent.compilation.artifacts import (
        ArtifactIntegrityError, preview_artifact, read_json_artifact, register_file,
        resolve_artifact, write_json_artifact,
    )

    seed = WorkRequest(request_id="integrity-000001", compilation_id="integrity",
                       agent_id="writer", objective="Create accepted evidence")
    ref = write_json_artifact(tmp_path, seed, "evidence.json", kind="report", value={"accepted": True})
    path = resolve_artifact(tmp_path, ref)
    assert ref.digest == hashlib.sha256(path.read_bytes()).hexdigest()
    mtime = path.stat().st_mtime_ns
    assert write_json_artifact(tmp_path, seed, "evidence.json", kind="report", value={"accepted": True}) == ref
    assert path.stat().st_mtime_ns == mtime
    path.write_text('{"accepted": false}', encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError):
        read_json_artifact(tmp_path, ref)
    assert preview_artifact(tmp_path, ref, limit=100)["issue_code"] == "artifact_integrity"

    def rewriting_agent(state):
        request = WorkRequest.model_validate(state["request"])
        replacement = register_file(tmp_path, request, ref.path, kind=ref.kind)
        assert replacement.artifact_id == ref.artifact_id and replacement.digest != ref.digest
        return {"result": WorkResult(request_id=request.request_id, outcome="completed",
                                     summary="Attempt to accept changed bytes", artifacts=[replacement]).model_dump(mode="json")}

    def decide(view):
        return Finish(summary="Checked") if view.last_result else Delegate(agent_id="writer", objective="Replay")

    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite")) as saver:
        built = graph.build_compilation_graph({"writer": interfaces.AgentSpec("Test writer", rewriting_agent)},
                                              checkpointer=saver)
        result = built.invoke({"compilation_id": "integrity", "goal": "Check content", "artifacts": [ref.model_dump()]},
                              {"configurable": {"thread_id": "integrity"}},
                              context=interfaces.CompilationContext(tmp_path, decide, "Check content"), durability="sync")
    assert result["result"]["outcome"] == "failed"
    assert result["result"]["issues"][0]["code"] == "artifact_integrity"
    assert result["artifacts"] == [ref.model_dump()]
    records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["event_type"] for row in records] == ["work_delegated", "work_result"]
    assert all(row["compilation_id"] == "integrity" and row["request_id"] == seed.request_id
               and row["agent_id"] == "writer" for row in records)
