"""Composition boundaries and frozen resources, using the existing fake SDK runner."""

import ast
import hashlib
import json
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
import pytest

from medbenchagent import prompts
from medbenchagent.compilation import graph, interfaces, run
from medbenchagent.compilation.contracts import Delegate, Finish, WorkRequest, WorkResult


def test_assembly_freezes_prompts_and_resumes_without_live_sources(tmp_path, monkeypatch):
    import medbenchagent.runtime.model_services as services
    from medbenchagent.compilation.demo import create_demo_source
    source = create_demo_source(tmp_path / "source")
    monkeypatch.setattr(services, "resolve_model_service", lambda kind, *_: f"{kind.lower()}-model")
    directory = tmp_path / "run"
    options = run.create_run(directory, compilation_id="freeze", goal="Freeze", source_manifest=source)
    with SqliteSaver.from_conn_string(str(directory / "checkpoints.sqlite")) as saver:
        saver.setup()
    original = prompts.load_prompt
    monkeypatch.setattr(prompts, "load_prompt", lambda name: original(name) + "\nCHANGED SOURCE\n")
    fresh = run.create_run(tmp_path / "fresh", compilation_id="fresh", goal="New", source_manifest=source)
    assert all(fresh.prompts[name].digest != options.prompts[name].digest for name in options.prompts)
    monkeypatch.setattr(prompts, "load_prompt", lambda _: pytest.fail("Resume read live prompts"))
    monkeypatch.setattr(services, "resolve_model_service", lambda *_: pytest.fail("Resume contacted model service"))
    assert run.resume_run(directory, "freeze") == options
    document = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    for selector in ("supervisor_mode", "coding_mode", "planner_mode"):
        with pytest.raises(ValueError, match="Extra inputs"):
            run.RunOptions.model_validate({**document, selector: "llm"})
    with monkeypatch.context() as changed:
        changed.setattr(run, "implementation_digest", lambda: "different-implementation")
        with pytest.raises(ValueError, match="incompatible"):
            run.resume_run(directory, "freeze")
    assert json.loads((directory / "run.json").read_text(encoding="utf-8")) == document
    document["prompts"]["coding"]["text"] += "tampered"
    (directory / "run.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="content digest"):
        run.resume_run(directory, "freeze")


def test_studio_assembles_frozen_run_without_live_discovery_or_model_calls(tmp_path, monkeypatch):
    from medbenchagent.compilation.demo import create_demo_source
    from medbenchagent.compilation.studio import graph as studio_graph
    import medbenchagent.runtime.model_services as services
    import medbenchagent.runtime.native as native
    from native_agent_fixtures import ScriptedChatModel

    source = create_demo_source(tmp_path / "source")
    monkeypatch.setattr(services, "resolve_model_service", lambda *_: "offline")
    directory = tmp_path / "studio"
    run.create_run(directory, compilation_id="studio", goal="Compile", source_manifest=source)
    monkeypatch.setattr(services, "resolve_model_service", lambda *_: pytest.fail("Studio rediscovered models"))
    monkeypatch.setattr(prompts, "load_prompt", lambda *_: pytest.fail("Studio reread live prompts"))
    monkeypatch.setattr(native, "chat_model", lambda *_: ScriptedChatModel(
        responder=lambda *_: pytest.fail("Assembly called a model")))
    graph = studio_graph({"configurable": {"directory": str(directory), "thread_id": "studio"}})
    assert graph.get_input_jsonschema()
    assert (directory / "studio-thread.txt").read_text() == "studio"
    assert not (directory / "checkpoints.sqlite").exists()


def test_agent_interfaces_do_not_depend_on_graph_or_sdk():
    root = Path(graph.__file__).parent
    assert all(not hasattr(graph, name) for name in (
        "AgentInput", "AgentOutput", "AgentSpec", "CompilationContext", "SupervisorView",
    ))
    for name in ("interfaces", "graph", "demo", "__main__"):
        tree = ast.parse((root / f"{name}.py").read_text(encoding="utf-8"))
        dependencies = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        if name in {"interfaces", "demo"}:
            assert "graph" not in dependencies and "medbenchagent.compilation.graph" not in dependencies
        if name in {"interfaces", "graph", "__main__"}:
            assert not {"coding", "coding_worker", "supervisor"}.intersection(dependencies)
        if name in {"interfaces", "graph"}:
            assert not any("openhands" in module for module in dependencies)
            assert not any(module.startswith("medbenchagent.agents") for module in dependencies)
    for path in (root.parent / "agents").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        dependencies = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        dependencies += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                         for alias in node.names]
        assert not {"medbenchagent.compilation.graph", "medbenchagent.compilation.assembly",
                    "medbenchagent.runtime.graph_runner"}.intersection(dependencies), path
    assert not hasattr(__import__("medbenchagent.compilation.demo", fromlist=["demo_agents"]), "demo_agents")
    with pytest.raises((FileNotFoundError, ValueError)):
        prompts.load_prompt("missing.md")


def test_artifact_digest_detects_changes_and_cannot_be_reregistered_over_an_accepted_ref(tmp_path):
    from medbenchagent.compilation.artifacts import (
        ArtifactIntegrityError, preview_artifact, read_json_artifact, register_file,
        resolve_artifact, write_json_artifact,
    )

    seed = WorkRequest(request_id="integrity-000001", compilation_id="integrity",
                       agent_id="writer", operation="write", objective="Create accepted evidence")
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
        return Finish(summary="Checked", goal_result="blocked") if view.last_result else Delegate(agent_id="writer", operation="write", objective="Replay")

    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite")) as saver:
        built = graph.build_compilation_graph({"writer": interfaces.AgentSpec("Test writer", rewriting_agent, operations=("write",))},
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
