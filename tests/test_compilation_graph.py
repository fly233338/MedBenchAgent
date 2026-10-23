"""Two focused acceptance tests for the scripted core, not autonomous model quality."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from langgraph.checkpoint.sqlite import SqliteSaver

from medbenchagent.compilation.artifacts import read_json_artifact, resolve_artifact
from medbenchagent.compilation.contracts import ArtifactRef
from medbenchagent.compilation.demo import (
    create_synthetic_inputs, scripted_supervisor,
)
from medbenchagent.compilation.graph import build_compilation_graph
from medbenchagent.compilation.interfaces import CompilationContext
from medbenchagent.compilation.assembly import assemble_compilation
from medbenchagent.compilation.run import RunOptions
from medbenchagent.data.dataset import FileDatasetReader
from medbenchagent.planning.contracts import DataRequirement, SemanticTaskSpec


def _private_planner_checkpoint(saver, config):
    # Read namespaces supplied by LangGraph, without constructing one ourselves.
    return next(
        item.checkpoint["channel_values"] for item in saver.list(config)
        if "planning_pass" in item.checkpoint["channel_values"]
    )


def test_multi_round_collaboration_uses_evidence_and_isolates_state(tmp_path):
    options = RunOptions(compilation_id="test", goal="Derive a Case-linked synthetic task", supervisor_mode="scripted",
                         coding_mode="scripted", initial_evidence=False, pause_for_input=False,
                         supervisor_settings=None, coding_settings=None, prompts={})
    with assemble_compilation(options, tmp_path) as (agents, _), \
            SqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite")) as saver:
        graph = build_compilation_graph(agents, checkpointer=saver)
        traces = []
        for initial_evidence in (False, True):
            identifier = "known" if initial_evidence else "raw"
            directory = tmp_path / identifier
            inputs = create_synthetic_inputs(
                directory, identifier, initial_evidence=initial_evidence,
            )
            config = {"configurable": {"thread_id": identifier}}
            output = graph.invoke(
                {
                    "compilation_id": identifier, "goal": "Derive a Case-linked synthetic task",
                    "artifacts": [ref.model_dump(mode="json") for ref in inputs],
                },
                config,
                context=CompilationContext(directory, scripted_supervisor(directory), options.goal),
                durability="sync",
            )
            assert output["status"] == "completed"
            trace = [entry["request"]["agent_id"] for entry in output["history"]]
            traces.append(trace)
            request_ids = [entry["request"]["request_id"] for entry in output["history"]]
            assert len(request_ids) == len(set(request_ids))
            assert all(identifier in item for item in request_ids)
            assert all(row["result"]["outcome"] != "failed" for row in output["history"])
            assert "planning_pass" not in output and "pending_requirement_id" not in output
            private = _private_planner_checkpoint(saver, config)
            assert private["planning_pass"] == 2
            assert private["pending_requirement_id"] is None

            refs = [ArtifactRef.model_validate(raw) for raw in output["artifacts"]]
            for ref in refs:
                assert resolve_artifact(directory, ref).is_file()
            plans = [
                SemanticTaskSpec.model_validate(read_json_artifact(directory, ref)["tasks"]["synthetic-double"])
                for ref in refs if ref.kind == "task_plan"
            ]
            assert [plan.feasibility for plan in plans] == ["needs_data", "feasible"]
            assert plans[0].requirement_ids == ["double-value"]
            assert plans[1].requirement_ids == []
            requirement = DataRequirement.model_validate(read_json_artifact(
                directory, next(ref for ref in refs if ref.kind == "data_requirement"),
            ))
            derived = read_json_artifact(
                directory, next(ref for ref in refs if ref.kind == "derived_view"),
            )
            assert derived["dataset"] == requirement.dataset.model_dump()
            assert derived["requirement_id"] == requirement.requirement_id
            assert [row["case_id"] for row in derived["rows"]] == requirement.case_ids
            assert [row["doubled_value"] for row in derived["rows"]] == [4, 10]
            assert all(row["source_evidence_ids"] == ["value"] for row in derived["rows"])

            reader = FileDatasetReader(resolve_artifact(directory, inputs[0]))
            assert reader.get_manifest().dataset_id == "synthetic"
            page = reader.list_cases(0, 1)
            assert page.next_offset == 1 and reader.list_cases(1, 1).next_offset is None
            case = reader.get_case(page.items[0].case_id)
            image = reader.get_evidence(case.case_id, "image")
            asset = reader.get_asset(image.value)
            assert asset.asset_id != asset.source_path
            assert Path(reader.resolve_asset(asset.asset_id).path).is_file()
            assert case.relations[0].source_refs[0].source_id == "raw"
        assert traces == [
            ["coding", "task_planner", "coding", "task_planner"],
            ["task_planner", "coding", "task_planner"],
        ]


def _demo_process(directory, *arguments):
    environment = {**os.environ, "PYTHONUTF8": "1", "LANGSMITH_TRACING": "false"}
    # Each invocation has a fresh interpreter and SDK-independent module graph.
    completed = subprocess.run(
        [
            sys.executable, "-m", "medbenchagent.compilation", *arguments,
            "--directory", str(directory), "--compilation-id", "resume-demo",
        ],
        capture_output=True, text=True, encoding="utf-8", env=environment,
        timeout=60, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_checkpoint_resumes_in_a_new_process_without_repeating_completed_work(tmp_path):
    directory = tmp_path / "restart"
    before = _demo_process(directory, "run", "--pause-for-input")
    assert before["status"] == "waiting" and len(before["questions"]) == 1
    assert before["supervisor_mode"] == before["coding_mode"] == before["planner_mode"] == "scripted"
    saved_options = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    assert saved_options["supervisor_mode"] == "scripted"
    assert saved_options["supervisor_settings"] is saved_options["coding_settings"] is None
    assert saved_options["coding_mode"] == "scripted"
    assert len(before["workstreams"]) == 1
    assert before["history"][0]["workstream_id"] == before["history"][2]["workstream_id"]
    assert [row["agent_id"] for row in before["history"]] == [
        "coding", "task_planner", "coding",
    ]
    files = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (directory / "artifacts").rglob("*") if path.is_file()
    }
    after = _demo_process(directory, "resume", "--answer", "Continue")
    assert after["status"] == "completed" and after["questions"] == []
    assert json.loads((directory / "run.json").read_text(encoding="utf-8")) == saved_options
    assert after["history"][:-1] == before["history"]
    assert after["workstreams"] == before["workstreams"]
    assert after["history"][-1]["agent_id"] == "task_planner"
    assert after["artifacts"][:len(before["artifacts"])] == before["artifacts"]
    for path, original in files.items():
        assert (path.read_bytes(), path.stat().st_mtime_ns) == original
    with SqliteSaver.from_conn_string(str(directory / "checkpoints.sqlite")) as saver:
        private = _private_planner_checkpoint(
            saver, {"configurable": {"thread_id": "resume-demo"}},
        )
        assert private["planning_pass"] == 2 and private["pending_requirement_id"] is None
    final_plan = next(ref for ref in reversed(after["artifacts"]) if ref["kind"] == "task_plan")
    assert read_json_artifact(
        directory, ArtifactRef.model_validate(final_plan),
    )["tasks"]["synthetic-double"]["feasibility"] == "feasible"


@pytest.mark.parametrize("supervisor_mode", ["scripted", "llm"])
@pytest.mark.parametrize("coding_mode", ["scripted", "openhands"])
def test_cli_modes_freeze_independent_models_and_resume(tmp_path, monkeypatch, capsys,
                                                       supervisor_mode, coding_mode):
    from native_agent_fixtures import ScriptedChatModel, call, current_view

    from test_compilation_coding import FakeRunner
    import medbenchagent.compilation.coding as coding
    from medbenchagent.compilation.__main__ import main
    from medbenchagent.compilation.demo import DERIVED_OUTPUT, OVERVIEW_OUTPUT
    from medbenchagent.compilation.coding_worker import CodingCandidate
    import medbenchagent.runtime.native as transport
    import medbenchagent.runtime.model_services as services
    import medbenchagent.settings as settings

    monkeypatch.setattr(settings, "load_project_env", lambda: None)
    for name in (*settings._ENVIRONMENT_OVERRIDES, "MEDBENCH_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEDBENCH_LLM_API_KEY", "supervisor-secret")
    monkeypatch.setenv("MEDBENCH_CODING_LLM_API_KEY", "coding-secret")
    configuration = tmp_path / "models.toml"
    configuration.write_text(
        '[llm]\nmodel="supervisor-model"\nbase_url="http://supervisor.invalid/v1"\n'
        '[coding.llm]\nmodel="coding-model"\nbase_url="http://coding.invalid/v1"\n', encoding="utf-8",
    )
    discoveries = []

    def discover(kind, url, model, key):
        discoveries.append(kind)
        assert key == ("coding-secret" if kind == "Coding" else "supervisor-secret")
        return model

    monkeypatch.setattr(services, "resolve_model_service", discover)

    def perform(payload):
        assert payload["settings"]["llm"]["model"] == "coding-model"
        assert payload["settings"]["llm"]["base_url"] == "http://coding.invalid/v1"
        inputs = {item["kind"]: Path(item["read_path"]) for item in payload["inputs"]}
        reader = FileDatasetReader(inputs["dataset"])
        manifest = reader.get_manifest()
        dataset = {"dataset_id": manifest.dataset_id, "revision": manifest.revision}
        if "data_requirement" in inputs:
            requirement = json.loads(inputs["data_requirement"].read_text(encoding="utf-8"))
            data = {"dataset": dataset, "requirement_id": requirement["requirement_id"], "rows": [
                {"case_id": case_id, "doubled_value": reader.get_evidence(case_id, "value").value * 2,
                 "source_evidence_ids": ["value"]} for case_id in requirement["case_ids"]
            ]}
            kind = "derived_view"
        else:
            data = {"dataset": dataset, "case_ids": [case.case_id for case in reader.list_cases().items],
                    "fields": ["value", "view"]}
            kind = "dataset_overview"
        (Path(payload["output_directory"]) / "view.json").write_text(json.dumps(data), encoding="utf-8")
        return CodingCandidate(request_id=payload["request"]["request_id"], outcome="completed",
                               summary="Synthetic data prepared", files=[{"path": "view.json", "kind": kind}])

    runner = FakeRunner(perform)
    monkeypatch.setattr(coding, "run_local_worker", runner)

    def respond(messages, index):
        view = current_view(messages)
        refs = view["available_artifacts"]
        kinds = {ref["kind"] for ref in refs}
        last = view["last_result"]
        if len(view["history"]) == 1 and "dataset_overview" in kinds and not view["user_answers"]:
            decision = {"action": "ask_user", "question": "Continue the synthetic exercise?"}
        elif last and not last["needs"] and "task_plan" in kinds and "derived_view" in kinds:
            decision = {"action": "finish", "summary": "Synthetic exercise complete"}
        else:
            needs_coding = "dataset_overview" not in kinds or bool(last and last["needs"])
            decision = {
                "action": "delegate", "agent_id": "coding" if needs_coding else "task_planner",
                "objective": "Inspect or prepare the requested data" if needs_coding else "Design the task",
                "input_artifact_ids": [ref["artifact_id"] for ref in refs],
                "constraints": [DERIVED_OUTPUT if "data_requirement" in kinds else OVERVIEW_OUTPUT],
                "workstream_id": (view["workstreams"][0]["workstream_id"]
                                  if needs_coding and view["workstreams"] else None),
            }
        return call("SupervisorResponse", {"decision": decision}, index)

    def model(settings, limits):
        assert settings.model == "supervisor-model"
        return ScriptedChatModel(responder=respond)
    monkeypatch.setattr(transport, "chat_model", model)
    directory = tmp_path / "run"
    common = ["--directory", str(directory), "--compilation-id", "modes"]
    arguments = ["run", *common, "--supervisor", supervisor_mode, "--coding", coding_mode]
    if supervisor_mode == "llm" or coding_mode == "openhands":
        arguments.extend(["--config", str(configuration)])
    if supervisor_mode == "scripted":
        arguments.append("--pause-for-input")
    main(arguments)
    before = json.loads(capsys.readouterr().out)
    assert before["status"] == "waiting"
    saved = (directory / "run.json").read_text(encoding="utf-8")
    assert "supervisor-secret" not in saved and "coding-secret" not in saved
    assert discoveries == (["Supervisor"] if supervisor_mode == "llm" else []) + (
        ["Coding"] if coding_mode == "openhands" else [])
    configuration.write_text("not valid TOML anymore", encoding="utf-8")
    monkeypatch.setenv("MEDBENCH_LLM_MODEL", "changed")
    monkeypatch.setenv("MEDBENCH_LLM_BASE_URL", "http://changed.invalid/v1")
    monkeypatch.setattr(services, "resolve_model_service", lambda *_args: pytest.fail("Resume must not discover models"))
    main(["resume", *common, "--answer", "Continue"])
    after = json.loads(capsys.readouterr().out)
    assert after["status"] == "completed"
    assert after["supervisor_mode"] == supervisor_mode and after["coding_mode"] == coding_mode
    assert after["planner_mode"] == "scripted"
    assert after["supervisor_model"] == ("supervisor-model" if supervisor_mode == "llm" else None)
    assert after["coding_model"] == ("coding-model" if coding_mode == "openhands" else None)
    assert (directory / "run.json").read_text(encoding="utf-8") == saved
    assert after["workstreams"] == before["workstreams"] and len(after["workstreams"]) == 1
    assert all(row["outcome"] != "failed" for row in after["history"])
    assert len(runner.payloads) == (2 if coding_mode == "openhands" else 0)
    with pytest.raises(SystemExit) as rejected:
        main(["resume", *common, "--coding", "scripted"])
    assert rejected.value.code == 2
