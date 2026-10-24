import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from medbenchagent.runtime import sessions
from medbenchagent.runtime.construction import record_event
from medbenchagent.runtime.intake import InputDraft, SelectedFile, select_dataset, validate_draft
from medbenchagent.runtime.native import invocation_config
from medbenchagent.settings import load_supervisor_settings, load_coding_settings
from medbenchagent.data.task_data import (
    DatasetProfile,
    SourceManifest,
    seal_profile,
    snapshot_profile,
    ProfileReader,
    write_once,
)
from medbenchagent.data.dataset_catalog import register_dataset

from native_agent_fixtures import ScriptedChatModel


def test_native_usage_and_unavailable_usage_never_record_messages(tmp_path, monkeypatch):
    context = SimpleNamespace(
        runtime_directory=tmp_path,
        settings=load_supervisor_settings(),
        compilation_id="run",
        agent_id="review",
        invocation_id="review-one",
        request=SimpleNamespace(operation="review_plan", workstream_id=None),
    )

    def reply(_, number):
        return AIMessage(
            content="patient-body-never-log",
            usage_metadata={"input_tokens": 17, "output_tokens": 3, "total_tokens": 20}
            if number == 1
            else None,
        )

    model = ScriptedChatModel(responder=reply)
    for _ in range(2):
        model.invoke(
            [HumanMessage(content="private-prompt")],
            config=invocation_config({}, context),
        )
    text = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert "private-prompt" not in text and "patient-body-never-log" not in text
    rows = [json.loads(line) for line in text.splitlines()]
    starts = [row for row in rows if row["event_type"] == "model_started"]
    ends = [row for row in rows if row["event_type"] == "model_finished"]
    assert [row["model_call_id"] for row in starts] == [row["model_call_id"] for row in ends]
    assert ends[0]["input_tokens"] == 17 and ends[0]["output_tokens"] == 3
    assert not ends[1]["token_usage_available"] and "input_tokens" not in ends[1]
    original = Path.open

    def unavailable(path, *args, **kwargs):
        if path.name == "events.jsonl":
            raise OSError("read-only diagnostics")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", unavailable)
    assert (
        model.invoke([HumanMessage(content="work")], config=invocation_config({}, context)).content
        == "patient-body-never-log"
    )
    record_event(tmp_path / "events.jsonl", "ignored", value=object())


def test_profile_derived_members_survive_workspace_changes(tmp_path):
    runtime = tmp_path / "runtime"
    workspace = runtime / "coding/ws/workspace"
    workspace.mkdir(parents=True)
    derived = workspace / "measurements.csv"
    derived.write_bytes(b"id,value\n1,4\n")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "source.txt").write_bytes(b"original")
    profile = DatasetProfile(
        dataset={"dataset_id": "data", "revision": "r1"},
        sources=[
            {"source_id": "raw", "root": str(raw)},
            {"source_id": "derived", "root": str(workspace)},
        ],
        description="Measured evidence",
        inspected_scope="one row",
        observations=[
            {
                "observation_id": "one",
                "description": "Measured",
                "source_refs": [
                    {"source_id": "raw", "path": "source.txt"},
                    {"source_id": "derived", "path": "measurements.csv"},
                ],
            }
        ],
    )
    manifest = workspace / "profile.json"
    write_once(manifest, seal_profile(profile, workspace).model_dump_json().encode())
    output = snapshot_profile(
        manifest,
        tmp_path / "runs/test/artifacts/profile",
        runtime_directory=runtime,
    )
    derived.unlink()
    manifest.unlink()
    reader = ProfileReader(output)
    assert reader.profile.sources[0].root == str(raw)
    assert reader.profile.sources[1].root == "sources/derived"
    assert str(runtime) not in output.read_text(encoding="utf-8")


def test_named_run_freezes_one_dataset_and_copy_has_no_execution_state(tmp_path, monkeypatch):
    import medbenchagent.runtime.model_services as models

    monkeypatch.setattr(models, "resolve_model_service", lambda *_: "offline")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "labels.csv").write_bytes(b"id,label\n1,A\n")
    record = sessions.create_session(
        "first",
        root=tmp_path / "runs",
        runtime_root=tmp_path / "runtime",
    )
    draft = InputDraft(
        goal="Inspect",
        source_dirs=[str(raw)],
        key_files=[SelectedFile(path=str(raw / "labels.csv"))],
        goal_kind="profile",
    )
    options = sessions.freeze_session(record["directory"], draft)
    directory = Path(record["directory"])
    runtime = Path(options.runtime_directory)
    assert runtime == tmp_path / "runtime/first" and (runtime / "resources/coding-skills").is_dir()
    assert not (directory / "resources").exists() and (directory / "inputs/sources.json").is_file()
    with pytest.raises(ValueError, match="frozen"):
        sessions.freeze_session(directory, draft)
    copied = sessions.copy_session(
        directory,
        "second",
        root=tmp_path / "runs",
        runtime_root=tmp_path / "runtime",
    )
    assert copied["draft"] == sessions.load_session(directory)["draft"] and not copied["messages"]
    assert not (Path(copied["directory"]) / "run.json").exists()
    catalog = tmp_path / "catalog.json"
    source = SourceManifest.model_validate_json(Path(options.source_manifest).read_bytes())
    register_dataset(
        source.dataset.dataset_id,
        options.source_manifest,
        kind="source",
        catalog=catalog,
    )
    selected = select_dataset(source.dataset.dataset_id, draft, catalog=catalog)
    assert validate_draft(selected, Path(copied["directory"]))["source"].dataset == source.dataset
    selected.key_files[0].purpose = "new evidence definition"
    assert (
        validate_draft(selected, Path(copied["directory"]))["source"].dataset.revision
        != source.dataset.revision
    )


def test_coding_rejects_candidate_escape_and_overlapping_roots(tmp_path):
    from medbenchagent.agents.data_processing.coding.adapter import OpenHandsCodingAgent
    from medbenchagent.agents.data_processing.coding.skills import freeze_skills
    from medbenchagent.compilation.contracts import WorkRequest

    data, runtime = tmp_path / "data", tmp_path / "runtime"
    settings = load_coding_settings()
    settings = replace(settings, llm=replace(settings.llm, model="offline"))
    request = WorkRequest(
        compilation_id="run",
        request_id="one",
        agent_id="coding",
        operation="execute",
        workstream_id="ws",
        objective="Inspect",
    )

    def escape(payload):
        return {
            "request_id": "one",
            "outcome": "completed",
            "summary": "Candidate",
            "files": [{"path": "../foreign.txt", "kind": "report"}],
        }

    agent = OpenHandsCodingAgent(
        settings,
        instructions="Test",
        skills=freeze_skills(runtime, settings.skill_names),
        runner=escape,
    )
    with pytest.raises(ValueError):
        agent.execute_candidate(data, request, "Inspect", runtime_directory=runtime)
    with pytest.raises(ValueError, match="separate"):
        agent.execute_candidate(runtime, request, "Inspect", runtime_directory=runtime)


def test_dataset_revision_is_not_used_as_a_filesystem_path(tmp_path):
    from medbenchagent.runtime.intake import prepare_input, register_draft

    source = tmp_path / "raw"
    source.mkdir()
    original = SourceManifest(
        dataset={"dataset_id": "data", "revision": "../../outside"},
        sources=[{"source_id": "raw", "root": str(source)}],
    )
    manifest = tmp_path / "external.json"
    manifest.write_text(original.model_dump_json(), encoding="utf-8")
    output = tmp_path / "run"
    draft = InputDraft(
        goal="Inspect",
        dataset_id="data",
        source_dirs=[str(source)],
        source_manifest=str(manifest),
        goal_kind="profile",
    )
    path = prepare_input(draft, output)["source_manifest"]
    assert path.resolve().is_relative_to(output.resolve())
    assert SourceManifest.model_validate_json(path.read_bytes()).dataset == original.dataset
    entry = register_draft(output, draft, catalog=tmp_path / "catalog.json")
    assert Path(entry["manifest"]) == path and entry["revision"] == "../../outside"
