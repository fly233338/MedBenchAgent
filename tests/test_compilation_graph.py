"""Two focused acceptance tests for the scripted core, not autonomous model quality."""

import json
import os
from pathlib import Path
import subprocess
import sys

from langgraph.checkpoint.sqlite import SqliteSaver

from medbenchagent.compilation.artifacts import read_json_artifact, resolve_artifact
from medbenchagent.compilation.contracts import ArtifactRef
from medbenchagent.compilation.demo import (
    create_synthetic_inputs, demo_agents, scripted_supervisor,
)
from medbenchagent.compilation.graph import CompilationContext, build_compilation_graph
from medbenchagent.data.dataset import FileDatasetReader
from medbenchagent.planning.contracts import DataRequirement, SemanticTaskSpec


def _private_planner_checkpoint(saver, config):
    # Read namespaces supplied by LangGraph, without constructing one ourselves.
    return next(
        item.checkpoint["channel_values"] for item in saver.list(config)
        if "planning_pass" in item.checkpoint["channel_values"]
    )


def test_multi_round_collaboration_uses_evidence_and_isolates_state(tmp_path):
    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite")) as saver:
        graph = build_compilation_graph(demo_agents(), checkpointer=saver)
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
                context=CompilationContext(directory, scripted_supervisor(directory)),
                durability="sync",
            )
            assert output["status"] == "completed"
            trace = [entry["agent_id"] for entry in output["history"]]
            traces.append(trace)
            request_ids = [entry["request_id"] for entry in output["history"]]
            assert len(request_ids) == len(set(request_ids))
            assert all(identifier in item for item in request_ids)
            assert all(row["outcome"] != "failed" for row in output["history"])
            assert "planning_pass" not in output and "pending_requirement_id" not in output
            private = _private_planner_checkpoint(saver, config)
            assert private["planning_pass"] == 2
            assert private["pending_requirement_id"] is None

            refs = [ArtifactRef.model_validate(raw) for raw in output["artifacts"]]
            for ref in refs:
                assert resolve_artifact(directory, ref).is_file()
            plans = [
                SemanticTaskSpec.model_validate(read_json_artifact(directory, ref))
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
    assert before["supervisor_mode"] == before["agent_mode"] == "scripted"
    saved_options = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    assert saved_options["supervisor_mode"] == "scripted"
    assert saved_options["model_settings"] is None
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
    )["feasibility"] == "feasible"
