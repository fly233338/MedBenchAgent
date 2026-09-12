import json

import pytest

from medbenchagent.runtime.dependencies import publish, require, snapshot_dataset, verify_resume
from medbenchagent.data.dataset import prepare_annotation_dataset


def test_upstream_change_invalidates_transitive_delivery_without_deleting_old_versions(tmp_path):
    context = {"run_directory": str(tmp_path)}
    for kind in (
        "DataConfig",
        "Requirement",
        "Materials",
        "PlanningConfig",
        "TemplateConfig",
        "GenerationConfig",
        "AuditConfig",
    ):
        publish(context, kind, {"value": 1}, ())
    for kind in ("Dataset", "Plan", "Templates", "Generation", "Audit"):
        publish(context, kind, {"path": f"{kind}.json"})
    require(context, "Audit")
    old = context["artifact_versions"]["Audit"]
    publish(context, "Dataset", {"path": "new.json"})
    with pytest.raises(ValueError, match="stale"):
        require(context, "Audit")
    index = json.loads((tmp_path / ".runtime/artifacts.json").read_text())
    assert old in index["versions"]


def test_new_run_snapshots_same_path_again_and_keeps_old_input(tmp_path):
    from PIL import Image

    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (2, 2)).save(source / "im.png")
    row = {
        entity: {f"{entity}_id": None, "annotation": {}} for entity in ("patient", "case", "study")
    }
    row["image"] = {
        "image_id": "im",
        "image_path": "im.png",
        "modality": "CT",
        "anatomy": None,
        "annotation": {},
    }
    annotation = source / "annotation.jsonl"
    annotation.write_text(json.dumps(row) + "\n")
    (source / "image_manifest.jsonl").write_text(
        json.dumps({"image_id": "im", "png_path": "im.png", "width": 2, "height": 2}) + "\n"
    )
    first = snapshot_dataset(annotation, tmp_path / "run1/input")
    row["image"]["modality"] = "MRI"
    annotation.write_text(json.dumps(row) + "\n")
    second = snapshot_dataset(annotation, tmp_path / "run2/input")
    a, _ = prepare_annotation_dataset(first, tmp_path / "run1/cache")
    b, _ = prepare_annotation_dataset(second, tmp_path / "run2/cache")
    assert a["fields"]["image.modality"]["top_values"] == ["CT"]
    assert b["fields"]["image.modality"]["top_values"] == ["MRI"]
    assert snapshot_dataset(annotation, first.parent).read_bytes() == first.read_bytes()


def test_resume_rejects_obsolete_contract(tmp_path):
    with pytest.raises(ValueError, match="start a new run"):
        verify_resume({"run_directory": str(tmp_path)})


def test_nested_specialist_interrupt_resumes_its_checkpoint(tmp_path, monkeypatch):
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.types import Command
    from medbenchagent.runtime.harness import compile_agent, result

    parent_actions = iter(["delegate", "finish"])
    child_actions = iter(["compute", "ask_user", "finish"])
    calls = []

    def decide(instructions, prompt):
        tool = next(child_actions if instructions.startswith("child") else parent_actions)
        return {
            "tool": tool,
            "arguments": {"question": "Which coordinate convention?", "reference": "mapping:1"}
            if tool == "ask_user"
            else {},
            "reason": "Needed",
        }

    monkeypatch.setattr("medbenchagent.runtime.harness.complete_json", decide)

    def build(connection):
        def compute(context):
            calls.append(1)
            return result("success", "Inspected")

        child = compile_agent(
            "Child",
            "child",
            {"compute": compute},
            lambda c: {},
            lambda c: result("complete", "Prepared"),
        )

        def delegate(context):
            output = child.invoke({"context": context})
            return result("success", "Child complete", decisions=output["context"]["decisions"])

        return compile_agent(
            "Parent",
            "parent",
            {"delegate": delegate},
            lambda c: {},
            lambda c: result("complete", "Done"),
            checkpointer=SqliteSaver(connection),
        )

    context = {
        "run_id": "test",
        "operation_directory": str(tmp_path / "ops"),
        "construction_events_path": str(tmp_path / "events.jsonl"),
    }
    config = {"configurable": {"thread_id": "nested"}}
    with sqlite3.connect(tmp_path / "checkpoints.sqlite", check_same_thread=False) as connection:
        initial = build(connection).invoke({"context": context}, config)
        assert initial["__interrupt__"][0].value["role"] == "Child"
    with sqlite3.connect(tmp_path / "checkpoints.sqlite", check_same_thread=False) as connection:
        resumed = build(connection).invoke(Command(resume="xyxy pixels"), config)
    assert resumed["result"]["status"] == "complete"
    assert calls == [1]
    assert list(resumed["context"]["decisions"].values()) == ["xyxy pixels"]
