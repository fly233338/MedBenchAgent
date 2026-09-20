"""Global finalization, quarantine, accounting and atomic publication."""

import json
from pathlib import Path
import pytest

from medbenchagent.agents import finalization as agent
from medbenchagent.audit.hard import read_json, read_rows, write_rows
from medbenchagent.runtime.dependencies import publish, require
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.planning.protocol import template_specification
from test_dataset_cards import prepared_dataset
from test_generation import _task, _grounding


def export_context(tmp_path, monkeypatch, datasets=("a",)):
    root = tmp_path / "run"
    root.mkdir()
    c = {"run_id": "test", "run_directory": str(root), "operation_directory": str(root / "op"),
         "construction_events_path": str(root / "events.jsonl"), "benchmark_info": {"sample_count": len(datasets) + 1, "language": "en"},
         "selected_dataset_ids": list(datasets), "dataset_task_plans": {}, "dataset_templates": {},
         "dataset_assets": {}, "dataset_generations": {}, "seed": 7, "limits": {"max_calls": 30}}
    for kind in ("DataConfig", "TemplateConfig", "GenerationConfig", "FinalizationConfig", "DatasetSelection"):
        publish(c, kind, {}, ())
    counts = {}
    for dataset in datasets:
        entry = prepared_dataset(tmp_path, dataset, monkeypatch)
        c["dataset_assets"][dataset] = {"entry": entry}
        leaf = _task("letter", ["image.annotation.status"])
        leaf.pop("question_templates")
        leaf.update(eligible_count=2, concept_id="core", mapping_grounding=_grounding(leaf["protocol"]))
        plan = {"dataset": {"dataset_id": dataset}, "benchmark_info": c["benchmark_info"], "tasks": {"leaf": leaf},
                "concepts": {"core": {"origin": "core", "description": "Status", "category": "test"}}}
        plan_path = root / f"{dataset}-plan.json"
        atomic_write_json(plan_path, plan)
        c["dataset_task_plans"][dataset] = str(plan_path)
        template_path = root / f"{dataset}-templates.json"
        atomic_write_json(template_path, {"templates": {"leaf": ["Status?", "Which status?", "Identify status."]},
            "reviews": {"leaf": {"specification": template_specification(leaf["protocol"], "en").model_dump(),
                                  "review": {"verdict": "approved", "reason": "Sound"}}}})
        c["dataset_templates"][dataset] = str(template_path)
        image = next(read_rows(entry["image_manifest_path"]))
        row = {"id": "old", "task_id": "leaf", "dataset": dataset, "image_id": image["image_id"],
               "image_path": image["png_path"], "question": "Status?", "options": ["benign", "malignant"], "answer": "A"}
        raw_path = root / f"{dataset}-raw.jsonl"
        write_rows(raw_path, [row])
        c["dataset_generations"][dataset] = {"raw_output_path": str(raw_path)}
        counts[dataset] = {"leaf": 1}
        publish(c, f"dataset/{dataset}/Dataset", {"annotation": entry["annotation_path"]}, ("DataConfig",))
        publish(c, f"dataset/{dataset}/Plan", {"path": str(plan_path)}, (f"dataset/{dataset}/Dataset",))
        publish(c, f"dataset/{dataset}/Templates", {"path": str(template_path)}, (f"dataset/{dataset}/Plan", "TemplateConfig"))
    allocation = {"counts": counts, "requested_count": len(datasets) + 1, "allocated_count": len(datasets), "planning_shortfall": 1}
    c["allocation_path"] = str(root / "allocation.json")
    atomic_write_json(c["allocation_path"], allocation)
    publish(c, "Allocation", {"path": c["allocation_path"]}, tuple(f"dataset/{d}/Templates" for d in datasets))
    for dataset in datasets:
        republish_raw(c, dataset)
    return c


def republish_raw(c, dataset="a"):
    publish(c, f"dataset/{dataset}/Generation", {"raw": c["dataset_generations"][dataset]["raw_output_path"]},
            ("Allocation", f"dataset/{dataset}/Plan", f"dataset/{dataset}/Templates"))


def call(c, function, **arguments):
    c["_calls"] = c.get("_calls", 0) + 1
    c["operation_directory"] = str(Path(c["run_directory"]) / f"op-{c['_calls']}")
    outcome = function(c, **arguments)
    c.update(outcome["updates"])
    return outcome


def validate(c):
    for dataset in c["selected_dataset_ids"]:
        if sum(read_json(c["allocation_path"])["counts"][dataset].values()):
            call(c, agent.inspect_dataset, dataset_id=dataset)
    call(c, agent.assemble_candidates)
    call(c, agent.balance_options)
    return call(c, agent.validate_delivery)


def test_publish_shortfall_and_replay_without_modifying_delivery(tmp_path, monkeypatch):
    c = export_context(tmp_path, monkeypatch, ("a", "b"))
    assert validate(c)["status"] == "success"
    call(c, agent.publish_delivery)
    assert c["item_count"] == 2 and c["total_shortfall"] == 1
    root = Path(c["delivery_path"])
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    assert set(before) == {"benchmark.jsonl", "benchmark_manifest.json", "audit_report.json", "quarantine.jsonl", "complete.json"}
    manifest = read_json(c["manifest_path"])
    assert manifest["schema_version"] == "benchmark-manifest.v4"
    assert set(manifest["raw_source_paths"]) == {"a", "b"}
    assert set(manifest["tasks"]) == {"a::leaf", "b::leaf"}
    assert all((root / row["image_path"]).is_file() for row in read_rows(root / "benchmark.jsonl"))
    agent.restore_delivery(c)
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}
    assert agent.finish(c)["status"] == "complete"


@pytest.mark.parametrize("bad", [{"dataset": "wrong"}, {"task_id": "missing"}, {"image_id": "missing"}, {"answer": "B"}, {"question": "Leaks benign"}])
def test_invalid_raw_is_quarantined_without_blocking_valid_items(tmp_path, monkeypatch, bad):
    c = export_context(tmp_path, monkeypatch)
    path = Path(c["dataset_generations"]["a"]["raw_output_path"])
    row = next(read_rows(path))
    write_rows(path, [row, {**row, **bad}])
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{broken JSON\n")
    republish_raw(c)
    assert validate(c)["status"] == "success"
    call(c, agent.publish_delivery)
    report = read_json(c["audit_artifacts"]["report"])
    assert report["counts"]["generated"] == 3
    assert report["counts"]["audit_rejected"] == 2
    assert report["counts"]["final"] == 1
    rejected = list(read_rows(c["audit_artifacts"]["quarantine"]))
    assert len(rejected) == 2 and rejected[1]["source"]["line"] == 3
    assert path.read_text(encoding="utf-8").endswith("{broken JSON\n")


@pytest.mark.parametrize("conflicting", [False, True])
def test_identity_duplicates_and_conflicts(tmp_path, monkeypatch, conflicting):
    c = export_context(tmp_path, monkeypatch)
    path = c["dataset_generations"]["a"]["raw_output_path"]
    row = next(read_rows(path))
    write_rows(path, [row, {**row, "id": "second", "question": "Which status?" if conflicting else "Status?"}])
    republish_raw(c)
    outcome = validate(c)
    report = read_json(Path(c["assembly_directory"]) / "audit_report.json")
    assert report["counts"]["audit_rejected"] == 0
    assert report["counts"]["assembly_removed"] == (2 if conflicting else 1)
    if conflicting:
        assert outcome["status"] == "blocked" and "No deliverable" in outcome["summary"]
        assert not (Path(c["run_directory"]) / "delivery").exists()
    else:
        assert outcome["status"] == "success"


def test_excess_is_reported_and_balancing_preserves_meaning(tmp_path, monkeypatch):
    c = export_context(tmp_path, monkeypatch)
    path = c["dataset_generations"]["a"]["raw_output_path"]
    first = next(read_rows(path))
    images = list(read_rows(c["dataset_assets"]["a"]["entry"]["image_manifest_path"]))
    second = {**first, "id": "two", "image_id": images[1]["image_id"], "answer": "B"}
    write_rows(path, [first, second])
    republish_raw(c)
    assert validate(c)["status"] == "success"
    call(c, agent.publish_delivery)
    report = read_json(c["audit_artifacts"]["report"])
    assert report["counts"]["generation_excess"] == 1
    assert report["counts"]["assembly_removed"] == 1
    assert report["rejection_reason_counts"]["allocation_excess"] == 1
    row = next(read_rows(c["audit_artifacts"]["audited"]))
    original = first if row["image_id"] == first["image_id"] else second
    assert row["options"][ord(row["answer"]) - 65] == original["options"][ord(original["answer"]) - 65]


def test_missing_generation_and_template_changes_block_finalization(tmp_path, monkeypatch):
    c = export_context(tmp_path, monkeypatch, ("a", "b"))
    old = c["artifact_versions"].pop("dataset/b/Generation")
    with pytest.raises(ValueError, match="Generation"):
        agent.initialize(c)
    c["artifact_versions"]["dataset/b/Generation"] = old
    publish(c, "dataset/a/Templates", {"changed": True}, ("dataset/a/Plan", "TemplateConfig"))
    with pytest.raises(ValueError, match="stale"):
        agent.initialize(c)


def test_final_package_tampering_cannot_be_published(tmp_path, monkeypatch):
    c = export_context(tmp_path, monkeypatch)
    assert validate(c)["status"] == "success"
    path = Path(c["assembly_directory"]) / "package/benchmark.jsonl"
    row = next(read_rows(path))
    write_rows(path, [{**row, "answer": "Z"}])
    with pytest.raises(ValueError, match="files changed"):
        agent.publish_delivery(c)
    assert not (Path(c["run_directory"]) / "delivery").exists()


@pytest.mark.parametrize("after_rename", [False, True])
def test_publication_failure_and_resume(tmp_path, monkeypatch, after_rename):
    from medbenchagent.runtime import run_export
    c = export_context(tmp_path, monkeypatch)
    assert validate(c)["status"] == "success"
    before = dict(c)
    if after_rename:
        original = agent.restore_delivery
        monkeypatch.setattr(agent, "restore_delivery", lambda _: (_ for _ in ()).throw(RuntimeError("after rename")))
    else:
        original = run_export.atomic_write_json
        monkeypatch.setattr(run_export, "atomic_write_json", lambda *a: (_ for _ in ()).throw(RuntimeError("before rename")))
    with pytest.raises(RuntimeError):
        agent.publish_delivery(c)
    assert (Path(c["run_directory"]) / "delivery").exists() == after_rename
    if after_rename:
        monkeypatch.setattr(agent, "restore_delivery", original)
        recovered = agent.initialize(before)
        require(recovered, "Delivery")
        assert agent.available_tools(recovered) == {"finish"}
    else:
        monkeypatch.setattr(run_export, "atomic_write_json", original)
        call(c, agent.publish_delivery)
    assert (Path(c["run_directory"]) / "delivery/complete.json").is_file()


def test_finalization_agent_uses_tools_and_reads_anomalies(tmp_path, monkeypatch):
    c = export_context(tmp_path, monkeypatch)
    actions = iter([("inspect_inputs", {}), ("inspect_dataset", {"dataset_id": "a"}),
                    ("read_issues", {"dataset_id": "a"}), ("assemble_candidates", {}),
                    ("balance_options", {}), ("validate_delivery", {}), ("publish_delivery", {}), ("finish", {})])
    def model(system, prompt):
        tool, arguments = next(actions)
        assert tool in json.loads(prompt)["tools"]
        return {"tool": tool, "arguments": arguments, "reason": "Inspect and finalize"}
    monkeypatch.setattr("medbenchagent.runtime.harness.complete_json", model)
    outcome = agent.compile_finalization_agent().invoke({"context": c}, {"recursion_limit": 50})
    assert outcome["result"]["status"] == "complete"
    assert outcome["context"]["item_count"] == 1
