import json
from pathlib import Path

import pytest
from openpyxl import Workbook
from PIL import Image

from medbenchagent.agents import source_inventory_worker
from medbenchagent.agents.coding_worker import CodingResult, CodingTask, OpenHandsCodingWorker
from medbenchagent.data.contracts import Source
from medbenchagent.data.source_report import validate_source_report
from medbenchagent.data.sources import inventory_source
from medbenchagent.runtime.dependencies import publish, require
from medbenchagent.runtime.openhands_worker import replay_source_script


def breast_source(tmp_path):
    source_root = tmp_path / "raw"
    images = source_root / "images"
    images.mkdir(parents=True)
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["CaseID", "Image_filename", "Mask_tumor_filename", "Classification"])
    rows = []
    for number in range(1, 257):
        image = f"case{number:03}.png"
        mask = f"case{number:03}_tumor.png" if number % 2 else ""
        label = "benign" if number <= 128 else "malignant"
        sheet.append([str(number), image, mask, label])
        Image.new("L", (2, 2)).save(images / image)
        if mask:
            Image.new("L", (2, 2)).save(images / mask)
        rows.append({"source_id": "original", "row_id": str(number), "image_id": image,
                     "image_path": f"images/{image}",
                     "mask_paths": [f"images/{mask}"] if mask else [], "classification": label})
    workbook.save(source_root / "clinical.xlsx")
    source = Source(id="original", root=str(source_root))
    manifest = inventory_source(source, tmp_path / "bundle" / "source_manifests" / source.id)
    context = {"run_id": "run", "run_directory": str(tmp_path / "run"),
               "operation_directory": str(tmp_path / "run" / "op"),
               "construction_events_path": str(tmp_path / "run" / "events.jsonl"),
               "dataset_scope": "breast", "source_scope": source.id,
               "source_root": str(source_root), "source_manifest": str(manifest),
               "active_source": source.model_dump(), "sources": [source.model_dump()],
               "bundle_root": str(tmp_path / "bundle"),
               "coding_worker_backend": "openhands", "coding_worker_max_iterations": 5,
               "coding_worker_timeout_seconds": 30, "coding_worker_max_revisions": 3}
    publish(context, "Source", {"manifest": str(manifest)}, ())
    report = {"schema_version": "source-report.v1",
              "tables": [{"path": "clinical.xlsx", "rows": 256, "role": "clinical annotation"}],
              "population": {"kind": "table", "table_path": "clinical.xlsx", "row_id_field": "CaseID",
                             "image_id_field": "Image_filename", "image_path_field": "Image_filename",
                             "image_prefix": "images", "mask_fields": ["Mask_tumor_filename"],
                             "mask_prefix": "images", "classification_field": "Classification"},
              "record_count": 256, "unique_images": 256, "mask_references": 128,
              "classification_counts": {"benign": 128, "malignant": 128}, "uncertainties": []}
    workspace = tmp_path / "worker"
    workspace.mkdir()
    report_path = workspace / "source_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    population_path = workspace / "image_population.jsonl"
    population_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    script_path = workspace / "inspect_source.py"
    script_path.write_text("# reproducible source inventory\n", encoding="utf-8")
    return context, report_path, population_path, script_path


def test_breast_report_is_independently_verified_and_freezes_fingerprint(tmp_path, monkeypatch):
    context, report, population, script = breast_source(tmp_path)
    checked = validate_source_report(context, report, population, tmp_path / "verified")
    assert checked["rows"] == checked["unique_images"] == 256
    assert checked["mask_references"] == 128
    assert checked["classification_counts"] == {"benign": 128, "malignant": 128}

    class Worker:
        calls = []

        def run(self, task):
            self.calls.append(task)
            return CodingResult(status="complete", conversation_id=task.conversation_id or "00000000-0000-0000-0000-000000000001",
                                outputs={"source_report.json": str(report), "image_population.jsonl": str(population),
                                         "inspect_source.py": str(script)})

    worker = Worker()
    monkeypatch.setattr(source_inventory_worker, "coding_worker", lambda _: worker)
    result = source_inventory_worker.build_source_inventory(context)
    assert result["status"] == "success"
    context.update(result["updates"])
    assert require(context, "SourceFingerprint")
    assert context["source_population_reference"]
    assert len(worker.calls) == 1
    from medbenchagent.agents.preparation import bound

    Path(context["source_report_path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="content changed"):
        bound(context, "SourceFingerprint")


def test_pipeline_runs_one_inventory_task_without_model_inspect_loop(tmp_path, monkeypatch):
    from medbenchagent.agents.data_pipeline import compile_pipeline

    context, report, population, script = breast_source(tmp_path)
    context["limits"] = {"max_calls": 1}
    calls = []

    class Worker:
        def run(self, task):
            calls.append(task.kind)
            return CodingResult(status="complete", conversation_id="00000000-0000-0000-0000-000000000001",
                                outputs={"source_report.json": str(report),
                                         "image_population.jsonl": str(population),
                                         "inspect_source.py": str(script)})

    monkeypatch.setattr(source_inventory_worker, "coding_worker", lambda _: Worker())
    monkeypatch.setattr("medbenchagent.runtime.harness.complete_validated_json",
                        lambda *args, **kwargs: pytest.fail("Inventory should not need another model decision"))
    state = compile_pipeline().invoke({"context": context}, {"recursion_limit": 8})
    assert calls == ["source_inventory"]
    assert require(state["context"], "SourceFingerprint")


@pytest.mark.parametrize("change,match", [
    ("count", "counts differ"), ("row", "differs from source"),
    ("extra", "extra rows"), ("path", "does not exist"),
])
def test_false_worker_report_cannot_publish_population(tmp_path, change, match):
    context, report_path, population_path, _ = breast_source(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = population_path.read_text(encoding="utf-8").splitlines()
    if change == "count":
        report["unique_images"] = 255
        report_path.write_text(json.dumps(report), encoding="utf-8")
    elif change == "row":
        row = json.loads(rows[3])
        row["classification"] = "malignant"
        rows[3] = json.dumps(row)
        population_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    elif change == "extra":
        population_path.write_text("\n".join(rows + [rows[-1]]) + "\n", encoding="utf-8")
    else:
        row = json.loads(rows[3])
        row["image_path"] = "missing.png"
        rows[3] = json.dumps(row)
        population_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        match = "differs from source"
    with pytest.raises(ValueError, match=match):
        validate_source_report(context, report_path, population_path, tmp_path / "verified")
    assert not (Path(context["bundle_root"]) / "preparation" / "original").exists()


def test_source_change_invalidates_worker_report(tmp_path):
    context, report, population, _ = breast_source(tmp_path)
    source_file = Path(context["source_root"]) / "clinical.xlsx"
    source_file.write_bytes(source_file.read_bytes() + b"x")
    with pytest.raises(ValueError, match="Source changed"):
        validate_source_report(context, report, population, tmp_path / "verified")


def test_worker_repair_uses_new_conversation_and_readonly_source_mount(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "records.csv").write_text("id,path\n1,image.png\n", encoding="utf-8")
    manifest = inventory_source(Source(id="s", root=str(raw)), tmp_path / "manifest")
    workspace = tmp_path / "worker"
    seen = []

    def run(command, **kwargs):
        from types import SimpleNamespace
        if command[1:3] == ["image", "inspect"] or "-c" in command:
            return SimpleNamespace(returncode=0)
        if command[1:3] == ["container", "inspect"]:
            return SimpleNamespace(returncode=1)
        if command[1] in {"logs", "rm"}:
            return SimpleNamespace(returncode=0)
        if command[1] == "wait":
            return SimpleNamespace(returncode=0, stdout="0\n")
        request = json.loads((workspace / "state" / "coding_task.json").read_text(encoding="utf-8"))
        seen.append((request, command))
        (workspace / "outputs" / "adapter.py").write_text("print('candidate')\n", encoding="utf-8")
        (workspace / "state" / "coding_result_dataset_adapter.json").write_text(json.dumps({
            "status": "complete", "conversation_id": request["conversation_id"],
            "outputs": {"adapter.py": "/workspace/outputs/adapter.py"}}), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("medbenchagent.agents.coding_worker.subprocess.run", run)
    task = CodingTask(task_id="run:s:adapter", kind="dataset_adapter", source_root=str(raw),
                      source_manifest=str(manifest), workspace=str(workspace),
                      objective="Write adapter", required_outputs=["adapter.py"], model="openai/test",
                      base_url="http://localhost")
    worker = OpenHandsCodingWorker()
    first = worker.run(task)
    cached = worker.run(task)
    assert cached.status == "complete" and len(seen) == 1
    second = worker.run(task.model_copy(update={"attempt": 2, "feedback": "Repair validation"}))
    assert first.status == second.status == "complete"
    assert first.conversation_id != second.conversation_id
    assert len(seen) == 2
    assert not (workspace / "source_snapshot").exists()
    assert any(str(raw) in item for item in seen[0][1])
    assert any("dst=/workspace/source,readonly" in item for item in seen[0][1])
    assert "source_manifest" not in seen[0][0]


def test_special_population_requires_existing_review_path(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "annotations.json").write_text('{"records": [{"id": "a"}]}', encoding="utf-8")
    Image.new("L", (2, 2)).save(raw / "image.png")
    Image.new("L", (2, 2)).save(raw / "mask.png")
    manifest = inventory_source(Source(id="s", root=str(raw)), tmp_path / "manifest")
    context = {"source_root": str(raw), "source_manifest": str(manifest), "source_scope": "s"}
    report = {"schema_version": "source-report.v1", "tables": [],
              "population": {"kind": "special", "source_files": ["annotations.json"],
                             "required_operation": "nested_array_join",
                             "reason": "The image identity requires joining nested records"},
              "record_count": 0, "unique_images": 1, "mask_references": 1,
              "classification_counts": {}, "uncertainties": []}
    report_path = tmp_path / "source_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    population = tmp_path / "image_population.jsonl"
    population.write_text(json.dumps({"source_id": "s", "image_id": "a", "image_path": "image.png",
                                      "mask_paths": ["mask.png"]}) + "\n", encoding="utf-8")
    verified = validate_source_report(context, report_path, population, tmp_path / "verified")
    assert verified["assurance"] == "human_review_required"
    assert verified["unique_images"] == 1


def test_direct_json_population_is_independently_enumerated(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("a.png", "b.png", "a_mask.png"):
        Image.new("L", (2, 2)).save(raw / name)
    objects = {"records": [
        {"id": "a", "image": "a.png", "masks": ["a_mask.png"], "class": "benign"},
        {"id": "b", "image": "b.png", "masks": [], "class": "malignant"},
    ]}
    (raw / "records.json").write_text(json.dumps(objects), encoding="utf-8")
    manifest = inventory_source(Source(id="s", root=str(raw)), tmp_path / "manifest")
    context = {"source_root": str(raw), "source_manifest": str(manifest), "source_scope": "s"}
    report = {"schema_version": "source-report.v1", "tables": [],
              "population": {"kind": "json", "path": "records.json", "collection": "/records",
                             "row_id_field": "id", "image_id_field": "id", "image_path_field": "image",
                             "mask_fields": ["masks"], "classification_field": "class"},
              "record_count": 2, "unique_images": 2, "mask_references": 1,
              "classification_counts": {"benign": 1, "malignant": 1}, "uncertainties": []}
    report_path = tmp_path / "source_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    population_path = tmp_path / "image_population.jsonl"
    population_path.write_text("".join(json.dumps(row) + "\n" for row in [
        {"source_id": "s", "row_id": "a", "image_id": "a", "image_path": "a.png",
         "mask_paths": ["a_mask.png"], "classification": "benign"},
        {"source_id": "s", "row_id": "b", "image_id": "b", "image_path": "b.png",
         "mask_paths": [], "classification": "malignant"},
    ]), encoding="utf-8")
    verified = validate_source_report(context, report_path, population_path, tmp_path / "verified")
    assert verified["assurance"] == "independent"
    assert verified["unique_images"] == 2 and verified["mask_references"] == 1


def test_adapter_worker_only_publishes_candidate_code_after_spec(tmp_path, monkeypatch):
    from medbenchagent.agents import adapter_worker, data_pipeline
    from test_preparation_evidence import grounded_spec, setup_source

    context = setup_source(tmp_path, monkeypatch)
    context.update(run_id="run", construction_events_path=str(tmp_path / "events.jsonl"),
                   coding_worker_max_iterations=5, coding_worker_timeout_seconds=30,
                   coding_worker_max_revisions=2)
    context.update(data_pipeline.save_semantic_spec(context, spec=grounded_spec(context))["updates"])
    calls = []

    class Worker:
        def run(self, task):
            calls.append(task.kind)
            path = Path(task.workspace) / "adapter.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("pass\n", encoding="utf-8")
            return CodingResult(status="complete", conversation_id="00000000-0000-0000-0000-000000000001",
                                outputs={"adapter.py": str(path)})

    monkeypatch.setattr(adapter_worker, "coding_worker", lambda _: Worker())
    outcome = adapter_worker.develop_dataset_adapter(context)
    assert outcome["status"] == "success"
    context.update(outcome["updates"])
    assert require(context, "DatasetAdapter")
    assert calls == ["dataset_adapter"]


def test_inspection_script_must_replay_identically(tmp_path):
    workspace = tmp_path / "worker"
    for name in ("source", "outputs", "state", "work"):
        (workspace / name).mkdir(parents=True)
    (workspace / "source" / "count.txt").write_text("2", encoding="utf-8")
    script = workspace / "outputs" / "inspect_source.py"
    script.write_text(
        "import os\nfrom pathlib import Path\n"
        "root = Path(os.environ['MEDBENCH_SOURCE_ROOT'])\n"
        "out = Path(os.environ['MEDBENCH_WORK_DIR'])\n"
        "(out / 'source_report.json').write_text(root.joinpath('count.txt').read_text())\n"
        "(out / 'image_population.jsonl').write_text('image\\n')\n",
        encoding="utf-8",
    )
    request = {"task_id": "run:source:inventory"}
    assert replay_source_script(request, workspace) == ""
    assert (workspace / "outputs" / "source_report.json").read_text(encoding="utf-8") == "2"
    assert json.loads((workspace / "state" / "source_inventory_replayed.json").read_text())["task_id"] == request["task_id"]
    script.write_text(
        "import os, uuid\nfrom pathlib import Path\n"
        "out = Path(os.environ['MEDBENCH_WORK_DIR'])\n"
        "(out / 'source_report.json').write_text(str(uuid.uuid4()))\n"
        "(out / 'image_population.jsonl').write_text('image\\n')\n",
        encoding="utf-8",
    )
    assert "different outputs" in replay_source_script(request, workspace)
