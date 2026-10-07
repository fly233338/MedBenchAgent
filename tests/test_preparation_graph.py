import json
import shutil
from pathlib import Path
from PIL import Image
from langgraph.checkpoint.memory import InMemorySaver
from medbenchagent.preparation.graph import compile_preparation_graph
from medbenchagent.preparation.contracts import Stage, StageResult, SemanticQuestion
from medbenchagent.data.sources import digest_file
from medbenchagent.data.models import ImageEntry


def mapping(filename="raw.jpg", target="images/view.jpg"):
    return {"population_policy": {"rule_id": "population", "unit_type": "case", "source_ids": ["s"],
        "identity_fields": ["id"], "inclusion": "all", "evidence_refs": ["s:annotation"]},
        "case_identity": {"rule_id": "id", "target_path": "case_id", "fields": ["id"], "evidence_refs": ["s:annotation"]},
        "grouping": {"rule_id": "group", "case_unit": "case", "key_fields": ["id"], "evidence_refs": ["s:annotation"]},
        "fields": [{"rule_id": "label", "source_path": "label", "disposition": "medical_evidence", "source_type": "string",
            "target_type": "string", "missingness": "unknown", "reason": "reference", "scope": "case", "evidence_refs": ["s:annotation"]}],
        "images": {target: {"source_id": "s", "source_path": filename, "source_format": "jpeg", "policy_id": "native"}},
        "field_semantics": {"label": "documented finding"}}


def state(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    Image.new("RGB", (4, 4), "red").save(raw / "raw.jpg")
    return {"run_id": "r", "dataset_id": "d", "run_directory": str(tmp_path / "runtime"),
        "preparation_root": str(tmp_path / "runtime/preparation"), "bundle_root": str(tmp_path / "prepared/d"),
        "sources": [{"id": "s", "root": str(raw)}], "source_roots": {"s": str(raw)},
        "source_manifests": {}, "source_snapshot_digest": "frozen"}


class Worker:
    def __init__(self, fail_image=False, question=False, revise=False):
        self.requests = []
        self.fail_image, self.question, self.revise = fail_image, question, revise

    def run(self, request):
        self.requests.append(request)
        root = Path(request.workspace) / "work/outputs"
        root.mkdir(parents=True, exist_ok=True)
        if self.question and request.stage is Stage.UNDERSTAND and request.attempt == 1:
            return StageResult(stage=request.stage, attempt=request.attempt, conversation_id=request.conversation_id,
                execution_key=request.execution_key, status="needs_input", questions=[SemanticQuestion(issue_id="meaning",
                    stage=request.stage, question="Which finding?", impact="mapping", blocking=True)])
        if self.revise and request.stage is Stage.PREPARE_TEXT and request.attempt == 1:
            return StageResult(stage=request.stage, attempt=request.attempt, conversation_id=request.conversation_id,
                execution_key=request.execution_key, status="blocked", control="mapping_revision_required", diagnostic="source conflict")
        if request.stage is Stage.UNDERSTAND:
            (root / "mapping.json").write_text(json.dumps(mapping()), encoding="utf-8")
            (root / "image_conversion_plan.json").write_text('{"policies":{"native":{}}}', encoding="utf-8")
        elif request.stage is Stage.PREPARE_TEXT:
            (root / "cases.jsonl").write_text(json.dumps({"case_id":"c", "evidence":[
                {"id":"i", "type":"asset", "name":"image", "value":"images/view.jpg"},
                {"id":"l", "type":"value", "name":"label", "value":"yes"}]}) + "\n", encoding="utf-8")
        else:
            source = Path(request.source_roots["s"]) / "raw.jpg"
            target = root / "images/view.jpg"
            target.parent.mkdir()
            shutil.copy2(source, target)
            row = ImageEntry(path="images/view.jpg", format="jpeg", width=4, height=4, file_digest=digest_file(target),
                             source_id="s", source_path="raw.jpg", source_digest=digest_file(source))
            (root / "image_manifest.jsonl").write_text(row.model_dump_json() + "\n", encoding="utf-8")
            if self.fail_image and request.attempt == 1:
                target.write_bytes(b"bad")
        return StageResult(stage=request.stage, attempt=request.attempt, conversation_id=request.conversation_id,
            execution_key=request.execution_key, input_digest=request.input_digest, status="submitted",
            outputs={p.relative_to(root).as_posix(): str(p) for p in root.rglob("*") if p.is_file()})


def test_three_stages_publish_minimal_dataset_and_recover(tmp_path):
    worker = Worker(fail_image=True)
    graph = compile_preparation_graph(worker=worker, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id":"d"}}
    initial = state(tmp_path)
    result = graph.invoke(initial, config)
    assert result["status"] == "complete", result.get("diagnostics")
    assert [r.stage for r in worker.requests] == [Stage.UNDERSTAND, Stage.PREPARE_TEXT, Stage.PREPARE_IMAGE, Stage.PREPARE_IMAGE]
    root = Path(result["publication"]["revision_path"])
    assert {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} == {
        "cases.jsonl", "image_manifest.jsonl", "metadata.json", "images/view.jpg"}
    assert (root / "images/view.jpg").read_bytes() == (Path(initial["source_roots"]["s"]) / "raw.jpg").read_bytes()
    assert not (Path(initial["bundle_root"]) / "cases.jsonl").exists()
    graph.invoke(result, config)
    assert len(worker.requests) == 4


def test_mapping_revision_invalidates_text(tmp_path):
    worker = Worker(revise=True)
    result = compile_preparation_graph(worker=worker).invoke(state(tmp_path))
    assert result["status"] == "complete"
    assert [r.stage for r in worker.requests] == [Stage.UNDERSTAND, Stage.PREPARE_TEXT, Stage.UNDERSTAND, Stage.PREPARE_TEXT, Stage.PREPARE_IMAGE]


def test_semantic_question_resumes_same_graph(tmp_path):
    worker = Worker(question=True)
    graph = compile_preparation_graph(worker=worker)
    result = graph.invoke(state(tmp_path))
    assert result["status"] == "needs_input"
    result["user_decisions"] = [{"issue_id":"meaning", "answer":"finding"}]
    result = graph.invoke(result)
    assert result["status"] == "complete"
