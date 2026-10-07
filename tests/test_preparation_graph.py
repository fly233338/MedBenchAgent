import json
from pathlib import Path
from PIL import Image
from langgraph.checkpoint.memory import InMemorySaver
from medbenchagent.preparation.graph import compile_preparation_graph
from medbenchagent.preparation.contracts import Stage, StageResult, SemanticQuestion
from medbenchagent.data.sources import digest_file
from medbenchagent.data.models import ImageEntry


def mapping(filename="raw.jpg", convert=False):
    return {"population_policy": {"rule_id": "population", "unit_type": "case", "source_ids": ["s"],
        "identity_fields": ["id"], "inclusion": "all", "evidence_refs": ["s:annotation"]},
        "case_identity": {"rule_id": "id", "target_path": "case_id", "fields": ["id"], "evidence_refs": ["s:annotation"]},
        "grouping": {"rule_id": "group", "case_unit": "case", "key_fields": ["id"], "evidence_refs": ["s:annotation"]},
        "fields": [{"rule_id": "label", "source_path": "label", "disposition": "medical_evidence", "source_type": "string",
            "target_type": "string", "missingness": "unknown", "reason": "reference", "scope": "case", "evidence_refs": ["s:annotation"]}],
        "image_rules": [{"rule_id": "image", "source_id": "s", "pattern": filename, "source_format": "jpeg", "policy_id": "native", "action": "convert" if convert else "direct"}],
        "record_sources": [{"source_id": "s", "path": "records.csv", "format": "csv"}],
        "field_semantics": {"label": "documented finding"}}


def state(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    Image.new("RGB", (4, 4), "red").save(raw / "raw.jpg")
    (raw / "records.csv").write_text("id,label\nc,yes\n", encoding="utf-8")
    return {"run_id": "r", "dataset_id": "d", "run_directory": str(tmp_path / "runtime"),
        "preparation_root": str(tmp_path / "runtime/preparation"), "bundle_root": str(tmp_path / "prepared/d"),
        "sources": [{"id": "s", "root": str(raw)}], "source_roots": {"s": str(raw)},
        "source_manifests": {}, "source_snapshot_digest": "frozen"}


class Worker:
    def __init__(self, fail_image=False, question=False, revise=False, convert=False):
        self.requests = []
        self.convert = convert or fail_image
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
        from medbenchagent.preparation.contracts import PreparationMapping, ImageConversionPlan
        from medbenchagent.preparation.images import resolve_image
        locator = resolve_image(PreparationMapping.model_validate(mapping(convert=self.convert)),
            ImageConversionPlan(policies={"native": {}}), request.source_roots,
            request.context["bundle_root"], "s", "raw.jpg")
        if request.stage is Stage.UNDERSTAND:
            (root / "mapping.json").write_text(json.dumps(mapping(convert=self.convert)), encoding="utf-8")
            (root / "image_conversion_plan.json").write_text('{"policies":{"native":{}}}', encoding="utf-8")
        elif request.stage is Stage.PREPARE_TEXT:
            (root / "cases.jsonl").write_text(json.dumps({"case_id":"c", "evidence":[
                {"id":"i", "type":"asset", "name":"image", "value":locator},
                {"id":"l", "type":"value", "name":"label", "value":"yes"}]}) + "\n", encoding="utf-8")
            (root.parent / "accounting.json").write_text(json.dumps({"records_seen":1, "records_consumed":1,
                "cases_emitted":1, "excluded":{}, "skipped":{}}), encoding="utf-8")
        else:
            source = Path(request.source_roots["s"]) / "raw.jpg"
            target = root / "images" / Path(locator).name
            target.parent.mkdir()
            Image.open(source).save(target, format="PNG")
            row = ImageEntry(path=locator, format="png", width=4, height=4, file_digest=digest_file(target),
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
        "cases.jsonl", "image_manifest.jsonl", "metadata.json"}
    assert list((Path(initial["bundle_root"]) / "images").glob("*.png"))
    assert not (Path(initial["bundle_root"]) / "cases.jsonl").exists()
    graph.invoke(result, config)
    assert len(worker.requests) == 4


def test_mapping_revision_invalidates_text(tmp_path):
    worker = Worker(revise=True)
    result = compile_preparation_graph(worker=worker).invoke(state(tmp_path))
    assert result["status"] == "complete"
    assert [r.stage for r in worker.requests] == [Stage.UNDERSTAND, Stage.PREPARE_TEXT, Stage.UNDERSTAND, Stage.PREPARE_TEXT]


def test_semantic_question_resumes_same_graph(tmp_path):
    worker = Worker(question=True)
    graph = compile_preparation_graph(worker=worker)
    result = graph.invoke(state(tmp_path))
    assert result["status"] == "needs_input"
    result["user_decisions"] = [{"issue_id":"meaning", "answer":"finding"}]
    result = graph.invoke(result)
    assert result["status"] == "complete"


def test_direct_sources_skip_image_without_copy(tmp_path):
    initial = state(tmp_path)
    worker = Worker()
    result = compile_preparation_graph(worker=worker).invoke(initial)
    assert result["status"] == "complete", result.get("diagnostics")
    assert [r.stage for r in worker.requests] == [Stage.UNDERSTAND, Stage.PREPARE_TEXT]
    assert not (Path(initial["bundle_root"]) / "images").exists()
    root = Path(result["publication"]["revision_path"])
    assert (root / "image_manifest.jsonl").read_text() == ""
    case = json.loads((root / "cases.jsonl").read_text())
    assert Path(case["evidence"][0]["value"]) == Path(initial["source_roots"]["s"]) / "raw.jpg"
