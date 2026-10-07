import pytest
import json
from pathlib import Path
from types import SimpleNamespace

from medbenchagent.data.contracts import Source
from medbenchagent.data.sources import inventory_source
from medbenchagent.preparation.contracts import Stage, StageResult, ValidationResult
from medbenchagent.preparation.worker import OpenHandsWorker
import medbenchagent.preparation.graph as preparation_graph
from medbenchagent.preparation.graph import compile_preparation_graph


class _StageWorker:
    def __init__(self, *, blocking=False):
        self.blocking = blocking
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        output_root = Path(request.workspace) / "outputs"
        output_root.mkdir(parents=True, exist_ok=True)
        if request.stage is Stage.UNDERSTAND:
            values = {
                "source_report.json": "{}",
                "image_population.jsonl": "{}\n",
                "mapping.json": "{}",
                "image_conversion_plan.json": "{}",
                "semantic_questions.json": json.dumps({"questions": []}),
                "inspect_source.py": "print('ok')\n",
                "submission.json": json.dumps({"requested_outcome": "submitted"}),
            }
        else:
            values = {
                "scripts/prepare.py": "print('prepare')\n",
                "cases.jsonl": "{}\n",
                "asset_manifest.jsonl": "{}\n",
                "provenance.jsonl": "{}\n",
                "conversion_config.json": "{}",
                "grouping_policy.json": "{}",
                "execution_report.json": "{}",
                "output_manifest.json": "{}",
                "submission.json": json.dumps({"requested_outcome": "submitted"}),
            }
        outputs = {}
        for name, value in values.items():
            path = output_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value, encoding="utf-8")
            outputs[name] = str(path)
        status = "needs_input" if self.blocking and request.stage is Stage.UNDERSTAND else "submitted"
        return StageResult(
            stage=request.stage, run_id=request.run_id, dataset_id=request.dataset_id,
            attempt=request.attempt, attempt_id=request.attempt_id,
            conversation_id=request.conversation_id, execution_key=request.execution_key,
            status=status, outputs=outputs, input_digest=request.input_digest,
            token_count=request.prompt_token_count,
            question_refs=["question-1"] if status == "needs_input" else [],
        )


def _state(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "records.txt").write_text("record", encoding="utf-8")
    manifest = tmp_path / "manifest"
    inventory_source(Source(id="source", root=str(source_root)), manifest)
    return {
        "run_id": "run", "dataset_id": "dataset", "run_directory": str(tmp_path / "run"),
        "preparation_root": str(tmp_path / "run" / "preparation" / "dataset"),
        "source_snapshot_digest": "snapshot",
        "sources": [Source(id="source", root=str(source_root)).model_dump()],
        "source_roots": {"source": str(source_root)},
        "source_manifests": {"source": str(manifest)},
        "preparation_attempt_limits": {"understand": 2, "prepare_text": 2, "prepare_image": 2, "mapping_revision_cycles": 1},
    }


def _accepted_handoff(outputs, state):
    return ValidationResult(status="accepted", stage=Stage.UNDERSTAND, receipt_ref=str(Path(state["preparation_root"]) / "validation" / "understand.json"))


def _accepted_candidate(outputs, state):
    return ValidationResult(status="accepted", stage=Stage.PREPARE_TEXT, report={"accepted_cases": 1}, candidate_digest="")


def test_two_stage_graph_uses_one_production_orchestrator(tmp_path, monkeypatch):
    worker = _StageWorker()
    monkeypatch.setattr(preparation_graph, "validate_understand_submission", _accepted_handoff)
    monkeypatch.setattr(preparation_graph, "validate_candidate_submission", _accepted_candidate)
    monkeypatch.setattr(preparation_graph, "publish_preparation", lambda state, result, validation: {"status": "complete", "bundle": str(tmp_path / "bundle"), "dataset_entry": {"dataset_id": "dataset"}, "cases_path": "cases", "access_manifest": {}, "card_path": "card"})
    result = compile_preparation_graph(worker=worker).invoke(_state(tmp_path))

    assert result["status"] == "complete"
    assert [request.stage for request in worker.requests] == [Stage.UNDERSTAND, Stage.PREPARE_TEXT, Stage.PREPARE_IMAGE]
    assert result["accepted_handoff"]["mapping"]["name"] == "mapping.json"


def test_blocking_semantic_question_stops_before_prepare(tmp_path):
    worker = _StageWorker(blocking=True)
    state = _state(tmp_path)
    result = compile_preparation_graph(worker=worker).invoke(state)

    assert result["status"] == "needs_input"
    assert result["stage"] == "understand"
    assert not (Path(state["preparation_root"]) / "attempts" / "prepare_text").exists()


def test_resume_uses_run_frozen_worker_config(tmp_path, monkeypatch):
    settings = SimpleNamespace(llm=SimpleNamespace(
        model="frozen-chat", base_url="http://127.0.0.1:8005/v1",
        reasoning_effort="medium", timeout_seconds=17.0, max_retries=2,
    ))
    monkeypatch.setattr(preparation_graph, "get_settings", lambda: settings)
    monkeypatch.setattr(preparation_graph, "validate_understand_submission", _accepted_handoff)
    monkeypatch.setattr(preparation_graph, "validate_candidate_submission", _accepted_candidate)
    monkeypatch.setattr(preparation_graph, "publish_preparation", lambda state, result, validation: {"status": "complete", "bundle": str(tmp_path / "bundle"), "dataset_entry": {"dataset_id": "dataset"}, "cases_path": "cases", "access_manifest": {}, "card_path": "card"})
    worker = _StageWorker()

    result = compile_preparation_graph(worker=worker).invoke(_state(tmp_path))

    assert result["status"] == "complete"
    request = worker.requests[0]
    assert request.model == "frozen-chat"
    assert request.base_url == "http://127.0.0.1:8005/v1"
    assert request.reasoning_effort == "medium"
    assert request.llm_timeout_seconds == 17.0
    assert request.llm_max_retries == 2


def test_mapping_revision_creates_new_understand_and_prepare_text_attempts(tmp_path, monkeypatch):
    class RevisionWorker(_StageWorker):
        def run(self, request):
            result = super().run(request)
            if request.stage is Stage.PREPARE_TEXT and len([item for item in self.requests if item.stage is Stage.PREPARE_TEXT]) == 1:
                return result.model_copy(update={
                    "status": "blocked", "control": "mapping_revision_required",
                    "diagnostic": "accepted mapping conflicts with source evidence",
                })
            return result

    worker = RevisionWorker()
    monkeypatch.setattr(preparation_graph, "validate_understand_submission", _accepted_handoff)
    monkeypatch.setattr(
        preparation_graph, "validate_candidate_submission",
        lambda outputs, state: ValidationResult(status="accepted", stage=Stage.PREPARE_TEXT, report={"accepted_cases": 1}),
    )
    monkeypatch.setattr(
        preparation_graph, "publish_preparation",
        lambda state, result, validation: {"status": "complete", "revision": "r1", "bundle": str(tmp_path / "bundle"), "dataset_entry": {"dataset_id": "dataset"}, "cases_path": "cases", "access_manifest": {}, "card_path": "card"},
    )

    result = compile_preparation_graph(worker=worker).invoke(_state(tmp_path))

    assert result["status"] == "complete"
    assert [request.stage for request in worker.requests] == [Stage.UNDERSTAND, Stage.PREPARE_TEXT, Stage.UNDERSTAND, Stage.PREPARE_TEXT, Stage.PREPARE_IMAGE]
    assert [request.attempt for request in worker.requests] == [1, 1, 2, 2, 1]
    assert result["mapping_revision_cycles"] == 1


class _ProtocolRevisionRuntime:
    def __init__(self):
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        output_root = Path(request.workspace) / "outputs"
        output_root.mkdir(parents=True, exist_ok=True)
        if request.stage is Stage.PREPARE_TEXT and len([item for item in self.requests if item.stage is Stage.PREPARE_TEXT]) == 1:
            path = output_root / "submission.json"
            path.write_text(json.dumps({
                "requested_outcome": "blocked",
                "summary": "accepted semantics conflict",
                "diagnostic_codes": ["mapping_revision_required"],
            }), encoding="utf-8")
            return StageResult(
                stage=request.stage, attempt=request.attempt, attempt_id=request.attempt_id,
                conversation_id=request.conversation_id, execution_key=request.execution_key,
                status="blocked", outputs={"submission.json": str(path)},
                input_digest=request.input_digest,
            )
        names = (STAGE_OUTPUTS[request.stage] if request.stage in STAGE_OUTPUTS else ())
        outputs = {}
        for name in names:
            path = output_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"requested_outcome": "submitted"}) if name == "submission.json" else "{}\n",
                encoding="utf-8",
            )
            outputs[name] = str(path)
        return StageResult(
            stage=request.stage, attempt=request.attempt, attempt_id=request.attempt_id,
            conversation_id=request.conversation_id, execution_key=request.execution_key,
            status="submitted", outputs=outputs, input_digest=request.input_digest,
        )


STAGE_OUTPUTS = {
    Stage.UNDERSTAND: ("source_report.json", "image_population.jsonl", "mapping.json", "image_conversion_plan.json",
                       "semantic_questions.json", "inspect_source.py", "submission.json"),
    Stage.PREPARE_IMAGE: ("asset_manifest.jsonl", "presentation_manifest.jsonl", "mask_features.jsonl", "submission.json"),
    Stage.PREPARE_TEXT: ("image_requests.jsonl", "scripts/prepare.py", "cases.jsonl", "asset_manifest.jsonl",
                    "provenance.jsonl", "conversion_config.json", "grouping_policy.json",
                    "execution_report.json", "output_manifest.json", "submission.json"),
}


def test_real_worker_protocol_routes_mapping_revision_with_repair_bases(tmp_path, monkeypatch):
    runtime = _ProtocolRevisionRuntime()
    worker = OpenHandsWorker(runtime=runtime)
    monkeypatch.setattr(preparation_graph, "validate_understand_submission", _accepted_handoff)
    monkeypatch.setattr(
        preparation_graph, "validate_candidate_submission",
        lambda outputs, state: ValidationResult(status="accepted", stage=Stage.PREPARE_TEXT, report={"accepted_cases": 1}),
    )
    monkeypatch.setattr(
        preparation_graph, "publish_preparation",
        lambda state, result, validation: {"status": "complete", "revision": "r1",
                                            "bundle": str(tmp_path / "bundle"),
                                            "dataset_entry": {"dataset_id": "dataset"},
                                            "cases_path": "cases", "access_manifest": {},
                                            "card_path": "card"},
    )

    result = compile_preparation_graph(worker=worker).invoke(_state(tmp_path))

    assert result["status"] == "complete"
    assert [item.stage for item in runtime.requests] == [Stage.UNDERSTAND, Stage.PREPARE_TEXT, Stage.UNDERSTAND, Stage.PREPARE_TEXT, Stage.PREPARE_IMAGE]
    assert len({item.attempt_id for item in runtime.requests}) == 5
    assert len({item.conversation_id for item in runtime.requests}) == 5
    assert runtime.requests[2].previous_attempt_root == runtime.requests[1].workspace
    assert runtime.requests[3].previous_attempt_root == runtime.requests[1].workspace
    assert (Path(runtime.requests[2].workspace) / "inputs" / "previous_reference.json").is_file()
    assert not (Path(runtime.requests[2].workspace) / "inputs" / "previous" / "outputs").exists()
    context = json.loads((Path(runtime.requests[2].workspace) / "inputs" / "context.json").read_text(encoding="utf-8"))
    assert context["facts"]["revision_basis"]["handoff_digest"]
    assert context["facts"]["revision_basis"]["artifacts"]["mapping"]["path"].endswith("mapping.json")
    assert context["validation_feedback"]["issues"]


@pytest.fixture(autouse=True)
def isolate_packaging_in_graph_routing_tests(monkeypatch, tmp_path):
    monkeypatch.setattr(preparation_graph, "assemble_candidate", lambda state: tmp_path)
    monkeypatch.setattr(preparation_graph, "validate_prepare_candidate", lambda *args, **kwargs:
        ValidationResult(status="accepted", stage=Stage.PREPARE_IMAGE, report={"accepted_cases": 1}))
