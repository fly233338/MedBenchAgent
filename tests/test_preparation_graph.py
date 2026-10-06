import json
from pathlib import Path
from types import SimpleNamespace

from medbenchagent.data.contracts import Source
from medbenchagent.data.sources import inventory_source
from medbenchagent.preparation.contracts import Stage, StageResult, ValidationResult
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
        "preparation_attempt_limits": {"understand": 2, "prepare": 2, "mapping_revision_cycles": 1},
    }


def _accepted_handoff(outputs, state):
    return ValidationResult(status="accepted", stage=Stage.UNDERSTAND, receipt_ref=str(Path(state["preparation_root"]) / "validation" / "understand.json"))


def _accepted_candidate(outputs, state):
    return ValidationResult(status="accepted", stage=Stage.PREPARE, report={"accepted_cases": 1}, candidate_digest="")


def test_two_stage_graph_uses_one_production_orchestrator(tmp_path, monkeypatch):
    worker = _StageWorker()
    monkeypatch.setattr(preparation_graph, "validate_understand_submission", _accepted_handoff)
    monkeypatch.setattr(preparation_graph, "validate_candidate_submission", _accepted_candidate)
    monkeypatch.setattr(preparation_graph, "publish_preparation", lambda state, result, validation: {"status": "complete", "bundle": str(tmp_path / "bundle"), "dataset_entry": {"dataset_id": "dataset"}, "cases_path": "cases", "access_manifest": {}, "card_path": "card"})
    result = compile_preparation_graph(worker=worker).invoke(_state(tmp_path))

    assert result["status"] == "complete"
    assert [request.stage for request in worker.requests] == [Stage.UNDERSTAND, Stage.PREPARE]
    assert result["accepted_handoff"]["mapping"]["name"] == "mapping.json"


def test_blocking_semantic_question_stops_before_prepare(tmp_path):
    worker = _StageWorker(blocking=True)
    state = _state(tmp_path)
    result = compile_preparation_graph(worker=worker).invoke(state)

    assert result["status"] == "needs_input"
    assert result["stage"] == "understand"
    assert not (Path(state["preparation_root"]) / "attempts" / "prepare").exists()


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


def test_mapping_revision_creates_new_understand_and_prepare_attempts(tmp_path, monkeypatch):
    class RevisionWorker(_StageWorker):
        def run(self, request):
            result = super().run(request)
            if request.stage is Stage.PREPARE and len([item for item in self.requests if item.stage is Stage.PREPARE]) == 1:
                return result.model_copy(update={
                    "status": "blocked", "control": "mapping_revision_required",
                    "diagnostic": "accepted mapping conflicts with source evidence",
                })
            return result

    worker = RevisionWorker()
    monkeypatch.setattr(preparation_graph, "validate_understand_submission", _accepted_handoff)
    monkeypatch.setattr(
        preparation_graph, "validate_candidate_submission",
        lambda outputs, state: ValidationResult(status="accepted", stage=Stage.PREPARE, report={"accepted_cases": 1}),
    )
    monkeypatch.setattr(
        preparation_graph, "publish_preparation",
        lambda state, result, validation: {"status": "complete", "revision": "r1", "bundle": str(tmp_path / "bundle"), "dataset_entry": {"dataset_id": "dataset"}, "cases_path": "cases", "access_manifest": {}, "card_path": "card"},
    )

    result = compile_preparation_graph(worker=worker).invoke(_state(tmp_path))

    assert result["status"] == "complete"
    assert [request.stage for request in worker.requests] == [Stage.UNDERSTAND, Stage.PREPARE, Stage.UNDERSTAND, Stage.PREPARE]
    assert [request.attempt for request in worker.requests] == [1, 1, 2, 2]
    assert result["mapping_revision_cycles"] == 1
