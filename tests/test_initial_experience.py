from dataclasses import replace
import sqlite3
from types import SimpleNamespace

import pytest

from medbenchagent.data.dataset_catalog import DatasetCatalogError, load_dataset_catalog
from medbenchagent.runtime import sessions
from medbenchagent.runtime.conversation_service import ConversationService
from medbenchagent.runtime.graph_runner import GraphRunner
from medbenchagent.runtime.session_supervisor import SessionDecision
from medbenchagent.settings import get_settings


def _service(tmp_path, monkeypatch):
    sessions.create_experiment("pilot", root=tmp_path / "experiments")
    configured = replace(get_settings(), paths=replace(get_settings().paths,
                                                         dataset_catalog=tmp_path / "datasets" / "dataset_catalog.json"),
                         dataset_sources=())
    monkeypatch.setattr("medbenchagent.runtime.conversation_service.load_settings", lambda *_args: configured)
    return ConversationService("pilot", root=tmp_path / "experiments"), configured.paths.dataset_catalog


def _raw_dataset(tmp_path):
    root = tmp_path / "BrEaST"
    root.mkdir()
    (root / "clinical.csv").write_text("image,label\na.png,benign\n", encoding="utf-8")
    (root / "a.png").write_bytes(b"image")
    return root


def test_missing_catalog_is_initialized_and_corruption_is_preserved(tmp_path):
    path = tmp_path / "datasets" / "dataset_catalog.json"
    assert load_dataset_catalog(path) == []
    assert path.read_text(encoding="utf-8").strip() == '{\n  "datasets": []\n}'
    path.write_text("", encoding="utf-8")
    with pytest.raises(DatasetCatalogError, match="Cannot read"):
        load_dataset_catalog(path)
    assert path.read_text(encoding="utf-8") == ""


def test_failed_submission_is_visible_but_not_history(tmp_path, monkeypatch):
    service, _ = _service(tmp_path, monkeypatch)
    monkeypatch.setattr("medbenchagent.runtime.conversation_service.classify_requirement",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad decision")))
    with pytest.raises(ValueError, match="bad decision"):
        service.submit("prepare BrEaST")
    record = service.record()
    assert record["messages"] == record["runs"] == []
    assert record["submissions"][0]["status"] == "failed"
    assert record["submissions"][0]["text"] == "prepare BrEaST"


def test_processing_submission_is_reconciled(tmp_path, monkeypatch):
    service, _ = _service(tmp_path, monkeypatch)
    record = service.record()
    record["submissions"].append({"id": "stale", "text": "prepare", "status": "processing"})
    sessions.save_experiment(record, root=service.root)
    service.reconcile()
    assert service.record()["submissions"][0]["status"] == "failed"
    assert service.record()["messages"] == []


def test_idle_reconcile_reads_session_once(tmp_path, monkeypatch):
    service, _ = _service(tmp_path, monkeypatch)
    original = sessions.load_experiment
    reads = []

    def load(*args, **kwargs):
        reads.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(sessions, "load_experiment", load)
    service.reconcile()
    assert len(reads) == 1


def test_checkpoint_recovery_compiles_once_per_database(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    sqlite3.connect(runs_root / "checkpoints.sqlite").close()
    compilations = []
    state_reads = []

    def factory(*, checkpointer):
        compilations.append(checkpointer)
        return SimpleNamespace(get_state=lambda config: state_reads.append(config) or SimpleNamespace(values={}))

    runs = [{"id": str(index), "thread_id": str(index), "runs_root": str(runs_root)}
            for index in range(2)]
    results = GraphRunner(graph_factory=factory).inspect_many(runs)
    assert len(compilations) == 1
    assert len(state_reads) == 2
    assert all(value["status"] == "interrupted" for value in results.values())


def test_v1_session_is_upgraded_on_next_save(tmp_path):
    record = sessions.create_experiment("pilot", root=tmp_path)
    record.pop("submissions")
    record["schema_version"] = "mba-session.v1"
    sessions.save_experiment(record, root=tmp_path)
    migrated = sessions.load_experiment("pilot", root=tmp_path)
    assert migrated["schema_version"] == "mba-session.v2"
    assert migrated["submissions"] == []
    assert migrated["messages"] == record["messages"]


def test_cancelled_intake_allows_new_request(tmp_path, monkeypatch):
    service, _ = _service(tmp_path, monkeypatch)
    monkeypatch.setattr("medbenchagent.runtime.conversation_service.classify_requirement",
                        lambda *_args, **_kwargs: SessionDecision.model_validate({
                            "tasks": [{"kind": "prepare_dataset"}], "reply": "Let's prepare it."}))
    service.submit("prepare a dataset")
    with pytest.raises(ValueError, match="pending preparation"):
        service.submit("another request")
    service.cancel_intake()
    assert service.pending_intake() is None
    assert service.record()["submissions"][0]["status"] == "cancelled"


def test_new_dataset_intake_and_resume(tmp_path, monkeypatch):
    service, _ = _service(tmp_path, monkeypatch)
    root = _raw_dataset(tmp_path)
    monkeypatch.setattr("medbenchagent.runtime.conversation_service.classify_requirement",
                        lambda *_args, **_kwargs: SessionDecision.model_validate({
                            "tasks": [{"kind": "prepare_dataset"}],
                            "reply": "The source root is optional and the ID will keep its case."}))
    monkeypatch.setattr("medbenchagent.runtime.conversation_service.parse_answer",
                        lambda _field, text: text)
    first = service.submit("I'd like to prepare a dataset")
    assert first.question["field"] == "dataset_name"
    assert "source root" not in first.reply
    assert "ID" not in first.reply
    assert service.record()["messages"] == service.record()["runs"] == []
    reopened = ConversationService("pilot", root=service.root)
    assert reopened.pending_intake()["current"] == 0
    second = reopened.answer_intake("BrEaST")
    assert second.question["field"] == "source_root"
    assert reopened.pending_intake()["current"] == 1
    done = reopened.answer_intake(str(root))
    assert len(done.run_ids) == 1
    assert "BrEaST (ID: breast)" in done.reply
    assert str(root) in done.reply
    record = reopened.record()
    assert record["runs"][0]["dataset_id"] == "breast"
    assert record["runs"][0]["dataset_name"] == "BrEaST"
    assert record["runs"][0]["source_root"] == str(root)
    assert record["submissions"][0]["status"] == "accepted"
    assert record["submissions"][0]["question_batch_status"] == "completed"
    assert record["messages"][-1]["text"] == done.reply
    assert all("keep its case" not in message["text"] for message in record["messages"])
    assert reopened.pending_intake() is None


def test_invalid_intake_answer_stays_on_current_question(tmp_path, monkeypatch):
    service, _ = _service(tmp_path, monkeypatch)
    monkeypatch.setattr("medbenchagent.runtime.conversation_service.classify_requirement",
                        lambda *_args, **_kwargs: SessionDecision.model_validate({
                            "tasks": [{"kind": "prepare_dataset", "dataset_name": "BrEaST"}],
                            "reply": "Let's prepare it."}))
    monkeypatch.setattr("medbenchagent.runtime.conversation_service.parse_answer",
                        lambda _field, text: text)
    service.submit("prepare BrEaST")
    with pytest.raises(ValueError, match="unavailable"):
        service.answer_intake(str(tmp_path / "missing"))
    assert service.pending_intake()["current"] == 0
    assert service.pending_intake()["questions"][0]["answer"] is None


