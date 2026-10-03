import sqlite3
from types import SimpleNamespace

from medbenchagent.data.dataset_catalog import load_dataset_catalog
from medbenchagent.runtime import sessions
from medbenchagent.runtime.conversation_service import ConversationService
from medbenchagent.runtime.graph_runner import GraphRunner


def test_missing_catalog_is_initialized_and_corruption_is_preserved(tmp_path):
    path = tmp_path / "datasets" / "dataset_catalog.json"
    assert load_dataset_catalog(path) == []
    path.write_text("", encoding="utf-8")
    try:
        load_dataset_catalog(path)
    except Exception:
        pass
    assert path.read_text(encoding="utf-8") == ""


def test_submit_creates_one_untyped_run(tmp_path):
    sessions.create_experiment("pilot", root=tmp_path)
    service = ConversationService("pilot", root=tmp_path)
    submission = service.submit("prepare a dataset and build a benchmark")
    record = service.record()
    assert len(submission.run_ids) == 1
    assert record["schema_version"] == "mba-session"
    assert record["runs"][0]["requirement"] == "prepare a dataset and build a benchmark"
    assert "kind" not in record["runs"][0]
    assert record["messages"][0]["role"] == "user"


def test_idle_reconcile_reads_session_once(tmp_path, monkeypatch):
    sessions.create_experiment("pilot", root=tmp_path)
    service = ConversationService("pilot", root=tmp_path)
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

    runs = [{"id": str(index), "thread_id": str(index), "runs_root": str(runs_root)} for index in range(2)]
    results = GraphRunner(graph_factory=factory).inspect_many(runs)
    assert len(compilations) == 1
    assert len(state_reads) == 2
    assert all(value["status"] == "interrupted" for value in results.values())

