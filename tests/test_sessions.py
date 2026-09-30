import pytest
from filelock import FileLock

from medbenchagent.agents.source_discovery import discover_source
from medbenchagent.runtime import sessions
from medbenchagent.runtime.conversation_service import ConversationService
from medbenchagent.runtime.graph_runner import RunEvent


def test_experiment_config_and_source_upsert(tmp_path):
    record = sessions.create_experiment("breast-pilot", root=tmp_path)
    assert record["runs"] == []
    sessions.update_config_value("breast-pilot", ("llm", "model"), "local-model", root=tmp_path)
    assert sessions.config_values("breast-pilot", root=tmp_path)["llm"]["model"] == "local-model"
    source = {"dataset_id": "breast", "name": "BrEaST", "description": "",
              "sources": [{"id": "original", "root": str(tmp_path), "kind": "raw",
                           "annotation": "annotation.jsonl"}]}
    sessions.upsert_dataset_source("breast-pilot", source, root=tmp_path)
    sessions.upsert_dataset_source("breast-pilot", source, root=tmp_path)
    assert len(sessions.config_values("breast-pilot", root=tmp_path)["dataset_sources"]) == 1
    with pytest.raises(ValueError, match="Conflicting"):
        sessions.upsert_dataset_source("breast-pilot", {**source, "name": "Other"}, root=tmp_path)


def test_remove_experiment_only_deletes_idle_workspace(tmp_path):
    sessions.create_experiment("pilot", root=tmp_path)
    sessions.create_experiment("other", root=tmp_path)
    lock = FileLock(str(tmp_path / "pilot" / "run.lock"))
    with lock.acquire(timeout=0):
        with pytest.raises(ValueError, match="in use"):
            sessions.remove_experiment("pilot", root=tmp_path)
    sessions.remove_experiment("pilot", root=tmp_path)
    assert not (tmp_path / "pilot").exists()
    assert (tmp_path / "other" / "session.json").is_file()


def test_independent_requests_are_queued_without_dependencies(tmp_path):
    record = sessions.create_experiment("pilot", root=tmp_path)
    first = sessions.queue_run(record, "prepare and build", root=tmp_path)
    second = sessions.queue_run(record, "inspect datasets", root=tmp_path)
    assert first["thread_id"] != second["thread_id"]
    assert sessions.pop_pending(record, root=tmp_path)["id"] == first["id"]
    assert sessions.pop_pending(record, root=tmp_path)["id"] == second["id"]


def test_run_question_and_answer_are_archived_once(tmp_path):
    record = sessions.create_experiment("pilot", root=tmp_path)
    run = sessions.queue_run(record, "prepare the dataset", root=tmp_path)
    service = ConversationService("pilot", root=tmp_path)
    event = RunEvent(run["id"], "needs_input", {"question": "What does status 0 mean?"})
    service._apply(event)
    service._apply(event)
    assert [item["text"] for item in service.record()["messages"]] == ["What does status 0 mean?"]
    service.answer(run["id"], "It means benign")
    service.answer(run["id"], "It means benign")
    assert [item["text"] for item in service.record()["messages"]] == [
        "What does status 0 mean?", "It means benign"]


def test_source_discovery_persists_decision_before_config(monkeypatch, tmp_path):
    root = tmp_path / "BrEaST"
    root.mkdir()
    (root / "clinical.csv").write_text("image,label\nimage.png,benign\n", encoding="utf-8")
    (root / "image.png").write_bytes(b"image")
    sessions.create_experiment("pilot", root=tmp_path / "experiments")
    monkeypatch.setattr("medbenchagent.agents.source_discovery.complete_json",
                        lambda *_args, **_kwargs: {"root": str(root), "dataset_name": "BrEaST"})
    run_directory = tmp_path / "run"
    context = {"run_directory": str(run_directory), "user_requirement": f"Prepare BrEaST at {root}",
               "experiment_name": "pilot", "experiments_root": str(tmp_path / "experiments")}
    entry = discover_source(context, "breast")
    assert (run_directory / "discovery/sources/breast/source_decision.json").is_file()
    assert entry == discover_source(context, "breast")
