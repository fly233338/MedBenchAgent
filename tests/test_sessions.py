import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from medbenchagent.runtime import sessions
from medbenchagent.runtime.session_supervisor import classify_requirement
from medbenchagent.agents.source_discovery import discover_source
from medbenchagent.tui import MBAApp, ConfigScreen, ConversationScreen, ExperimentMenu
from medbenchagent.settings import load_settings


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


def test_independent_runs_and_dependency(tmp_path):
    record = sessions.create_experiment("pilot", root=tmp_path)
    first = sessions.queue_run(record, "prepare_dataset", "prepare breast", "breast", root=tmp_path)
    second = sessions.queue_run(record, "build_benchmark", "build breast", depends_on=first["id"],
                                root=tmp_path)
    assert first["thread_id"] != second["thread_id"]
    assert sessions.pop_pending(record, root=tmp_path)["id"] == first["id"]
    assert sessions.pop_pending(record, root=tmp_path) is None
    sessions.update_run(record, first["id"], status="complete", root=tmp_path)
    assert sessions.pop_pending(record, root=tmp_path)["id"] == second["id"]
    sessions.add_message(record, "user", "next task", root=tmp_path)
    assert sessions.load_experiment("pilot", root=tmp_path)["messages"][-1]["text"] == "next task"


def test_failed_preparation_blocks_queued_build(tmp_path):
    record = sessions.create_experiment("pilot", root=tmp_path)
    preparation = sessions.queue_run(record, "prepare_dataset", "prepare", "breast", root=tmp_path)
    build = sessions.queue_run(record, "build_benchmark", "build", depends_on=preparation["id"],
                               root=tmp_path)
    sessions.block_dependents(record, preparation["id"], root=tmp_path)
    assert next(run for run in record["runs"] if run["id"] == build["id"])["status"] == "blocked"
    assert sessions.pop_pending(record, root=tmp_path)["id"] == preparation["id"]


def test_session_supervisor_contract(monkeypatch):
    monkeypatch.setattr("medbenchagent.runtime.session_supervisor.complete_json",
                        lambda *_args, **_kwargs: {"tasks": [
                            {"kind": "prepare_dataset", "dataset_id": "breast", "requested_dataset_ids": None},
                            {"kind": "build_benchmark", "dataset_id": None,
                             "requested_dataset_ids": ["breast"]}], "reply": ""})
    decision = classify_requirement("prepare then build", known_datasets=[], history=[])
    assert [task.kind for task in decision.tasks] == ["prepare_dataset", "build_benchmark"]


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
    assert (run_directory / ".runtime/source_discovery/breast/source_decision.json").is_file()
    assert entry == discover_source(context, "breast")
    assert sessions.config_values("pilot", root=tmp_path / "experiments")["dataset_sources"][0]["dataset_id"] == "breast"


def test_source_discovery_asks_for_complete_directory(monkeypatch, tmp_path):
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    complete = tmp_path / "complete"
    complete.mkdir()
    (complete / "clinical.csv").write_text("image,label\na.png,benign\n", encoding="utf-8")
    (complete / "a.png").write_bytes(b"image")
    monkeypatch.setattr("medbenchagent.agents.source_discovery.complete_json",
                        lambda *_args: {"root": str(incomplete), "dataset_name": None})
    questions = []
    monkeypatch.setattr("medbenchagent.agents.source_discovery.interrupt",
                        lambda question: questions.append(question["question"]) or str(complete))
    entry = discover_source({"run_directory": str(tmp_path / "run"),
                             "user_requirement": "Prepare the dataset"}, "dataset")
    assert questions and "tables and images" in questions[0]
    assert entry["sources"][0]["root"] == str(complete)


def test_no_task_reply_does_not_create_run(monkeypatch, tmp_path):
    record = sessions.create_experiment("pilot", root=tmp_path)
    monkeypatch.setattr("medbenchagent.runtime.session_supervisor.complete_json",
                        lambda *_args, **_kwargs: {"tasks": [], "reply": "Please describe a preparation or benchmark task."})
    decision = classify_requirement("hello", known_datasets=[], history=[])
    assert decision.reply
    assert sessions.load_experiment("pilot", root=tmp_path)["runs"] == record["runs"]


def test_resume_settings_ignore_changed_experiment_toml(monkeypatch, tmp_path):
    sessions.create_experiment("pilot", root=tmp_path / "experiments")
    config = sessions.config_path("pilot", root=tmp_path / "experiments")
    settings = load_settings(config)
    saved = tmp_path / "effective_config.json"
    saved.write_text(json.dumps(settings.snapshot()), encoding="utf-8")
    config.write_text("invalid = [", encoding="utf-8")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    (runs_root / "checkpoints.sqlite").touch()
    monkeypatch.setattr("medbenchagent.tui._compile_graph",
                        lambda _checkpointer: SimpleNamespace(get_state=lambda _config: SimpleNamespace(
                            values={"context": {"effective_config_path": str(saved)}})))
    screen = ConversationScreen("pilot")
    screen._app = SimpleNamespace(experiments_root=tmp_path / "experiments")
    resumed = screen._settings_for_run({"runs_root": str(runs_root), "thread_id": "run"}, resume=True)
    assert resumed.llm.model == settings.llm.model


def test_tui_create_config_conversation(tmp_path):
    async def check():
        app = MBAApp(experiments_root=Path(tmp_path))
        async with app.run_test() as pilot:
            assert isinstance(app.screen, ExperimentMenu)
            menu = app.screen
            menu.query_one("#new-name").value = "pilot"
            menu.create()
            await pilot.pause()
            assert isinstance(app.screen, ConfigScreen)
            tree = app.screen.query_one("#config-tree")
            llm = next(node for node in tree.root.children if node.label.plain == "llm")
            model = next(node for node in llm.children if node.label.plain.startswith("model ="))
            tree.select_node(model)
            await pilot.pause()
            field = app.screen.query_one("#field-value")
            assert not field.disabled
            field.value = "server-model"
            field.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert sessions.config_values("pilot", root=tmp_path)["llm"]["model"] == "server-model"
            app.screen.action_save()
            await pilot.pause()
            assert isinstance(app.screen, ConversationScreen)
            assert (tmp_path / "pilot" / "config.toml").is_file()

    asyncio.run(check())


def test_experiment_row_opens_with_enter(tmp_path):
    sessions.create_experiment("first", root=tmp_path)
    sessions.create_experiment("second", root=tmp_path)

    async def check():
        app = MBAApp(experiments_root=tmp_path)
        async with app.run_test() as pilot:
            assert isinstance(app.screen, ExperimentMenu)
            await pilot.press("down", "enter")
            await pilot.pause()
            assert isinstance(app.screen, ConversationScreen)
            assert app.screen.experiment_name == "first"

    asyncio.run(check())


def test_experiment_search_and_keyboard_navigation(tmp_path):
    sessions.create_experiment("first", root=tmp_path)
    sessions.create_experiment("second", root=tmp_path)

    async def check():
        app = MBAApp(experiments_root=tmp_path)
        async with app.run_test() as pilot:
            await pilot.press("ctrl+f", "f", "i", "r")
            await pilot.pause()
            table = app.screen.query_one("#experiments")
            assert table.row_count == 1
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ConversationScreen)
            assert app.screen.experiment_name == "first"

    asyncio.run(check())


def test_experiment_row_opens_with_mouse(tmp_path):
    sessions.create_experiment("pilot", root=tmp_path)

    async def check():
        app = MBAApp(experiments_root=tmp_path)
        async with app.run_test() as pilot:
            await pilot.click("#experiments", offset=(2, 1))
            await pilot.pause()
            assert isinstance(app.screen, ConversationScreen)

    asyncio.run(check())


def test_run_row_shows_details_with_enter(tmp_path):
    record = sessions.create_experiment("pilot", root=tmp_path)
    run = sessions.queue_run(record, "prepare_dataset", "Prepare", "breast", root=tmp_path)
    record["pending_runs"].clear()
    sessions.update_run(record, run["id"], root=tmp_path,
                        status="complete", summary="Prepared breast")

    async def check():
        app = MBAApp(experiments_root=tmp_path)
        async with app.run_test() as pilot:
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ConversationScreen)
            table = app.screen.query_one("#runs")
            table.focus()
            await pilot.press("enter")
            await pilot.pause()
            detail = app.screen.query_one("#run-detail")
            assert any("Prepared breast" in str(line) for line in detail.lines)

    asyncio.run(check())
