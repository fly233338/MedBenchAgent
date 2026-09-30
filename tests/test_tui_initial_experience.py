import asyncio
import threading
from types import SimpleNamespace

import pytest
from textual.widgets import Select

from medbenchagent.runtime import sessions
from medbenchagent.runtime.construction import record_event
from medbenchagent.runtime.graph_runner import RunEvent
from medbenchagent import tui
from medbenchagent.tui import MBAApp, ChatMessage, ConversationScreen, ExperimentMenu, RemoveExperimentScreen, _brand


def test_plain_text_status_and_incremental_run_rows(tmp_path, monkeypatch):
    sessions.create_experiment("pilot", root=tmp_path)

    async def check():
        app = MBAApp(experiments_root=tmp_path)
        async with app.run_test() as pilot:
            app.open_experiment("pilot")
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConversationScreen)
            screen._set_status("[type=literal_error, input_value='prepare']")
            assert "[type=literal_error" in str(screen.query_one("#session-status").content)
            record = sessions.load_experiment("pilot", root=tmp_path)
            run = sessions.queue_run(record, "prepare", root=tmp_path)
            screen.refresh_record()
            table = screen.query_one("#runs")
            assert table.row_count == 1
            monkeypatch.setattr(table, "clear", lambda: pytest.fail("full table rebuild"))
            sessions.update_run(record, run["id"], root=tmp_path, status="running")
            screen.refresh_record()
            assert str(table.get_cell_at((0, 1))) == "running"
            with monkeypatch.context() as patch:
                patch.setattr(screen.service, "record", lambda: pytest.fail("artifact reread session"))
                screen._on_event(RunEvent(run["id"], "artifact", {"name": "test", "path": "asset"}))
            screen.running = True
            screen.action_edit_config()
            await pilot.pause()
            app.action_request_exit()
            assert screen.stop_event.is_set()
            assert screen.exit_requested
            await pilot.pause()
            assert app.screen is screen
            screen.classifying = True
            screen._submission_failed("Interrupted submission")
            assert app.is_running  # the run must finish before the app exits
            screen._execution_ended()
            await pilot.pause()
            assert not app.is_running

    asyncio.run(check())


def test_question_inputs_distinguish_absolute_paths_from_commands(tmp_path, monkeypatch):
    sessions.create_experiment("pilot", root=tmp_path)

    async def check():
        app = MBAApp(experiments_root=tmp_path)
        async with app.run_test() as pilot:
            app.open_experiment("pilot")
            await pilot.pause()
            screen = app.screen
            field = screen.query_one("#message")
            screen.pending_question = True
            screen.current_run_id = "run"
            run_answers = []
            monkeypatch.setattr(screen.service, "answer", lambda _run, value: run_answers.append(value))
            monkeypatch.setattr(screen, "_execute_run", lambda **_kwargs: None)
            field.value = "/mnt/data/raw"
            await pilot.press("enter")
            assert run_answers == ["/mnt/data/raw"]

    asyncio.run(check())


def test_conversation_shows_optimistic_message_then_reconciles_reply(tmp_path, monkeypatch):
    sessions.create_experiment("pilot", root=tmp_path)

    async def check():
        app = MBAApp(experiments_root=tmp_path)
        async with app.run_test() as pilot:
            app.open_experiment("pilot")
            await app.workers.wait_for_complete()
            await pilot.pause()
            screen = app.screen
            monkeypatch.setattr(screen, "_submit_task", lambda _value: None)
            field = screen.query_one("#message")
            field.value = "Prepare BrEaST"
            await pilot.press("enter")
            await pilot.pause()
            bubbles = list(screen.query(ChatMessage))
            assert [bubble.full_text for bubble in bubbles] == ["Prepare BrEaST", "⚙ Thinking…"]
            assert screen.record["messages"] == []

            record = sessions.load_experiment("pilot", root=tmp_path)
            sessions.add_message(record, "user", "Prepare BrEaST", root=tmp_path, save=False)
            sessions.add_message(record, "assistant", "Ready to prepare.", root=tmp_path)
            screen._submitted(SimpleNamespace(question=None, run_ids=()))
            await pilot.pause()
            bubbles = list(screen.query(ChatMessage))
            assert [bubble.full_text for bubble in bubbles] == ["Prepare BrEaST", "Ready to prepare."]
            assert bubbles[0].message_id == record["messages"][0]["id"]
            assert bubbles[1].styles.width.value == 95
            app.set_class(False, "compact")
            await pilot.pause()
            assert bubbles[1].styles.width.value == 75

    asyncio.run(check())


def test_run_question_card_archives_answer(tmp_path, monkeypatch):
    sessions.create_experiment("pilot", root=tmp_path)

    async def check():
        app = MBAApp(experiments_root=tmp_path)
        async with app.run_test() as pilot:
            app.open_experiment("pilot")
            await app.workers.wait_for_complete()
            await pilot.pause()
            screen = app.screen
            record = sessions.load_experiment("pilot", root=tmp_path)
            run = sessions.queue_run(record, "prepare", root=tmp_path)
            event = RunEvent(run["id"], "needs_input", {"question": "What does status 0 mean?"})
            screen.service._apply(event)
            screen._on_event(event)
            await pilot.pause()
            assert screen.query_one("#question-card").display
            assert "status 0" in str(screen.query_one("#run-question").content)
            monkeypatch.setattr(screen, "_execute_run", lambda **_kwargs: None)
            field = screen.query_one("#message")
            field.value = "Benign"
            await pilot.press("enter")
            await pilot.pause()
            assert not screen.query_one("#question-card").display
            assert [item["text"] for item in screen.service.record()["messages"]] == [
                "What does status 0 mean?", "Benign"]
            assert [bubble.full_text for bubble in screen.query(ChatMessage)] == [
                "What does status 0 mean?", "Benign"]

    asyncio.run(check())


def test_agent_progress_filter_keeps_live_and_saved_events(tmp_path):
    record = sessions.create_experiment("pilot", root=tmp_path)
    run = sessions.queue_run(record, "prepare", root=tmp_path)
    record["pending_runs"].clear()
    folder = tmp_path / "run"
    folder.mkdir()
    sessions.update_run(record, run["id"], root=tmp_path, status="complete", run_directory=str(folder))
    events = folder / "control" / "events.jsonl"
    record_event(events, "tool_started", agent="SupervisorAgent", operation="prepare_data", status="running")
    record_event(events, "tool_result", agent="DataProcessingAgent", operation="inspect_table",
                 status="success", summary="Table inspected")

    async def check():
        app = MBAApp(experiments_root=tmp_path)
        async with app.run_test() as pilot:
            app.open_experiment("pilot")
            await app.workers.wait_for_complete()
            await pilot.pause()
            screen = app.screen
            progress = screen.query_one("#progress")
            assert any("prepare_data" in str(line) for line in progress.lines)
            assert any("inspect_table" in str(line) for line in progress.lines)
            screen.query_one("#agent-filter", Select).value = "DataProcessingAgent"
            await pilot.pause()
            assert any("inspect_table" in str(line) for line in progress.lines)
            assert not any("prepare_data" in str(line) for line in progress.lines)
            record_event(events, "worker_started", agent="DatasetProcessingAgent",
                         operation="coding_worker", status="running")
            screen._flush_progress()
            screen.query_one("#agent-filter", Select).value = "DatasetProcessingAgent"
            await pilot.pause()
            assert any("coding_worker" in str(line) for line in progress.lines)
            assert "DatasetProcessing" in str(screen.query_one("#active-agent").content)
            screen.query_one("#agent-filter", Select).value = "*"
            await pilot.pause()
            assert any("prepare_data" in str(line) for line in progress.lines)
            screen._on_event(RunEvent("new-run", "error", {"status": "blocked", "summary": "Invalid config"}))
            screen._flush_progress()
            await pilot.pause()
            assert screen._progress_run_id == "new-run"
            assert any("Invalid" in str(line) for line in progress.lines)
            assert any("Invalid config" in entry[1] for entry in screen._progress_entries)

    asyncio.run(check())


def test_sidebar_drag_and_toggle(tmp_path):
    sessions.create_experiment("pilot", root=tmp_path)

    async def check():
        app = MBAApp(experiments_root=tmp_path)
        async with app.run_test() as pilot:
            app.open_experiment("pilot")
            await app.workers.wait_for_complete()
            await pilot.pause()
            screen = app.screen
            side = screen.query_one("#side-pane")
            original = side.outer_size.width
            await pilot.mouse_down("#side-resize-handle")
            await pilot.hover("#chat-pane", offset=(5, 5))
            await pilot.pause()
            assert side.outer_size.width > original
            assert screen.query_one("#chat-pane").size.width >= 24
            expanded = side.outer_size.width
            await pilot.hover("#side-pane", offset=(expanded - 1, 1))
            await pilot.pause()
            assert 18 <= side.outer_size.width < expanded
            await pilot.mouse_up("#side-pane", offset=(1, 1))
            await pilot.resize_terminal(70, 24)
            await pilot.pause()
            assert screen.query_one("#chat-pane").size.width >= 24
            screen.action_toggle_details()
            assert not side.display
            assert not screen.query_one("#side-resize-handle").display
            screen.action_toggle_details()
            assert side.display
            assert screen.query_one("#side-resize-handle").display

    asyncio.run(check())


def test_conversation_mount_remains_responsive_during_recovery(tmp_path, monkeypatch):
    from medbenchagent.runtime.conversation_service import ConversationService

    sessions.create_experiment("pilot", root=tmp_path)
    release = threading.Event()
    original = ConversationService.reconcile

    def slow_reconcile(service):
        release.wait(timeout=3)
        return original(service)

    monkeypatch.setattr(ConversationService, "reconcile", slow_reconcile)

    async def check():
        app = MBAApp(experiments_root=tmp_path)
        async with app.run_test() as pilot:
            app.open_experiment("pilot")
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConversationScreen)
            assert "Loading experiment" in str(screen.query_one("#session-status").content)
            release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert screen.service is not None
            assert "Ready" in str(screen.query_one("#session-status").content)

    try:
        asyncio.run(check())
    finally:
        release.set()


def test_menu_colored_brand_and_confirmed_remove(tmp_path, monkeypatch):
    sessions.create_experiment("pilot", root=tmp_path)
    sessions.create_experiment("other", root=tmp_path)
    monkeypatch.setattr(tui, "_supports_block_glyphs", lambda: True)
    brand = _brand()
    lines = brand.plain.splitlines()
    assert len(lines) == 6
    assert len(set(map(len, lines))) == 1
    assert "░" in brand.plain
    assert max(map(len, lines)) <= 72
    assert len({span.style for span in brand.spans}) > 3
    monkeypatch.setattr(tui, "_supports_block_glyphs", lambda: False)
    fallback = _brand()
    assert fallback.plain == "MedBenchAgent"
    assert len({span.style for span in fallback.spans}) > 3

    async def check():
        app = MBAApp(experiments_root=tmp_path)
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            menu = app.screen
            assert isinstance(menu, ExperimentMenu)
            table = menu.query_one("#experiments")
            selected = str(table.get_row_at(table.cursor_row)[0])

            menu._dispatch_command("/remove")
            await pilot.pause()
            assert isinstance(app.screen, RemoveExperimentScreen)
            assert app.screen.experiment_name == selected
            await pilot.press("escape")
            await pilot.pause()
            assert (tmp_path / selected / "session.json").is_file()

            menu._dispatch_command("/remove")
            await pilot.pause()
            app.screen.query_one("#confirm-remove").press()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not (tmp_path / selected).exists()
            assert len(sessions.list_experiments(root=tmp_path)) == 1

    asyncio.run(check())


def test_quit_waits_for_run_then_returns_to_experiments(tmp_path):
    sessions.create_experiment("pilot", root=tmp_path)

    async def check():
        app = MBAApp(experiments_root=tmp_path)
        async with app.run_test() as pilot:
            app.open_experiment("pilot")
            await pilot.pause()
            screen = app.screen
            screen.running = True
            screen.action_back()
            assert screen.stop_event.is_set()
            assert app.screen is screen
            screen._execution_ended()
            await pilot.pause()
            assert app.is_running
            assert not isinstance(app.screen, ConversationScreen)

    asyncio.run(check())


@pytest.mark.parametrize("screen_name", ["menu", "config", "conversation"])
def test_escape_exits_from_each_screen(tmp_path, screen_name):
    sessions.create_experiment("pilot", root=tmp_path)

    async def check():
        app = MBAApp(experiments_root=tmp_path)
        async with app.run_test() as pilot:
            if screen_name != "menu":
                app.open_experiment("pilot", configure=screen_name == "config")
                await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert not app.is_running

    asyncio.run(check())


def test_ctrl_c_copies_selected_input_without_exiting(tmp_path):
    sessions.create_experiment("pilot", root=tmp_path)

    async def check():
        app = MBAApp(experiments_root=tmp_path)
        async with app.run_test() as pilot:
            app.open_experiment("pilot")
            await app.workers.wait_for_complete()
            await pilot.pause()
            field = app.screen.query_one("#message")
            field.value = "copy this path"
            field.focus()
            await pilot.press("ctrl+shift+a", "ctrl+c")
            await pilot.pause()
            assert app.clipboard == "copy this path"
            assert app.is_running

    asyncio.run(check())


def test_all_collapsed_detail_reopens(tmp_path):
    sessions.create_experiment("pilot", root=tmp_path)

    async def check():
        app = MBAApp(experiments_root=tmp_path)
        async with app.run_test() as pilot:
            app.open_experiment("pilot")
            await app.workers.wait_for_complete()
            await pilot.pause()
            screen = app.screen
            for name in ("agents", "events", "runs"):
                screen.query_one(f"#{name}-toggle").press()
                await pilot.pause()
            assert not screen.query_one("#side-pane").display
            assert not screen.query_one("#side-resize-handle").display
            screen._layout_panels()
            screen._dispatch_command("/details")
            await pilot.pause()
            assert screen.query_one("#side-pane").display
            assert not screen._collapsed_panels
            for name in ("runs", "agents", "events"):
                assert all(child.display for child in screen.query_one(f"#{name}-panel").children)
    asyncio.run(check())
