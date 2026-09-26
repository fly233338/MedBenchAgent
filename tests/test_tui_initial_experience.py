import asyncio

import pytest

from medbenchagent.runtime import sessions
from medbenchagent.runtime.graph_runner import RunEvent
from medbenchagent.tui import MBAApp, ConversationScreen


def test_plain_text_preview_and_incremental_run_rows(tmp_path, monkeypatch):
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
            screen._reply_delta("First")
            screen._reply_delta(" part")
            screen._flush_reply()
            assert "MBA  First part" in str(screen.query_one("#reply-preview").content)
            screen._reply_queue.put(None)
            screen._reply_delta("Repaired")
            screen._flush_reply()
            assert "MBA  Repaired" == str(screen.query_one("#reply-preview").content)
            screen._reply_reset()
            screen._flush_reply()
            assert not screen.query_one("#reply-scroll").display
            record = sessions.load_experiment("pilot", root=tmp_path)
            run = sessions.queue_run(record, "prepare_dataset", "prepare", "breast", root=tmp_path)
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
            answers = []
            commands = []
            monkeypatch.setattr(screen, "_answer_intake", answers.append)
            monkeypatch.setattr(screen, "_dispatch_command", commands.append)
            screen.pending_intake = True

            field.value = "/public/home/data/BrEaST"
            await pilot.press("enter")
            assert answers == ["/public/home/data/BrEaST"]
            assert commands == []

            screen.classifying = False
            field.value = "/cancel"
            await pilot.press("enter")
            assert commands == ["/cancel"]

            screen.pending_intake = False
            screen.pending_question = True
            screen.current_run_id = "run"
            run_answers = []
            monkeypatch.setattr(screen.service, "answer", lambda _run, value: run_answers.append(value))
            monkeypatch.setattr(screen, "_execute_run", lambda **_kwargs: None)
            field.value = "/mnt/data/raw"
            await pilot.press("enter")
            assert run_answers == ["/mnt/data/raw"]

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
def test_ctrl_c_exits_from_each_screen(tmp_path, screen_name):
    sessions.create_experiment("pilot", root=tmp_path)

    async def check():
        app = MBAApp(experiments_root=tmp_path)
        async with app.run_test() as pilot:
            if screen_name != "menu":
                app.open_experiment("pilot", configure=screen_name == "config")
                await pilot.pause()
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert not app.is_running

    asyncio.run(check())
