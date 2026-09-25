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
            assert "MBA  First part" in str(screen.query_one("#reply-preview").content)
            screen._reply_reset()
            assert not screen.query_one("#reply-preview").display
            record = sessions.load_experiment("pilot", root=tmp_path)
            run = sessions.queue_run(record, "prepare_dataset", "prepare", "breast", root=tmp_path)
            screen.refresh_record()
            table = screen.query_one("#runs")
            assert table.row_count == 1
            monkeypatch.setattr(table, "clear", lambda: pytest.fail("full table rebuild"))
            sessions.update_run(record, run["id"], root=tmp_path, status="running")
            screen.refresh_record()
            assert table.get_cell_at((0, 1)) == "running"
            monkeypatch.setattr(screen.service, "record", lambda: pytest.fail("artifact reread session"))
            screen._on_event(RunEvent(run["id"], "artifact", {"name": "test", "path": "asset"}))
            screen.running = True
            app.action_request_exit()
            assert screen.stop_event.is_set()
            assert screen.exit_requested

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
