import asyncio
from pathlib import Path

import pytest
from textual.widgets import Input, OptionList, TextArea

from medbenchagent.runtime import sessions
from medbenchagent.runtime.intake import InputDraft
from medbenchagent.tui.app import MBAApp
from medbenchagent.tui.screens import RunsScreen, NameScreen
from medbenchagent.tui.dialogs import ConnectionScreen, DraftScreen
from medbenchagent.tui.widgets import Composer


@pytest.mark.parametrize("size", [(80, 24), (100, 35), (140, 45)])
def test_run_entry_creation_commands_and_rendering(tmp_path, monkeypatch, size):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    async def check():
        app = MBAApp(root=tmp_path / "runs", runtime_root=tmp_path / "runtime")
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            assert isinstance(app.screen, RunsScreen)
            assert not (tmp_path / "runs").exists()
            screenshots = Path(".runtime/tui-review")
            screenshots.mkdir(parents=True, exist_ok=True)
            app.save_screenshot(path=str(screenshots), filename=f"home-{size[0]}.svg")
            await pilot.click("#create-run")
            assert isinstance(app.screen, NameScreen)
            app.screen.query_one("#run-name", Input).value = "research"
            await pilot.click("#confirm-name")
            for _ in range(100):
                if app.selected_run == "research" and app.view.get("status") == "draft":
                    break
                await pilot.pause(0.02)
            assert app.selected_run == "research" and app.view["status"] == "draft"
            composer = app.query_one(Composer)
            assert composer.region.bottom <= size[1] and composer.region.height >= 3
            composer.focus()
            composer.load_text("/conf")
            await pilot.pause()
            assert app.query_one("#command-options", OptionList).display
            await pilot.press("tab", "enter")
            assert isinstance(app.screen, ConnectionScreen)
            assert "[llm]" in app.screen.query_one("#advanced-config", TextArea).text
            await pilot.press("escape")
            composer.focus()
            composer.load_text("Build a dataset task 👩‍💻 é")
            await pilot.press("shift+enter")
            assert "\n" in composer.text
            await pilot.press("enter")
            assert isinstance(app.screen, DraftScreen)
            await pilot.press("escape")
            composer.load_text("Build 4 questions from the selected dataset.")
            app.chat("INPUT REQUIRED", "Do these views belong to the same case?")
            app.view = {"status": "waiting", "enabled_agents": ["supervisor", "data_processing", "coding"]}
            app.render_status()
            app.render_agents()
            await pilot.pause()
            app.save_screenshot(path=str(screenshots), filename=f"workbench-{size[0]}.svg")
            await pilot.resize_terminal(70, 20)
            await pilot.pause()
            await pilot.resize_terminal(*size)
            await pilot.pause()
            assert composer.region.bottom <= size[1]
            app.theme = "textual-light"
            await pilot.pause()
            app.save_screenshot(path=str(screenshots), filename=f"light-{size[0]}.svg")
    asyncio.run(check())


def test_history_focus_and_other_run_activity_do_not_mix(tmp_path):
    from textual.containers import VerticalScroll
    from medbenchagent.runtime.activity import ActivityReader
    from medbenchagent.runtime.construction import record_event

    record = sessions.create_session("history", root=tmp_path / "runs", runtime_root=tmp_path / "runtime")
    sessions.save_draft(record["directory"], InputDraft(goal="Review old evidence"))
    async def check():
        app = MBAApp(root=tmp_path / "runs", runtime_root=tmp_path / "runtime")
        async with app.run_test(size=(100, 35)) as pilot:
            await pilot.pause()
            app.pop_screen()
            app.select_run("history")
            await pilot.pause(0.2)
            active = app.reader
            old = ActivityReader(tmp_path / "other")
            record_event(old.path, "coding_started", agent_id="coding", request_id="old")
            app.activities(old, "other", old.read())
            assert app.reader is active and not app.latest_activity
            for index in range(30):
                app.chat("RESULT", f"Completed work {index}")
            await pilot.pause()
            history = app.query_one("#conversation", VerticalScroll)
            history.scroll_to(y=5, animate=False)
            await pilot.pause()
            position = history.scroll_y
            app.chat("RESULT", "New activity")
            await pilot.pause()
            assert history.scroll_y == position and app.unread > 0
            app.action_copy_run()
            await pilot.pause()
            app.screen.query_one("#run-name", Input).value = "copied"
            await pilot.click("#confirm-name")
            for _ in range(100):
                if app.selected_run == "copied":
                    break
                await pilot.pause(0.02)
            assert app.selected_run == "copied"
            assert not (tmp_path / "runs/copied/run.json").exists()
    asyncio.run(check())


def test_run_configuration_reuses_existing_local_defaults(tmp_path, monkeypatch):
    import medbenchagent.settings as settings
    from medbenchagent.runtime.intake import connection_values
    source = settings.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace('model = ""', 'model = "existing-model"', 1)
    (tmp_path / "config").mkdir()
    (tmp_path / "config/local.toml").write_text(source, encoding="utf-8")
    monkeypatch.setattr(settings, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("MEDBENCH_CONFIG", raising=False)
    record = sessions.create_session("local", root=tmp_path / "runs", runtime_root=tmp_path / "runtime")
    assert connection_values(Path(record["directory"]) / "config.toml")["model"] == "existing-model"
