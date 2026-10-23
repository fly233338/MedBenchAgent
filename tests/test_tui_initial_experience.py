import asyncio
import pytest
from medbenchagent.runtime.sessions import create_experiment
from medbenchagent.tui.app import MBAApp
from textual.widgets import TextArea


def test_new_rejects_existing_name_before_configuration(tmp_path):
    create_experiment("example", root=tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        create_experiment("example", root=tmp_path)


@pytest.mark.parametrize("size", [(80, 24), (140, 45)])
def test_chat_workbench_keeps_composer_visible_and_opens_settings(tmp_path, monkeypatch, size):
    from medbenchagent.tui.widgets import Composer
    from medbenchagent.tui.dialogs import ConnectionScreen
    async def check():
        app = MBAApp(root=tmp_path)
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            assert app.experiment == "default"
            composer = app.query_one(Composer)
            assert composer.region.bottom <= size[1] and composer.region.height >= 3
            composer.load_text("用数据目录构建 4 道题")
            await pilot.press("shift+enter")
            assert "\n" in composer.text
            await pilot.click("#settings")
            await pilot.pause()
            assert isinstance(app.screen, ConnectionScreen)
            assert "[llm]" in app.screen.query_one("#advanced-config", TextArea).text
            await pilot.press("escape")
            dispatched = []
            monkeypatch.setattr(app, "execute", lambda *args: dispatched.append(args))
            app.selected_run = "prepared-run"
            app.view = {"status": "ready"}
            app.render_status()
            await pilot.click("#resume")
            assert dispatched[0][2] == "prepared-run" and dispatched[0][-1] is False
            app.running = False
    asyncio.run(check())


def test_history_reading_and_activity_selection_do_not_jump_or_mix_runs(tmp_path):
    from textual.containers import VerticalScroll
    from medbenchagent.runtime.activity import ActivityReader
    from medbenchagent.runtime.construction import record_event
    async def check():
        app = MBAApp(root=tmp_path / "experiments")
        async with app.run_test(size=(100, 35)) as pilot:
            active = ActivityReader(tmp_path / "active")
            old = ActivityReader(tmp_path / "old")
            record_event(old.path, "coding_started", agent_id="coding", request_id="old")
            app.reader, app.selected_run = active, "active"
            app.activities(old, "default", "old", old.read())
            assert not app.latest_activity
            for index in range(30):
                app.chat("回执", f"已完成工作 {index}")
            await pilot.pause()
            history = app.query_one("#conversation", VerticalScroll)
            history.scroll_to(y=5, animate=False)
            await pilot.pause()
            position = history.scroll_y
            app.chat("回执", "后台刚完成的新工作")
            await pilot.pause()
            assert history.scroll_y == position and app.unread > 0
            assert app.query_one("#new-content").display
    asyncio.run(check())


def test_first_launch_prefers_existing_local_configuration(tmp_path, monkeypatch):
    import medbenchagent.settings as settings
    from medbenchagent.runtime.intake import connection_values
    source = settings.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace('model = ""', 'model = "existing-model"', 1)
    (tmp_path / "config").mkdir()
    (tmp_path / "config/local.toml").write_text(source, encoding="utf-8")
    monkeypatch.setattr(settings, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("MEDBENCH_CONFIG", raising=False)
    async def check():
        app = MBAApp(root=tmp_path / "experiments")
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            assert connection_values(tmp_path / "experiments/default/config.toml")["model"] == "existing-model"
    asyncio.run(check())
