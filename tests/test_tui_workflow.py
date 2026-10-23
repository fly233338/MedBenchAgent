import asyncio
from pathlib import Path
import threading

import tomlkit
from textual.widgets import TextArea

from breast_fixtures import create_breast_source
from coding_fixtures import FakeRunner
from native_agent_fixtures import ScriptedChatModel
from test_breast_workflow import BreastScript
from medbenchagent.agents.data_processing.coding.adapter import OpenHandsCodingAgent
from medbenchagent.compilation.contracts import ArtifactRef
from medbenchagent.data.task_data import SourceManifest
from medbenchagent.settings import DEFAULT_CONFIG_PATH
from medbenchagent.tui.app import MBAApp
from medbenchagent.tui.dialogs import DraftScreen, ActivityScreen
from medbenchagent.tui.widgets import Composer


def test_tui_directory_to_partial_delivery_live_activity_pause_and_answer(tmp_path, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    import medbenchagent.tui.app as ui
    import medbenchagent.runtime.graph_runner as runner_module
    import medbenchagent.runtime.model_services as services
    import medbenchagent.runtime.native as native
    import medbenchagent.compilation.policy as policy
    import medbenchagent.agents.data_processing.coding.adapter as coding

    raw = tmp_path / "BrEaST source"
    script = BreastScript(tmp_path / "unknown", raw, create_breast_source(raw))
    config = tomlkit.parse(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    for section in ("supervisor", "planner", "data_processing_agent", "generation_agent", "review_agent"):
        config[section]["context_chars"] = 2000000
        config[section]["loop"].update(context_tokens=500000, summary_trigger_tokens=300000, summary_keep_tokens=10000)
    template = tmp_path / "config.toml"
    template.write_text(tomlkit.dumps(config), encoding="utf-8")
    monkeypatch.setattr(ui, "check_connections", lambda *_: {})
    monkeypatch.setattr(services, "resolve_model_service", lambda *_: "offline")
    monkeypatch.setattr(native, "chat_model", lambda *_: ScriptedChatModel(responder=script.respond))
    original_initial = runner_module.initial_state
    def initial(directory, options):
        script.directory = directory
        script.dataset = SourceManifest.model_validate_json(Path(options.source_manifest).read_bytes()).dataset
        state = original_initial(directory, options)
        for row in state["artifacts"]:
            ref = ArtifactRef.model_validate(row)
            script.known[ref.artifact_id] = script.latest[ref.kind] = ref
        return state
    monkeypatch.setattr(runner_module, "initial_state", initial)
    original_check = policy.check_result
    def checked(directory, request, result):
        result = original_check(directory, request, result)
        script.observe(result)
        script.completed.append((request, result))
        return result
    monkeypatch.setattr(policy, "check_result", checked)
    entered, release = threading.Event(), threading.Event()
    def produce(payload):
        if payload["request"]["objective"] == "profile":
            entered.set()
            assert release.wait(20)
        return script.produce(payload)
    runner = FakeRunner(produce)
    monkeypatch.setattr(coding, "OpenHandsCodingAgent", lambda settings, **kw: OpenHandsCodingAgent(settings, **kw, runner=runner))

    async def until(pilot, predicate, seconds=60):
        for _ in range(int(seconds * 20)):
            if predicate():
                return
            await pilot.pause(0.05)
        raise AssertionError("TUI did not reach the expected state")

    async def check():
        app = MBAApp(root=tmp_path / "experiments", config_template=template)
        async with app.run_test(size=(100, 35)) as pilot:
            app.query_one(Composer).load_text(f'使用 "{raw}" 构建 4 道题，保持缺失与未知关系。')
            await pilot.press("enter")
            assert isinstance(app.screen, DraftScreen)
            app.screen.query_one("#key-files", TextArea).load_text(f"{script.workbook} | 标签表")
            await pilot.click("#start-draft")
            await until(pilot, lambda: entered.is_set() and "coding" in app.latest_activity)
            assert app.running and app.latest_activity["coding"]["running"]
            await pilot.click("#details")
            assert isinstance(app.screen, ActivityScreen)
            await pilot.press("escape")
            await pilot.click("#pause")
            assert app.stop_event.is_set()
            release.set()
            await until(pilot, lambda: not app.running and app.view.get("status") == "paused")
            await pilot.click("#resume")
            await until(pilot, lambda: not app.running and bool(app.view.get("questions")))
            question_id = app.view["question_id"]
            app.query_one(Composer).focus()
            app.query_one(Composer).load_text("Continue")
            await pilot.press("enter")
            await until(pilot, lambda: not app.running and app.view.get("status") == "stopped")
            assert app.view["goal_result"] == "partial"
            assert app.view["counts"]["accepted"] == 3 and app.view["target_count"] == 4
            await until(pilot, lambda: app.latest_activity.get("supervisor", {}).get("event", {}).get("decision") == "finish")
            assert len(app.view["history"]) == 11
            messages = app.service.record()["messages"]
            assert len([m for m in messages if m.get("id") == "answer-" + question_id]) == 1
            assert script.coding_calls.count("profile") == 1
            assert app.query_one(Composer).region.bottom <= 35
            identifier = app.selected_run
            (tmp_path / "workbench.svg").write_text(app.export_screenshot(), encoding="utf-8")
        reopened = MBAApp(root=tmp_path / "experiments")
        async with reopened.run_test(size=(80, 24)) as pilot:
            await until(pilot, lambda: reopened.view.get("status") == "stopped")
            assert reopened.selected_run == identifier and not reopened.running
            assert reopened.view["goal_result"] == "partial"
            assert reopened.query_one(Composer).region.bottom <= 24
    try:
        asyncio.run(check())
    finally:
        release.set()
