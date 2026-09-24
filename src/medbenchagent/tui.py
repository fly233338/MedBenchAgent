"""Keyboard-first persistent terminal interface for MedBenchAgent."""

from __future__ import annotations

import queue
import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, RichLog, Static, Tree

from medbenchagent.data.dataset_catalog import load_dataset_catalog
from medbenchagent.runtime import sessions
from medbenchagent.runtime.session_supervisor import classify_requirement
from medbenchagent.settings import active_settings, load_settings, run_settings
from medbenchagent.workflow.graph import compile_benchmark_graph


class ExperimentMenu(Screen):
    BINDINGS = [Binding("ctrl+q", "app.quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Experiments · Select a row or enter a new name", id="menu-title")
        yield DataTable(id="experiments")
        with Horizontal():
            yield Input(placeholder="new-experiment-name", id="new-name")
            yield Button("Create", id="create", variant="primary")
        yield Static("Names use lowercase letters, digits and hyphens.", id="menu-message")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_experiments()
        self.query_one("#experiments", DataTable).focus()

    def on_screen_resume(self) -> None:
        self.refresh_experiments()

    def refresh_experiments(self) -> None:
        table = self.query_one("#experiments", DataTable)
        table.clear(columns=True)
        table.add_columns("Experiment", "Runs", "Latest status")
        for record in sessions.list_experiments(root=self.app.experiments_root):
            status = record["runs"][-1]["status"] if record["runs"] else "new"
            table.add_row(record["name"], str(len(record["runs"])), status, key=record["name"])

    @on(DataTable.RowSelected, "#experiments")
    def open_row(self, event: DataTable.RowSelected) -> None:
        self.app.open_experiment(str(event.row_key.value))

    @on(Button.Pressed, "#create")
    @on(Input.Submitted, "#new-name")
    def create(self) -> None:
        name = self.query_one("#new-name", Input).value.strip()
        try:
            sessions.create_experiment(name, root=self.app.experiments_root,
                                       template=self.app.config_template)
        except (ValueError, OSError) as exc:
            self.query_one("#menu-message", Static).update(str(exc))
            return
        self.app.open_experiment(name, configure=True)


class ConfigScreen(Screen):
    BINDINGS = [Binding("ctrl+s", "save", "Enter conversation"),
                Binding("escape", "back", "Back")]

    def __init__(self, name: str, *, new: bool = False) -> None:
        super().__init__()
        self.experiment_name = name
        self.new = new
        self.selected: tuple[str, ...] | None = None
        self.selected_value: object = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(f"{self.experiment_name} · Configuration (defaults are shown)")
        yield Tree("Configuration", id="config-tree")
        yield Label("Select a leaf to edit. dataset_sources is managed by preparation.", id="field-label")
        with Horizontal():
            yield Input(id="field-value", disabled=True)
            yield Button("Apply value", id="apply", disabled=True)
            yield Button("Enter conversation", id="save", variant="primary")
        yield Static("", id="config-message")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_tree()

    def refresh_tree(self) -> None:
        tree = self.query_one("#config-tree", Tree)
        tree.root.remove_children()
        values = sessions.config_values(self.experiment_name, root=self.app.experiments_root)
        values.setdefault("generation", {}).setdefault("max_total_tasks_per_image", None)

        def add(parent, data: dict, path: tuple[str, ...] = ()) -> None:
            for key, value in data.items():
                current = (*path, key)
                if isinstance(value, dict):
                    node = parent.add(key, data=current)
                    add(node, value, current)
                    node.expand()
                else:
                    label = f"{key} = {value!r}"
                    parent.add_leaf(label, data=(current, value))

        add(tree.root, values)
        tree.root.expand()

    @on(Tree.NodeSelected, "#config-tree")
    def select_field(self, event: Tree.NodeSelected) -> None:
        data = event.node.data
        if not isinstance(data, tuple) or len(data) != 2 or not isinstance(data[0], tuple):
            return
        path, value = data
        editable = path[0] != "dataset_sources"
        self.selected, self.selected_value = path, value
        self.query_one("#field-label", Label).update(".".join(path))
        field = self.query_one("#field-value", Input)
        field.value = "" if value is None else str(value)
        field.disabled = not editable
        self.query_one("#apply", Button).disabled = not editable
        if editable:
            field.focus()

    @on(Button.Pressed, "#apply")
    @on(Input.Submitted, "#field-value")
    def apply_value(self) -> None:
        if self.selected is None or self.selected[0] == "dataset_sources":
            return
        raw = self.query_one("#field-value", Input).value
        try:
            if self.selected_value is None:
                value = int(raw) if raw.strip() else None
            elif isinstance(self.selected_value, bool):
                if raw.lower() not in {"true", "false"}:
                    raise ValueError("Enter true or false")
                value = raw.lower() == "true"
            elif isinstance(self.selected_value, int):
                value = int(raw)
            elif isinstance(self.selected_value, float):
                value = float(raw)
            else:
                value = raw
            sessions.update_config_value(self.experiment_name, self.selected, value,
                                         root=self.app.experiments_root)
        except (ValueError, TypeError, KeyError) as exc:
            self.query_one("#config-message", Static).update(str(exc))
            return
        self.query_one("#config-message", Static).update("Saved. Changes apply to the next run.")
        self.refresh_tree()

    def action_save(self) -> None:
        self.app.pop_screen()
        if self.new:
            self.app.push_screen(ConversationScreen(self.experiment_name))

    @on(Button.Pressed, "#save")
    def save_button(self) -> None:
        self.action_save()

    def action_back(self) -> None:
        self.app.pop_screen()


class ConversationScreen(Screen):
    BINDINGS = [Binding("ctrl+e", "edit_config", "Config"),
                Binding("ctrl+r", "resume_run", "Resume"),
                Binding("escape", "back", "Experiments"),
                Binding("ctrl+q", "app.quit", "Quit")]

    def __init__(self, name: str) -> None:
        super().__init__()
        self.experiment_name = name
        self.record: dict = {}
        self.running = False
        self.pending_question = False
        self.answer_queue: queue.Queue[str | None] = queue.Queue()
        self.current_run_id: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(f"MBA · {self.experiment_name}", id="session-title")
        yield Static("Ready", id="session-status")
        with Horizontal():
            with Vertical():
                yield RichLog(id="conversation", wrap=True, highlight=True)
                yield Input(placeholder="Describe a task or answer the current question", id="message")
            with Vertical():
                yield DataTable(id="runs")
                yield RichLog(id="run-detail", wrap=True, highlight=True)
                yield RichLog(id="progress", wrap=True, highlight=True)
        with Horizontal():
            yield Button("Resume run", id="resume")
            yield Button("Config", id="config")
            yield Button("Experiments", id="back")
        yield Footer()

    def on_mount(self) -> None:
        self.record = sessions.load_experiment(self.experiment_name, root=self.app.experiments_root)
        table = self.query_one("#runs", DataTable)
        table.add_columns("Type", "Status", "Run")
        for message in self.record["messages"]:
            self._write_message(message["role"], message["text"])
        self.refresh_runs()
        self.query_one("#run-detail", RichLog).write("Select a run to see its frozen inputs.")
        self.query_one("#message", Input).focus()
        self.set_timer(0.1, self._start_next)

    def _write_message(self, role: str, text: str) -> None:
        self.query_one("#conversation", RichLog).write(f"{role}: {text}")

    def _set_status(self, text: str) -> None:
        self.query_one("#session-status", Static).update(text)

    def _log_progress(self, text: str) -> None:
        if text:
            self.query_one("#progress", RichLog).write(text)

    def _run_initialized(self, run_id: str, metadata: dict) -> None:
        sessions.update_run(self.record, run_id, root=self.app.experiments_root,
                            run_id=metadata.get("run_id"),
                            run_directory=metadata.get("run_directory"),
                            effective_config_path=metadata.get("effective_config_path"),
                            catalog_snapshot_path=metadata.get("task_catalog_snapshot_path"))
        self.refresh_runs()

    def refresh_runs(self) -> None:
        table = self.query_one("#runs", DataTable)
        table.clear()
        for run in self.record["runs"]:
            table.add_row(run["kind"], run["status"], run["id"][:8], key=run["id"])

    @on(DataTable.RowSelected, "#runs")
    def show_run(self, event: DataTable.RowSelected) -> None:
        run = next(run for run in self.record["runs"] if run["id"] == event.row_key.value)
        detail = (f"{run['kind']} · {run['status']}\n{run['summary']}\n"
                  f"Run: {run['run_directory']}\nConfig: {run['effective_config_path']}\n"
                  f"Catalog: {run['catalog_snapshot_path']}\n"
                  f"Discovered source: {run.get('source_decision_path')}")
        snapshot_path = run.get("effective_config_path")
        if snapshot_path and Path(snapshot_path).is_file():
            snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
            detail += "\nEnvironment overrides: " + ", ".join(snapshot.get("environment_overrides", []))
            detail += "\nEffective settings:\n" + json.dumps(snapshot["settings"], ensure_ascii=False, indent=2)
        if run.get("selected_assets"):
            detail += "\nSelected assets:\n" + json.dumps(run["selected_assets"], ensure_ascii=False, indent=2)
        view = self.query_one("#run-detail", RichLog)
        view.clear()
        view.write(detail)

    @on(Input.Submitted, "#message")
    def submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if self.pending_question:
            sessions.add_message(self.record, "user", text, run_id=self.current_run_id,
                                 root=self.app.experiments_root)
            self._write_message("you", text)
            self.pending_question = False
            self.answer_queue.put(text)
            self._set_status("Running")
            return
        if self.running or any(run["status"] in {"running", "interrupted", "waiting_input"}
                               for run in self.record["runs"]):
            self._set_status("Resume the unfinished run before starting another task.")
            return
        sessions.add_message(self.record, "user", text, root=self.app.experiments_root)
        self._write_message("you", text)
        self.running = True
        event.input.disabled = True
        self._set_status("Supervisor is interpreting the request")
        self._classify(text, list(self.record["messages"][:-1]))

    @work(thread=True)
    def _classify(self, text: str, history: list[dict]) -> None:
        try:
            settings = load_settings(sessions.config_path(self.experiment_name, root=self.app.experiments_root))
            with active_settings(settings):
                known = [{"dataset_id": e["dataset_id"], "name": e["name"]}
                         for e in load_dataset_catalog()]
                known += [{"dataset_id": e["dataset_id"], "name": e["name"]}
                          for e in settings.dataset_sources]
                decision = classify_requirement(text, known_datasets=known, history=history)
            self.app.call_from_thread(self._classified, text, decision)
        except Exception as exc:
            self.app.call_from_thread(self._classification_failed, str(exc))

    def _classification_failed(self, message: str) -> None:
        self.running = False
        self.query_one("#message", Input).disabled = False
        self._set_status(f"Supervisor error: {message}")

    def _classified(self, text: str, decision: Any) -> None:
        if decision.reply:
            sessions.add_message(self.record, "assistant", decision.reply,
                                 root=self.app.experiments_root)
            self._write_message("MBA", decision.reply)
        if not decision.tasks:
            self.running = False
            self.query_one("#message", Input).disabled = False
            self._set_status("Ready")
            return
        dependency = None
        for task in decision.tasks:
            run = sessions.queue_run(self.record, task.kind, text, task.dataset_id,
                                     task.requested_dataset_ids, dependency,
                                     root=self.app.experiments_root)
            dependency = run["id"]
        self.running = False
        self.refresh_runs()
        self._start_next()

    def _start_next(self) -> None:
        if self.running:
            return
        unfinished = next((run for run in reversed(self.record["runs"])
                           if run["status"] in {"running", "interrupted", "waiting_input"}), None)
        if unfinished:
            self._set_status("Unfinished run available. Press Ctrl+R to resume.")
            return
        run = sessions.pop_pending(self.record, root=self.app.experiments_root)
        if run:
            self._launch_run(run, resume=False)

    def _settings_for_run(self, run: dict, *, resume: bool):
        if not resume:
            return load_settings(sessions.config_path(self.experiment_name,
                                                      root=self.app.experiments_root))
        runs_root = Path(run["runs_root"])
        if not (runs_root / "checkpoints.sqlite").is_file():
            raise ValueError("Run checkpoint database is missing")
        connection = sqlite3.connect(str(runs_root / "checkpoints.sqlite"))
        try:
            graph = compile_benchmark_graph(checkpointer=SqliteSaver(connection))
            snapshot = graph.get_state({"configurable": {"thread_id": run["thread_id"]}})
            path = snapshot.values.get("context", {}).get("effective_config_path")
        finally:
            connection.close()
        if not path:
            raise ValueError("Run checkpoint or frozen configuration is missing")
        with run_settings(path) as saved:
            return replace(saved, paths=replace(saved.paths, runs_root=runs_root))

    def _launch_run(self, run: dict, *, resume: bool) -> None:
        try:
            settings = self._settings_for_run(run, resume=resume)
        except Exception as exc:
            sessions.update_run(self.record, run["id"], root=self.app.experiments_root,
                                status="blocked", summary=f"Invalid experiment configuration: {exc}")
            sessions.block_dependents(self.record, run["id"], root=self.app.experiments_root)
            self._set_status(f"Invalid experiment configuration: {exc}")
            self.refresh_runs()
            return
        self.running = True
        self.current_run_id = run["id"]
        self.query_one("#message", Input).disabled = True
        self._set_status(f"Running {run['kind']}")
        sessions.update_run(self.record, run["id"], root=self.app.experiments_root,
                            status="running", runs_root=run.get("runs_root") or str(settings.paths.runs_root))
        self.refresh_runs()
        self._execute_run(run["id"], resume, settings)

    @work(thread=True)
    def _execute_run(self, run_id: str, resume: bool, settings) -> None:
        from medbenchagent.cli import run_interactive_build

        run = next(run for run in self.record["runs"] if run["id"] == run_id)
        errors: list[str] = []

        def output(text: str) -> None:
            if text.startswith(("Benchmark construction failed:", "Local preflight failed:")):
                errors.append(text)
            self.app.call_from_thread(self._log_progress, text)

        def initialized(metadata: dict) -> None:
            self.app.call_from_thread(self._run_initialized, run_id, metadata)

        def ask(prompt: str) -> str:
            self.app.call_from_thread(self._show_question, prompt)
            answer = self.answer_queue.get()
            if answer is None:
                raise EOFError
            return answer

        try:
            with active_settings(settings):
                status = run_interactive_build(
                    settings=settings, input_fn=ask, output=output,
                    initial_requirement=None if resume else run["requirement"],
                    resume=run["thread_id"] if resume else None,
                    thread_id=run["thread_id"], dataset_id=run["dataset_id"],
                    dataset_ids=run["requested_dataset_ids"], show_settings=False,
                    experiment_name=self.experiment_name, experiments_root=str(self.app.experiments_root),
                    run_metadata_callback=initialized,
                )
        except Exception as exc:
            self.app.call_from_thread(self._run_finished, run_id,
                                      {"status": "failed", "summary": str(exc)})
            return
        try:
            connection = sqlite3.connect(str(Path(run["runs_root"]) / "checkpoints.sqlite"))
            try:
                graph = compile_benchmark_graph(checkpointer=SqliteSaver(connection))
                snapshot = graph.get_state({"configurable": {"thread_id": run["thread_id"]}})
                values = snapshot.values or {}
            finally:
                connection.close()
            context = values.get("context", {})
            selected_assets = {dataset_id: asset.get("entry", {})
                               for dataset_id, asset in context.get("dataset_assets", {}).items()}
            run_status = ("waiting_input" if snapshot.tasks and
                          any(task.interrupts for task in snapshot.tasks) else values.get("status"))
            if status != 0 and run_status not in {"complete", "blocked", "failed"}:
                error_text = errors[-1] if errors else ""
                run_status = "blocked" if "Code or runtime contract changed" in error_text else "interrupted"
            changes = {"run_id": values.get("run_id"),
                       "run_directory": values.get("run_directory"),
                       "effective_config_path": context.get("effective_config_path"),
                       "catalog_snapshot_path": context.get("task_catalog_snapshot_path"),
                       "source_decision_path": context.get("source_decision_path"),
                       "selected_assets": selected_assets,
                       "summary": errors[-1] if errors else values.get("summary", ""),
                       "status": run_status or ("interrupted" if status == 130 else "failed")}
        except Exception as exc:
            changes = {"status": "interrupted" if status == 130 else "failed",
                       "summary": f"Could not inspect checkpoint: {exc}"}
        self.app.call_from_thread(self._run_finished, run_id, changes)

    def _show_question(self, prompt: str) -> None:
        self.pending_question = True
        self.query_one("#message", Input).disabled = False
        self.query_one("#message", Input).focus()
        self._set_status(prompt)

    def _run_finished(self, run_id: str, changes: dict) -> None:
        run = sessions.update_run(self.record, run_id, root=self.app.experiments_root, **changes)
        self.running = False
        self.current_run_id = None
        self.pending_question = False
        self.query_one("#message", Input).disabled = False
        self.refresh_runs()
        reply = f"{run['kind']}: {run['status']}. {run['summary']}"
        sessions.add_message(self.record, "assistant", reply, run_id=run_id,
                             root=self.app.experiments_root)
        self._write_message("MBA", reply)
        self._set_status("Ready" if run["status"] == "complete" else reply)
        if run["status"] == "complete":
            self._start_next()
        elif run["status"] in {"blocked", "failed"}:
            sessions.block_dependents(self.record, run_id, root=self.app.experiments_root)
            self.refresh_runs()

    def action_resume_run(self) -> None:
        if self.running:
            return
        run = next((run for run in reversed(self.record["runs"])
                    if run["status"] in {"running", "interrupted", "waiting_input"}), None)
        if run:
            self._launch_run(run, resume=True)
        else:
            self._set_status("No unfinished run to resume.")

    @on(Button.Pressed, "#resume")
    def resume_button(self) -> None:
        self.action_resume_run()

    def action_edit_config(self) -> None:
        self.app.push_screen(ConfigScreen(self.experiment_name))

    @on(Button.Pressed, "#config")
    def config_button(self) -> None:
        self.action_edit_config()

    def action_back(self) -> None:
        if self.running:
            self._set_status("Wait for the current step or exit; its checkpoint is preserved.")
            return
        self.app.pop_screen()

    @on(Button.Pressed, "#back")
    def back_button(self) -> None:
        self.action_back()


class MBAApp(App):
    CSS = """
    Screen { layout: vertical; }
    #menu-title, #session-title, #session-status, #run-detail, #config-message,
    #menu-message, #field-label { padding: 0 1; }
    #experiments, #config-tree { height: 1fr; }
    #conversation { width: 2fr; height: 1fr; border: solid $accent; }
    #runs { height: 35%; }
    #run-detail { height: 30%; overflow: auto; }
    #progress { height: 35%; border: solid $surface-lighten-2; }
    Horizontal { height: auto; }
    ConversationScreen > Horizontal { height: 1fr; }
    ConversationScreen > Horizontal:last-of-type { height: auto; }
    ConversationScreen Horizontal > Vertical { height: 1fr; }
    Button { margin: 0 1; }
    Input { width: 1fr; }
    """

    def __init__(self, *, experiments_root: Path = sessions.EXPERIMENTS_ROOT,
                 config_template: Path | None = None) -> None:
        super().__init__()
        self.experiments_root = experiments_root
        self.config_template = config_template

    def on_mount(self) -> None:
        self.push_screen(ExperimentMenu())

    def open_experiment(self, name: str, *, configure: bool = False) -> None:
        if configure:
            self.push_screen(ConfigScreen(name, new=True))
        else:
            self.push_screen(ConversationScreen(name))
