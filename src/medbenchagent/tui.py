"""Keyboard-first persistent terminal interface for MedBenchAgent."""

from __future__ import annotations

import queue
import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Input, Label, RichLog, Static, Tree

from medbenchagent.runtime import sessions
from medbenchagent.settings import active_settings, load_settings, run_settings


def _compile_graph(checkpointer):
    from medbenchagent.workflow.graph import compile_benchmark_graph

    return compile_benchmark_graph(checkpointer=checkpointer)


class ExperimentMenu(Screen):
    BINDINGS = [Binding("ctrl+f", "focus_search", "Search"),
                Binding("ctrl+n", "focus_new", "New"),
                Binding("ctrl+o", "open_selected", "Open"),
                Binding("escape", "focus_list", "Experiments"),
                Binding("ctrl+q", "app.quit", "Quit")]

    def __init__(self) -> None:
        super().__init__()
        self._entries: list[dict] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="menu-shell"):
            yield Static("MBA", id="menu-brand")
            yield Static("MEDBENCHAGENT  /  TERMINAL WORKSPACE", id="menu-subtitle")
            with Vertical(id="menu-panel"):
                yield Static("EXPERIMENTS", id="menu-title")
                with Horizontal(id="menu-search-bar"):
                    yield Input(placeholder="Search experiments  ·  Ctrl+F", id="search")
                    yield Button("Open", id="open", variant="primary")
                yield DataTable(id="experiments", cursor_type="row", zebra_stripes=True)
                yield Static("↑↓ choose  ·  Enter open  ·  Ctrl+N create", id="menu-message")
                with Horizontal(id="menu-create-bar"):
                    yield Input(placeholder="New experiment name", id="new-name")
                    yield Button("Create", id="create")
            yield Static("Names use lowercase letters, digits and hyphens.", id="menu-hint")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#experiments", DataTable).add_columns("EXPERIMENT", "RUNS", "STATUS")
        self.refresh_experiments()
        self.query_one("#experiments", DataTable).focus()

    def on_screen_resume(self) -> None:
        self.refresh_experiments()

    def refresh_experiments(self) -> None:
        self._entries = sessions.list_experiments(root=self.app.experiments_root)
        self._render_entries()

    def _render_entries(self) -> None:
        table = self.query_one("#experiments", DataTable)
        table.clear()
        query = self.query_one("#search", Input).value.strip().casefold()
        matches = [record for record in self._entries if query in record["name"].casefold()]
        table.styles.height = min(16, max(4, len(matches) + 2))
        for record in matches:
            status = record["runs"][-1]["status"] if record["runs"] else "new"
            status_color = {"complete": "#63d6bd", "blocked": "#e8ac70",
                            "failed": "#eb7d83", "running": "#a9c7f6"}.get(status, "#aab9ca")
            table.add_row(record["name"], str(len(record["runs"])),
                          Text(status.upper(), style=status_color), key=record["name"])
        if not matches:
            message = "No matching experiments. Create one below." if query else "No experiments yet. Create one below."
            self.query_one("#menu-message", Static).update(message)
        else:
            self.query_one("#menu-message", Static).update(
                "↑↓ choose  ·  Enter open  ·  Ctrl+N create")

    @on(Input.Changed, "#search")
    def filter_experiments(self) -> None:
        self._render_entries()

    @on(Input.Submitted, "#search")
    @on(Button.Pressed, "#open")
    def open_selected(self) -> None:
        self.action_open_selected()

    def action_open_selected(self) -> None:
        table = self.query_one("#experiments", DataTable)
        if table.row_count:
            self.app.open_experiment(str(table.get_row_at(table.cursor_row)[0]))

    def action_focus_search(self) -> None:
        self.query_one("#search", Input).focus()

    def action_focus_new(self) -> None:
        self.query_one("#new-name", Input).focus()

    def action_focus_list(self) -> None:
        self.query_one("#experiments", DataTable).focus()

    @on(DataTable.RowHighlighted, "#experiments")
    def preview_row(self, event: DataTable.RowHighlighted) -> None:
        record = next((entry for entry in self._entries if entry["name"] == event.row_key.value), None)
        if record:
            status = record["runs"][-1]["status"] if record["runs"] else "new"
            self.query_one("#menu-message", Static).update(
                f"{record['name']}  ·  {len(record['runs'])} runs  ·  {status}  ·  Enter to open")

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
        yield Static(f"MBA / {self.experiment_name} / CONFIGURATION", id="config-title")
        yield Tree("Configuration", id="config-tree")
        yield Label("Enter expands a section. Select a value to edit; dataset_sources is managed by preparation.", id="field-label")
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
        expanded = set()

        def remember(node) -> None:
            if node.is_expanded and isinstance(node.data, tuple):
                expanded.add(node.data)
            for child in node.children:
                remember(child)

        remember(tree.root)
        tree.root.remove_children()
        values = sessions.config_values(self.experiment_name, root=self.app.experiments_root)
        values.setdefault("generation", {}).setdefault("max_total_tasks_per_image", None)

        def add(parent, data: dict, path: tuple[str, ...] = ()) -> None:
            for key, value in data.items():
                current = (*path, key)
                if isinstance(value, dict):
                    node = parent.add(key, data=current)
                    add(node, value, current)
                    if current in expanded or (not expanded and current == ("llm",)):
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
        self._progress_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self.current_run_id: str | None = None

    def compose(self) -> ComposeResult:
        yield Static(f"MBA · {self.experiment_name}", id="session-title")
        yield Static("Ready", id="session-status")
        with Horizontal(id="session-body"):
            with Vertical(id="chat-pane"):
                yield RichLog(id="conversation", wrap=True, highlight=True, min_width=1)
                yield Input(placeholder="Describe a task or answer the current question", id="message")
            with Vertical(id="side-pane"):
                yield DataTable(id="runs", cursor_type="row", zebra_stripes=True)
                yield RichLog(id="run-detail", wrap=True, highlight=True, min_width=1)
                yield RichLog(id="progress", wrap=True, highlight=True, min_width=1, max_lines=600)
        with Horizontal(id="session-actions"):
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
        self.set_interval(0.1, self._flush_progress)
        self.set_timer(0.1, self._start_next)

    def _write_message(self, role: str, text: str) -> None:
        label = "YOU" if role in {"user", "you"} else "MBA"
        color = "#9ec3ef" if label == "YOU" else "#72d8c2"
        line = Text()
        line.append(f" {label} ", style=f"bold {color}")
        line.append("  " + text)
        self.query_one("#conversation", RichLog).write(line)

    def _set_status(self, text: str) -> None:
        self.query_one("#session-status", Static).update(text)

    def _log_progress(self, text: str) -> None:
        if text:
            color = ("#72d8c2" if text.startswith(("[done]", "[artifact]")) else
                     "#e8ac70" if text.startswith(("[waiting]", "Benchmark construction failed:")) else
                     "#a5b8c8")
            self.query_one("#progress", RichLog).write(Text(text, style=color))

    def _flush_progress(self, limit: int | None = 100) -> None:
        count = 0
        while limit is None or count < limit:
            try:
                line = self._progress_queue.get_nowait()
            except queue.Empty:
                break
            self._log_progress(line)
            count += 1

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
            from medbenchagent.data.dataset_catalog import load_dataset_catalog
            from medbenchagent.runtime.session_supervisor import classify_requirement

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
        from langgraph.checkpoint.sqlite import SqliteSaver

        runs_root = Path(run["runs_root"])
        if not (runs_root / "checkpoints.sqlite").is_file():
            raise ValueError("Run checkpoint database is missing")
        connection = sqlite3.connect(str(runs_root / "checkpoints.sqlite"))
        try:
            graph = _compile_graph(SqliteSaver(connection))
            snapshot = graph.get_state({"configurable": {"thread_id": run["thread_id"]}})
            path = snapshot.values.get("context", {}).get("effective_config_path")
        finally:
            connection.close()
        if not path:
            raise ValueError("Run checkpoint or frozen configuration is missing")
        with run_settings(path) as saved:
            return replace(saved, paths=replace(saved.paths, runs_root=runs_root))

    def _launch_run(self, run: dict, *, resume: bool) -> None:
        settings = None
        if not resume:
            try:
                settings = self._settings_for_run(run, resume=False)
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
        from langgraph.checkpoint.sqlite import SqliteSaver
        from medbenchagent.cli import run_interactive_build

        run = next(run for run in self.record["runs"] if run["id"] == run_id)
        errors: list[str] = []
        if resume:
            try:
                settings = self._settings_for_run(run, resume=True)
            except Exception as exc:
                self.app.call_from_thread(self._run_finished, run_id,
                                          {"status": "blocked", "summary": f"Cannot resume run: {exc}"})
                return

        def output(text: str) -> None:
            if text.startswith(("Benchmark construction failed:", "Local preflight failed:")):
                errors.append(text)
            self._progress_queue.put(text)

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
                graph = _compile_graph(SqliteSaver(connection))
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
        self._flush_progress(limit=None)
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
    App, Screen { background: #0d131b; color: #dce6ef; }
    Screen { layout: vertical; }
    Header { background: #15212b; color: #dce6ef; }
    Footer { background: #15212b; color: #a5b8c8; }
    ExperimentMenu { align: center middle; }
    #menu-shell { width: 84; max-width: 100%; height: 1fr; align: center middle; padding: 1 2; }
    #menu-brand { height: 3; content-align: center middle; color: #72d8c2; text-style: bold; }
    #menu-subtitle { height: 2; content-align: center middle; color: #869bab; }
    #menu-panel { width: 100%; height: auto; max-height: 26; background: #151e28; padding: 1 2; }
    #menu-title { height: 2; color: #72d8c2; text-style: bold; padding: 0 1; }
    #menu-search-bar, #menu-create-bar { width: 100%; height: 3; }
    #menu-create-bar { border-top: solid #263c45; padding-top: 1; height: 4; }
    #menu-message { height: 2; color: #a5b8c8; padding: 0 1; }
    #menu-hint { height: 2; content-align: center middle; color: #708696; }
    #experiments { width: 100%; height: 8; background: #151e28; border: none; }
    #experiments:focus { border: none; }
    #config-tree { height: 1fr; border: round #315456; background: #151e28; }
    #config-title { height: 2; padding: 0 2; color: #72d8c2; text-style: bold; }
    #field-label, #config-message { padding: 0 1; color: #a5b8c8; }
    #session-title { height: 2; padding: 0 2; color: #72d8c2; text-style: bold; }
    #session-status { height: 2; padding: 0 2; color: #a5b8c8; }
    #session-body { width: 100%; height: 1fr; }
    #chat-pane { width: 2fr; height: 1fr; min-width: 36; }
    #side-pane { width: 1fr; height: 1fr; min-width: 28; }
    #conversation { width: 100%; height: 1fr; border: round #315456; background: #111b25; }
    #message { width: 100%; height: 3; }
    #runs { width: 100%; height: 30%; border: round #315456; background: #151e28; }
    #run-detail { width: 100%; height: 35%; border: round #263c45; background: #111b25; }
    #progress { width: 100%; height: 35%; border: round #263c45; background: #111b25; }
    #session-actions { width: 100%; height: 3; background: #151e28; }
    #menu-search-bar, #menu-create-bar, ConfigScreen > Horizontal { height: auto; }
    Button { margin: 0 1; }
    #search, #new-name, #field-value { width: 1fr; }
    Input:focus { border: tall #72d8c2; }
    """

    TITLE = "MBA · MedBenchAgent"

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
