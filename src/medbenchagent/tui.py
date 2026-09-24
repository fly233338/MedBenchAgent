"""Keyboard-first persistent terminal interface for MedBenchAgent."""

from __future__ import annotations

import queue
import json
from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Input, Label, RichLog, Static, Tree

from medbenchagent.runtime import sessions
from medbenchagent.runtime.conversation_service import ConversationService
from medbenchagent.runtime.graph_runner import RunEvent


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
            yield Static("MedBenchAgent", id="menu-brand")
            yield Static("Experiments  /  choose a workspace or create a new one", id="menu-subtitle")
            with Vertical(id="menu-panel"):
                yield Static("RECENT EXPERIMENTS", id="menu-title")
                with Horizontal(id="menu-search-bar"):
                    yield Input(placeholder="Search experiments  ·  Ctrl+F", id="search")
                    yield Button("Open", id="open", variant="primary")
                yield DataTable(id="experiments", cursor_type="row", zebra_stripes=True)
                yield Static("↑↓ select  ·  Enter open  ·  Ctrl+N new", id="menu-message")
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
                "↑↓ select  ·  Enter open  ·  Ctrl+N new")

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
        yield Label("Edit values for future runs; dataset_sources is managed by preparation.", id="field-label")
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
                Binding("ctrl+n", "focus_queue", "Queue task"),
                Binding("ctrl+b", "toggle_details", "Details"),
                Binding("escape", "back", "Experiments"),
                Binding("ctrl+q", "app.quit", "Quit")]

    def __init__(self, name: str) -> None:
        super().__init__()
        self.experiment_name = name
        self.service: ConversationService | None = None
        self.record: dict = {}
        self.running = False
        self.classifying = False
        self.pending_question = False
        self.current_run_id: str | None = None
        self._displayed_messages: set[str] = set()
        self._progress_queue: queue.SimpleQueue[str] = queue.SimpleQueue()

    def compose(self) -> ComposeResult:
        yield Static(f"MedBenchAgent  /  {self.experiment_name}", id="session-title")
        yield Static("Ready", id="session-status")
        with Horizontal(id="session-body"):
            with Vertical(id="chat-pane"):
                yield RichLog(id="conversation", wrap=True, highlight=True, min_width=1)
                yield Input(placeholder="Describe a task", id="message")
            with Vertical(id="side-pane"):
                yield DataTable(id="runs", cursor_type="row", zebra_stripes=True)
                yield RichLog(id="run-detail", wrap=True, highlight=True, min_width=1)
                yield RichLog(id="progress", wrap=True, highlight=True, min_width=1, max_lines=600)
                with Horizontal(id="queue-bar"):
                    yield Input(placeholder="New task while answering", id="queued-task")
                    yield Button("Queue task", id="queue")
        with Horizontal(id="session-actions"):
            yield Button("Details  ^B", id="toggle-side")
            yield Button("Resume run", id="resume")
            yield Button("Config", id="config")
            yield Button("Experiments", id="back")
        yield Footer()

    def on_mount(self) -> None:
        self.service = ConversationService(self.experiment_name, root=self.app.experiments_root)
        self.record = self.service.reconcile()
        self.query_one("#runs", DataTable).add_columns("Type", "Status", "Run")
        self.refresh_record()
        self.query_one("#run-detail", RichLog).write("Select a run to inspect frozen inputs.")
        self.query_one("#message", Input).focus()
        self.set_interval(0.1, self._flush_progress)
        self.set_timer(0.1, self._start_next)

    def refresh_record(self) -> None:
        self.record = self.service.record()
        for message in self.record["messages"]:
            if message["id"] not in self._displayed_messages:
                self._write_message(message["role"], message["text"])
                self._displayed_messages.add(message["id"])
        self.refresh_runs()

    def _write_message(self, role: str, content: str) -> None:
        label = "YOU" if role in {"user", "you"} else "MBA"
        color = "#9ec3ef" if label == "YOU" else "#72d8c2"
        line = Text()
        line.append(f" {label} ", style=f"bold {color}")
        line.append("  " + content)
        self.query_one("#conversation", RichLog).write(line)

    def _set_status(self, value: str) -> None:
        self.query_one("#session-status", Static).update(value)

    def _log_progress(self, value: str) -> None:
        if value:
            color = ("#72d8c2" if value.startswith(("[done]", "[artifact]")) else
                     "#e8ac70" if value.startswith(("[waiting]", "[failed]")) else "#a5b8c8")
            self.query_one("#progress", RichLog).write(Text(value, style=color))

    def _flush_progress(self, limit: int | None = 100) -> None:
        count = 0
        while limit is None or count < limit:
            try:
                value = self._progress_queue.get_nowait()
            except queue.Empty:
                break
            self._log_progress(value)
            count += 1

    def refresh_runs(self) -> None:
        table = self.query_one("#runs", DataTable)
        table.clear()
        for run in self.record["runs"]:
            table.add_row(run["kind"], run["status"], run["id"][:8], key=run["id"])

    @on(DataTable.RowSelected, "#runs")
    def show_run(self, event: DataTable.RowSelected) -> None:
        run = next(run for run in self.record["runs"] if run["id"] == event.row_key.value)
        detail = (f"{run['kind']} · {run['status']}\n{run['summary']}\n"
                  f"Session Run: {run['id']}\nLangGraph thread: {run['thread_id']}\n"
                  f"Run directory: {run['run_directory']}\n"
                  f"Frozen config: {run['effective_config_path']}\n"
                  f"Catalog snapshot: {run['catalog_snapshot_path']}\n"
                  f"Source decision: {run.get('source_decision_path')}")
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
        value = event.value.strip()
        event.input.value = ""
        if not value:
            return
        if self.pending_question and self.current_run_id:
            try:
                self.service.answer(self.current_run_id, value)
            except Exception as exc:
                self._set_status(str(exc))
                return
            run_id = self.current_run_id
            self.pending_question = False
            self.current_run_id = None
            self.refresh_record()
            self._execute_run(resume_id=run_id, answer=value)
            return
        self._queue_task(value)

    @on(Input.Submitted, "#queued-task")
    @on(Button.Pressed, "#queue")
    def queue_task(self) -> None:
        field = self.query_one("#queued-task", Input)
        value = field.value.strip()
        field.value = ""
        if value:
            self._queue_task(value)
        else:
            field.focus()

    def _queue_task(self, value: str) -> None:
        if self.classifying:
            self._set_status("Wait for the current request to be classified.")
            return
        self.classifying = True
        self._set_status("Interpreting request…")
        self._submit_task(value)

    @work(thread=True)
    def _submit_task(self, value: str) -> None:
        try:
            result = self.service.submit(value)
            self.app.call_from_thread(self._submitted, result)
        except Exception as exc:
            self.app.call_from_thread(self._submission_failed, str(exc))

    def _submitted(self, result) -> None:
        self.classifying = False
        self.refresh_record()
        if result.run_ids:
            self._set_status(f"Queued {len(result.run_ids)} run(s)")
        else:
            self._set_status("Ready")
        self._start_next()

    def _submission_failed(self, message: str) -> None:
        self.classifying = False
        self.refresh_record()
        self._set_status(f"Supervisor error: {message}")

    def _start_next(self) -> None:
        if self.running or self.pending_question:
            return
        self.refresh_record()
        unfinished = next((run for run in reversed(self.record["runs"])
                           if run["status"] in {"running", "interrupted", "waiting_input"}), None)
        if unfinished:
            if unfinished["status"] == "waiting_input":
                self._execute_run(resume_id=unfinished["id"])
            else:
                self._set_status("Unfinished run available. Press Ctrl+R to resume.")
            return
        if self.record["pending_runs"]:
            self._execute_run()

    def _execute_run(self, *, resume_id: str | None = None, answer: str | None = None) -> None:
        if self.running:
            return
        self.running = True
        self._set_status("Resuming run" if resume_id else "Running queued tasks")
        self._execute(resume_id, answer)

    @work(thread=True)
    def _execute(self, resume_id: str | None, answer: str | None) -> None:
        try:
            events = (self.service.resume(resume_id, answer=answer) if resume_id
                      else self.service.drive())
            for event in events:
                if event.kind == "progress":
                    self._progress_queue.put(event.payload.get("text", ""))
                else:
                    self.app.call_from_thread(self._on_event, event)
        except Exception as exc:
            self.app.call_from_thread(self._set_status, f"Run error: {exc}")
        finally:
            self.app.call_from_thread(self._execution_ended)

    def _on_event(self, event: RunEvent) -> None:
        self.refresh_record()
        if event.kind == "initialized":
            self._set_status(f"Running {event.run_session_id[:8]}")
            self._progress_queue.put(f"[run] {event.payload.get('run_directory')}")
        elif event.kind == "artifact":
            self._progress_queue.put(f"[artifact] {event.payload}")
        elif event.kind == "needs_input":
            self.pending_question = True
            self.current_run_id = event.run_session_id
            question = event.payload["question"]
            self._set_status(question)
            self.query_one("#message", Input).placeholder = question
            self.query_one("#message", Input).focus()
        elif event.kind in {"finished", "error"}:
            self._flush_progress(limit=None)
            self._set_status(f"{event.payload.get('status')}: {event.payload.get('summary', '')}")
            self.query_one("#message", Input).placeholder = "Describe a task"

    def _execution_ended(self) -> None:
        self._flush_progress(limit=None)
        self.running = False
        self.refresh_record()
        if not self.pending_question:
            self._start_next()

    def action_resume_run(self) -> None:
        if self.running:
            return
        run = next((run for run in reversed(self.record["runs"])
                    if run["status"] in {"running", "interrupted", "waiting_input"}), None)
        if run:
            self._execute_run(resume_id=run["id"])
        else:
            self._set_status("No unfinished run to resume.")

    @on(Button.Pressed, "#resume")
    def resume_button(self) -> None:
        self.action_resume_run()

    def action_focus_queue(self) -> None:
        self.query_one("#queued-task", Input).focus()

    def action_toggle_details(self) -> None:
        sidebar = self.query_one("#side-pane", Vertical)
        sidebar.display = not sidebar.display

    @on(Button.Pressed, "#toggle-side")
    def toggle_details_button(self) -> None:
        self.action_toggle_details()

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
    App, Screen { background: #101319; color: #d9dce5; }
    Screen { layout: vertical; }
    Footer { background: #171c24; color: #9aa6b6; }
    Button { margin: 0 1 0 0; min-width: 10; background: #242b36; color: #d9dce5; border: none; }
    Button:focus { background: #34445f; }
    Input { background: #1a2029; border: tall #323c4b; }
    Input:focus { border: tall #82aaff; }
    DataTable { background: #171c24; color: #d9dce5; }

    #menu-shell { width: 100%; height: 1fr; padding: 1 3 0 3; }
    #menu-brand { height: 1; color: #a8c4ff; text-style: bold; }
    #menu-subtitle { height: 2; color: #8492a4; }
    #menu-panel { width: 100%; height: 1fr; background: #171c24; padding: 1 2; }
    #menu-title { height: 1; color: #a8c4ff; text-style: bold; }
    #menu-search-bar, #menu-create-bar { width: 100%; height: 3; }
    #menu-create-bar { background: #171c24; }
    #menu-message { height: 1; color: #9aa6b6; }
    #menu-hint { height: 1; color: #718095; }
    #experiments { width: 100%; height: 1fr; min-height: 3; border: none; }
    #experiments:focus { border: none; }
    #search, #new-name, #field-value, #queued-task { width: 1fr; }
    #open, #create { width: 12; }

    ConfigScreen { padding: 1 2 0 2; }
    #config-title { height: 1; color: #a8c4ff; text-style: bold; }
    #config-tree { height: 1fr; background: #171c24; border: none; }
    #field-label, #config-message { height: 1; color: #9aa6b6; }
    ConfigScreen > Horizontal { height: 3; }

    #session-title { height: 1; padding: 0 2; color: #a8c4ff; text-style: bold; }
    #session-status { height: 1; padding: 0 2; color: #9aa6b6; }
    #session-body { width: 100%; height: 1fr; }
    #chat-pane { width: 1fr; height: 1fr; min-width: 24; padding: 0 1 0 2; }
    #side-pane { width: 36; height: 1fr; min-width: 26; background: #171c24; padding: 0 1; }
    #conversation { width: 100%; height: 1fr; background: #101319; border: none; }
    #message { width: 100%; height: 3; }
    #runs { width: 100%; height: 30%; min-height: 4; border: none; }
    #run-detail { width: 100%; height: 35%; min-height: 4; border-top: solid #323c4b; }
    #progress { width: 100%; height: 1fr; min-height: 3; border-top: solid #323c4b; }
    #queue-bar { width: 100%; height: 3; }
    #queue { width: 12; }
    #session-actions { width: 100%; height: 3; background: #171c24; padding: 0 2; }
    """

    TITLE = "MBA · MedBenchAgent"

    def __init__(self, *, experiments_root: Path = sessions.EXPERIMENTS_ROOT,
                 config_template: Path | None = None) -> None:
        super().__init__()
        self.experiments_root = experiments_root
        self.config_template = config_template

    def on_mount(self) -> None:
        self.push_screen(ExperimentMenu())

    def get_system_commands(self, screen: Screen):
        yield from super().get_system_commands(screen)
        if isinstance(screen, ExperimentMenu):
            yield SystemCommand("Search experiments", "Focus the experiment search", screen.action_focus_search)
            yield SystemCommand("New experiment", "Focus the new experiment field", screen.action_focus_new)
            yield SystemCommand("Open experiment", "Open the highlighted experiment", screen.action_open_selected)
        elif isinstance(screen, ConversationScreen):
            yield SystemCommand("Queue task", "Focus the task queue field", screen.action_focus_queue)
            yield SystemCommand("Run details", "Toggle the run detail sidebar", screen.action_toggle_details)
            yield SystemCommand("Resume run", "Resume an unfinished run", screen.action_resume_run)
            yield SystemCommand("Experiment config", "Edit configuration for future runs", screen.action_edit_config)
        elif isinstance(screen, ConfigScreen):
            yield SystemCommand("Enter conversation", "Return to the experiment", screen.action_save)

    def open_experiment(self, name: str, *, configure: bool = False) -> None:
        if configure:
            self.push_screen(ConfigScreen(name, new=True))
        else:
            self.push_screen(ConversationScreen(name))

