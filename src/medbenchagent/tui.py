"""Keyboard-first persistent terminal interface for MedBenchAgent."""

from __future__ import annotations

import queue
import json
from pathlib import Path

from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Input, Label, OptionList, RichLog, Static, Tree
from textual.widgets.option_list import Option

from medbenchagent.runtime import sessions
from medbenchagent.runtime.conversation_service import ConversationService
from medbenchagent.runtime.graph_runner import RunEvent


_BRAND_LINES = (
    "███╗   ███╗██████╗  █████╗",
    "████╗ ████║██╔══██╗██╔══██╗",
    "██╔████╔██║██████╔╝███████║",
    "██║╚██╔╝██║██╔══██╗██╔══██║",
    "██║ ╚═╝ ██║██████╔╝██║  ██║",
    "╚═╝     ╚═╝╚═════╝ ╚═╝  ╚═╝",
)


def _brand() -> Text:
    title = Text()
    for index, line in enumerate(_BRAND_LINES):
        title.append(line, style=f"bold {('#4386f7', '#5598ff', '#6babff', '#82bdff', '#73aafa', '#4e8ef1')[index]}")
        if index < len(_BRAND_LINES) - 1:
            title.append("\n")
    return title


def _update_slash_list(options: OptionList, value: str,
                       commands: tuple[tuple[str, str], ...]) -> None:
    if not value.startswith("/"):
        options.display = False
        return
    query = value[1:].casefold()
    matches = [(name, description) for name, description in commands if name.startswith(query)]
    options.set_options(Option(f"/{name}    {description}", id=name) for name, description in matches)
    options.display = bool(matches)
    if matches:
        options.highlighted = 0


def _navigate_slash_list(event: events.Key, field: Input, options: OptionList) -> None:
    if not options.display or not field.has_focus or event.key not in {"down", "up"}:
        return
    if options.option_count:
        current = options.highlighted or 0
        options.highlighted = (current + (1 if event.key == "down" else -1)) % options.option_count
    event.prevent_default()
    event.stop()


class ExperimentMenu(Screen):
    COMMANDS = (("new", "Create an experiment"), ("open", "Open the selected experiment"),
                ("quit", "Exit MedBenchAgent"))

    def __init__(self) -> None:
        super().__init__()
        self._entries: list[dict] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="menu-shell"):
            yield Static(_brand(), id="menu-brand")
            yield Static("Choose a workspace or create a new one  /  experiments", id="menu-subtitle")
            with Vertical(id="menu-panel"):
                yield Static("RECENT EXPERIMENTS", id="menu-title")
                with Horizontal(id="menu-search-bar"):
                    yield Input(placeholder="Search experiments or type / for commands", id="search")
                    yield Button("Open", id="open", variant="primary")
                yield OptionList(id="menu-commands")
                yield DataTable(id="experiments", cursor_type="row", zebra_stripes=True)
                yield Static("", id="menu-message")
            yield Static("Type / for commands  ·  Enter opens a selected experiment", id="menu-hint")

    def on_mount(self) -> None:
        self.query_one("#experiments", DataTable).add_columns("EXPERIMENT", "RUNS", "STATUS")
        self.query_one("#menu-commands", OptionList).display = False
        self.refresh_experiments()
        self.query_one("#search", Input).focus()

    def on_screen_resume(self) -> None:
        self.app.title = "MBA"
        self.refresh_experiments()

    def refresh_experiments(self) -> None:
        self._entries = sessions.list_experiments(root=self.app.experiments_root)
        self._render_entries()

    def _render_entries(self) -> None:
        table = self.query_one("#experiments", DataTable)
        table.clear()
        query = self.query_one("#search", Input).value.strip().casefold()
        if query.startswith("/"):
            query = ""
        matches = [record for record in self._entries if query in record["name"].casefold()]
        for record in matches:
            status = record["runs"][-1]["status"] if record["runs"] else "new"
            status_color = {"complete": "#63d6bd", "blocked": "#e8ac70",
                            "failed": "#eb7d83", "running": "#a9c7f6"}.get(status, "#aab9ca")
            table.add_row(record["name"], str(len(record["runs"])),
                          Text(status.upper(), style=status_color), key=record["name"])
        if not matches:
            message = "No matching experiments. Type /new to create." if query else "No experiments yet. Type /new to create."
            self.query_one("#menu-message", Static).update(message)
        else:
            self.query_one("#menu-message", Static).update("")

    @on(Input.Changed, "#search")
    def filter_experiments(self) -> None:
        _update_slash_list(self.query_one("#menu-commands", OptionList),
                           self.query_one("#search", Input).value, self.COMMANDS)
        self._render_entries()

    @on(Input.Submitted, "#search")
    def submit_search(self, event: Input.Submitted) -> None:
        if event.value.startswith("/"):
            self._dispatch_command(event.value)
        else:
            self.action_open_selected()

    @on(OptionList.OptionSelected, "#menu-commands")
    def select_command(self, event: OptionList.OptionSelected) -> None:
        self._dispatch_command("/" + str(event.option_id))

    def _dispatch_command(self, value: str) -> None:
        name = value[1:].strip().casefold()
        options = self.query_one("#menu-commands", OptionList)
        if name not in {item[0] for item in self.COMMANDS}:
            selected = options.highlighted_option
            name = selected.id if selected else name
        self.query_one("#search", Input).value = ""
        options.display = False
        if name == "new":
            self.app.push_screen(CreateExperimentScreen())
        elif name == "open":
            self.action_open_selected()
        elif name == "quit":
            self.app.exit()
        else:
            self.query_one("#menu-message", Static).update(f"Unknown command: /{name}")

    def on_key(self, event: events.Key) -> None:
        search = self.query_one("#search", Input)
        options = self.query_one("#menu-commands", OptionList)
        if search.has_focus and event.key == "down" and not options.display:
            self.query_one("#experiments", DataTable).focus()
            event.prevent_default()
            event.stop()
            return
        _navigate_slash_list(event, search, options)

    def action_open_selected(self) -> None:
        table = self.query_one("#experiments", DataTable)
        if table.row_count:
            self.app.open_experiment(str(table.get_row_at(table.cursor_row)[0]))

    @on(Button.Pressed, "#open")
    def open_button(self) -> None:
        self.action_open_selected()

    @on(DataTable.RowSelected, "#experiments")
    def open_row(self, event: DataTable.RowSelected) -> None:
        self.app.open_experiment(str(event.row_key.value))


class CreateExperimentScreen(Screen):
    def compose(self) -> ComposeResult:
        with Vertical(id="create-dialog"):
            yield Static("Create experiment", id="create-title")
            yield Input(placeholder="Experiment name", id="new-name")
            yield Static("Lowercase letters, digits and hyphens. Type /quit to cancel.", id="create-message")

    def on_mount(self) -> None:
        self.query_one("#new-name", Input).focus()

    @on(Input.Submitted, "#new-name")
    def create(self, event: Input.Submitted) -> None:
        name = event.value.strip()
        if name == "/quit":
            self.app.pop_screen()
            return
        try:
            sessions.create_experiment(name, root=self.app.experiments_root,
                                       template=self.app.config_template)
        except (ValueError, OSError) as exc:
            self.query_one("#create-message", Static).update(str(exc))
            return
        self.app.pop_screen()
        self.app.open_experiment(name, configure=True)


class ConfigScreen(Screen):
    COMMANDS = (("done", "Enter the conversation"), ("quit", "Return to the previous screen"))

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
        yield Static("", id="config-message")
        yield OptionList(id="config-commands")
        yield Input(placeholder="Type / for commands", id="config-command")

    def on_mount(self) -> None:
        self.query_one("#config-commands", OptionList).display = False
        self.refresh_tree()

    @on(Input.Changed, "#config-command")
    def filter_commands(self) -> None:
        _update_slash_list(self.query_one("#config-commands", OptionList),
                           self.query_one("#config-command", Input).value, self.COMMANDS)

    @on(Input.Submitted, "#config-command")
    def submit_command(self, event: Input.Submitted) -> None:
        self._dispatch_command(event.value)

    @on(OptionList.OptionSelected, "#config-commands")
    def select_command(self, event: OptionList.OptionSelected) -> None:
        self._dispatch_command("/" + str(event.option_id))

    def _dispatch_command(self, value: str) -> None:
        name = value[1:].strip().casefold() if value.startswith("/") else ""
        options = self.query_one("#config-commands", OptionList)
        if name not in {item[0] for item in self.COMMANDS}:
            selected = options.highlighted_option
            name = selected.id if selected else name
        self.query_one("#config-command", Input).value = ""
        options.display = False
        if name == "done":
            self.action_save()
        elif name == "quit":
            self.action_back()
        else:
            self.query_one("#config-message", Static).update(f"Unknown command: /{name}")

    def on_key(self, event: events.Key) -> None:
        _navigate_slash_list(event, self.query_one("#config-command", Input),
                             self.query_one("#config-commands", OptionList))

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

    def action_back(self) -> None:
        self.app.pop_screen()


class ConversationScreen(Screen):
    COMMANDS = (("details", "Toggle run details"), ("resume", "Resume an unfinished run"),
                ("config", "Edit future run settings"), ("queue", "Queue a request: /queue <text>"),
                ("quit", "Return to experiments"))

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
        path = sessions.experiment_path(self.experiment_name, self.app.experiments_root)
        yield Static(f"{self.experiment_name}    {path}", id="session-title")
        yield Static("Ready", id="session-status")
        with Horizontal(id="session-body"):
            with Vertical(id="chat-pane"):
                yield RichLog(id="conversation", wrap=True, highlight=True, min_width=1)
                yield OptionList(id="session-commands")
                yield Input(placeholder="Describe a task or type / for commands", id="message")
            with Vertical(id="side-pane"):
                yield DataTable(id="runs", cursor_type="row", zebra_stripes=True)
                yield RichLog(id="run-detail", wrap=True, highlight=True, min_width=1)
                yield RichLog(id="progress", wrap=True, highlight=True, min_width=1, max_lines=600)

    def on_mount(self) -> None:
        self.service = ConversationService(self.experiment_name, root=self.app.experiments_root)
        self.record = self.service.reconcile()
        self.query_one("#runs", DataTable).add_columns("Type", "Status", "Run")
        self.query_one("#session-commands", OptionList).display = False
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

    @on(Input.Changed, "#message")
    def filter_commands(self) -> None:
        _update_slash_list(self.query_one("#session-commands", OptionList),
                           self.query_one("#message", Input).value, self.COMMANDS)

    @on(OptionList.OptionSelected, "#session-commands")
    def select_command(self, event: OptionList.OptionSelected) -> None:
        self._dispatch_command("/" + str(event.option_id))

    def on_key(self, event: events.Key) -> None:
        _navigate_slash_list(event, self.query_one("#message", Input),
                             self.query_one("#session-commands", OptionList))

    def _dispatch_command(self, value: str) -> None:
        name, _, argument = value[1:].strip().partition(" ")
        name = name.casefold()
        options = self.query_one("#session-commands", OptionList)
        if name not in {item[0] for item in self.COMMANDS} and not argument:
            selected = options.highlighted_option
            name = selected.id if selected else name
        options.display = False
        field = self.query_one("#message", Input)
        field.value = ""
        if name == "details":
            self.action_toggle_details()
        elif name == "resume":
            self.action_resume_run()
        elif name == "config":
            self.action_edit_config()
        elif name == "queue":
            if argument.strip():
                self._queue_task(argument.strip())
            else:
                field.value = "/queue "
                field.focus()
                self._set_status("Enter the request after /queue and press Enter.")
        elif name == "quit":
            self.action_back()
        else:
            self._set_status(f"Unknown command: /{name}")

    @on(Input.Submitted, "#message")
    def submit(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if not value:
            return
        if value.startswith("/"):
            self._dispatch_command(value)
            return
        event.input.value = ""
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
                self._set_status("Unfinished run available. Type /resume to continue.")
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
            self.query_one("#message", Input).placeholder = "Describe a task or type / for commands"

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

    def action_toggle_details(self) -> None:
        sidebar = self.query_one("#side-pane", Vertical)
        sidebar.display = not sidebar.display

    def action_edit_config(self) -> None:
        self.app.push_screen(ConfigScreen(self.experiment_name))

    def action_back(self) -> None:
        if self.running:
            self.app.exit()
            return
        self.app.pop_screen()

class MBAApp(App):
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    App, Screen { background: #101319; color: #d9dce5; }
    Screen { layout: vertical; }
    Button { margin: 0 1 0 0; min-width: 10; background: #242b36; color: #d9dce5; border: none; }
    Button:focus { background: #34445f; }
    Input { background: #1a2029; border: tall #323c4b; }
    Input:focus { border: tall #82aaff; }
    DataTable { background: #171c24; color: #d9dce5; }
    OptionList { background: #202938; border: solid #4386f7; height: auto; max-height: 8; }

    #menu-shell { width: 100%; height: 1fr; padding: 1 3 0 3; }
    #menu-brand { height: 6; color: #5b9cff; text-style: bold; }
    #menu-subtitle { height: 2; color: #a5b3c9; }
    #menu-panel { width: 100%; height: 1fr; background: #171c24; padding: 1 2; }
    #menu-title { height: 1; color: #a8c4ff; text-style: bold; }
    #menu-search-bar { width: 100%; height: 3; }
    #menu-message { height: 1; color: #9aa6b6; }
    #menu-hint { height: 1; color: #718095; }
    #experiments { width: 100%; height: 1fr; min-height: 3; border: none; }
    #experiments:focus { border: none; }
    #search, #field-value { width: 1fr; }
    #open { width: 12; }

    CreateExperimentScreen { align: center middle; }
    #create-dialog { width: 62; max-width: 100%; height: auto; padding: 1 2;
                     background: #171c24; border: solid #4386f7; }
    #create-title { height: 1; color: #82bdff; text-style: bold; }
    #new-name { height: 3; }
    #create-message { height: 2; color: #9aa6b6; }

    ConfigScreen { padding: 1 2 0 2; }
    #config-title { height: 1; color: #a8c4ff; text-style: bold; }
    #config-tree { height: 1fr; background: #171c24; border: none; }
    #field-label, #config-message { height: 1; color: #9aa6b6; }
    ConfigScreen > Horizontal { height: 3; }
    #config-command { height: 3; }

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
    """

    TITLE = "MBA"

    def __init__(self, *, experiments_root: Path = sessions.EXPERIMENTS_ROOT,
                 config_template: Path | None = None) -> None:
        super().__init__()
        self.experiments_root = experiments_root
        self.config_template = config_template

    def on_mount(self) -> None:
        self.push_screen(ExperimentMenu())

    def open_experiment(self, name: str, *, configure: bool = False) -> None:
        self.title = name
        if configure:
            self.push_screen(ConfigScreen(name, new=True))
        else:
            self.push_screen(ConversationScreen(name))

