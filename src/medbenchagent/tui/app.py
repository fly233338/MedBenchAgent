"""Textual Run workbench. Compilation owns execution; screens own navigation."""
import threading
from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, DataTable, Static, OptionList, TextArea

from medbenchagent.settings import load_paths
from medbenchagent.runtime import sessions
from medbenchagent.runtime.activity import AGENTS, elapsed
from medbenchagent.runtime.conversation_service import ConversationService
from medbenchagent.runtime.intake import InputDraft, suggest_draft, check_connections
from medbenchagent.runtime.construction import sanitize
from .widgets import Composer, ChatMessage
from .dialogs import DraftScreen, ConnectionScreen, ActivityScreen, ArtifactsScreen
from .screens import RunsScreen, NameScreen, DatasetScreen

STATES = {"draft": "DRAFT", "ready": "READY", "running": "RUNNING", "paused": "PAUSED", "waiting": "WAITING",
          "stopped": "STOPPED", "interrupted": "INTERRUPTED", "external": "STUDIO"}
RESULTS = {"succeeded": "SUCCESS", "partial": "PARTIAL", "blocked": "BLOCKED", "failed": "FAILED"}
COMMANDS = {"new": "New run", "runs": "Browse runs", "dataset": "Select dataset",
            "config": "Configure", "pause": "Pause", "resume": "Start / resume",
            "events": "Trace", "artifacts": "Artifacts", "export": "Export", "help": "Keys"}


class MBAApp(App):
    TITLE = "MedD2B"
    CSS_PATH = "app.tcss"
    BINDINGS = [Binding("ctrl+c", "request_quit", "Quit", priority=True),
                Binding("ctrl+n", "new_run", "New run"), Binding("ctrl+r", "runs", "Runs"),
                Binding("ctrl+p", "commands", "Commands")]

    def __init__(self, *, root=None, runtime_root=None, config_template=None):
        super().__init__()
        paths = load_paths(config_template)
        self.root = Path(root or paths["runs_root"])
        self.runtime_root = Path(runtime_root or paths["runtime_root"])
        self.config_template = config_template
        self.selected_run = self.active_run = None
        self.running = self.exit_after = False
        self.stop_event = threading.Event()
        self.view, self.latest_activity = {}, {}
        self.draft = InputDraft()
        self.reader = None
        self.reading = False
        self.seen_messages = set()
        self.unread = self.selection = 0
        self.rendering_history = False
        self.pending_answer = None
        self.command_matches = []

    @property
    def service(self):
        return ConversationService(sessions.run_path(self.selected_run, self.root))

    def compose(self) -> ComposeResult:
        with Horizontal(id="toolbar"):
            yield Button("Runs", id="runs")
            yield Static("MedD2B", id="title")
            yield Button("Clone", id="copy-run")
            yield Button("New", id="new")
        yield Static("Select a run.", id="run-status")
        with Horizontal(id="navigation"):
            yield Button("Task", id="task")
            yield Button("Dataset", id="dataset")
            yield Button("Config", id="settings")
            yield Button("Trace", id="details")
            yield Button("Artifacts", id="products")
        with Horizontal(id="workbench"):
            with Vertical(id="conversation-column"):
                yield VerticalScroll(id="conversation")
                yield Static("", id="new-content")
            with Vertical(id="activity-column"):
                yield Static("AGENT ACTIVITY", classes="section-label")
                yield DataTable(id="agents", cursor_type="row")
                yield Static("Select an agent for details.", id="activity-preview")
        yield OptionList(id="command-options")
        yield Composer(id="message", soft_wrap=True)
        with Horizontal(id="controls"):
            yield Button("Pause", id="pause", disabled=True)
            yield Button("Start", id="resume", disabled=True, variant="primary")
            yield Static("Enter send · Shift+Enter newline · ^P commands", id="key-hint")
            yield Button("Send", id="send")

    def on_mount(self):
        self.query_one("#agents", DataTable).add_columns("Agent", "Activity", "Time")
        self.query_one("#new-content").display = False
        self.query_one("#command-options").display = False
        self.set_interval(0.25, self.poll_activity)
        self.action_runs()

    def on_resize(self, event):
        for screen in self.screen_stack:
            screen.set_class(event.size.width < 110, "compact")
            screen.set_class(event.size.width < 80 or event.size.height < 24, "too-small")
        if event.size.width < 80 or event.size.height < 24:
            self.notify("Resize to 80x24+", timeout=3)

    def action_runs(self):
        if isinstance(self.screen, RunsScreen):
            self.screen.refresh_runs()
        else:
            self.push_screen(RunsScreen())

    def action_new_run(self):
        self.push_screen(NameScreen(), lambda name: self.create_named_run(name) if name else None)

    def action_copy_run(self):
        if self.selected_run:
            source = self.service.directory
            self.push_screen(NameScreen(copy_name=self.selected_run), lambda name: self.create_named_run(name, source) if name else None)

    @work(thread=True, group="create-run", exclusive=True)
    def create_named_run(self, name, source=None):
        try:
            options = {"root": self.root, "runtime_root": self.runtime_root}
            record = (sessions.copy_session(source, name, **options) if source else
                      sessions.create_session(name, template=self.config_template, **options))
            self.call_from_thread(self.open_created, record["name"])
        except Exception as exc:
            self.call_from_thread(self.notify, str(sanitize(str(exc))), severity="error")

    def open_created(self, name):
        if isinstance(self.screen, RunsScreen):
            self.pop_screen()
        self.select_run(name)

    def select_run(self, name, *, preserve_scroll=False):
        history = self.query_one("#conversation", VerticalScroll)
        viewport = (history.is_vertical_scroll_end, history.scroll_y) if preserve_scroll else (True, 0)
        self.selected_run = name
        self.selection += 1
        if not preserve_scroll:
            self.unread = 0
            self.query_one("#new-content").display = False
            self.reader = self.service.activity_reader()
            self.latest_activity.clear()
        self.load_run(name, self.selection, viewport)

    @work(thread=True, group="load-view", exclusive=True)
    def load_run(self, name, selection, viewport):
        try:
            service = ConversationService(sessions.run_path(name, self.root))
            self.call_from_thread(self.loaded, name, selection, service.inspect(), service.record(), viewport)
        except Exception as exc:
            self.call_from_thread(self.notify, str(sanitize(str(exc))), severity="error")

    def loaded(self, name, selection, view, record, viewport):
        if (name, selection) != (self.selected_run, self.selection):
            return
        self.view = view
        self.draft = InputDraft.model_validate(record.get("draft", {}))
        self.rendering_history = True
        self.query_one("#conversation", VerticalScroll).remove_children()
        self.seen_messages.clear()
        history = view.get("history", [])
        if view["status"] == "draft":
            self.chat("SETUP", self.draft.goal or "Set input. Define goal. Run.")
        for after_work in range(len(history) + 1):
            for index, message in enumerate(record["messages"]):
                if message.get("after_work", 0) == after_work:
                    self.chat("you" if message["role"] == "user" else "INPUT REQUIRED", message["text"],
                              identity=message.get("id", f"message-{index}"))
            if after_work < len(history):
                item = history[after_work]
                self.chat(AGENTS.get(item["agent_id"], item["agent_id"]), item["summary"], identity=item["request_id"])
        self.show_question(view)
        if view["status"] == "stopped":
            self.chat("RESULT", f"{RESULTS.get(view.get('goal_result'), '')}\n{view.get('summary', '')}", identity="finish")
        if view.get("error"):
            self.chat("INTERRUPTED", view["error"], identity="execution-error")
        self.rendering_history = False
        history = self.query_one("#conversation", VerticalScroll)
        self.call_after_refresh(history.scroll_end, animate=False) if viewport[0] else self.call_after_refresh(history.scroll_to, y=viewport[1], animate=False)
        self.render_status()
        self.render_agents()

    def chat(self, role, text, *, identity=None):
        if not text or identity is not None and identity in self.seen_messages:
            return
        if identity is not None:
            self.seen_messages.add(identity)
        history = self.query_one("#conversation", VerticalScroll)
        follow = history.is_vertical_scroll_end
        history.mount(ChatMessage(role, str(sanitize(str(text)))))
        if self.rendering_history:
            return
        if follow:
            self.call_after_refresh(history.scroll_end, animate=False)
        else:
            self.unread += 1
            self.query_one("#new-content", Static).update(f"↓ {self.unread} new")
            self.query_one("#new-content").display = True

    def show_question(self, view):
        if view.get("questions"):
            question = view["questions"][0]
            self.chat("INPUT REQUIRED", question.get("question", str(question)),
                      identity="question-" + str(view.get("question_id")))
            self.query_one(Composer).focus()

    def render_status(self):
        active = self.running and self.selected_run == self.active_run
        status = "PAUSING" if active and self.stop_event.is_set() else "RUNNING" if active else STATES.get(self.view.get("status"), "Select run")
        if self.running and not active:
            status += f" · {self.active_run} RUNNING"
        quantity = ""
        if counts := self.view.get("counts"):
            quantity = f" · {self.view.get('count_kind', '')} {counts.get('accepted', '?')}/{counts.get('requested', '?')}"
        self.query_one("#run-status", Static).update(Text(f"{status}{quantity} · {RESULTS.get(self.view.get('goal_result'), '')}", overflow="ellipsis", no_wrap=True))
        self.query_one("#title", Static).update(Text(f"MedD2B / {self.selected_run or 'Runs'}", overflow="ellipsis", no_wrap=True))
        self.query_one("#pause", Button).disabled = not active or self.stop_event.is_set()
        self.query_one("#resume", Button).disabled = self.running or not self.selected_run or self.view.get("status") not in {"draft", "ready", "paused", "interrupted"}
        self.query_one("#resume", Button).label = "Start" if self.view.get("status") in {"draft", "ready"} else "Resume"
        for selector in ("#settings", "#task", "#dataset"):
            self.query_one(selector, Button).disabled = not self.selected_run or active

    def render_agents(self):
        table = self.query_one("#agents", DataTable)
        selected = table.cursor_row
        table.clear()
        enabled = set(self.view.get("enabled_agents", [])) | self.latest_activity.keys()
        for agent, label in AGENTS.items():
            if agent not in enabled:
                continue
            row = self.latest_activity.get(agent)
            text = row["action"] if row else "idle"
            if row and row["running"] and self.view.get("status") != "running":
                text = "last: " + text
            duration = f"{elapsed(row)}s" if row and row["running"] and self.running and self.selected_run == self.active_run else (
                f"{row['event']['elapsed_ms'] / 1000:.1f}s" if row and row["event"].get("elapsed_ms") is not None else "—")
            table.add_row(label, Text(text, overflow="ellipsis", no_wrap=True), duration, key=agent)
        if enabled:
            table.move_cursor(row=min(selected, len(enabled) - 1))

    def poll_activity(self):
        if not self.query("#conversation"):
            return
        if self.query_one("#conversation", VerticalScroll).is_vertical_scroll_end and self.unread:
            self.unread = 0
            self.query_one("#new-content").display = False
        if self.reader and not self.reading:
            self.reading = True
            self.read_activity(self.reader, self.selected_run)

    @work(thread=True, group="activity")
    def read_activity(self, reader, name):
        try:
            rows = reader.read()
            if self.is_running:
                self.call_from_thread(self.activities, reader, name, rows)
        finally:
            self.reading = False

    def activities(self, reader, name, rows):
        if reader is not self.reader or name != self.selected_run:
            return
        for row in rows:
            self.latest_activity[row["agent_id"]] = row
        if rows:
            self.render_agents()
            self.query_one("#activity-preview", Static).update(Text(rows[-1]["action"] + "\n" + rows[-1].get("summary", ""), overflow="ellipsis"))
            if isinstance(self.screen, ActivityScreen):
                self.screen.update_rows(list(reader.events))

    @on(DataTable.RowSelected, "#agents")
    def activity_selected(self, event):
        self.push_screen(ActivityScreen(list(self.reader.events) if self.reader else [], str(event.row_key.value)))

    def configure_task(self):
        if not self.selected_run:
            return self.action_new_run()
        if self.view.get("status") != "draft":
            self.notify("Run frozen. Clone to edit.")
            return
        self.push_screen(DraftScreen(self.draft.model_copy(deep=True), self.service.directory), self.draft_ready)

    def draft_ready(self, result):
        if result is None:
            return
        self.draft, start = result
        try:
            self.service.save_draft(self.draft)
        except (ValueError, OSError) as exc:
            self.notify(str(exc), severity="error")
            return
        if start and not self.running:
            self.start_execution(draft=self.draft)
        else:
            self.notify("Draft saved.")
            self.select_run(self.selected_run)

    @on(Composer.Submitted)
    def submitted(self, event):
        self.send(event.value)

    @on(TextArea.Changed, "#message")
    def command_changed(self):
        text = self.query_one(Composer).text
        self.command_matches = [key for key in COMMANDS if key.startswith(text[1:])] if text.startswith("/") and "\n" not in text else []
        options = self.query_one("#command-options", OptionList)
        options.clear_options()
        options.add_options([f"/{key}  {COMMANDS[key]}" for key in self.command_matches])
        options.display = bool(self.command_matches)
        if self.command_matches:
            options.highlighted = 0

    @on(Composer.CommandKey)
    def command_key(self, event):
        options = self.query_one("#command-options", OptionList)
        if not self.command_matches:
            return
        index = options.highlighted or 0
        if event.key == "tab":
            self.query_one(Composer).load_text("/" + self.command_matches[index])
        else:
            options.highlighted = (index + (1 if event.key == "down" else -1)) % len(self.command_matches)

    @on(OptionList.OptionSelected, "#command-options")
    def command_selected(self, event):
        self.send("/" + self.command_matches[event.option_index])

    def action_commands(self):
        self.query_one(Composer).load_text("/")
        self.query_one(Composer).focus()

    def send(self, text):
        text = text.strip()
        if not text:
            return
        if text.startswith("/"):
            command = text[1:].split()[0] if text[1:] else ""
            if command not in COMMANDS:
                self.notify("Unknown command. /help")
                return
            self.query_one(Composer).load_text("")
            self.dispatch(command)
            return
        if self.running:
            self.notify("Busy. Awaiting agent output.")
        elif self.view.get("questions"):
            self.pending_answer = text
            self.start_execution(answer=text)
        elif self.selected_run:
            if self.view.get("status") != "draft":
                self.notify("Run frozen. Clone to edit.")
                return
            self.draft = suggest_draft(text, self.draft)
            self.configure_task()
        else:
            self.action_new_run()

    def dispatch(self, action):
        if action == "new":
            return self.action_new_run()
        if action == "runs":
            return self.action_runs()
        if action == "help":
            self.notify("^N new · ^R runs · ^P commands · Tab complete · Enter send · ^C quit", timeout=12)
            return
        if not self.selected_run:
            self.notify("Select a run.")
            return
        if action in {"dataset", "config"} and self.running and self.selected_run == self.active_run:
            self.notify("Pause before editing.")
            return
        if action == "dataset":
            self.push_screen(DatasetScreen(self.draft, directory=self.service.directory, catalog=self.runtime_root.parent / "datasets.json", frozen=self.view.get("status") != "draft"), self.dataset_selected)
        elif action == "config":
            self.push_screen(ConnectionScreen(self.service.directory / "config.toml", self.draft))
        elif action == "pause":
            if self.running and self.selected_run == self.active_run:
                self.stop_event.set()
                self.render_status()
        elif action == "resume" and not self.running:
            self.configure_task() if self.view.get("status") == "draft" else self.start_execution()
        elif action == "events":
            self.push_screen(ActivityScreen(list(self.reader.events) if self.reader else [], "*"))
        elif action in {"artifacts", "export"}:
            self.push_screen(ArtifactsScreen(self.view))

    def dataset_selected(self, draft):
        if draft:
            self.draft = draft
            self.service.save_draft(draft)
            self.configure_task()

    @on(Button.Pressed)
    def pressed(self, event):
        if isinstance(self.screen, (RunsScreen, NameScreen, DatasetScreen, DraftScreen, ConnectionScreen, ActivityScreen, ArtifactsScreen)):
            return
        action = event.button.id
        if action == "copy-run":
            self.action_copy_run()
        elif action == "task":
            self.configure_task()
        elif action == "send":
            self.send(self.query_one(Composer).text)
        else:
            self.dispatch({"settings": "config", "details": "events", "products": "artifacts"}.get(action, action))

    def start_execution(self, *, draft=None, answer=None):
        if self.running or not self.selected_run:
            return
        self.running = True
        self.active_run = self.selected_run
        self.stop_event.clear()
        self.render_status()
        self.execute(self.selected_run, draft, answer, self.view.get("question_id") if answer is not None else None,
                     self.view.get("status") not in {"draft", "ready"})

    @work(thread=True, exclusive=True, group="execution")
    def execute(self, name, draft, answer, question_id, resume):
        try:
            service = ConversationService(sessions.run_path(name, self.root))
            if draft is not None:
                check_connections(service.directory / "config.toml", draft)
                if self.stop_event.is_set():
                    return
                service.submit_draft(draft)
                self.call_from_thread(self.started, name, draft)
            for event in service.run(resume=resume, answer=answer, question_id=question_id, stop_requested=self.stop_event.is_set):
                self.call_from_thread(self.show_event, name, event)
        except Exception as exc:
            self.call_from_thread(self.execution_error, name, str(sanitize(str(exc))))
        finally:
            self.call_from_thread(self.finished, name)

    def started(self, name, draft):
        if name == self.selected_run:
            self.view = {"status": "running", "goal": draft.goal, "run_directory": str(self.service.directory)}
            self.reader = self.service.activity_reader()
            self.latest_activity.clear()
            self.chat("you", draft.goal, identity="goal")
            self.query_one(Composer).load_text("")

    def show_event(self, name, event):
        if name != self.selected_run:
            return
        if event.kind == "initialized":
            self.view = event.payload
        elif event.kind == "answer_accepted":
            self.chat("you", event.payload["answer"], identity="answer-" + str(event.payload["question_id"]))
            self.query_one(Composer).load_text("")
            self.pending_answer = None
        elif event.kind == "progress":
            self.view = event.payload["view"]
            row = event.payload["work"]
            self.chat(AGENTS.get(row["agent_id"], row["agent_id"]), row["summary"], identity=row["request_id"])
        elif event.kind in {"finished", "needs_input", "paused"}:
            self.view = event.payload
            self.show_question(self.view)
            if event.kind == "finished":
                self.chat("RESULT", f"{RESULTS.get(self.view.get('goal_result'), '')}\n{self.view.get('summary', '')}", identity="finish")
        self.render_status()

    def execution_error(self, name, message):
        if name == self.selected_run:
            self.chat("ERROR", message)
        self.notify(f"{name}: {message}", severity="error")

    def finished(self, name):
        self.running = False
        self.active_run = None
        if self.exit_after:
            self.exit()
        elif name == self.selected_run:
            self.select_run(name, preserve_scroll=True)

    def action_request_quit(self):
        if self.running:
            self.exit_after = True
            self.stop_event.set()
            self.notify("Pausing at checkpoint. Exit pending.")
            self.render_status()
        else:
            self.exit()
