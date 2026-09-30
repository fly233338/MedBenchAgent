"""Keyboard-first persistent terminal interface for MedBenchAgent."""

from __future__ import annotations

import queue
import json
import sys
import threading
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

from rich.text import Text
from rich.markdown import Markdown
from medbenchagent.tui_components import ExperimentIndex, Prompt, Splitter, CommandDialog
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, DataTable, Input, Label, OptionList, RichLog, Select, Static, Tree, TextArea
from textual.widgets.option_list import Option

from medbenchagent.runtime import sessions

if TYPE_CHECKING:
    from medbenchagent.runtime.conversation_service import ConversationService
    from medbenchagent.runtime.graph_runner import RunEvent


_STATUS_COLORS = {"complete": "#63d6bd", "success": "#63d6bd", "blocked": "#e8ac70",
                  "waiting_input": "#e8ac70", "needs_input": "#e8ac70", "failed": "#eb7d83",
                  "interrupted": "#e8ac70", "running": "#a9c7f6"}
_MAX_EVENT_LINE_BYTES = 1_048_576


def _progress_color(value: str, status: str) -> str:
    if status in _STATUS_COLORS:
        return _STATUS_COLORS[status]
    if value.startswith(("[done]", "[artifact]")):
        return "#72d8c2"
    if value.startswith(("[waiting]", "[failed]")):
        return "#e8ac70"
    return "#a5b8c8"


def _recent_event_lines(path: Path, limit: int = 600) -> tuple[list[bytes], int]:
    """Read recent complete JSONL rows without scanning an entire long run log."""
    with path.open("rb") as stream:
        stream.seek(0, 2)
        end = stream.tell()
        cursor = end
        chunks = []
        newlines = 0
        while cursor and newlines <= limit and end - cursor < 4 * _MAX_EVENT_LINE_BYTES:
            size = min(65536, cursor)
            cursor -= size
            stream.seek(cursor)
            chunk = stream.read(size)
            chunks.append(chunk)
            newlines += chunk.count(b"\n")
    data = b"".join(reversed(chunks))
    if cursor and b"\n" in data:
        data = data.split(b"\n", 1)[1]
    complete, _, partial = data.rpartition(b"\n")
    if not complete and data.endswith(b"\n"):
        complete = data[:-1]
        partial = b""
    return complete.splitlines()[-limit:], end - len(partial)


def _supports_block_glyphs() -> bool:
    try:
        "█▄▀".encode(getattr(sys.stdout, "encoding", None) or "utf-8")
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def _brand() -> Text:
    glyphs = {
        "M": ("█▄█", "█▀█", "▀ ▀"),
        "E": ("█▀▀▀", "█▀▀ ", "▀▀▀▀"),
        "D": ("█▀▀▄", "█  █", "▀▀▀ "),
        "B": ("█▀▀▄", "█▀▀▄", "▀▀▀ "),
        "N": ("█▄ █", "█ ▀█", "▀  ▀"),
        "C": ("█▀▀▀", "█   ", "▀▀▀▀"),
        "H": ("█  █", "█▀▀█", "▀  ▀"),
        "A": ("▄▀▀▄", "█▀▀█", "▀  ▀"),
        "G": ("█▀▀▀", "█ ▀█", "▀▀▀▀"),
        "T": ("▀█▀", " █ ", " ▀ "),
    }
    colors = ("#65e5d4", "#69dfdf", "#70d7eb", "#77cef6", "#80c3ff",
              "#8bb7ff", "#98aafe", "#a99cfa", "#b991f1", "#c789e7",
              "#d483d9", "#df82c9", "#e78cbb")
    title = Text()
    if not _supports_block_glyphs():
        for letter, color in zip("MedBenchAgent", colors, strict=True):
            title.append(letter, style=f"bold {color}")
        return title
    for row in range(3):
        for index, letter in enumerate("MEDBENCHAGENT"):
            glyph = glyphs[letter][row]
            # C's open edge meets H's stem; E and N share a vertical boundary.
            if index and "MEDBENCHAGENT"[index - 1:index + 1] in {"CH", "EN"}:
                glyph = glyph[1:]
            title.append(glyph, style=f"bold {colors[index]}")
        if row < 2:
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


class HistoryView(VerticalScroll):
    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        if new_value <= 1 and old_value > new_value:
            self.screen.load_older_messages()


class ChatMessage(Static):
    """One conversation bubble, optionally awaiting a persisted message ID."""

    def __init__(self, role: str, content: str, *, message_id: str | None = None) -> None:
        super().__init__(Markdown(content) if role == "assistant" else Text(content), classes=f"chat-bubble {role}")
        self.role = role
        self.full_text = content
        self.message_id = message_id
        self.chat_row: Horizontal | None = None
        

    def show_text(self, content: str) -> None:
        self.update(Text(content))


class RemoveExperimentScreen(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, name: str) -> None:
        super().__init__()
        self.experiment_name = name

    def compose(self) -> ComposeResult:
        with Vertical(id="remove-dialog"):
            yield Static("Remove experiment?", id="remove-title")
            yield Static(self.experiment_name, id="remove-name")
            yield Static("Deletes this workspace and its conversation. External run artifacts and datasets remain.",
                         id="remove-message")
            with Horizontal(id="remove-actions"):
                yield Button("Remove", id="confirm-remove", variant="error")
                yield Button("Cancel", id="cancel-remove")

    @on(Button.Pressed, "#confirm-remove")
    def confirm(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#cancel-remove")
    def cancel(self) -> None:
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)


class ExperimentMenu(Screen):
    COMMANDS = (("new", "Create an experiment"), 
                ("remove", "Remove the selected experiment"),
                ("quit", "Exit MedBenchAgent"), ("help", "Commands and keys"))

    def __init__(self) -> None:
        super().__init__()
        self._entries: list[dict] = []
        self._visible_entries: list[tuple] = []
        self._scan_id = 0
        self._page = 0
        self._search_timer = None
        self._removing = False
        self._removal_message = ""

    def compose(self) -> ComposeResult:
        with Vertical(id="menu-shell"):
            yield Static(_brand(), id="menu-brand")
            
            with Vertical(id="menu-panel"):
                yield Static("RECENT EXPERIMENTS", id="menu-title")
                with Horizontal(id="menu-search-bar"):
                    yield Input(placeholder="Search experiments", id="search")
                    yield Button("Open", id="open", variant="primary")
                yield OptionList(id="menu-commands")
                yield DataTable(id="experiments", cursor_type="row", zebra_stripes=False)
                yield Static("", id="menu-message")
            with Horizontal(id="menu-pages"):
                yield Button("Previous", id="previous-page")
                yield Static("", id="page-label")
                yield Button("Next", id="next-page")

    def on_mount(self) -> None:
        self.query_one("#experiments", DataTable).add_columns("EXPERIMENT", "RUNS", "STATUS")
        self.query_one("#menu-commands", OptionList).display = False
        self.refresh_experiments()
        self.query_one("#search", Input).focus()

    def on_screen_resume(self) -> None:
        self.app.title = "MBA"
        self.refresh_experiments()

    def refresh_experiments(self) -> None:
        self._scan_id += 1
        self.query_one("#menu-message", Static).update("Loading experiments…")
        self._load_experiments(self._scan_id)

    @work(thread=True)
    def _load_experiments(self, scan_id: int) -> None:
        try:
            entries = []
            for batch in self.app.experiment_index.scan(self.app.experiments_root):
                entries.extend(batch)
                self.app.call_from_thread(self._loaded_experiments, scan_id, list(entries), "")
        except (ValueError, OSError) as exc:
            self.app.call_from_thread(self._loaded_experiments, scan_id, [], str(exc))
        else:
            self.app.call_from_thread(self._loaded_experiments, scan_id, entries, "")

    def _loaded_experiments(self, scan_id: int, entries: list[dict], error: str) -> None:
        if not self.is_mounted or scan_id != self._scan_id:
            return
        if not self.query("#experiments"):
            self.call_after_refresh(self._loaded_experiments, scan_id, entries, error)
            return
        self._entries = sorted(entries, key=lambda item: item["created_at"], reverse=True)
        self._render_entries()
        if error:
            self.query_one("#menu-message", Static).update(Text(error))
        elif self._removal_message:
            self.query_one("#menu-message", Static).update(Text(self._removal_message))
            self._removal_message = ""

    def _render_entries(self) -> None:
        table = self.query_one("#experiments", DataTable)
        query = self.query_one("#search", Input).value.strip().casefold()
        if query.startswith("/"):
            query = ""
        matches = [record for record in self._entries if query in record["name"].casefold()]
        pages = max(1, (len(matches) + 99) // 100)
        self.query_one("#menu-pages").display = pages > 1
        self._page = min(self._page, pages - 1)
        self.query_one("#page-label", Static).update(f"{self._page + 1} of {pages}")
        self.query_one("#previous-page", Button).disabled = self._page == 0
        self.query_one("#next-page", Button).disabled = self._page + 1 == pages
        rows = [(r["name"], r["run_count"], r["status"])
                for r in matches[self._page * 100:(self._page + 1) * 100]]
        if rows != self._visible_entries:
            selected = table.get_row_at(table.cursor_row)[0] if table.row_count else None
            old = {row[0]: row for row in self._visible_entries}
            names = [row[0] for row in rows]
            for name in old.keys() - set(names):
                table.remove_row(name)
            columns = list(table.columns)
            for name, count, status in rows:
                values = (name, str(count), Text(status.upper(), style=_STATUS_COLORS.get(status, "#aab9ca")))
                if name not in old:
                    table.add_row(*values, key=name)
                elif old[name] != (name, count, status):
                    for column, value in zip(columns, values):
                        table.update_cell(name, column, value)
            order = {name: index for index, name in enumerate(names)}
            table.sort(columns[0], key=lambda name: order[str(name)])
            self._visible_entries = rows
            if selected in names:
                table.move_cursor(row=names.index(selected))
        if not matches:
            message = "No matching experiments. Type /new to create." if query else "No experiments yet. Type /new to create."
            self.query_one("#menu-message", Static).update(Text(message))
        else:
            self.query_one("#menu-message", Static).update("")

    @on(Input.Changed, "#search")
    def filter_experiments(self) -> None:
        _update_slash_list(self.query_one("#menu-commands", OptionList),
                           self.query_one("#search", Input).value, self.COMMANDS)
        self._page = 0
        if self._search_timer:
            self._search_timer.stop()
        self._search_timer = self.set_timer(0.15, self._render_entries)

    @on(Button.Pressed, "#previous-page")
    @on(Button.Pressed, "#next-page")
    def change_page(self, event: Button.Pressed) -> None:
        self._page += 1 if event.button.id == "next-page" else -1
        self._render_entries()

    @on(Input.Submitted, "#search")
    def submit_search(self, event: Input.Submitted) -> None:
        if event.value.startswith("/"):
            self._dispatch_command(event.value)
        else:
            self._render_entries()
            self.action_open_selected()

    @on(OptionList.OptionSelected, "#menu-commands")
    def select_command(self, event: OptionList.OptionSelected) -> None:
        self._dispatch_command("/" + str(event.option_id))

    def _dispatch_command(self, value: str) -> None:
        if value.strip() == "/help":
            self.app.action_show_help()
            return
        name = value[1:].strip().casefold()
        options = self.query_one("#menu-commands", OptionList)
        if name not in {item[0] for item in self.COMMANDS}:
            selected = options.highlighted_option
            name = selected.id if selected else name
        table = self.query_one("#experiments", DataTable)
        selected_name = str(table.get_row_at(table.cursor_row)[0]) if table.row_count else None
        self.query_one("#search", Input).value = ""
        options.display = False
        if name == "new":
            self.app.push_screen(CreateExperimentScreen())
        elif name == "remove":
            if self._removing:
                self.query_one("#menu-message", Static).update("Removal is already in progress.")
            elif selected_name is None:
                self.query_one("#menu-message", Static).update("Select an experiment to remove.")
            else:
                self.app.push_screen(RemoveExperimentScreen(selected_name),
                                     lambda confirmed: self._confirm_remove(selected_name, confirmed))
        elif name == "quit":
            self.app.exit()
        else:
            self.query_one("#menu-message", Static).update(Text(f"Unknown command: /{name}"))

    def _confirm_remove(self, name: str, confirmed: bool) -> None:
        if not confirmed:
            return
        self._removing = True
        self.query_one("#menu-message", Static).update(f"Removing {name}…")
        self._remove_workspace(name)

    @work(thread=True)
    def _remove_workspace(self, name: str) -> None:
        try:
            sessions.remove_experiment(name, root=self.app.experiments_root)
        except Exception as exc:
            self.app.call_from_thread(self._remove_finished, name, str(exc))
        else:
            self.app.call_from_thread(self._remove_finished, name, "")

    def _remove_finished(self, name: str, error: str) -> None:
        if not self.is_mounted:
            return
        self._removing = False
        if error:
            self.query_one("#menu-message", Static).update(Text(error))
        else:
            self._removal_message = f"Removed experiment {name}."
            self.refresh_experiments()

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
            yield Static("a–z  0–9  hyphen", id="create-message")

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
            self.query_one("#create-message", Static).update(Text(str(exc)))
            return
        self.app.pop_screen()
        self.app.open_experiment(name, configure=True)


class ConfigScreen(Screen):
    COMMANDS = (("done", "Enter the conversation"), ("quit", "Return to the previous screen"), ("help", "Commands and keys"))

    def __init__(self, name: str, *, new: bool = False) -> None:
        super().__init__()
        self.experiment_name = name
        self.new = new
        self.selected: tuple[str, ...] | None = None
        self.selected_value: object = None

    def compose(self) -> ComposeResult:
        yield Static(f"{self.experiment_name}   Configuration", id="config-title")
        yield Tree("Configuration", id="config-tree")
        yield Label("Future run settings", id="field-label")
        with Horizontal():
            yield Input(id="field-value", disabled=True)
            yield Button("Apply value", id="apply", disabled=True)
        yield Static("", id="config-message")
        yield OptionList(id="config-commands")
        yield Input(placeholder="Command", id="config-command")

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
        if value.strip() == "/help":
            self.app.action_show_help()
            return
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
            self.query_one("#config-message", Static).update(Text(f"Unknown command: /{name}"))

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
        had_tree = bool(tree.root.children)
        tree.root.remove_children()
        values = sessions.config_values(self.experiment_name, root=self.app.experiments_root)
        values.setdefault("generation", {}).setdefault("max_total_tasks_per_image", None)

        def add(parent, data: dict, path: tuple[str, ...] = ()) -> None:
            for key, value in data.items():
                current = (*path, key)
                if isinstance(value, dict):
                    node = parent.add(key, data=current)
                    add(node, value, current)
                    if current in expanded or (not had_tree and current == ("llm",)):
                        node.expand()
                else:
                    label = f"{key} = {value!r}"
                    node = parent.add_leaf(Text(label), data=(current, value))
                    if current == self.selected:
                        tree.move_cursor(node)
                        self.selected_value = value

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
        except (ValueError, TypeError, KeyError, OSError) as exc:
            self.query_one("#config-message", Static).update(Text(str(exc)))
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
                ("quit", "Return to experiments"), ("help", "Commands and keys"))

    def __init__(self, name: str) -> None:
        super().__init__()
        self.experiment_name = name
        self.service: ConversationService | None = None
        self.record: dict = {}
        self.running = False
        self.classifying = False
        self.pending_question = False
        self._run_question = ""
        self.current_run_id: str | None = None
        self._displayed_messages: set[str] = set()
        self._message_widgets: dict[str, ChatMessage] = {}
        self._optimistic_messages: list[ChatMessage] = []
        self._history_loaded = False
        self._thinking_row: Horizontal | None = None
        self._history_start = None
        self._loading_history = False
        self._agent_status_dirty = False
        self._progress_queue: queue.SimpleQueue[tuple[str | None, str]] = queue.SimpleQueue()
        self._progress_entries: deque[tuple[str | None, str, str]] = deque(maxlen=600)
        self._progress_run_id: str | None = None
        self._event_path: Path | None = None
        self._event_offset = 0
        self._event_partial = b""
        self._discard_event_line = False
        self._event_generation = 0
        self._event_reading = False
        self._event_initial = False
        self._record_generation = 0
        self._agents: list[str] = []
        self._agent_statuses: dict[tuple[str, str, str], dict[str, str]] = {}
        self._context_tokens: int | None = None
        self._context_window: int | None = None
        self._context_model: str = ""
        self._status_text = "Ready"
        self._context_summary = "◉ n/a"
        self._sidebar_width: int | None = None
        self._panel_weights = [0.25, 0.25, 0.5]
        self._collapsed_panels: set[str] = set()
        self._layout_scheduled = False
        self.stop_event = threading.Event()
        self.exit_requested = False
        self.back_requested = False

    def compose(self) -> ComposeResult:
        yield Static(self.experiment_name, id="session-title")
        yield Static("Ready", id="session-status")
        with Horizontal(id="session-body"):
            with Vertical(id="chat-pane"):
                yield HistoryView(id="conversation")
                with VerticalScroll(id="question-card"):
                    yield Static("Input required", id="question-title")
                    yield Static("", id="run-question")
        
                yield OptionList(id="session-commands")
                yield Prompt(placeholder="Message", id="message", highlight_cursor_line=False)
            yield Splitter(axis="x", id="side-resize-handle")
            with Vertical(id="side-pane"):
                with Vertical(id="runs-panel"):
                    yield Button("RUNS  ▾", id="runs-toggle")
                    yield DataTable(id="runs", cursor_type="none", zebra_stripes=True)
                yield Splitter(axis="y", id="runs-divider")
                with Vertical(id="agents-panel"):
                    yield Button("AGENTS  ▾", id="agents-toggle")
                    yield Static("Latest activity: waiting", id="active-agent")
                    yield DataTable(id="agent-status", cursor_type="none", zebra_stripes=True)
                yield Splitter(axis="y", id="events-divider")
                with Vertical(id="events-panel"):
                    yield Button("EVENTS  ▾", id="events-toggle")
                    yield Select([("All agents", "*")], value="*", allow_blank=False, id="agent-filter")
                    yield RichLog(id="progress", wrap=True, highlight=True, min_width=1, max_lines=600)

    def on_mount(self) -> None:
        self.query_one("#runs", DataTable).add_columns("Request", "Status", "Run")
        self.query_one("#agent-status", DataTable).add_columns("Agent", "Scope", "Status", "Operation", "Detail")
        self.query_one("#session-commands", OptionList).display = False
        self._set_status("Loading experiment…")
        self.query_one("#message", Prompt).focus()
        self._progress_timer = self.set_interval(0.1, self._flush_progress)
        self._load_session()

    @work(thread=True)
    def _load_session(self) -> None:
        try:
            from medbenchagent.runtime.conversation_service import ConversationService

            service = ConversationService(self.experiment_name, root=self.app.experiments_root)
            record = service.reconcile()
        except Exception as exc:
            self.app.call_from_thread(self._session_load_failed, str(exc))
        else:
            self.app.call_from_thread(self._session_loaded, service, record)

    def _session_loaded(self, service: ConversationService, record: dict) -> None:
        if not self.is_mounted:
            return
        self.service = service
        self.refresh_record(record)
        recent = next((run for run in reversed(record["runs"]) if run.get("run_directory")), None)
        if recent:
            self._follow_run(recent["id"], recent["run_directory"])
        self._history_loaded = True
        self._render_question()
        self._set_status("Ready")
        self._start_next(refresh=False)

    def _session_load_failed(self, error: str) -> None:
        if self.is_mounted:
            self._set_status(f"Could not open experiment: {error}")

    def refresh_record(self, record: dict | None = None, *, after=None) -> None:
        self.record = record if record is not None else self.service.record()
        if self._history_start is None:
            self._history_start = max(0, len(self.record["messages"]) - 100)
        questions = {(run["id"], run.get("pending_question")) for run in self.record["runs"]
                     if run["status"] == "waiting_input" and not run.get("pending_answer")}
        def awaiting_answer(message: dict) -> bool:
            return (message.get("run_id"), message["text"]) in questions

        added_ids = []
        for message in self.record["messages"][self._history_start:]:
            if awaiting_answer(message):
                continue
            if message["id"] not in self._displayed_messages:
                matched = next((item for item in self._optimistic_messages
                                if item.role == message["role"] and item.full_text == message["text"]), None)
                if matched:
                    matched.message_id = message["id"]
                    self._optimistic_messages.remove(matched)
                    bubble = matched
                else:
                    bubble = self._write_message(message["role"], message["text"], message_id=message["id"])
                self._message_widgets[message["id"]] = bubble
                self._displayed_messages.add(message["id"])
                added_ids.append(message["id"])
        if added_ids:
            self.call_after_refresh(self._position_messages, added_ids)
        self.refresh_runs()
        if after:
            after()

    def load_older_messages(self) -> None:
        if not self._history_start or self._loading_history:
            return
        view = self.query_one("#conversation", VerticalScroll)
        self._loading_history = True
        previous_height, previous_y = view.virtual_size.height, view.scroll_y
        self._history_start = max(0, self._history_start - 100)
        self.refresh_record(self.record)
        def restore():
            view.scroll_to(y=previous_y + view.virtual_size.height - previous_height, animate=False)
            self._loading_history = False
        self.call_after_refresh(restore)

    def _position_messages(self, added_ids: list[str]) -> None:
        view = self.query_one("#conversation", VerticalScroll)
        ordered = [message["id"] for message in self.record["messages"]
                   if message["id"] in self._message_widgets]
        positions = {message_id: index for index, message_id in enumerate(ordered)}
        for message_id in added_ids:
            row = self._message_widgets[message_id].chat_row
            index = positions[message_id]
            previous = self._message_widgets[ordered[index - 1]].chat_row if index else None
            if previous is not None and row is not previous:
                view.move_child(row, after=previous)
            elif previous is None and view.children and row is not view.children[0]:
                view.move_child(row, before=view.children[0])

    def _render_question(self) -> None:
        view = self.query_one("#run-question", Static)
        container = self.query_one("#question-card")
        if self.pending_question:
            container.display = True
            view.update(Text(self._run_question))
            self.query_one("#message", Prompt).placeholder = "Answer"
            return
        container.display = False
        self.query_one("#message", Prompt).placeholder = "Message"

    def _write_message(self, role: str, content: str, *, message_id: str | None = None,
                       reveal: bool = False, failed: bool = False) -> ChatMessage:
        role = "user" if role in {"user", "you"} else "assistant"
        view = self.query_one("#conversation", VerticalScroll)
        was_at_end = view.is_vertical_scroll_end
        bubble = ChatMessage(role, content, message_id=message_id)
        bubble.full_text = content
        if failed:
            bubble.add_class("failed")
        row = Horizontal(bubble, classes=f"chat-row {role}")
        bubble.chat_row = row
        view.mount(row)
        if was_at_end and not self._loading_history:
            self.call_after_refresh(view.scroll_end, animate=False)
        return bubble

    def _show_optimistic(self, content: str) -> ChatMessage:
        bubble = self._write_message("user", content)
        self._optimistic_messages.append(bubble)
        return bubble

    def _remove_optimistic(self, bubble: ChatMessage, *, restore_input: bool = False) -> None:
        if bubble in self._optimistic_messages:
            self._optimistic_messages.remove(bubble)
            bubble.chat_row.remove()
        if restore_input:
            field = self.query_one("#message", Prompt)
            field.value = bubble.full_text
            field.focus()

    def _show_thinking(self) -> None:
        if self._thinking_row is None:
            bubble = ChatMessage("assistant", "⚙ Thinking…")
            bubble.add_class("thinking")
            self._thinking_row = Horizontal(bubble, classes="chat-row assistant")
            self.query_one("#conversation", VerticalScroll).mount(self._thinking_row)
            self.query_one("#conversation", VerticalScroll).scroll_end(animate=False)

    def _hide_thinking(self) -> None:
        if self._thinking_row is not None:
            self._thinking_row.remove()
            self._thinking_row = None

    def _set_status(self, value: str) -> None:
        self._status_text = value
        self._render_status()

    def _render_status(self) -> None:
        self.query_one("#session-status", Static).update(
            Text(f"{self._status_text} · Context {self._context_summary}")
        )

    def _follow_run(self, run_id: str, run_directory: str | None) -> None:
        if self._progress_run_id == run_id:
            return
        self._progress_run_id = run_id
        self._context_tokens = None
        self._context_window = None
        self._context_model = ""
        self._context_summary = "◉ n/a"
        runtime_name = Path(run_directory).name if run_directory else ""
        self._set_status(f"Running {runtime_name} · {run_id[:8]}" if runtime_name else f"Running {run_id[:8]}")
        self._progress_entries.clear()
        self._agents.clear()
        self._agent_statuses.clear()
        self._event_path = Path(run_directory) / "control" / "events.jsonl" if run_directory else None
        self._event_offset = 0
        self._event_partial = b""
        self._discard_event_line = False
        self.query_one("#active-agent", Static).update("Latest activity: waiting")
        self.query_one("#agent-filter", Select).set_options([("All agents", "*")])
        self.query_one("#agent-filter", Select).value = "*"
        self._render_agent_status()
        self._event_generation += 1
        self._event_reading = False
        self._event_initial = True
        self._render_progress()
        self._poll_construction_events()

    def _add_agent(self, agent: str) -> None:
        if agent in self._agents:
            return
        self._agents.append(agent)
        selector = self.query_one("#agent-filter", Select)
        selected = selector.value
        selector.set_options([("All agents", "*"),
                              *((name.removesuffix("Agent"), name) for name in self._agents)])
        selector.value = selected if selected in {"*", *self._agents} else "*"

    def _update_agent_status(self, agent: str, dataset: str, source: str,
                             status: str, operation: str, detail: str) -> None:
        key = (agent, dataset, source)
        self._agent_statuses[key] = {"agent": agent.removesuffix("Agent"),
                                     "scope": "/".join(item for item in (dataset, source) if item),
                                     "status": status or "running", "operation": str(operation or "event"),
                                     "detail": detail}
        self._agent_status_dirty = True

    def _render_agent_status(self) -> None:
        self._agent_status_dirty = False
        table = self.query_one("#agent-status", DataTable)
        selected = self.query_one("#agent-filter", Select).value
        table.clear()
        for key, item in self._agent_statuses.items():
            if selected not in {"*", key[0]}:
                continue
            table.add_row(item["agent"], item["scope"],
                          Text(item["status"], style=_STATUS_COLORS.get(item["status"], "#aab9ca")),
                          item["operation"], item["detail"][:180], key="|".join(key))

    def _append_progress(self, agent: str | None, value: str, status: str = "", *, render: bool = True) -> None:
        if not value:
            return
        self._progress_entries.append((agent, value, status))
        if render and self.query_one("#agent-filter", Select).value in {"*", agent}:
            log = self.query_one("#progress", RichLog)
            log.write(Text(value, style=_progress_color(value, status)), scroll_end=log.is_vertical_scroll_end)

    def _render_progress(self) -> None:
        log = self.query_one("#progress", RichLog)
        log.clear()
        selected = self.query_one("#agent-filter", Select).value
        for agent, value, status in self._progress_entries:
            if selected in {"*", agent}:
                self._append_progress_render(log, value, status)

    def _append_progress_render(self, log: RichLog, value: str, status: str) -> None:
        log.write(Text(value, style=_progress_color(value, status)), scroll_end=True)

    @on(Select.Changed, "#agent-filter")
    def agent_filter_changed(self, _event: Select.Changed) -> None:
        self._render_progress()
        self._render_agent_status()

    def _read_construction_event(self, line: bytes, *, render: bool = True) -> None:
        try:
            event = json.loads(line)
        except (UnicodeDecodeError, ValueError):
            return
        if not isinstance(event, dict):
            return
        agent = event.get("agent")
        kind = event.get("event_type")
        if not isinstance(agent, str) or kind not in {
            "agent_step_started", "tool_started", "tool_result", "llm_call", "llm_validation_error",
            "coding_worker_event", "agent_started", "agent_finished", "worker_started", "worker_result",
            "finalization_started", "finalization_finished", "candidate_submitted"
        }:
            return
        operation = event.get("operation", "")
        status = event.get("status", "")
        label = agent.removesuffix("Agent")
        timestamp = event.get("timestamp", "")
        time = timestamp[11:19] if isinstance(timestamp, str) else ""
        dataset = str(event.get("dataset_id") or "")
        source = str(event.get("source_id") or "")
        if kind == "coding_worker_event":
            operation = str(event.get("operation") or "event")
            detail = f"OpenHands · {operation.removesuffix('Event')}"
            status = event.get("status", "running")
        elif kind == "agent_step_started":
            detail = "deciding"
        elif kind == "tool_started":
            detail = f"{operation} · running"
        elif kind == "tool_result":
            detail = f"{operation} · {status}: {str(event.get('summary', ''))[:240]}"
        elif kind == "llm_validation_error":
            detail = f"{operation} · invalid model output; repairing"
            status = "failed"
        elif kind in {"agent_started", "worker_started", "finalization_started"}:
            detail = f"{operation} · started"
            status = status or "running"
        elif kind in {"agent_finished", "worker_result", "finalization_finished", "candidate_submitted"}:
            detail = f"{operation} · {status}: {str(event.get('summary') or event.get('diagnostic') or '')[:240]}"
        else:
            latency = event.get("latency_ms")
            duration = f" ({latency / 1000:.1f}s)" if isinstance(latency, (int, float)) else ""
            detail = f"{operation} · model {status}{duration}"
        if kind == "llm_call":
            if isinstance(event.get("input_tokens"), int):
                self._context_tokens = event["input_tokens"]
            if isinstance(event.get("context_window_tokens"), int):
                self._context_window = event["context_window_tokens"]
            if isinstance(event.get("model"), str):
                self._context_model = event["model"]
            self._update_context_indicator()
        self._add_agent(agent)
        self._update_agent_status(agent, dataset, source, str(status), str(operation), detail)
        self._append_progress(agent, f"{time}  {label}: {detail}", status, render=render)
        if kind in {"agent_step_started", "tool_started", "coding_worker_event", "agent_started", "worker_started", "finalization_started"}:
            self.query_one("#active-agent", Static).update(f"Working: {label} · {operation}")
        elif kind in {"tool_result", "llm_call", "agent_finished", "worker_result", "finalization_finished", "candidate_submitted"}:
            self.query_one("#active-agent", Static).update(f"Latest: {label} · {operation}")

    def _update_context_indicator(self) -> None:
        if self._context_tokens is None:
            self._context_summary = "◉ n/a"
        elif self._context_window:
            ratio = self._context_tokens / self._context_window
            self._context_summary = f"◉ {ratio:.0%}"
        else:
            self._context_summary = f"◉ {self._context_tokens:,}"
        self._render_status()

    def _poll_construction_events(self) -> None:
        if self._event_path is None or self._event_reading:
            return
        self._event_reading = True
        try:
            if self._event_initial:
                lines, self._event_offset = _recent_event_lines(self._event_path)
                self._event_initial = False
            else:
                with self._event_path.open("rb") as stream:
                    stream.seek(self._event_offset)
                    data = self._event_partial + stream.read(65536)
                    self._event_offset = stream.tell()
                lines = data.split(b"\n")
                self._event_partial = lines.pop()
            for line in lines:
                if len(line) <= _MAX_EVENT_LINE_BYTES:
                    self._read_construction_event(line)
        except OSError:
            pass
        self._event_reading = False
        if self._agent_status_dirty:
            self._render_agent_status()

    @work(thread=True, group="event-read")
    def _read_event_batch(self, generation, path, offset, partial, discard, initial):
        lines = []
        try:
            if initial:
                lines, offset = _recent_event_lines(path)
            else:
                with path.open("rb") as stream:
                    stream.seek(0, 2)
                    if stream.tell() < offset:
                        offset, partial, discard = 0, b"", False
                    stream.seek(offset)
                    chunk = stream.read(65536)
                offset += len(chunk)
                data = partial + chunk
                if discard:
                    end = data.find(b"\n")
                    data = data[end + 1:] if end >= 0 else b""
                    discard = end < 0
                lines = data.split(b"\n")
                partial = lines.pop()
                if len(partial) > _MAX_EVENT_LINE_BYTES:
                    partial, discard = b"", True
        except OSError:
            pass
        self.app.call_from_thread(self._event_batch_loaded, generation, lines, offset, partial, discard)

    def _event_batch_loaded(self, generation, lines, offset, partial, discard):
        if not self.is_mounted or generation != self._event_generation:
            return
        self._event_reading = False
        self._event_initial = False
        self._event_offset, self._event_partial, self._discard_event_line = offset, partial, discard
        with self.app.batch_update():
            for line in lines:
                if len(line) <= _MAX_EVENT_LINE_BYTES:
                    self._read_construction_event(line)
            if self._agent_status_dirty:
                self._render_agent_status()

    def _flush_progress(self, limit: int = 100) -> None:
        count = 0
        while count < limit:
            try:
                run_id, value = self._progress_queue.get_nowait()
            except queue.Empty:
                break
            if run_id == self._progress_run_id or self._progress_run_id is None:
                if not (self._event_path and value.startswith("[tool]")) and not (
                    value.startswith(("[start]", "[done]"))
                    and value.endswith((" / decide", " / execute"))
                ):
                    self._append_progress(None, value)
            count += 1
        self._poll_construction_events()
        if self._agent_status_dirty:
            self._render_agent_status()

    def refresh_runs(self) -> None:
        table = self.query_one("#runs", DataTable)
        if table.row_count > len(self.record["runs"]):
            table.clear()
        for index, run in enumerate(self.record["runs"]):
            values = (run["requirement"][:48], Text(run["status"], style=_STATUS_COLORS.get(run["status"], "#aab9ca")),
                      run["id"][:8])
            if index >= table.row_count:
                table.add_row(*values, key=run["id"])
            else:
                for column, value in enumerate(values):
                    if table.get_cell_at((index, column)) != value:
                        table.update_cell_at((index, column), value)

    @on(TextArea.Changed, "#message")
    def filter_commands(self) -> None:
        _update_slash_list(self.query_one("#session-commands", OptionList),
                           self.query_one("#message", Prompt).value, self.COMMANDS)

    @on(OptionList.OptionSelected, "#session-commands")
    def select_command(self, event: OptionList.OptionSelected) -> None:
        self._dispatch_command("/" + str(event.option_id))

    def on_key(self, event: events.Key) -> None:
        _navigate_slash_list(event, self.query_one("#message", Prompt),
                             self.query_one("#session-commands", OptionList))

    def _dispatch_command(self, value: str) -> None:
        if value.strip() == "/help":
            self.app.action_show_help()
            return
        name, _, argument = value[1:].strip().partition(" ")
        name = name.casefold()
        options = self.query_one("#session-commands", OptionList)
        options.display = False
        if self.service is None and name != "quit":
            self._set_status("Experiment is still loading")
            return
        field = self.query_one("#message", Prompt)
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

    @on(Prompt.Submitted, "#message")
    def submit(self, event: Prompt.Submitted) -> None:
        value = event.value.strip()
        if not value:
            return
        command = value[1:].split(" ", 1)[0].casefold() if value.startswith("/") else ""
        if command in {name for name, _ in self.COMMANDS}:
            self._dispatch_command(value)
            return
        if value.startswith("/") and not self.pending_question:
            self._set_status(f"Unknown command: /{command}")
            return
        if self.service is None:
            self._set_status("Experiment is still loading")
            return
        if self.exit_requested or self.back_requested:
            self._set_status("Stopping; resume the saved run before submitting more work.")
            return
        if self.classifying:
            self._set_status("Wait for the current request to finish")
            return
        event.input.value = ""
        if self.pending_question and self.current_run_id:
            bubble = self._show_optimistic(value)
            try:
                self.service.answer(self.current_run_id, value)
            except Exception as exc:
                self._remove_optimistic(bubble, restore_input=True)
                self._set_status(str(exc))
                return
            run_id = self.current_run_id
            self.pending_question = False
            self._run_question = ""
            self.query_one("#question-card").display = False
            self.current_run_id = None
            self.refresh_record()
            self._execute_run(resume_id=run_id, answer=value)
            return
        self._queue_task(value)

    def _queue_task(self, value: str) -> None:
        if self.exit_requested or self.back_requested:
            self._set_status("Stopping; resume the saved run before submitting more work.")
            return
        if self.classifying:
            self._set_status("Wait for the current request to be submitted.")
            return
        self.classifying = True
        self._show_optimistic(value)
        self._show_thinking()
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
        self._hide_thinking()
        self.refresh_record(after=lambda: self._submission_ready(result))

    def _submission_ready(self, result):
        self._render_question()
        if result.run_ids:
            self._set_status(f"Queued {len(result.run_ids)} run(s)")
        else:
            self._set_status("Ready")
        if not self._finish_navigation():
            self._start_next(refresh=False)

    def _submission_failed(self, message: str) -> None:
        self.classifying = False
        self._hide_thinking()
        self.refresh_record()
        self._set_status(f"Supervisor error: {message}")
        self._finish_navigation()

    def _start_next(self, *, refresh: bool = True) -> None:
        if self.service is None or self.running or self.pending_question or self.exit_requested or self.back_requested:
            return
        if refresh:
            self.refresh_record(after=lambda: self._start_next(refresh=False))
            return
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
        if resume_id:
            run = next((item for item in self.record["runs"] if item["id"] == resume_id), None)
            if run and run.get("run_directory"):
                self._follow_run(run["id"], run["run_directory"])
        self.running = True
        self._show_thinking()
        self._set_status("Resuming run" if resume_id else "Running queued tasks")
        self._execute(resume_id, answer)

    @work(thread=True)
    def _execute(self, resume_id: str | None, answer: str | None) -> None:
        try:
            events = (self.service.resume(resume_id, answer=answer,
                                          stop_requested=self.stop_event.is_set) if resume_id
                      else self.service.drive(stop_requested=self.stop_event.is_set))
            for event in events:
                if event.kind == "progress":
                    self._progress_queue.put((event.run_session_id, event.payload.get("text", "")))
                else:
                    self.app.call_from_thread(self._on_event, event)
        except Exception as exc:
            self.app.call_from_thread(self._set_status, f"Run error: {exc}")
        finally:
            self.app.call_from_thread(self._execution_ended)

    def _on_event(self, event: RunEvent) -> None:
        if event.kind in {"initialized", "needs_input", "finished", "error"}:
            self.refresh_record()
        if event.kind == "initialized":
            if event.payload.get("run_directory"):
                self._follow_run(event.run_session_id, event.payload["run_directory"])
            self._set_status(f"Running {event.run_session_id[:8]}")
            self._progress_queue.put((event.run_session_id, f"[run] {event.payload.get('run_directory')}"))
        elif event.kind == "artifact":
            self._progress_queue.put((event.run_session_id, f"[artifact] {event.payload}"))
        elif event.kind == "needs_input":
            self._hide_thinking()
            self.pending_question = True
            self.current_run_id = event.run_session_id
            question = event.payload["question"]
            self._run_question = question
            self._set_status("Answer the run question to continue")
            self._render_question()
            self.query_one("#message", Prompt).focus()
        elif event.kind in {"finished", "error"}:
            if event.run_session_id != self._progress_run_id:
                run = next((item for item in self.record["runs"] if item["id"] == event.run_session_id), None)
                self._follow_run(event.run_session_id, event.payload.get("run_directory") or
                                 (run.get("run_directory") if run else None))
            self._hide_thinking()
            self._flush_progress()
            self._progress_queue.put((event.run_session_id,
                                      f"[{event.payload.get('status', event.kind)}] {event.payload.get('summary', '')}"))
            if event.payload.get("recovery_hint"):
                self._progress_queue.put((event.run_session_id, event.payload["recovery_hint"]))
            self._set_status(f"{event.payload.get('status')}: {event.payload.get('summary', '')}")
            self.query_one("#message", Prompt).placeholder = "Message"
            if not self.pending_question:
                self.query_one("#question-card").display = False

    def _execution_ended(self) -> None:
        self._flush_progress()
        self._hide_thinking()
        self.running = False
        self.refresh_record(after=self._execution_record_loaded)

    def _execution_record_loaded(self):
        if self._finish_navigation():
            return
        if not self.pending_question:
            self._start_next(refresh=False)

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
        self.query_one("#side-resize-handle", Static).display = sidebar.display
        for splitter in self.query(Splitter):
            splitter.release()
        self._schedule_panel_layout()

    def _clamp_sidebar_width(self, width: int) -> int:
        return max(30, min(self.size.width - 28, width))

    @on(Splitter.Adjust)
    def resize_detail(self, event: Splitter.Adjust):
        if event.splitter.axis == "x":
            side = self.query_one("#side-pane")
            self._sidebar_width = self._clamp_sidebar_width(42 if event.reset else side.outer_size.width - event.delta)
            side.styles.width = self._sidebar_width
        else:
            panels = [self.query_one("#" + name + "-panel") for name in ("runs", "agents", "events")]
            sizes = [panel.size.height for panel in panels]
            index = 0 if event.splitter.id == "runs-divider" else 1
            minima = [3, 4, 5]
            pair = sizes[index] + sizes[index + 1]
            target = round(pair * (0.5 if index == 0 else 1 / 3)) if event.reset else sizes[index] + event.delta
            sizes[index] = max(minima[index], min(pair - minima[index + 1], target))
            sizes[index + 1] = pair - sizes[index]
            total = sum(sizes)
            self._panel_weights = [size / total for size in sizes]
            self._schedule_panel_layout()
        event.stop()

    @on(Button.Pressed, "#runs-toggle, #agents-toggle, #events-toggle")
    def toggle_detail_panel(self, event: Button.Pressed) -> None:
        panel_id = event.button.id.removesuffix("-toggle")
        if panel_id in self._collapsed_panels:
            self._collapsed_panels.remove(panel_id)
            event.button.label = f"{panel_id.upper()}  ▾"
        else:
            self._collapsed_panels.add(panel_id)
            event.button.label = f"{panel_id.upper()}  ▸"
        self._schedule_panel_layout()

    def _layout_panels(self):
        self._layout_scheduled = False
        if not self.query("#side-pane"):
            return
        height = max(12, self.query_one("#side-pane").size.height - 2)
        names = ("runs", "agents", "events")
        minima = (3, 4, 5)
        visible = [name not in self._collapsed_panels for name in names]
        sizes = [0 if not show else max(low, int(height * weight))
                 for low, weight, show in zip(minima, self._panel_weights, visible)]
        while sum(sizes) > height:
            index = max((i for i in range(3) if sizes[i]), key=lambda i: sizes[i] - minima[i])
            sizes[index] -= 1
        if sum(sizes) < height:
            sizes[next(i for i in (2, 1, 0) if visible[i])] += height - sum(sizes)
        for name, size, show in zip(names, sizes, visible):
            panel = self.query_one("#" + name + "-panel")
            panel.styles.height = max(1, size)
            for child in panel.children:
                child.display = show or child.id == f"{name}-toggle"

    def _schedule_panel_layout(self) -> None:
        if not self._layout_scheduled:
            self._layout_scheduled = True
            self.call_after_refresh(self._layout_panels)

    def on_resize(self, _event: events.Resize) -> None:
        if self.is_mounted and self.query("#side-pane"):
            if self._sidebar_width is not None:
                self._sidebar_width = self._clamp_sidebar_width(self._sidebar_width)
                self.query_one("#side-pane").styles.width = self._sidebar_width
            self._schedule_panel_layout()

    def on_unmount(self) -> None:
        self._progress_timer.stop()
        self._event_generation += 1
        self._record_generation += 1
        self.workers.cancel_group(self, "event-read")
        self.workers.cancel_group(self, "record-read")
        for splitter in self.query(Splitter):
            splitter.release()

    def action_edit_config(self) -> None:
        self.app.push_screen(ConfigScreen(self.experiment_name))

    def action_back(self) -> None:
        self.back_requested = True
        self.stop_event.set()
        if self.service is not None:
            self.service.runner.request_stop()
        if not self._finish_navigation():
            self._set_status("Stopping at the next saved run boundary…")

    def _finish_navigation(self) -> bool:
        if self.running or self.classifying:
            return False
        if self.exit_requested:
            self.app.exit()
            return True
        if self.back_requested:
            while self.app.screen is not self:
                self.app.pop_screen()
            self.app.pop_screen()
            return True
        return False

class MBAApp(App):
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [Binding("escape", "request_exit", "Exit"),
                Binding("ctrl+p", "commands", "Commands", priority=True)]
    CSS = """
    App, Screen { background: #101319; color: #d9dce5; }
    Screen { layout: vertical; }
    Button { margin: 0 1 0 0; min-width: 10; background: #242b36; color: #d9dce5; border: none; }
    Button:focus { background: #34445f; }
    Input { background: #1a2029; border: tall #323c4b; }
    Input:focus { border: tall #82aaff; }
    DataTable { background: #171c24; color: #d9dce5; }
    OptionList { background: #1a1e25; border: none; height: auto; max-height: 8; border-left: thick #657c99; }
    #menu-shell { width: 100%; height: auto; padding: 1 2; max-width: 100; max-height: 90%; }
    #menu-brand { height: auto; max-height: 4; content-align: center middle; margin-bottom: 1; }
    #menu-subtitle { height: 2; color: #a5b3c9; }
    #menu-panel { width: 100%; height: auto; background: #101319; padding: 0; }
    #menu-title { height: 1; color: #a8c4ff; text-style: bold; }
    #menu-search-bar { width: 100%; height: 3; }
    #menu-message { height: 1; color: #9aa6b6; }
    #menu-hint { height: 1; color: #718095; }
    #experiments { width: 100%; height: auto; min-height: 3; border: none; max-height: 18; }
    #experiments:focus { border: none; }
    #search, #field-value { width: 1fr; }
    #open { width: 12; height: 3; margin: 0; padding: 0; content-align: center middle; background: #34445f; color: #e5ecf5; }
    RemoveExperimentScreen { align: center middle; }
    #remove-dialog { width: 68; max-width: 100%; height: auto; padding: 1 2; background: #171c24; border: solid #eb7d83; }
    #remove-title { height: 1; color: #eb7d83; text-style: bold; }
    #remove-name { height: 1; color: #d9dce5; text-style: bold; }
    #remove-message { height: auto; color: #9aa6b6; }
    #remove-actions { height: 3; }
    CreateExperimentScreen { align: center middle; }
    #create-dialog { width: 62; max-width: 100%; height: auto; padding: 1 2; background: #171c24; border: solid #4386f7; }
    #create-title { height: 1; color: #82bdff; text-style: bold; }
    #new-name { height: 3; }
    #create-message { height: 2; color: #9aa6b6; }
    ConfigScreen { padding: 1 2 0 2; }
    #config-title { height: 1; color: #a8c4ff; text-style: bold; }
    #config-tree { height: 1fr; background: #171c24; border: none; }
    #field-label { height: 1; color: #9aa6b6; }
    #config-message, #menu-message { height: auto; max-height: 3; overflow-y: auto; color: #9aa6b6; }
    ConfigScreen > Horizontal { height: 3; }
    #config-command { height: 3; }
    #session-title { height: 1; padding: 0 2; color: #a8c4ff; text-style: bold; }
    #session-status { height: auto; max-height: 4; overflow-y: auto; padding: 0 2; color: #9aa6b6; }
    #session-body { width: 100%; height: 1fr; }
    #chat-pane { width: 1fr; height: 1fr; min-width: 24; padding: 0 1 0 2; }
    #side-resize-handle { width: 2; min-width: 2; height: 1fr; background: #101319; pointer: w-resize; color: #657c99; }
    #side-pane { width: 42; height: 1fr; min-width: 32; background: #171c24; padding: 0 1; overflow-y: auto; }
    #conversation { width: 100%; height: 1fr; background: #101319; border: none; }
    .chat-row { width: 100%; height: auto; margin: 0 0 1 0; }
    .chat-row.user { align: left top; }
    .chat-row.assistant { align: left top; }
    .chat-bubble { width: 75%; height: auto; min-height: 1; padding: 0 1; border: none; background: #101319; color: #d9dce5; }
    .chat-bubble.user { width: 70%; border: none; background: #1a1e25; padding: 1; border-left: thick #657c99; }
    .compact .chat-bubble { width: 95%; }
    .compact .chat-bubble.user { width: 95%; }
    .chat-bubble.failed { border: round #eb7d83; color: #eb9ca1; border-left: thick #eb7d83; }
    .chat-bubble.thinking { width: auto; color: #a5b8c8; }
    #question-card { height: auto; max-height: 10; display: none; margin: 1 0; padding: 0 1; border: none; background: #211d1a; border-left: thick #e8ac70; }
    #question-title { height: 1; color: #e8ac70; text-style: bold; }
    #run-question { height: auto; color: #e9e0d6; }
    #question-hint { height: 1; color: #baa993; }
    #message { width: 100%; height: 3; }
    #runs { width: 100%; height: 1fr; min-height: 1; border: none; }
    #active-agent { height: auto; max-height: 2; padding: 0 1; color: #a8c4ff; }
    #agent-status { width: 100%; height: 1fr; min-height: 1; border: none; }
    #agent-filter { width: 100%; height: 3; }
    #progress { width: 100%; height: 1fr; min-height: 2; border-top: solid #323c4b; }
    .compact #chat-pane { padding: 0 1; }
    .compact #side-pane { width: 38; min-width: 30; }
    .compact .chat-bubble, .compact .chat-bubble.user { width: 95%; }
    .compact #session-title { padding: 0 1; }
    ExperimentMenu { align: center middle; }
    #menu-pages { height: 3; align: center middle; }
    #page-label { width: 1fr; content-align: center middle; color: #89919d; }
    Input, Prompt { background: #1a1e25; border: none; border-left: thick #657c99; padding: 1; }
    Input:focus, Prompt:focus { border: none; border-left: thick #a9c7f6; }
    #open:hover, #open:focus { background: #496181; }
    #experiments > .datatable--header { background: #1a1e25; color: #89919d; text-style: none; }
    #experiments > .datatable--cursor { background: #34445f; color: #e5ecf5; }
    .splitter.y { height: 1; width: 100%; pointer: s-resize; color: #657c99; }
    .splitter:hover, .splitter:focus, .splitter.dragging { background: #34445f; color: #a9c7f6; }
    #runs-panel { height: 1fr; min-height: 3; }
    #agents-panel { height: 1fr; min-height: 4; }
    #events-panel { height: 2fr; min-height: 5; }
    CommandDialog { align: center middle; }
    #command-dialog { width: 70; max-width: 95%; height: auto; max-height: 90%; overflow-y: auto; background: #1a1e25; padding: 1 2; }
    #command-title { height: 1; color: #a9c7f6; }
    #command-options { height: auto; max-height: 12; }
    """

    TITLE = "MBA"

    def __init__(self, *, experiments_root: Path = sessions.EXPERIMENTS_ROOT,
                 config_template: Path | None = None) -> None:
        super().__init__()
        self.experiments_root = experiments_root
        self.config_template = config_template
        self.experiment_index = ExperimentIndex()

    def on_mount(self) -> None:
        self.push_screen(ExperimentMenu())

    def action_commands(self):
        screen = self.screen
        commands = getattr(screen, "COMMANDS", ())
        if not commands:
            return
        focused = screen.focused
        def selected(name):
            if focused and focused.is_mounted:
                focused.focus()
            if name:
                screen._dispatch_command("/" + name)
        self.push_screen(CommandDialog(commands), selected)

    def action_show_help(self):
        self.push_screen(CommandDialog(getattr(self.screen, "COMMANDS", ()), help_only=True))

    def action_request_exit(self) -> None:
        active = [screen for screen in self.screen_stack
                  if isinstance(screen, ConversationScreen) and (screen.running or screen.classifying)]
        if active:
            screen = active[-1]
            screen.exit_requested = True
            screen.stop_event.set()
            if screen.service is not None:
                screen.service.runner.request_stop()
            screen._set_status("Stopping at the next saved run boundary…")
            while self.screen is not screen:
                self.pop_screen()
        else:
            self.exit()

    def action_quit(self) -> None:
        self.action_request_exit()

    def on_resize(self, event: events.Resize) -> None:
        self.set_class(event.size.width < 100, "compact")

    def open_experiment(self, name: str, *, configure: bool = False) -> None:
        self.title = name
        if configure:
            self.push_screen(ConfigScreen(name, new=True))
        else:
            self.push_screen(ConversationScreen(name))



