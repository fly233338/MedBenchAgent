"""Presentation-only input and experiment index helpers."""

from pathlib import Path
import threading

from textual import events, on
from textual.binding import Binding
from textual.message import Message
from textual.widgets import OptionList, TextArea, Static, Input
from textual.screen import ModalScreen
from textual.containers import Vertical
from textual.widgets.option_list import Option

from medbenchagent.runtime import sessions


class ExperimentIndex:
    """Cache summaries, never complete conversation records."""

    def __init__(self):
        self.entries = {}
        self.lock = threading.Lock()

    def scan(self, root: Path):
        with self.lock:
            seen = set()
            batch = []
            for path in root.glob("*/session.json"):
                name = path.parent.name
                seen.add(name)
                stat = path.stat()
                signature = (stat.st_mtime_ns, stat.st_size)
                cached = self.entries.get(name)
                if cached is None or cached[0] != signature:
                    record = sessions.load_experiment(name, root=root)
                    runs = record["runs"]
                    summary = {"name": name, "created_at": record["created_at"],
                               "run_count": len(runs), "status": runs[-1]["status"] if runs else "new"}
                    self.entries[name] = (signature, summary)
                batch.append(self.entries[name][1])
                if len(batch) == 100:
                    yield batch
                    batch = []
            if batch:
                yield batch
            for name in self.entries.keys() - seen:
                del self.entries[name]


class Prompt(TextArea):
    """Multiline prompt with explicit submission and command navigation."""

    BINDINGS = [Binding("ctrl+shift+a", "select_all", show=False)]

    class Submitted(Message):
        def __init__(self, field):
            super().__init__()
            self.input = field
            self.value = field.text

        @property
        def control(self):
            return self.input

    @property
    def value(self):
        return self.text

    @value.setter
    def value(self, text):
        self.load_text(text)
        self.move_cursor(self.document.end)

    def on_mount(self):
        self.fit_height()

    def on_text_area_changed(self):
        self.fit_height()

    def on_resize(self):
        self.fit_height()

    def fit_height(self):
        height = self.wrapped_document.height + 2
        self.styles.height = min(max(3, height), max(3, self.app.size.height // 3))

    async def _on_key(self, event: events.Key):
        options = self.screen.query_one("#session-commands", OptionList)
        if event.key in {"up", "down"} and options.display:
            if options.option_count:
                options.highlighted = ((options.highlighted or 0) +
                                       (1 if event.key == "down" else -1)) % options.option_count
            event.stop()
            event.prevent_default()
            return
        if event.key in {"enter", "shift+enter", "ctrl+j"}:
            event.stop()
            event.prevent_default()
            if event.key == "enter":
                self.post_message(self.Submitted(self))
            else:
                self.replace("\n", *self.selection)
            return
        await super()._on_key(event)




class CommandDialog(ModalScreen):
    BINDINGS = [Binding("escape", "close", show=False, priority=True)]

    def __init__(self, commands, *, help_only=False):
        super().__init__()
        self.commands = commands
        self.help_only = help_only

    def compose(self):
        with Vertical(id="command-dialog"):
            yield Static("Help" if self.help_only else "Commands", id="command-title")
            if not self.help_only:
                yield Input(placeholder="Search commands", id="command-search")
                yield OptionList(id="command-options")
            else:
                yield Static("\n".join(f"/{name}  {description}" for name, description in self.commands))
                yield Static("\nCtrl+P  Commands\nEnter  Submit    Shift+Enter / Ctrl+J  New line\n"
                             "Tab  Focus divider    Arrows  Resize    Home  Reset\nEscape  Close help")
                yield Static("\nMouse not responding in tmux? Check `tmux show -v mouse` and mouse bindings. "
                             "Enable with `tmux set -g mouse on`. Terminal mouse-pointer shapes vary.")

    def on_mount(self):
        if not self.help_only:
            self.filter_options()
            self.query_one(Input).focus()

    @on(Input.Changed)
    def filter_options(self):
        query = self.query_one(Input).value.casefold()
        options = self.query_one(OptionList)
        options.set_options([Option(f"/{name}  {description}", id=name)
                             for name, description in self.commands if query in f"{name} {description}".casefold()])
        if options.option_count:
            options.highlighted = 0

    def on_key(self, event):
        if not self.help_only and event.key in {"up", "down"}:
            options = self.query_one(OptionList)
            if options.option_count:
                options.highlighted = ((options.highlighted or 0) + (1 if event.key == "down" else -1)) % options.option_count
            event.stop()
            event.prevent_default()

    @on(Input.Submitted)
    def submit(self):
        option = self.query_one(OptionList).highlighted_option
        if option:
            self.dismiss(option.id)

    @on(OptionList.OptionSelected)
    def select(self, event):
        self.dismiss(event.option_id)

    def action_close(self):
        self.dismiss(None)
