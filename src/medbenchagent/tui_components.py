"""Presentation-only input and experiment index helpers."""

from pathlib import Path
import threading

from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.widgets import OptionList, TextArea

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
