"""Core experiment interactions backed exclusively by Compilation."""
import json
import sys
import threading
from pathlib import Path

import tomlkit
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, DataTable, Input, RichLog, Select, Static, TabbedContent, TabPane, TextArea

from medbenchagent.compilation.contracts import GoalSpec
from medbenchagent.runtime import sessions
from medbenchagent.runtime.conversation_service import ConversationService

def _supports_block_glyphs() -> bool:
    try:
        "█▄▀".encode(getattr(sys.stdout, "encoding", None) or "utf-8")
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def _brand() -> Text:
    glyphs = {
        "M": ("█   █", "██ ██", "█ █ █", "█   █", "█   █"),
        "E": ("█▀▀▀", "█   ", "█▀▀ ", "█   ", "▀▀▀▀"),
        "D": ("█▀▀▄", "█  █", "█  █", "█  █", "▀▀▀ "),
        "2": ("▀▀▀▄", "   █", "▄▀▀ ", "█   ", "▀▀▀▀"),
        "B": ("█▀▀▄", "█  █", "█▀▀▄", "█  █", "▀▀▀ "),
        "N": ("█  █", "██ █", "█ ██", "█  █", "█  █"),
        "C": ("▄▀▀▀", "█   ", "█   ", "█   ", "▀▄▄▄"),
        "H": ("█  █", "█  █", "█▀▀█", "█  █", "█  █"),
        "A": ("▄▀▀▄", "█  █", "█▀▀█", "█  █", "█  █"),
        "G": ("▄▀▀▀", "█   ", "█ ▀█", "█  █", "▀▄▄▀"),
        "T": ("▀█▀", " █ ", " █ ", " █ ", " █ "),
    }
    colors = ("#65e5d4", "#69dfdf", "#70d7eb", "#77cef6", "#80c3ff",
              "#8bb7ff", "#98aafe", "#a99cfa", "#b991f1", "#c789e7",
              "#d483d9", "#df82c9", "#e78cbb")
    title = Text()
    if not _supports_block_glyphs():
        for letter, color in zip("MedD2B", colors[:6], strict=True):
            title.append(letter, style=f"bold {color}")
        return title
    rows = [" ".join(glyphs[letter][row] for letter in "MEDD2B") for row in range(5)]
    column_colors = []
    for index, letter in enumerate("MEDD2B"):
        column_colors.extend([colors[index]] * len(glyphs[letter][0]))
        if index < 5:
            column_colors.append(colors[index])
    # A one-cell down/right shadow never overwrites a letter's foreground.
    width = len(rows[0]) + 1
    for y in range(6):
        for x in range(width):
            char = rows[y][x] if y < 5 and x < width - 1 else " "
            shadow = y > 0 and x > 0 and rows[y - 1][x - 1] != " "
            if char != " ":
                title.append(char, style=f"bold {column_colors[x]}")
            else:
                title.append("░" if shadow else " ", style="#303845")
        if y < 5:
            title.append("\n")
    return title



class MBAApp(App):
    TITLE = "MedD2B"
    CSS = """
    Screen { background: #151a23; color: #d8e2ee; }
    #brand { height: 7; margin: 1 2; }
    Horizontal { height: auto; margin: 0 1; }
    Input, Select { width: 1fr; }
    Button { margin: 0 1; }
    #runs { height: 9; }
    #goal { height: 5; }
    #events { height: 1fr; min-height: 8; }
    #config { height: 1fr; }
    #products { height: 1fr; }
    Label { margin: 0 2; }
    """
    BINDINGS = [("ctrl+q", "quit", "Quit")]

    def __init__(self, *, root=sessions.EXPERIMENTS_ROOT, config_template=None):
        super().__init__()
        self.root = Path(root)
        self.config_template = config_template
        self.experiment = None
        self.selected_run = None
        self.stop_event = threading.Event()
        self.running = False

    def compose(self) -> ComposeResult:
        yield Static(_brand(), id="brand")
        with Horizontal():
            yield Select([(name, name) for name in sessions.list_experiments(self.root)], prompt="Experiment", id="experiment")
            yield Input(placeholder="New experiment name", id="new-name")
            yield Button("Create", id="create")
        with TabbedContent():
            with TabPane("Compilation"):
                yield DataTable(id="runs")
                with Horizontal():
                    yield Select([("Raw sources", "source_manifest"), ("Canonical Dataset", "dataset_manifest")], value="source_manifest", allow_blank=False, id="source-kind")
                    yield Input(placeholder="Manifest path", id="source")
                with Horizontal():
                    yield Select([("Benchmark", "benchmark"), ("Profile", "profile"), ("Scenario plan", "scenarios"), ("Executable Leaf plan", "executable_leaves")], value="benchmark", allow_blank=False, id="goal-kind")
                    yield Input(placeholder="Sample count (config default)", id="count")
                    yield Select([("Allow limited delivery", "allow"), ("Require full delivery", "full")], value="allow", allow_blank=False, id="limited")
                yield TextArea("Compile and deliver a traceable benchmark from the supplied data.", id="goal")
                with Horizontal():
                    yield Button("Run", id="run", variant="primary")
                    yield Button("Stop after current step", id="stop")
                    yield Button("Resume selected", id="resume")
                    yield Input(placeholder="Answer to pending question", id="answer")
                yield RichLog(id="events", wrap=True, markup=False)
            with TabPane("Artifacts"):
                yield RichLog(id="products", wrap=True, markup=False)
            with TabPane("Configuration"):
                yield TextArea(id="config")
                yield Button("Save for new runs", id="save-config")

    def on_mount(self):
        table = self.query_one("#runs", DataTable)
        table.add_columns("Run", "State", "Goal result", "Goal")
        table.cursor_type = "row"

    @on(Select.Changed, "#experiment")
    def select_experiment(self, event):
        if event.value is Select.BLANK:
            return
        self.experiment = str(event.value)
        self.selected_run = None
        self.query_one("#config", TextArea).load_text(sessions.config_path(self.experiment, self.root).read_text(encoding="utf-8"))
        self.refresh_runs()

    def refresh_runs(self):
        if not self.experiment:
            return
        service = ConversationService(self.experiment, root=self.root)
        table = self.query_one("#runs", DataTable)
        table.clear()
        for row in service.record()["runs"]:
            try:
                view = service.inspect(row["id"])
                status, result = view["status"], view.get("goal_result") or ""
            except (ValueError, OSError):
                status, result = "unavailable", ""
            table.add_row(row["id"], status, result, row["goal"], key=row["id"])

    @on(DataTable.RowSelected, "#runs")
    def select_run(self, event):
        self.selected_run = str(event.row_key.value)
        view = ConversationService(self.experiment, root=self.root).inspect(self.selected_run)
        self.query_one("#events", RichLog).write(json.dumps({k: v for k, v in view.items() if k != "artifacts"}, ensure_ascii=False, indent=2))
        products = self.query_one("#products", RichLog)
        products.clear()
        for ref in view.get("artifacts", []):
            products.write(f"{ref['kind']}  {ref['artifact_id']}\n{Path(view['run_directory']) / ref['path']}")

    @on(Button.Pressed)
    def button(self, event):
        try:
            action = event.button.id
            if action == "create":
                name = self.query_one("#new-name", Input).value.strip()
                sessions.create_experiment(name, root=self.root, template=self.config_template)
                selector = self.query_one("#experiment", Select)
                selector.set_options([(n, n) for n in sessions.list_experiments(self.root)])
                selector.value = name
            elif action == "save-config":
                if not self.experiment:
                    raise ValueError("Select an experiment")
                content = self.query_one("#config", TextArea).text
                tomlkit.parse(content)
                sessions.config_path(self.experiment, self.root).write_text(content, encoding="utf-8")
                self.notify("Saved. Existing runs keep their frozen configuration.")
            elif action == "stop":
                self.stop_event.set()
            elif action in {"run", "resume"}:
                if not self.experiment or self.running:
                    raise ValueError("Select an experiment and wait for the active run")
                options = {}
                if action == "run":
                    kind = self.query_one("#goal-kind", Select).value
                    options["goal_spec"] = GoalSpec(kind="plan" if kind in {"scenarios", "executable_leaves"} else kind,
                        plan_level=kind if kind in {"scenarios", "executable_leaves"} else None,
                        allow_limited_delivery=self.query_one("#limited", Select).value == "allow")
                    source = self.query_one("#source", Input).value.strip()
                    if not source:
                        raise ValueError("Enter a source/Dataset manifest")
                    options[self.query_one("#source-kind", Select).value] = Path(source)
                    count = self.query_one("#count", Input).value.strip()
                    options["sample_count"] = int(count) if count else None
                    options["goal"] = self.query_one("#goal", TextArea).text
                elif not self.selected_run:
                    raise ValueError("Select a run to resume")
                self.running = True
                self.query_one("#experiment", Select).disabled = True
                self.query_one("#create", Button).disabled = True
                self.stop_event.clear()
                answer = self.query_one("#answer", Input).value.strip() or None
                self.execute(self.experiment, self.selected_run if action == "resume" else None, options, answer)
        except (ValueError, OSError) as exc:
            self.notify(str(exc), severity="error")

    @work(thread=True, exclusive=True)
    def execute(self, experiment, run_id, options, answer):
        try:
            service = ConversationService(experiment, root=self.root)
            resume = run_id is not None
            if not resume:
                row = service.submit(**options)
                run_id = row["id"]
            self.call_from_thread(self.set_active_run, run_id)
            for event in service.run(run_id, resume=resume, answer=answer if resume else None, stop_requested=self.stop_event.is_set):
                self.call_from_thread(self.show_event, event)
        except Exception as exc:
            self.call_from_thread(self.notify, str(exc), severity="error")
        finally:
            self.call_from_thread(self.finished)

    def set_active_run(self, identifier):
        self.selected_run = identifier
        self.refresh_runs()

    def show_event(self, event):
        self.query_one("#events", RichLog).write(event.payload.get("text") or json.dumps(event.payload, ensure_ascii=False, indent=2))
        if event.kind == "finished":
            products = self.query_one("#products", RichLog)
            products.clear()
            for ref in event.payload.get("artifacts", []):
                products.write(f"{ref['kind']}: {Path(event.payload['run_directory']) / ref['path']}")

    def finished(self):
        self.running = False
        self.query_one("#experiment", Select).disabled = False
        self.query_one("#create", Button).disabled = False
        self.refresh_runs()
