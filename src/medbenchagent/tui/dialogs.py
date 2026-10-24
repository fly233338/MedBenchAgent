import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from rich.text import Text
from textual import on, work
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Collapsible,
    DirectoryTree,
    Input,
    Label,
    RichLog,
    Select,
    SelectionList,
    Static,
    Switch,
    TextArea,
)

from medbenchagent.compilation.contracts import ArtifactRef
from medbenchagent.compilation.artifacts import read_json_artifact, resolve_artifact
from medbenchagent.runtime.activity import AGENTS, PRODUCTS, artifact_summary
from medbenchagent.runtime.intake import (
    InputDraft,
    SelectedFile,
    input_path,
    validate_draft,
    connection_values,
    save_connection,
    save_advanced_config,
    set_session_credentials,
    probe_connections,
)
from medbenchagent.runtime.construction import sanitize
from medbenchagent.generation.publication import export_benchmark


class Dialog(ModalScreen):
    BINDINGS = [("escape", "cancel", "Close")]

    def action_cancel(self):
        self.dismiss(None)


class PathPicker(Dialog):
    def __init__(self, *, directory=True, start=None):
        super().__init__()
        self.directory = directory
        self.start = input_path(start or str(Path.cwd()))
        if not self.start.is_dir():
            self.start = self.start.parent

    def compose(self):
        with Vertical(classes="dialog picker"):
            yield Label(
                "Source directory" if self.directory else "Key file",
                classes="dialog-title",
            )
            with Horizontal():
                yield Input(str(self.start), id="browse-path")
                yield Button("Go", id="browse-go")
            yield DirectoryTree(str(self.start), id="path-tree")
            yield Static("", id="picker-error")
            with Horizontal():
                yield Button("Select", id="use-path", variant="primary")
                yield Button("Cancel", id="cancel")

    @on(DirectoryTree.DirectorySelected)
    @on(DirectoryTree.FileSelected)
    def selected(self, event):
        self.query_one("#browse-path", Input).value = str(event.path)

    @on(Button.Pressed)
    def pressed(self, event):
        event.stop()
        try:
            if event.button.id == "cancel":
                self.dismiss(None)
            elif event.button.id == "browse-go":
                path = input_path(self.query_one("#browse-path", Input).value)
                if not path.is_dir():
                    raise ValueError("Directory not readable.")
                self.query_one(DirectoryTree).path = path
            elif event.button.id == "use-path":
                path = input_path(self.query_one("#browse-path", Input).value)
                if not (path.is_dir() if self.directory else path.is_file()):
                    raise ValueError("Select a directory." if self.directory else "Select a file.")
                self.dismiss(str(path))
        except (ValueError, OSError) as exc:
            self.query_one("#picker-error", Static).update(Text(str(exc)))


class DraftScreen(Dialog):
    def __init__(self, draft, output):
        super().__init__()
        self.draft, self.output = draft, output

    def compose(self):
        with Vertical(classes="dialog"):
            yield Label("Task", classes="dialog-title")
            with VerticalScroll():
                yield Input(
                    self.draft.dataset_id or "",
                    placeholder="Dataset ID · auto",
                    id="draft-dataset-id",
                )
                yield Label("Sources · one path per line")
                yield TextArea(
                    "\n".join(self.draft.source_dirs),
                    id="sources",
                    classes="short-area",
                )
                with Horizontal():
                    yield Button("+ Directory", id="add-dir")
                    yield Button("+ Key file", id="add-file")
                with Collapsible(
                    title="Key files · path | purpose",
                    collapsed=not self.draft.key_files,
                    id="key-files-panel",
                ):
                    yield TextArea(
                        "\n".join(f"{f.path} | {f.purpose}" for f in self.draft.key_files),
                        id="key-files",
                        classes="short-area",
                    )
                yield Select(
                    [
                        ("Profile", "profile"),
                        ("Scenarios", "scenarios"),
                        ("Executable tasks", "executable_leaves"),
                        ("Benchmark", "benchmark"),
                    ],
                    value=self.draft.goal_kind,
                    allow_blank=False,
                    id="draft-kind",
                )
                with Horizontal(id="quantity"):
                    yield Input(
                        str(self.draft.sample_count or ""),
                        placeholder="Count · default if blank",
                        type="integer",
                        id="draft-count",
                    )
                    yield Input(self.draft.language, placeholder="Language", id="draft-language")
                    yield Label("Partial")
                    yield Switch(self.draft.allow_limited_delivery, id="draft-limited")
                yield Label("Goal")
                yield TextArea(self.draft.goal, id="draft-goal", classes="short-area")
                with Collapsible(
                    title="Canonical input",
                    collapsed=not self.draft.canonical_manifest,
                ):
                    yield Label("Manifest path")
                    yield Input(self.draft.canonical_manifest or "", id="canonical")
            yield Static("", id="draft-error")
            with Horizontal():
                yield Button("Start", id="start-draft", variant="primary")
                yield Button("Save", id="save-draft")
                yield Button("Cancel", id="cancel")

    def on_mount(self):
        self.query_one("#quantity").display = self.draft.goal_kind == "benchmark"

    @on(Select.Changed, "#draft-kind")
    def goal_changed(self, event):
        self.query_one("#quantity").display = event.value == "benchmark"

    def read_draft(self):
        kind = self.query_one("#draft-kind", Select).value
        count = self.query_one("#draft-count", Input).value.strip()
        canonical = self.query_one("#canonical", Input).value.strip() or None
        files = []
        for line in self.query_one("#key-files", TextArea).text.splitlines():
            if line.strip():
                path, _, purpose = line.partition("|")
                files.append(
                    SelectedFile(path=path.strip(), purpose=purpose.strip() or "Selected key file")
                )
        return InputDraft(
            goal=self.query_one("#draft-goal", TextArea).text,
            source_dirs=[]
            if canonical
            else [
                line.strip()
                for line in self.query_one("#sources", TextArea).text.splitlines()
                if line.strip()
            ],
            key_files=[] if canonical else files,
            goal_kind=kind,
            sample_count=int(count) if count and kind == "benchmark" else None,
            allow_limited_delivery=self.query_one("#draft-limited", Switch).value,
            canonical_manifest=canonical,
            source_manifest=None if canonical else self.draft.source_manifest,
            dataset_id=self.query_one("#draft-dataset-id", Input).value.strip() or None,
            language=self.query_one("#draft-language", Input).value.strip() or "en",
        )

    @on(Button.Pressed)
    def pressed(self, event):
        event.stop()
        action = event.button.id
        if action == "cancel":
            self.dismiss(None)
        elif action in {"add-dir", "add-file"}:

            def chosen(path):
                if path:
                    area = self.query_one(
                        "#sources" if action == "add-dir" else "#key-files",
                        TextArea,
                    )
                    area.load_text(
                        (
                            area.text
                            + "\n"
                            + path
                            + (" | Key file" if action == "add-file" else "")
                        ).strip()
                    )
                    if action == "add-file":
                        self.query_one("#key-files-panel", Collapsible).collapsed = False

            self.app.push_screen(PathPicker(directory=action == "add-dir"), chosen)
        elif action in {"start-draft", "save-draft"}:
            try:
                draft = self.read_draft()
                if action == "start-draft":
                    self.query_one("#start-draft", Button).disabled = True
                    self.query_one("#draft-error", Static).update("Checking...")
                    self.validate(draft)
                else:
                    self.dismiss((draft, False))
            except (ValueError, OSError) as exc:
                self.query_one("#draft-error", Static).update(Text(str(exc)))

    @work(thread=True, exclusive=True)
    def validate(self, draft):
        try:
            validate_draft(draft, self.output / "draft-check")
            error = ""
        except Exception as exc:
            error = str(sanitize(str(exc)))
        if self.is_mounted and self.app.is_running:
            self.app.call_from_thread(self.validated, draft, error)

    def validated(self, draft, error):
        if not self.is_mounted:
            return
        if error:
            self.query_one("#draft-error", Static).update(Text(error))
            self.query_one("#start-draft", Button).disabled = False
        else:
            self.dismiss((draft, True))


class ConnectionScreen(Dialog):
    def __init__(self, config, draft):
        super().__init__()
        self.config, self.draft = config, draft
        self.values = connection_values(config)
        self.frozen = (config.parent / "run.json").exists()

    def compose(self):
        with Vertical(classes="dialog"):
            yield Label("Config · frozen" if self.frozen else "Config", classes="dialog-title")
            with VerticalScroll():
                yield Label("Endpoint")
                yield Input(self.values["base_url"], id="base-url", disabled=self.frozen)
                yield Select(
                    [(Text(self.values["model"]), self.values["model"])]
                    if self.values["model"]
                    else [],
                    value=self.values["model"] or Select.NULL,
                    prompt="Select model",
                    id="chat-model",
                    disabled=self.frozen,
                )
                yield Input(password=True, placeholder="API key · session only", id="chat-key")
                with Collapsible(title="Coding · local process", collapsed=False):
                    yield Input(
                        self.values["worker_venv"],
                        placeholder="Worker venv",
                        id="worker-venv",
                        disabled=self.frozen,
                    )
                    yield Input(
                        self.values["coding_base_url"],
                        placeholder="Endpoint · inherit if blank",
                        id="coding-url",
                        disabled=self.frozen,
                    )
                    yield Select(
                        [(Text(self.values["coding_model"]), self.values["coding_model"])]
                        if self.values["coding_model"]
                        else [],
                        value=self.values["coding_model"] or Select.NULL,
                        prompt="Inherit / select model",
                        id="coding-model",
                        disabled=self.frozen,
                    )
                    yield Input(
                        password=True,
                        placeholder="Coding key · session only",
                        id="coding-key",
                    )
                    yield SelectionList(
                        *[
                            (name, name, name in self.values["skills"])
                            for name in ("tabular-inspection", "medical-image", "mask-geometry")
                        ],
                        id="skills",
                        disabled=self.frozen,
                    )
                with Collapsible(title="Advanced", collapsed=True):
                    yield TextArea(
                        (self.config.parent / "run.json").read_text(encoding="utf-8")
                        if self.frozen
                        else self.config.read_text(encoding="utf-8"),
                        id="advanced-config",
                        classes="config-area",
                        read_only=self.frozen,
                    )
                    yield Button("Save TOML", id="save-advanced", disabled=self.frozen)
            yield Static("Keys stay in memory.", id="connection-status")
            with Horizontal():
                yield Button("Probe", id="probe")
                yield Button("Save", id="save-connection", variant="primary", disabled=self.frozen)
                yield Button("Close", id="cancel")

    def values_now(self):
        def model(identifier):
            value = self.query_one(identifier, Select).value
            return "" if value is Select.NULL else str(value)

        return {
            "base_url": self.query_one("#base-url", Input).value.strip(),
            "model": model("#chat-model"),
            "coding_base_url": self.query_one("#coding-url", Input).value.strip(),
            "coding_model": model("#coding-model"),
            "worker_venv": self.query_one("#worker-venv", Input).value.strip(),
            "skills": self.query_one("#skills", SelectionList).selected,
        }

    @on(Button.Pressed)
    def pressed(self, event):
        event.stop()
        try:
            if event.button.id == "cancel":
                self.dismiss(None)
            elif event.button.id == "save-advanced":
                text = self.query_one("#advanced-config", TextArea).text
                save_advanced_config(self.config, text)
                self.dismiss(True)
            else:
                values = self.values_now()
                set_session_credentials(
                    self.query_one("#chat-key", Input).value,
                    self.query_one("#coding-key", Input).value,
                )
                if event.button.id == "save-connection":
                    save_connection(self.config, values)
                    self.dismiss(True)
                elif event.button.id == "probe":
                    self.query_one("#probe", Button).disabled = True
                    self.query_one("#connection-status", Static).update("Probing...")
                    self.probe(values)
        except (ValueError, OSError) as exc:
            self.query_one("#connection-status", Static).update(Text(str(exc)))

    @work(thread=True, exclusive=True)
    def probe(self, values):
        try:
            with TemporaryDirectory(prefix="medbench-config-") as folder:
                path = Path(folder) / "config.toml"
                path.write_text(self.config.read_text(encoding="utf-8"), encoding="utf-8")
                save_connection(path, values)
                result = probe_connections(path, self.draft)
            if self.is_mounted and self.app.is_running:
                self.app.call_from_thread(self.probed, result, "Connected. Select model.")
        except Exception as exc:
            if self.is_mounted and self.app.is_running:
                self.app.call_from_thread(self.probed, {}, str(sanitize(str(exc))))

    def probed(self, result, message):
        if not self.is_mounted:
            return
        self.query_one("#probe", Button).disabled = False
        self.query_one("#connection-status", Static).update(Text(message))
        for key, identifier in (("model", "#chat-model"), ("coding_model", "#coding-model")):
            if key in result:
                select = self.query_one(identifier, Select)
                previous = select.value
                select.set_options([(Text(name), name) for name in result[key]])
                select.value = (
                    previous
                    if previous in result[key]
                    else result[key][0]
                    if len(result[key]) == 1
                    else Select.NULL
                )


class ActivityScreen(Dialog):
    def __init__(self, rows, selected):
        super().__init__()
        self.rows, self.selected = rows, selected

    def compose(self):
        with Vertical(classes="dialog"):
            yield Label("Trace · last 80", classes="dialog-title")
            yield Select(
                [("All", "*")] + [(label, key) for key, label in AGENTS.items()],
                value=self.selected,
                allow_blank=False,
                id="activity-filter",
            )
            yield RichLog(id="activity-detail", wrap=True, markup=False, max_lines=1500)
            yield Button("Close", id="cancel")

    def on_mount(self):
        self.render_rows()

    @on(Select.Changed)
    def changed(self):
        self.render_rows()

    def render_rows(self):
        log = self.query_one(RichLog)
        log.clear()
        selected = self.query_one(Select).value
        for row in [row for row in self.rows if selected in {"*", row["agent_id"]}][-80:]:
            log.write(
                Text(f"{row['label']} · {row['action']}\n{row['detail']}\n"),
                scroll_end=False,
            )

    def update_rows(self, rows):
        log = self.query_one(RichLog)
        follow, position = log.is_vertical_scroll_end, log.scroll_y
        self.rows = rows
        self.render_rows()
        if follow:
            self.call_after_refresh(log.scroll_end, animate=False)
        else:
            self.call_after_refresh(log.scroll_to, y=position, animate=False)

    @on(Button.Pressed)
    def close(self):
        self.dismiss(None)


class ExportScreen(Dialog):
    def __init__(self, source):
        super().__init__()
        self.source = source

    def compose(self):
        with Vertical(classes="dialog small-dialog"):
            yield Label("Export · new directory")
            yield Input(str(self.source.parent / "exported-benchmark"), id="export-path")
            yield Static("", id="export-status")
            with Horizontal():
                yield Button("Export", id="export-now", variant="primary")
                yield Button("Close", id="cancel")

    @on(Button.Pressed)
    def pressed(self, event):
        if event.button.id == "cancel":
            self.dismiss(None)
        else:
            self.query_one("#export-now", Button).disabled = True
            self.export(input_path(self.query_one(Input).value))

    @work(thread=True)
    def export(self, destination):
        try:
            path = export_benchmark(self.source, destination)
            message = f"Exported: {path}"
        except (OSError, ValueError) as exc:
            message = str(exc)
        if self.is_mounted and self.app.is_running:
            self.app.call_from_thread(self.exported, message)

    def exported(self, message):
        if self.is_mounted:
            self.query_one("#export-status", Static).update(Text(message))
            self.query_one("#export-now", Button).disabled = False


class ArtifactsScreen(Dialog):
    def __init__(self, view):
        super().__init__()
        self.directory = Path(view["run_directory"])
        self.refs = [ArtifactRef.model_validate(row) for row in view.get("artifacts", [])]
        self.selected = None

    def compose(self):
        with Vertical(classes="dialog"):
            yield Label("Artifacts", classes="dialog-title")
            yield Select(
                [
                    (Text(f"{index + 1}. {PRODUCTS.get(ref.kind, ref.kind)}"), index)
                    for index, ref in enumerate(self.refs)
                ],
                prompt="Select artifact",
                id="artifact-choice",
            )
            yield TextArea(read_only=True, id="artifact-summary")
            with Collapsible(title="Raw", collapsed=True):
                yield TextArea(read_only=True, id="artifact-raw", classes="config-area")
            with Horizontal():
                yield Button("Copy path", id="copy-artifact")
                yield Button("Open", id="open-artifact")
                yield Button("Export", id="export-artifact", disabled=True)
                yield Button("Close", id="cancel")

    @on(Select.Changed)
    def selected_artifact(self, event):
        if event.value is Select.NULL:
            return
        self.selected = self.refs[event.value]
        try:
            path = resolve_artifact(self.directory, self.selected)
            self.query_one("#artifact-summary", TextArea).load_text(
                str(sanitize(artifact_summary(self.directory, self.selected), limit=16000))
            )
            self.query_one("#artifact-raw", TextArea).load_text(
                str(path)
                + "\n"
                + self.selected.model_dump_json(indent=2)
                + "\n"
                + json.dumps(
                    sanitize(read_json_artifact(self.directory, self.selected)),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            self.query_one("#export-artifact", Button).disabled = (
                self.selected.kind != "delivery_manifest"
            )
        except (OSError, ValueError) as exc:
            self.query_one("#artifact-summary", TextArea).load_text(str(exc))

    @on(Button.Pressed)
    def pressed(self, event):
        event.stop()
        if event.button.id == "cancel":
            self.dismiss(None)
        elif self.selected:
            try:
                path = resolve_artifact(self.directory, self.selected)
            except (OSError, ValueError) as exc:
                self.query_one("#artifact-summary", TextArea).load_text(str(sanitize(str(exc))))
                return
            if event.button.id == "copy-artifact":
                self.app.copy_to_clipboard(str(path))
                self.notify("Copied.")
            elif event.button.id == "open-artifact":
                if os.name == "nt":
                    os.startfile(path.parent)
                else:
                    subprocess.Popen(
                        ["xdg-open", str(path.parent)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
            elif event.button.id == "export-artifact":
                self.app.push_screen(ExportScreen(path.parent))
