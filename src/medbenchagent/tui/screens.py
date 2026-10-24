"""Run navigation and small creation/selection screens over existing services."""
from pathlib import Path
from uuid import uuid4

from rich.text import Text
from textual import on, work
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Label, OptionList, Static

from medbenchagent.runtime import sessions
from medbenchagent.runtime.conversation_service import ConversationService
from medbenchagent.runtime.intake import select_dataset, register_draft
from medbenchagent.data.dataset_catalog import list_datasets, remove_registration
from .dialogs import Dialog


class RunsScreen(Screen):
    BINDINGS = [("escape", "back", "Back"), ("n", "new", "New")]

    def compose(self):
        with Vertical(id="run-home"):
            yield Label("MedD2B", id="home-title")
            yield Static("dataset → benchmark", id="home-description")
            with Horizontal(id="run-home-actions"):
                yield Input(placeholder="Filter runs...", id="run-search")
                yield Button("New run", id="create-run", variant="primary")
                yield Button("Clone", id="clone-run")
            yield OptionList(id="run-list")
            yield Static("Loading...", id="run-empty")
            yield Static("↑↓ select · Enter open · ^N new · ^C quit", classes="muted")

    def on_mount(self):
        self.rows = []
        self.filtered = []
        self.refresh_runs()

    @work(thread=True, exclusive=True)
    def refresh_runs(self):
        rows = sessions.list_runs(self.app.root)
        for row in rows:
            try:
                row["status"] = ConversationService(row["directory"]).inspect()["status"]
            except (ValueError, OSError, KeyError):
                row["status"] = "unavailable"
        if self.is_mounted:
            self.app.call_from_thread(self.loaded, rows)

    def loaded(self, rows):
        self.rows = rows
        self.filter_runs()

    @on(Input.Changed, "#run-search")
    def filter_runs(self):
        query = self.query_one("#run-search", Input).value.casefold()
        self.filtered = [row for row in self.rows if query in (row["name"] + " " + str(row.get("draft", {}))).casefold()]
        listing = self.query_one("#run-list", OptionList)
        listing.clear_options()
        labels = {"draft": "DRAFT", "ready": "READY", "running": "RUNNING", "waiting": "WAITING", "paused": "PAUSED",
                  "stopped": "STOPPED", "interrupted": "INTERRUPTED", "external": "Studio", "unavailable": "UNAVAILABLE"}
        listing.add_options([Text(f"{row['name']}  ·  {labels.get(row['status'], row['status'])}\n{row.get('draft', {}).get('goal') or 'No goal set'}") for row in self.filtered])
        self.query_one("#run-empty", Static).update("No runs. Create one." if not self.rows else "No matches.")
        self.query_one("#run-empty").display = not self.filtered
        self.query_one("#clone-run", Button).disabled = not self.filtered
        if self.filtered:
            listing.highlighted = 0

    @on(OptionList.OptionSelected, "#run-list")
    def selected(self, event):
        name = self.filtered[event.option_index]["name"]
        self.app.pop_screen()
        self.app.select_run(name)

    @on(Button.Pressed)
    def pressed(self, event):
        event.stop()
        if event.button.id == "create-run":
            self.app.action_new_run()
        elif event.button.id == "clone-run" and self.filtered:
            row = self.filtered[self.query_one("#run-list", OptionList).highlighted or 0]
            self.app.push_screen(NameScreen(copy_name=row["name"]),
                                 lambda name: self.app.create_named_run(name, Path(row["directory"])) if name else None)

    def action_new(self):
        self.app.action_new_run()

    def action_back(self):
        if self.app.selected_run:
            self.app.pop_screen()
        else:
            self.app.action_request_quit()


class NameScreen(Dialog):
    def __init__(self, copy_name=None):
        super().__init__()
        self.copy_name = copy_name

    def compose(self):
        with Vertical(classes="dialog small-dialog"):
            yield Label("Clone run" if self.copy_name else "New run", classes="dialog-title")
            yield Static("One dataset. One compilation.")
            yield Input((self.copy_name + "-copy") if self.copy_name else "run-" + uuid4().hex[:8], id="run-name")
            yield Static("Name: a-z, 0-9, hyphens.", classes="muted")
            yield Static("", id="name-error")
            with Horizontal():
                yield Button("Create", id="confirm-name", variant="primary")
                yield Button("Cancel", id="cancel")

    def on_mount(self):
        self.query_one(Input).focus()

    @on(Input.Submitted)
    def submitted(self):
        self.confirm()

    @on(Button.Pressed)
    def pressed(self, event):
        event.stop()
        self.dismiss(None) if event.button.id == "cancel" else self.confirm()

    def confirm(self):
        try:
            name = self.query_one("#run-name", Input).value.strip()
            path = sessions.run_path(name, self.app.root)
            if path.exists():
                raise ValueError("Name taken.")
            self.dismiss(name)
        except ValueError as exc:
            self.query_one("#name-error", Static).update(Text(str(exc)))


class DatasetScreen(Dialog):
    def __init__(self, draft, *, directory, catalog, frozen=False):
        super().__init__()
        self.draft, self.directory, self.catalog, self.frozen = draft, Path(directory), Path(catalog), frozen
        self.entries = {}
        self.keys = []

    def compose(self):
        with Vertical(classes="dialog"):
            yield Label("Dataset", classes="dialog-title")
            yield Static(Text("\n".join(self.draft.source_dirs) or self.draft.canonical_manifest or "No input"), id="dataset-current")
            yield OptionList(id="datasets")
            yield Static("Source references only.", id="dataset-status")
            with Horizontal():
                yield Button("Select", id="select-dataset", disabled=self.frozen, variant="primary")
                yield Button("Register", id="register-dataset", disabled=self.frozen)
                yield Button("Unlink", id="remove-dataset", disabled=self.frozen)
                yield Button("Close", id="cancel")

    def on_mount(self):
        self.reload()

    def reload(self):
        self.entries = list_datasets(self.catalog)
        self.keys = sorted(self.entries)
        listing = self.query_one("#datasets", OptionList)
        listing.clear_options()
        listing.add_options([Text(f"{key} · {self.entries[key]['kind']}\n{self.entries[key].get('revision', '')}") for key in self.keys])
        if self.keys:
            listing.highlighted = 0

    @on(Button.Pressed)
    def pressed(self, event):
        event.stop()
        try:
            action = event.button.id
            if action == "cancel":
                self.dismiss(None)
            elif action == "register-dataset":
                self.register_current()
            elif self.keys:
                key = self.keys[self.query_one("#datasets", OptionList).highlighted or 0]
                if action == "select-dataset":
                    self.dismiss(select_dataset(key, self.draft, catalog=self.catalog))
                elif action == "remove-dataset":
                    remove_registration(key, self.catalog)
                    self.reload()
                    self.query_one("#dataset-status", Static).update("Unlinked.")
        except (ValueError, OSError, KeyError) as exc:
            self.query_one("#dataset-status", Static).update(Text(str(exc)))

    @work(thread=True, exclusive=True)
    def register_current(self):
        try:
            register_draft(self.directory, self.draft, catalog=self.catalog)
            self.app.call_from_thread(self.registered)
        except Exception as exc:
            if self.is_mounted:
                self.app.call_from_thread(self.query_one("#dataset-status", Static).update, Text(str(exc)))

    def registered(self):
        if self.is_mounted:
            self.reload()
            self.query_one("#dataset-status", Static).update("Registered.")
