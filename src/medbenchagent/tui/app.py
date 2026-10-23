"""Single-column chat workbench; execution and projections belong to runtime services."""
import threading
import os
from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, DataTable, Static

from medbenchagent.runtime import sessions
from medbenchagent.runtime.activity import AGENTS, OPERATIONS, elapsed
from medbenchagent.runtime.construction import sanitize
from medbenchagent.runtime.conversation_service import ConversationService
from medbenchagent.runtime.intake import InputDraft, suggest_draft, check_connections
from .widgets import Composer, ChatMessage
from .dialogs import DraftScreen, ConnectionScreen, HistoryScreen, ActivityScreen, ArtifactsScreen

STATES = {"ready": "准备启动", "running": "运行中", "paused": "已暂停", "waiting": "等待回答",
          "stopped": "已结束", "interrupted": "异常中断，可恢复", "unavailable": "运行不可用", "external": "由 Studio 管理"}
RESULTS = {"succeeded": "目标完成", "partial": "部分交付", "blocked": "目标受阻", "failed": "目标失败"}


class MBAApp(App):
    TITLE = "MedD2B"
    CSS = """
    Screen { background: #151a23; color: #d8e2ee; }
    #toolbar, #controls { height: 3; }
    #title { width: 1fr; padding: 1 1; color: #7ccdc6; text-style: bold; }
    Button { min-width: 8; width: auto; margin: 0; }
    #run-status { height: 1; padding: 0 1; color: #a6b5ce; }
    #agents { height: 6; margin: 0 1; background: #1b2330; }
    #conversation { height: 1fr; min-height: 3; padding: 0 2; }
    .chat-message { height: auto; margin: 1 0 0 0; }
    #new-content { height: 1; color: #7ccdc6; text-align: center; }
    #message { height: 3; margin: 0 1; border: round #4d617b; }
    #key-hint { width: 1fr; padding: 1 1; color: #8494a8; }
    ModalScreen { align: center middle; background: #101620b0; }
    .dialog { width: 90%; max-width: 105; height: 90%; padding: 1 2; border: round #5d8492; background: #1b2330; }
    .small-dialog { height: auto; max-height: 20; }
    .dialog-title { height: 2; text-style: bold; color: #7ccdc6; }
    .dialog Horizontal { height: 3; }
    .dialog Input, .dialog Select { width: 1fr; }
    .dialog Label { height: auto; }
    .dialog VerticalScroll { height: 1fr; }
    .short-area { height: 4; }
    .config-area { height: 12; }
    #path-tree, #activity-detail, #history-runs, #artifact-summary { height: 1fr; }
    #skills { height: 5; }
    #draft-error, #connection-status, #picker-error, #export-status { height: auto; max-height: 4; color: #e5bd87; }
    """
    BINDINGS = [("ctrl+q", "request_quit", "退出"), ("ctrl+n", "new_task", "新任务")]

    def __init__(self, *, root=sessions.EXPERIMENTS_ROOT, config_template=None):
        super().__init__()
        self.root, self.config_template = Path(root), config_template
        self.experiment = "default"
        self.selected_run = self.active_run = None
        self.active_experiment = None
        self.running = False
        self.exit_after = False
        self.stop_event = threading.Event()
        self.view = {}
        self.draft = InputDraft()
        self.reader = None
        self.reading = False
        self.latest_activity = {}
        self.scope_labels = {}
        self.seen_messages = set()
        self.unread = 0
        self.pending_answer = None
        self.selection = 0
        self.rendering_history = False

    @property
    def service(self):
        return ConversationService(self.experiment, root=self.root)

    def compose(self) -> ComposeResult:
        with Horizontal(id="toolbar"):
            yield Static("MedD2B", id="title")
            yield Button("历史", id="history")
            yield Button("产物", id="products")
            yield Button("活动", id="details")
            yield Button("设置", id="settings")
        yield Static("选择数据，描述目标，即可开始。", id="run-status")
        yield DataTable(id="agents", cursor_type="row")
        yield VerticalScroll(id="conversation")
        yield Static("", id="new-content")
        yield Composer(id="message", soft_wrap=True)
        with Horizontal(id="controls"):
            yield Button("新任务", id="new")
            yield Button("暂停", id="pause", disabled=True)
            yield Button("继续", id="resume", disabled=True)
            yield Static("Enter 发送 · Shift+Enter 换行", id="key-hint")
            yield Button("发送", id="send", variant="primary")

    def on_mount(self):
        if not (self.root / "default" / "session.json").exists():
            from medbenchagent.settings import PROJECT_ROOT, load_project_env
            load_project_env()
            configured = os.getenv("MEDBENCH_CONFIG")
            existing = (PROJECT_ROOT / Path(configured).expanduser()).resolve() if configured else PROJECT_ROOT / "config" / "local.toml"
            template = self.config_template or (existing if existing.is_file() else None)
            sessions.create_experiment("default", root=self.root, template=template)
        self.query_one("#agents", DataTable).add_columns("Agent", "当前 / 最近动作", "耗时")
        self.query_one("#new-content").display = False
        self.query_one("#agents").display = False
        record = self.service.record()
        self.draft = InputDraft.model_validate(record.get("draft", {}))
        if record["runs"]:
            self.select_run(record["runs"][-1]["id"])
        else:
            self.chat("MedD2B", "描述任务并选择本地数据目录。系统会自动准备运行文件；遇到需要确认的事实，会在这里询问。")
        self.set_interval(0.25, self.poll_activity)
        self.query_one(Composer).focus()

    def chat(self, role, text, *, identity=None):
        if not text or identity in self.seen_messages:
            return
        if identity is not None:
            self.seen_messages.add(identity)
        history = self.query_one("#conversation", VerticalScroll)
        follow = history.is_vertical_scroll_end
        history.mount(ChatMessage(role, str(text) if role == "你" else str(sanitize(str(text)))))
        if self.rendering_history:
            return
        if follow:
            self.call_after_refresh(history.scroll_end, animate=False)
        else:
            self.unread += 1
            self.query_one("#new-content", Static).update(f"↓ {self.unread} 条新消息 · 滚动到底部查看")
            self.query_one("#new-content").display = True

    def select_run(self, run_id, *, preserve_scroll=False):
        history = self.query_one("#conversation", VerticalScroll)
        viewport = (history.is_vertical_scroll_end, history.scroll_y) if preserve_scroll else (True, 0)
        if not preserve_scroll:
            self.unread = 0
            self.query_one("#new-content").display = False
        self.selection += 1
        self.selected_run = run_id
        if not preserve_scroll or self.reader is None:
            self.reader = self.service.activity_reader(run_id)
            self.latest_activity.clear()
            self.scope_labels.clear()
        self.load_run(self.experiment, run_id, self.selection, viewport)

    @work(thread=True, group="load-view", exclusive=True)
    def load_run(self, experiment, run_id, selection, viewport):
        try:
            service = ConversationService(experiment, root=self.root)
            view, record = service.inspect(run_id), service.record()
            self.call_from_thread(self.loaded, experiment, run_id, selection, view, record, viewport)
        except Exception as exc:
            self.call_from_thread(self.notify, str(sanitize(str(exc))), severity="error")

    def loaded(self, experiment, run_id, selection, view, record, viewport):
        if (experiment, run_id, selection) != (self.experiment, self.selected_run, self.selection):
            return
        self.view = view
        self.rendering_history = True
        self.query_one("#conversation", VerticalScroll).remove_children()
        self.seen_messages.clear()
        history = view.get("history", [])
        for after_work in range(len(history) + 1):
            for index, message in enumerate(record["messages"]):
                if message.get("run_id") == run_id and message.get("after_work", 0) == after_work:
                    key = message.get("id", f"message-{index}")
                    self.chat("你" if message["role"] == "user" else "需要你的回答", message["text"], identity=key)
            if after_work < len(history):
                item = history[after_work]
                self.chat(AGENTS.get(item["agent_id"], item["agent_id"]), item["summary"], identity=item["request_id"])
        self.show_question(view)
        if view.get("status") == "stopped":
            self.chat("结果", f"{RESULTS.get(view.get('goal_result'), '')}\n{view.get('summary', '')}", identity="finish")
        if view.get("error"):
            self.chat("运行中断", view["error"], identity="execution-error")
        self.rendering_history = False
        history = self.query_one("#conversation", VerticalScroll)
        if viewport[0]:
            self.call_after_refresh(history.scroll_end, animate=False)
        else:
            self.call_after_refresh(history.scroll_to, y=viewport[1], animate=False)
        self.render_status()
        self.render_agents()

    def show_question(self, view):
        questions = view.get("questions", [])
        if questions:
            question = questions[0]
            text = question.get("question", str(question)) if isinstance(question, dict) else str(question)
            self.chat("需要你的回答", text, identity="question-" + str(view.get("question_id", text)))
            self.query_one(Composer).focus()

    def render_status(self):
        active = self.running and (self.experiment, self.selected_run) == (self.active_experiment, self.active_run)
        status = "正在暂停 · 当前工作结束后保存" if self.running and self.stop_event.is_set() else "运行中" if active else STATES.get(self.view.get("status"), "准备启动")
        if self.running and self.active_run is None:
            status = "正在检查输入和模型连接…" if not self.stop_event.is_set() else "正在取消启动…"
        background = " · 另一个任务在后台运行" if self.running and self.active_run and not active else ""
        summary = self.view.get("goal", self.draft.goal) or "描述目标并选择数据目录"
        result = RESULTS.get(self.view.get("goal_result"), "")
        counts = self.view.get("counts")
        quantity = (f"{self.view['count_kind']} {counts.get('accepted', '?')}/{counts.get('requested', '?')} · 缺口 {counts.get('shortfall', '?')}"
                    if counts else f"目标 {self.view['target_count']} 题" if self.view.get("target_count") else summary)
        self.query_one("#run-status", Static).update(Text(f"{status}{background} · {quantity} · {result}", overflow="ellipsis", no_wrap=True))
        self.query_one("#title", Static).update(Text(f"MedD2B · {self.experiment}", overflow="ellipsis", no_wrap=True))
        self.query_one("#pause", Button).disabled = not self.running or self.stop_event.is_set()
        self.query_one("#resume", Button).disabled = self.running or not self.selected_run or self.view.get("status") not in {"ready", "paused", "interrupted"}
        self.query_one("#resume", Button).label = "开始" if self.view.get("status") == "ready" else "继续"
        self.query_one("#settings", Button).disabled = self.running

    def render_agents(self):
        table = self.query_one("#agents", DataTable)
        selected = table.cursor_row
        table.clear()
        enabled = set(self.view.get("enabled_agents", [])) | self.latest_activity.keys()
        for agent, label in AGENTS.items():
            if agent not in enabled:
                continue
            row = self.latest_activity.get(agent)
            if row:
                identity = row.get("parent_request_id") or row.get("request_id")
                scope = self.scope_labels.get(identity, "")
                text = row["action"] + (" · " + scope if scope else "")
                if row["running"] and not (self.running and (self.experiment, self.selected_run) == (self.active_experiment, self.active_run)):
                    text = "上次：" + text
                duration = f"{elapsed(row)}s" if row["running"] and self.running and (self.experiment, self.selected_run) == (self.active_experiment, self.active_run) else f"{row['event']['elapsed_ms'] / 1000:.1f}s" if row["event"].get("elapsed_ms") is not None else "—"
            else:
                previous = next((work for work in reversed(self.view.get("history", [])) if work["agent_id"] == agent), None)
                text = {"completed": "本次工作完成", "blocked": "本次工作受阻", "failed": "本次工作失败"}.get(previous["outcome"], "已返回") if previous else "暂无活动记录" if self.view.get("status") != "ready" else "待委派"
                duration = "—"
            if agent == "supervisor" and self.view.get("status") == "stopped":
                text = "已结束 · " + RESULTS.get(self.view.get("goal_result"), "")
            table.add_row(label, text[:max(20, self.size.width - 35)], duration, key=agent)
        table.styles.height = min(9, max(2, len(enabled) + 1))
        table.display = bool(enabled)
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
            self.read_activity(self.reader, self.experiment, self.selected_run)
        self.render_agents()

    @work(thread=True, group="activity")
    def read_activity(self, reader, experiment, run_id):
        try:
            rows = reader.read()
            if self.is_running:
                self.call_from_thread(self.activities, reader, experiment, run_id, rows)
        finally:
            self.reading = False

    def activities(self, reader, experiment, run_id, rows):
        if not self.query("#agents"):
            return
        if reader is not self.reader or (experiment, run_id) != (self.experiment, self.selected_run):
            return
        for row in rows:
            self.latest_activity[row["agent_id"]] = row
            event = row["event"]
            if event["event_type"] == "work_delegated":
                objects = [key for scope in event.get("scope", []) for key in scope.get("object_ids", [])]
                self.scope_labels[row["request_id"]] = ", ".join(objects) or OPERATIONS.get(row["operation"], row["operation"])
        if rows and isinstance(self.screen, ActivityScreen):
            self.screen.update_rows(list(reader.events))
        self.render_agents()

    @on(DataTable.RowSelected, "#agents")
    def activity_selected(self, event):
        self.push_screen(ActivityScreen(list(self.reader.events) if self.reader else [], str(event.row_key.value)))

    def action_new_task(self):
        self.push_screen(DraftScreen(self.draft.model_copy(deep=True), self.service.folder / "runs"), self.draft_ready)

    def draft_ready(self, result):
        if result is None:
            return
        self.draft, start = result
        self.service.save_draft(self.draft)
        if start:
            if self.running:
                self.notify("草稿已保存；当前运行结束或暂停后再开始")
            else:
                self.start_execution(draft=self.draft)
        else:
            self.notify("草稿已保存")

    @on(Composer.Submitted)
    def submitted(self, event):
        self.send(event.value)

    def send(self, text):
        if not text.strip():
            return
        if self.running:
            self.notify("当前正在运行。可用“新任务”保存草稿；需要回答时会在这里提示。")
            return
        if self.view.get("questions") and self.selected_run:
            self.pending_answer = text.strip()
            self.start_execution(run_id=self.selected_run, answer=self.pending_answer)
        else:
            self.draft = suggest_draft(text, self.draft)
            self.action_new_task()

    @on(Button.Pressed)
    def pressed(self, event):
        action = event.button.id
        if action == "new":
            self.action_new_task()
        elif action == "send":
            self.send(self.query_one(Composer).text)
        elif action == "pause":
            self.stop_event.set()
            self.render_status()
        elif action == "resume" and self.selected_run and not self.running:
            self.start_execution(run_id=self.selected_run)
        elif action == "settings" and not self.running:
            self.push_screen(ConnectionScreen(sessions.config_path(self.experiment, self.root), self.draft))
        elif action == "history":
            self.push_screen(HistoryScreen(sessions.list_experiments(self.root), self.experiment, self.service.record()["runs"]), self.history_selected)
        elif action == "products" and self.view.get("run_directory"):
            self.push_screen(ArtifactsScreen(self.view))
        elif action == "details":
            self.push_screen(ActivityScreen(list(self.reader.events) if self.reader else [], "*"))

    def history_selected(self, result):
        if result:
            self.experiment, run_id = result
            record = self.service.record()
            self.draft = InputDraft.model_validate(record.get("draft", {}))
            if run_id or record["runs"]:
                self.select_run(run_id or record["runs"][-1]["id"])

    def start_execution(self, *, draft=None, run_id=None, answer=None):
        if self.running:
            return
        self.running = True
        self.active_run, self.active_experiment = run_id, self.experiment
        self.stop_event.clear()
        self.render_status()
        self.execute(self.experiment, draft, run_id, answer, self.view.get("question_id") if answer is not None else None,
                     run_id is not None and self.view.get("status") != "ready")

    @work(thread=True, exclusive=True, group="execution")
    def execute(self, experiment, draft, run_id, answer, question_id, resume):
        phase = "connection"
        try:
            service = ConversationService(experiment, root=self.root)
            if run_id is None:
                check_connections(sessions.config_path(experiment, self.root), draft)
                if self.stop_event.is_set():
                    return
                phase = "input"
                row = service.submit_draft(draft)
                run_id = row["id"]
                self.call_from_thread(self.started, experiment, row, draft)
            for event in service.run(run_id, resume=resume, answer=answer, question_id=question_id, stop_requested=self.stop_event.is_set):
                self.call_from_thread(self.show_event, experiment, event)
        except Exception as exc:
            self.call_from_thread(self.execution_error, str(sanitize(str(exc))), run_id is None and phase == "connection")
        finally:
            self.call_from_thread(self.finished, experiment, run_id)

    def started(self, experiment, row, draft):
        self.active_run = row["id"]
        if self.experiment == experiment:
            self.selected_run = row["id"]
            self.selection += 1
            self.view = {"status": "running", "goal": draft.goal, "run_directory": row["directory"]}
            self.reader = self.service.activity_reader(row["id"])
            self.latest_activity.clear()
            self.scope_labels.clear()
            self.seen_messages.clear()
            self.query_one("#conversation", VerticalScroll).remove_children()
            self.chat("你", draft.goal)
            self.query_one(Composer).load_text("")
            self.render_status()

    def show_event(self, experiment, event):
        if (experiment, event.run_session_id) != (self.experiment, self.selected_run):
            return
        if event.kind == "initialized":
            self.view = event.payload
        elif event.kind == "answer_accepted":
            self.chat("你", self.pending_answer or "", identity="answer-" + str(event.payload.get("question_id")))
            self.query_one(Composer).load_text("")
            self.pending_answer = None
        elif event.kind == "progress" and event.payload.get("work"):
            self.view = event.payload["view"]
            row = event.payload["work"]
            self.chat(AGENTS.get(row["agent_id"], row["agent_id"]), row["summary"], identity=row["request_id"])
        elif event.kind in {"finished", "needs_input", "paused"}:
            self.view = event.payload
            self.show_question(self.view)
            if event.kind == "finished":
                self.chat("结果", f"{RESULTS.get(self.view.get('goal_result'), '')}\n{self.view.get('summary', '')}", identity="finish")
        self.render_status()

    def execution_error(self, message, before_run):
        self.chat("运行提示", message)
        if before_run and not self.exit_after:
            self.call_after_refresh(self.push_screen, ConnectionScreen(sessions.config_path(self.experiment, self.root), self.draft))

    def finished(self, experiment, run_id):
        self.running = False
        self.active_run = None
        if self.exit_after:
            self.exit()
            return
        if (experiment, run_id) == (self.experiment, self.selected_run) and run_id:
            self.select_run(run_id, preserve_scroll=True)
        self.render_status()

    def action_request_quit(self):
        if self.running:
            self.exit_after = True
            self.stop_event.set()
            self.notify("已请求暂停；当前工作保存后退出")
            self.render_status()
        else:
            self.exit()
