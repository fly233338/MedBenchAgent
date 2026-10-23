"""Bounded, disposable activity projections over diagnostic events and formal products."""
import json
from collections import deque
from datetime import datetime
from pathlib import Path

from medbenchagent.runtime.construction import sanitize

AGENTS = {"supervisor": "Supervisor", "data_processing": "Data Processing", "coding": "  ↳ Coding",
          "task_planner": "Task Planning", "generation": "Generation", "review": "Review", "delivery": "Delivery"}
OPERATIONS = {"profile": "调研来源", "prepare_task_data": "准备任务数据", "discover_scenarios": "发现场景",
              "design_leaves": "设计 / 修订任务", "generate": "生成题目", "review_plan": "审查计划",
              "review_candidate": "审查候选", "publish": "交付", "execute": "执行代码"}


class ActivityReader:
    def __init__(self, directory, *, max_events=400):
        self.path = Path(directory) / "events.jsonl"
        self.events = deque(maxlen=max_events)
        self.offset = 0
        self.partial = b""
        self.identity = None
        self.seen = deque(maxlen=2000)
        self.error = ""

    def read(self):
        try:
            stat = self.path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if self.identity != identity or stat.st_size < self.offset:
                self.offset, self.partial = 0, b""
                self.events.clear()
                self.seen.clear()
                self.identity = identity
            with self.path.open("rb") as stream:
                if self.offset == 0 and stat.st_size > 262144:
                    stream.seek(stat.st_size - 262144)
                    stream.readline()
                    self.offset = stream.tell()
                stream.seek(self.offset)
                content = stream.read(262144)
                self.offset = stream.tell()
            lines = (self.partial + content).split(b"\n")
            self.partial = lines.pop()
            if len(self.partial) > 524288:
                self.partial = b""
            fresh = []
            for line in lines:
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict) or not event.get("event_id") or event["event_id"] in self.seen:
                        continue
                    self.seen.append(event["event_id"])
                    if row := activity_view(event):
                        self.events.append(row)
                        fresh.append(row)
                except (ValueError, UnicodeDecodeError, TypeError):
                    continue
            self.error = ""
            return fresh
        except FileNotFoundError:
            return []
        except OSError:
            self.error = "活动日志暂不可读；运行状态仍以 Checkpoint 为准"
            return []


def activity_view(event):
    agent = event.get("agent_id")
    if agent not in AGENTS:
        return None
    kind = event.get("event_type", "")
    status = event.get("status", "")
    labels = {"model_started": "调用模型", "model_finished": "模型已返回", "model_failed": "模型调用失败",
              "tool_started": "调用工具", "tool_finished": "工具已返回", "tool_failed": "工具失败，可返工",
              "work_delegated": "已委派", "work_result": "工作已返回", "coding_started": "执行代码",
              "coding_finished": "Coding 已返回", "coding_action": "执行工具", "coding_observation": "收到工具回执",
              "coding_error": "Coding 执行错误", "review_submitted": "审查报告已提交"}
    if kind not in labels:
        return None
    action = labels[kind]
    if event.get("tool_name"):
        action += " · " + event["tool_name"]
    if kind == "work_result":
        outcome = event.get("outcome", "")
        action = {"completed": "本次工作完成", "blocked": "本次工作受阻", "failed": "本次工作失败",
                  "needs_input": "需要补充信息", "progress": "已取得进展"}.get(outcome, action)
        if agent == "review" and any(i.get("code", "").startswith("review_") for i in event.get("issues", [])):
            action = "审查完成 · 需修订"
    if kind == "review_submitted" and event.get("approved") is False:
        action = "审查完成 · 未批准"
    running = kind in {"model_started", "tool_started", "work_delegated", "coding_started", "coding_action"}
    return {"id": event["event_id"], "timestamp": event.get("timestamp", ""), "agent_id": agent,
            "request_id": event.get("request_id"), "parent_request_id": event.get("parent_request_id"),
            "label": AGENTS[agent], "action": action, "operation": event.get("operation", ""),
            "running": running, "status": status, "summary": str(event.get("summary", "")),
            "detail": json.dumps(sanitize(event), ensure_ascii=False, indent=2), "event": event}


def elapsed(row):
    try:
        return max(0, int((datetime.now().astimezone() - datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))).total_seconds()))
    except (ValueError, KeyError):
        return 0


def artifact_summary(directory, ref):
    from medbenchagent.compilation.artifacts import read_json_artifact
    value = read_json_artifact(Path(directory), ref)
    kind = ref.kind
    if kind == "dataset_profile":
        return f"数据调研 · {value.get('description', '')}\n范围：{value.get('coverage', '')}\n未知项：" + "；".join(value.get("unknowns", []))
    if kind == "task_plan":
        pending = set(value.get("requirements", {})) - value.get("acceptances", {}).keys()
        return f"任务计划 · {len(value.get('scenarios', {}))} 个场景 / {len(value.get('tasks', {}))} 个任务 / {len(pending)} 个未解决需求"
    if kind == "task_data":
        return f"任务数据 · {value.get('case_count', 0)} 个 Case · {value.get('backend_id', '')}"
    if kind == "review_report":
        return f"审查 · {'已批准' if value.get('approved') else '未批准'}\n" + "\n".join(item["summary"] for item in value.get("findings", []))
    if "counts" in value:
        counts = value["counts"]
        return f"{kind} · 请求 {counts.get('requested', '?')} / 接受 {counts.get('accepted', '?')} / 缺口 {counts.get('shortfall', '?')}"
    return kind + "\n" + json.dumps(sanitize(value, limit=1500), ensure_ascii=False, indent=2)
