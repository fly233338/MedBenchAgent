"""Bounded, disposable activity projections over diagnostic events and formal products."""
import json
from collections import deque
from datetime import datetime
from pathlib import Path

from medbenchagent.runtime.construction import sanitize

AGENTS = {"supervisor": "Supervisor", "data_processing": "Data Processing", "coding": "  ↳ Coding",
          "task_planner": "Task Planning", "generation": "Generation", "review": "Review", "delivery": "Delivery", "runtime": "Run"}
OPERATIONS = {"profile": "profile", "prepare_task_data": "prepare data", "discover_scenarios": "Scenarios",
              "design_leaves": "design", "generate": "generate", "review_plan": "review plan",
              "review_candidate": "review candidate", "publish": "Delivery", "execute": "coding"}
PRODUCTS = {"source_manifest": "Sources", "dataset_profile": "Profile", "task_plan": "Plan",
            "data_requirement": "Requirements", "task_data": "TaskData", "generation_spec": "Generation spec",
            "generation_candidate": "Candidate", "review_report": "Review", "audit_report": "Audit",
            "delivery_manifest": "Delivery", "benchmark_request": "Target", "allocation": "Allocation", "generation_batch": "Batch"}


class ActivityReader:
    def __init__(self, directory, *, max_events=400):
        self.path = Path(directory) / "events.jsonl"
        self.events = deque(maxlen=max_events)
        self.offset = 0
        self.partial = b""
        self.identity = None
        self.seen = deque(maxlen=2000)
        self.error = ""
        self.work_started = {}

    def read(self):
        try:
            stat = self.path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if self.identity != identity or stat.st_size < self.offset:
                self.offset, self.partial = 0, b""
                self.events.clear()
                self.seen.clear()
                self.work_started.clear()
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
                    request_id = event.get("request_id")
                    if event.get("event_type") == "work_delegated":
                        self.work_started[request_id] = event.get("timestamp")
                        if len(self.work_started) > 400:
                            self.work_started.pop(next(iter(self.work_started)))
                    if event.get("event_type") == "work_result" and request_id in self.work_started:
                        start = self.work_started.pop(request_id)
                        if start and event.get("timestamp"):
                            event["elapsed_ms"] = max(0, int((datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00")) - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds() * 1000))
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
            self.error = "Trace unavailable. Checkpoint remains authoritative."
            return []


def activity_view(event):
    agent = event.get("agent_id")
    if agent not in AGENTS:
        return None
    kind = event.get("event_type", "")
    status = event.get("status", "")
    labels = {"model_started": "model start", "model_finished": "model done", "model_failed": "model failed",
              "run_started": "run start", "run_resumed": "run resume", "run_paused": "run paused",
              "run_waiting": "awaiting input", "run_finished": "run done", "execution_failed": "interrupted",
              "answer_accepted": "answer accepted", "work_started": "work start", "work_failed": "work failed",
              "artifact_submitted": "artifact committed", "repair_requested": "repair needed", "repair_exhausted": "repair exhausted",
              "decision_rejected": "decision rejected", "context_budget": "context checked",
              "tool_started": "tool start", "tool_finished": "tool done", "tool_failed": "tool failed",
              "work_delegated": "delegated", "work_result": "work done", "coding_started": "coding",
              "coding_finished": "coding done", "coding_action": "tool start", "coding_observation": "tool result",
              "coding_error": "coding error", "coding_failed": "coding failed", "review_submitted": "review committed", "supervisor_decision": "decision"}
    if kind not in labels:
        return None
    action = labels[kind]
    if kind == "supervisor_decision":
        action = {"delegate": "delegate → " + AGENTS.get(event.get("target_agent"), ""),
                  "ask_user": "awaiting input", "finish": "STOPPED · " + {"succeeded": "SUCCESS", "partial": "PARTIAL", "blocked": "BLOCKED", "failed": "FAILED"}.get(event.get("goal_result"), "")}.get(event.get("decision"), action)
    if kind == "coding_finished" and event.get("outcome") in {"blocked", "failed"}:
        action = "coding blocked" if event["outcome"] == "blocked" else "coding failed"
    if event.get("tool_name"):
        action += " · " + event["tool_name"]
    if kind == "model_finished":
        action += (f" · {event.get('input_tokens', '?')} → {event.get('output_tokens', '?')} tokens"
                   if event.get("token_usage_available") else " · usage unavailable")
    if kind == "work_result":
        outcome = event.get("outcome", "")
        action = {"completed": "completed", "blocked": "blocked", "failed": "failed",
                  "needs_input": "needs input", "progress": "progress"}.get(outcome, action)
        if agent == "review" and any(i.get("code", "").startswith("review_") for i in event.get("issues", [])):
            action = "review · rework"
    if kind == "review_submitted" and event.get("approved") is False:
        action = "review · rejected"
    running = kind in {"model_started", "tool_started", "work_delegated", "work_started", "coding_started", "coding_action", "run_started", "run_resumed"}
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
    if kind == "source_manifest":
        return "Sources\n" + "\n".join(source["root"] for source in value["sources"]) + f"\nKey file：{len(value.get('key_files', []))}"
    if kind == "dataset_profile":
        return f"Profile · {value.get('coverage', '')}\n{value.get('description', '')}\nUnknown: " + "; ".join(value.get("unknowns", []))
    if kind == "task_plan":
        pending = set(value.get("requirements", {})) - value.get("acceptances", {}).keys()
        return f"Plan · {len(value.get('scenarios', {}))} scenarios / {len(value.get('tasks', {}))} tasks / {len(pending)} pending"
    if kind == "task_data":
        return f"TaskData · {value.get('case_count', 0)} cases · {value.get('backend_id', '')}"
    if kind == "review_report":
        return f"Review · {'approved' if value.get('approved') else 'rejected'}\n" + "\n".join(item["summary"] for item in value.get("findings", []))
    if kind == "audit_report":
        return f"Audit · {'pass' if value.get('passed') else 'fail'} · {value.get('checked_count', 0)} checked"
    if kind == "data_requirement":
        return "Requirements · " + value["purpose"] + "\nChecks: " + "; ".join(value.get("acceptance_criteria", []))
    if "counts" in value:
        counts = value["counts"]
        return f"{PRODUCTS.get(kind, kind)} · requested {counts.get('requested', '?')} / accepted {counts.get('accepted', '?')} / shortfall {counts.get('shortfall', '?')}"
    if kind == "benchmark_request":
        return f"Target: {value['sample_count']} items"
    return PRODUCTS.get(kind, "Artifact")
