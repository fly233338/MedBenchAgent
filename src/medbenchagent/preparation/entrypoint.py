"""Freeze inputs and invoke the production preparation graph with checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from medbenchagent.data.contracts import Source
from medbenchagent.data.sources import inventory_source, verify_source
from medbenchagent.workflow.clarifications import answer_payload
from medbenchagent.runtime.construction import record_event
from medbenchagent.runtime.dataset_runtime import dataset_runtime
from medbenchagent.settings import get_settings

from .contracts import UserDecision, digest_file, digest_payload
from .graph import compile_preparation_graph
from .worker import OpenHandsWorker


def freeze_sources(context: dict[str, Any], workspace: Path):
    """Inventory new sources or verify existing manifests once at entry."""
    sources = [Source.model_validate(item) for item in context["sources"]]
    if not sources or len({item.id for item in sources}) != len(sources):
        raise ValueError("Preparation requires sources with unique IDs")
    manifests = {}
    digests = {}
    for source in sources:
        manifest = workspace / "work/inputs/source_manifests" / source.id
        if (manifest / "source.json").is_file():
            saved = Source.model_validate_json((manifest / "source.json").read_text(encoding="utf-8"))
            if saved != source:
                raise ValueError(f"Source identity changed: {source.id}")
            verify_source(manifest)
        else:
            inventory_source(source, manifest)
        manifests[source.id] = str(manifest)
        digests[source.id] = digest_payload({
            item.relative_to(manifest).as_posix(): digest_file(item)
            for item in sorted(manifest.rglob("*")) if item.is_file()
        })
    return [item.model_dump(mode="json") for item in sources], manifests, digest_payload(digests)


def preparation_decisions(context):
    decisions = {item["issue_id"]: item for item in context.get("user_decisions", [])
                 if isinstance(item, dict) and item.get("issue_id")}
    for issue in context.get("issues", {}).values():
        if (issue.get("role") != "DataProcessingAgent"
                or issue.get("dataset_id") != context["dataset_scope"]
                or issue.get("status") != "resolved"):
            continue
        payload = answer_payload(context, issue)
        issue_id = issue.get("preparation_issue_id", issue["issue_id"])
        decisions[issue_id] = UserDecision(
            issue_id=issue_id, decision_id=issue["issue_id"], dataset_id=context["dataset_scope"],
            answer=payload["text"], exchanges=payload["exchanges"],
            source_ids=issue.get("source_ids", []), fields=issue.get("fields", []),
            evidence_refs=issue.get("evidence_refs", []),
        ).model_dump(mode="json")
    return list(decisions.values())


def run_preparation(context: dict[str, Any]) -> dict[str, Any]:
    dataset_id = context["dataset_scope"]
    workspace = Path(context["run_directory"]) / "preparation" / dataset_id
    if context.get("preparation_revision"):
        workspace /= "repairs/" + context["preparation_revision"]
    workspace.mkdir(parents=True, exist_ok=True)
    sources, manifests, source_digest = freeze_sources(context, workspace)
    decisions = preparation_decisions(context)
    settings = get_settings().data_processing
    state = {
        "run_id": context["run_id"], "dataset_id": dataset_id,
        "run_directory": context["run_directory"], "preparation_root": str(workspace),
        "bundle_root": context["bundle_root"], "sources": sources,
        "source_roots": {item["id"]: item["root"] for item in sources},
        "source_manifests": manifests, "source_snapshot_digest": source_digest,
        "user_request": (context.get("user_request") or context.get("user_requirement") or "")
                        + ("\nSource repair findings: " + context["notes"] if context.get("notes") else ""),
        "user_decisions": decisions,
        "preparation_attempt_limits": context.get("preparation_attempt_limits", settings["preparation_attempt_limits"]),
        "preparation_stage_budgets": context.get("preparation_stage_budgets", settings["preparation_stage_budgets"]),
        **{key: context[key] for key in (
            "execution_runtime", "data_runtime", "construction_events_path", "model",
            "base_url", "reasoning_effort", "llm_timeout_seconds", "llm_max_retries",
        ) if key in context},
    }
    limits = state["preparation_attempt_limits"]
    config = {"configurable": {"thread_id": dataset_id},
              "recursion_limit": max(25, 2 * (limits["understand"] + limits["prepare_text"] + limits["prepare_image"]) + 6)}
    with SqliteSaver.from_conn_string(str(workspace / "checkpoint.sqlite")) as checkpointer:
        graph = compile_preparation_graph(
            worker=OpenHandsWorker(runtime=dataset_runtime(context)), checkpointer=checkpointer,
        )
        snapshot = graph.get_state(config)
        if snapshot.values and snapshot.values["source_snapshot_digest"] != source_digest:
            raise ValueError("Preparation source snapshot changed; start a new run")
        if snapshot.values.get("status") == "complete":
            outcome = snapshot.values
        elif snapshot.next:
            # LangGraph resumes the pending node. Runtime persistence continues
            # only its unfinished conversation or reuses its saved result.
            outcome = graph.invoke(None, config)
        else:
            outcome = graph.invoke({**snapshot.values, **state}, config)
    status = outcome["status"]
    updates = {
        "preparation_workspace": str(workspace), "preparation_status": status,
        "source_snapshot_digest": source_digest, "user_decisions": decisions,
        **{key: outcome[key] for key in (
            "publication", "pending_questions", "diagnostics",
        ) if key in outcome},
    }
    if status == "complete":
        return {"status": "complete", "summary": "Prepared dataset published", "updates": updates}
    if status == "needs_input":
        return {"status": "needs_input", "summary": "Preparation needs a semantic decision", "updates": updates}
    issues = outcome.get("diagnostics", [])
    summary = "; ".join(item.get("message", "") for item in issues) or "Preparation stage blocked"
    return {"status": "blocked", "summary": summary, "updates": updates}


def run_preparation_agent(context):
    def event(kind, **values):
        if path := context.get("construction_events_path"):
            record_event(path, kind, run_id=context["run_id"], agent="DataProcessingAgent",
                         dataset_id=context["dataset_scope"], operation="process_dataset", **values)

    event("agent_started", status="running")
    try:
        outcome = run_preparation(context)
    except (ValueError, OSError) as exc:
        outcome = {"status": "blocked", "summary": str(exc), "updates": {}}
    event("agent_finished", status=outcome["status"], diagnostic=outcome["summary"])
    return outcome


def _coding_worker_identity() -> dict[str, str]:
    import os
    import subprocess
    from medbenchagent.runtime.dataset_runtime import WORKER_STARTUP_TIMEOUT_SECONDS
    settings = get_settings().data_processing
    venv = Path(settings.get("coding_worker_venv", ".venv-coding")).resolve()
    executable = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    requirements = Path(__file__).resolve().parents[3] / "config" / "coding-worker.requirements.txt"
    identity = {
        "venv": str(venv),
        "python": str(executable),
        "requirements_digest": digest_file(requirements) if requirements.is_file() else "",
    }
    if executable.is_file():
        try:
            probe = subprocess.run(
                [str(executable), "-c", "import importlib.metadata,sys; print(sys.version); print(importlib.metadata.version('openhands-sdk'))"],
                capture_output=True, text=True, timeout=WORKER_STARTUP_TIMEOUT_SECONDS, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            probe = None
        if probe is not None and probe.returncode == 0:
            lines = probe.stdout.splitlines()
            identity["python_version"] = lines[0] if lines else ""
            identity["openhands_version"] = lines[1] if len(lines) > 1 else ""
    return identity


def initialize_preparation(context, sources, dataset_id):
    from medbenchagent.data.dataset_catalog import validate_dataset_id
    from medbenchagent.data.execution import select_backend
    validate_dataset_id(dataset_id)
    sources = [Source.model_validate(source) for source in sources]
    bundle = (get_settings().paths.prepared_datasets_root / dataset_id).resolve()
    if any(bundle.is_relative_to(Path(source.root).resolve()) for source in sources):
        raise ValueError("Prepared output must be outside source roots")
    runtime = dict(context.get("execution_runtime") or select_backend(get_settings().data_processing))
    if runtime["backend"] == "local":
        runtime["coding_worker"] = _coding_worker_identity()
    return dict(context, sources=[source.model_dump() for source in sources], dataset_id=dataset_id,
                dataset_scope=dataset_id, agent_role="DataProcessingAgent", execution_runtime=runtime,
                bundle_root=str(bundle), source_scope=None)
