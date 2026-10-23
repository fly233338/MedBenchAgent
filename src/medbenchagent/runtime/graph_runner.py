"""Shared Compilation execution and projections for CLI, TUI and application services."""

from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock, Timeout
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from medbenchagent.compilation.assembly import assemble_compilation
from medbenchagent.compilation.artifacts import register_file, write_json_artifact, read_json_artifact
from medbenchagent.compilation.contracts import ArtifactRef, WorkRecord, WorkRequest
from medbenchagent.compilation.graph import build_compilation_graph
from medbenchagent.compilation.interfaces import record_summary
from medbenchagent.compilation.run import RunOptions, resume_run
from medbenchagent.data.dataset import snapshot_dataset
from medbenchagent.data.task_data import SourceManifest


@dataclass(frozen=True)
class RunEvent:
    run_session_id: str
    kind: str
    payload: dict


def initial_state(directory, options):
    identifier = options.compilation_id
    request = WorkRequest(compilation_id=identifier, request_id=f"{identifier}-input",
                          agent_id="input", operation="bind", objective="Bind source inputs")
    if options.source_manifest:
        path = Path(options.source_manifest)
        source = SourceManifest.model_validate_json(path.read_bytes())
        for item in source.sources:
            item.root = str((path.parent / item.root).resolve())
        refs = [write_json_artifact(directory, request, "sources.json", kind="source_manifest", value=source.model_dump())]
    else:
        path = snapshot_dataset(Path(options.dataset_manifest), directory / "artifacts" / request.request_id / "dataset")
        refs = [register_file(directory, request, path.relative_to(directory).as_posix(), kind="dataset")]
    if options.benchmark_request:
        refs.append(write_json_artifact(directory, request, "target.json", kind="benchmark_request", value=options.benchmark_request))
    return {"compilation_id": identifier, "goal": options.goal, "artifacts": [r.model_dump() for r in refs]}


def _enabled_agents(options):
    roles = ["supervisor"]
    if "data_processing" in options.professional_settings:
        roles.extend(["data_processing", "coding"])
    if options.planner_settings is not None:
        roles.append("task_planner")
    roles.extend(role for role in ("generation", "review") if role in options.professional_settings)
    if options.goal_spec.kind == "benchmark":
        roles.append("delivery")
    return roles


def _view(values, directory, options, interrupts=(), *, active=False, errors=()):
    status = values.get("status", "ready")
    if interrupts:
        status = "waiting"
    elif status == "stopped":
        pass
    elif active:
        status = "running"
    elif values and status != "paused":
        status = "interrupted"
    counts, count_kind = None, None
    for kind in ("delivery_manifest", "generation_candidate"):
        product = next((row for row in reversed(values.get("artifacts", [])) if row["kind"] == kind), None)
        if product:
            try:
                counts = read_json_artifact(directory, ArtifactRef.model_validate(product)).get("counts")
                count_kind = "交付" if kind == "delivery_manifest" else "候选"
            except (OSError, ValueError):
                pass
            break
    return {"compilation_id": options.compilation_id, "run_directory": str(directory), "goal": options.goal,
            "target_count": (options.benchmark_request or {}).get("sample_count"), "counts": counts, "count_kind": count_kind,
            "error": str(errors[0]) if errors else None,
            "status": status, "goal_result": values.get("goal_result"), "summary": values.get("summary", ""),
            "feedback": values.get("feedback", []), "enabled_agents": _enabled_agents(options),
            "history": [record_summary(WorkRecord.model_validate(row)) for row in values.get("history", [])],
            "artifacts": values.get("artifacts", []), "answers": values.get("answers", []),
            "questions": [item.value for item in interrupts], "question_id": interrupts[0].id if interrupts else None}


def state_view(snapshot, directory, options, *, active=False):
    return _view(snapshot.values, directory, options,
                 [item for task in snapshot.tasks for item in task.interrupts], active=active,
                 errors=[task.error for task in snapshot.tasks if task.error])


def inspect_compilation(directory):
    directory = Path(directory).resolve()
    options = RunOptions.model_validate_json((directory / "run.json").read_bytes())
    active = False
    try:
        with FileLock(directory / "execution.lock", timeout=0):
            pass
    except Timeout:
        active = True
    if not (directory / "checkpoints.sqlite").is_file():
        return _view({}, directory, options, active=active)
    with SqliteSaver.from_conn_string(str(directory / "checkpoints.sqlite")) as saver:
        checkpoint = saver.get_tuple({"configurable": {"thread_id": options.compilation_id}})
    if checkpoint is None:
        return _view({}, directory, options, active=active)
    interrupts = [item for _, channel, value in checkpoint.pending_writes or []
                  if channel == "__interrupt__" for item in value]
    errors = [value for _, channel, value in checkpoint.pending_writes or [] if channel == "__error__"]
    return _view(checkpoint.checkpoint["channel_values"], directory, options, interrupts, active=active, errors=errors)


def execute_compilation(directory, *, resume=False, answer=None, question_id=None, stop_requested=None):
    directory = Path(directory).resolve()
    options = RunOptions.model_validate_json((directory / "run.json").read_bytes())
    identifier = options.compilation_id
    config = {"configurable": {"thread_id": identifier}, "recursion_limit": 300}
    with FileLock(directory / "execution.lock", timeout=0):
        if resume:
            options = resume_run(directory, identifier)
        with assemble_compilation(options, directory) as (agents, context), SqliteSaver.from_conn_string(str(directory / "checkpoints.sqlite")) as saver:
            graph = build_compilation_graph(agents, checkpointer=saver)
            snapshot = graph.get_state(config)
            questions = [item for task in snapshot.tasks for item in task.interrupts]
            if resume and snapshot.values:
                if snapshot.values.get("status") == "stopped" or not snapshot.next:
                    raise ValueError("This run has ended; create a new run")
                if bool(questions) != (answer is not None):
                    raise ValueError("Supply an answer only for a waiting user question")
                if answer is not None and (not answer.strip() or question_id is not None and questions[0].id != question_id):
                    raise ValueError("The answer is empty or belongs to a different question")
                invocation = Command(resume=answer) if questions else None
                if not questions:
                    graph.update_state(config, {"status": "running"})
            else:
                if answer is not None:
                    raise ValueError("There is no saved question to answer")
                if snapshot.values:
                    raise ValueError("This run has started; use resume")
                invocation = initial_state(directory, options)
            yield RunEvent(identifier, "initialized", state_view(graph.get_state(config), directory, options, active=True))
            stream = graph.stream(invocation, config, context=context, durability="sync", stream_mode="updates")
            try:
                for update in stream:
                    last_node = None
                    for node, value in update.items():
                        if node == "__interrupt__":
                            continue
                        last_node = node
                        if node == "ask_user" and answer is not None:
                            yield RunEvent(identifier, "answer_accepted", {"question_id": questions[0].id, "answer": answer,
                                "after_work": len(snapshot.values.get("history", []))})
                            answer = None
                        if node == "collect" and isinstance(value, dict) and value.get("history"):
                            work = record_summary(WorkRecord.model_validate(value["history"][-1]))
                            yield RunEvent(identifier, "progress", {"work": work, "text": work["summary"],
                                "view": state_view(graph.get_state(config), directory, options, active=True)})
                    saved = graph.get_state(config)
                    if stop_requested and stop_requested() and last_node and saved.next and saved.values.get("status") != "waiting":
                        stream.close()
                        graph.update_state(config, {"status": "paused"}, as_node=last_node)
                        yield RunEvent(identifier, "paused", state_view(graph.get_state(config), directory, options))
                        return
            finally:
                stream.close()
            view = state_view(graph.get_state(config), directory, options)
            yield RunEvent(identifier, "needs_input" if view["questions"] else "finished", view)
