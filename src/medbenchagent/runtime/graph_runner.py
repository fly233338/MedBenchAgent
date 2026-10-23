"""Shared Compilation execution and projections for CLI, TUI and application services."""

from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from medbenchagent.compilation.assembly import assemble_compilation
from medbenchagent.compilation.artifacts import register_file, write_json_artifact
from medbenchagent.compilation.contracts import WorkRecord, WorkRequest
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


def state_view(snapshot, directory):
    state = snapshot.values
    return {"compilation_id": state.get("compilation_id"), "run_directory": str(directory),
            "status": state.get("status", "ready"), "goal_result": state.get("goal_result"),
            "summary": state.get("summary", ""), "feedback": state.get("feedback", []),
            "history": [record_summary(WorkRecord.model_validate(r)) for r in state.get("history", [])],
            "artifacts": state.get("artifacts", []),
            "questions": [i.value for task in snapshot.tasks for i in task.interrupts]}


def inspect_compilation(directory):
    directory = Path(directory).resolve()
    options = RunOptions.model_validate_json((directory / "run.json").read_bytes())
    # Reading a checkpoint does not construct models or require credentials.
    with SqliteSaver.from_conn_string(str(directory / "checkpoints.sqlite")) as saver:
        checkpoint = saver.get_tuple({"configurable": {"thread_id": options.compilation_id}})
    if checkpoint is None:
        return {"compilation_id": options.compilation_id, "status": "ready", "artifacts": [], "history": [], "questions": []}
    values = checkpoint.checkpoint["channel_values"]
    return {"compilation_id": options.compilation_id, "run_directory": str(directory),
            "status": values.get("status", "ready"), "goal_result": values.get("goal_result"),
            "summary": values.get("summary", ""), "question": values.get("question"),
            "artifacts": values.get("artifacts", []),
            "history": [record_summary(WorkRecord.model_validate(r)) for r in values.get("history", [])]}


def execute_compilation(directory, *, resume=False, answer=None, stop_requested=None):
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
            if resume and snapshot.values:
                waiting = any(t.interrupts for t in snapshot.tasks)
                if waiting != (answer is not None):
                    raise ValueError("Supply an answer only for a waiting user question")
                invocation = Command(resume=answer) if waiting else None
            else:
                if answer is not None:
                    raise ValueError("There is no saved question to answer")
                if snapshot.values:
                    raise ValueError("This run has started; use resume")
                invocation = initial_state(directory, options)
            yield RunEvent(identifier, "initialized", {"run_directory": str(directory)})
            if not snapshot.values or snapshot.next:
                for update in graph.stream(invocation, config, context=context, durability="sync", stream_mode="updates"):
                    for node, value in update.items():
                        if node == "__interrupt__":
                            continue
                        text = node
                        if isinstance(value, dict) and value.get("request"):
                            request = value["request"]
                            text = f"{request['agent_id']}/{request['operation']}: {request['objective']}"
                        if isinstance(value, dict) and value.get("result"):
                            text = value["result"]["summary"]
                        yield RunEvent(identifier, "progress", {"text": text})
                    if stop_requested and stop_requested():
                        yield RunEvent(identifier, "interrupted", {"summary": "Stopped at a saved node boundary"})
                        return
            view = state_view(graph.get_state(config), directory)
            yield RunEvent(identifier, "needs_input" if view["questions"] else "finished", view)
