"""Independent CLI: load a run, assemble selected implementations, and invoke LangGraph."""

import argparse
import json
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .assembly import assemble_compilation
from .contracts import WorkRecord, WorkRequest
from .artifacts import register_file, write_json_artifact
from medbenchagent.data.dataset import snapshot_dataset
from medbenchagent.data.task_data import SourceManifest
from .demo import DEFAULT_DEMO_GOAL, create_synthetic_inputs
from .graph import build_compilation_graph
from .interfaces import record_summary
from .run import create_run, resume_run


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compilation with five native Agent roles, internal OpenHands and deterministic Delivery.",
    )
    parser.add_argument("command", choices=("run", "resume"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--compilation-id", required=True)
    parser.add_argument("--supervisor", choices=("scripted", "llm"),
                        help="run default: scripted; resume uses saved mode")
    parser.add_argument("--coding", choices=("scripted", "openhands"),
                        help="run default: scripted; resume uses saved mode")
    parser.add_argument("--planner", choices=("scripted", "llm"), help="run default: scripted")
    parser.add_argument("--dataset-manifest", type=Path, help="Canonical Dataset manifest to freeze for a new run")
    parser.add_argument("--source-manifest", type=Path, help="Raw SourceManifest with dataset identity and source directories")
    parser.add_argument("--data-processing", choices=("disabled", "llm"))
    parser.add_argument("--generation", choices=("disabled", "llm"))
    parser.add_argument("--review", choices=("disabled", "llm"))
    parser.add_argument("--delivery", action="store_true", help="Enable deterministic portable delivery")
    parser.add_argument("--all-agents", action="store_true", help="Enable all five model roles, OpenHands and Delivery")
    parser.add_argument("--sample-count", type=int, help="Frozen generation target; otherwise use benchmark.default_sample_count")
    parser.add_argument("--config", type=Path, help="TOML configuration for enabled agents in a new run")
    parser.add_argument("--goal", help="User goal for a new run; default depends on the selected input/capabilities")
    parser.add_argument("--initial-evidence", action="store_true")
    parser.add_argument("--pause-for-input", action="store_true", help="Scripted Supervisor demonstration only")
    parser.add_argument("--answer")
    args = parser.parse_args(argv)
    try:
        directory = args.directory.resolve()
        if args.command == "run":
            if args.answer is not None:
                raise ValueError("--answer is only valid for resume")
            options = create_run(
                directory, compilation_id=args.compilation_id,
                goal=args.goal if args.goal is not None else ("Compile and deliver a traceable benchmark from the supplied data; report actual products and limitations."
                    if args.all_agents or args.generation == "llm" else
                    "Inspect the supplied raw sources and publish an evidence-grounded data Profile."
                    if args.source_manifest is not None else
                    "Discover and publish an evidence-grounded semantic Task Plan for the supplied Dataset."
                    if args.dataset_manifest is not None or args.planner == "llm" else DEFAULT_DEMO_GOAL),
                supervisor_mode=args.supervisor or ("llm" if args.all_agents else "scripted"),
                coding_mode=args.coding or ("openhands" if args.all_agents or args.data_processing == "llm" else "scripted"),
                planner_mode=args.planner or ("llm" if args.all_agents else "scripted"), dataset_manifest=args.dataset_manifest,
                source_manifest=args.source_manifest,
                data_processing_mode=args.data_processing or ("llm" if args.all_agents else "disabled"),
                generation_mode=args.generation or ("llm" if args.all_agents else "disabled"),
                review_mode=args.review or ("llm" if args.all_agents else "disabled"),
                delivery_enabled=args.delivery or args.all_agents, sample_count=args.sample_count,
                config_path=args.config, initial_evidence=args.initial_evidence,
                pause_for_input=args.pause_for_input,
            )
        else:
            if (args.initial_evidence or args.pause_for_input or args.supervisor is not None
                    or args.coding is not None or args.config is not None or args.goal is not None):
                raise ValueError("Resume uses the original mode, model, goal and options")
            if (args.planner is not None or args.dataset_manifest is not None or args.source_manifest is not None
                    or args.data_processing is not None or args.generation is not None or args.review is not None
                    or args.delivery or args.all_agents or args.sample_count is not None):
                raise ValueError("Resume uses the original Planner and Dataset")
            options = resume_run(directory, args.compilation_id)
        identifier = options.compilation_id
        config = {"configurable": {"thread_id": identifier}, "recursion_limit": 60}
        with assemble_compilation(options, directory) as (agents, context), \
                SqliteSaver.from_conn_string(str(directory / "checkpoints.sqlite")) as saver:
            graph = build_compilation_graph(agents, checkpointer=saver)
            if args.command == "run":
                # Synthetic input creation belongs to this demo entrypoint, never Assembly.
                if options.source_manifest is not None:
                    source_path = Path(options.source_manifest)
                    source = SourceManifest.model_validate_json(source_path.read_bytes())
                    for item in source.sources:
                        item.root = str((source_path.parent / item.root).resolve())
                    request = WorkRequest(compilation_id=identifier, request_id=f"{identifier}-input",
                                          agent_id="input", objective="Bind raw sources")
                    inputs = [write_json_artifact(directory, request, "sources.json", kind="source_manifest", value=source.model_dump())]
                elif options.dataset_manifest is not None:
                    request = WorkRequest(compilation_id=identifier, request_id=f"{identifier}-input",
                                          agent_id="input", objective="Freeze the input Dataset")
                    path = snapshot_dataset(Path(options.dataset_manifest),
                                            directory / "artifacts" / request.request_id / "dataset")
                    inputs = [register_file(directory, request, path.relative_to(directory).as_posix(), kind="dataset")]
                else:
                    inputs = create_synthetic_inputs(directory, identifier, initial_evidence=options.initial_evidence)
                if options.benchmark_request is not None:
                    request = WorkRequest(compilation_id=identifier, request_id=f"{identifier}-target",
                                          agent_id="input", objective="Freeze the benchmark target")
                    inputs.append(write_json_artifact(directory, request, "target.json", kind="benchmark_request", value=options.benchmark_request))
                invocation = {"compilation_id": identifier, "goal": options.goal,
                              "artifacts": [ref.model_dump(mode="json") for ref in inputs]}
            else:
                snapshot = graph.get_state(config)
                if not snapshot.values:
                    raise ValueError("No state exists for this Compilation")
                pending = any(task.interrupts for task in snapshot.tasks)
                if pending and args.answer is None:
                    raise ValueError("This run is waiting for --answer")
                if not pending and args.answer is not None:
                    raise ValueError("There is no pending question to answer")
                invocation = Command(resume=args.answer) if pending else None
            if args.command == "run" or graph.get_state(config).next:
                graph.invoke(invocation, config, context=context, durability="sync")
            snapshot = graph.get_state(config)
            state = snapshot.values
            print(json.dumps({
                "compilation_id": identifier, "supervisor_mode": options.supervisor_mode,
                "coding_mode": options.coding_mode, "planner_mode": options.planner_mode,
                "data_processing_mode": options.data_processing_mode, "generation_mode": options.generation_mode,
                "review_mode": options.review_mode, "delivery_enabled": options.delivery_enabled,
                "supervisor_model": options.supervisor_settings["llm"]["model"] if options.supervisor_settings else None,
                "coding_model": options.coding_settings["llm"]["model"] if options.coding_settings else None,
                "planner_model": options.planner_settings["llm"]["model"] if options.planner_settings else None,
                "status": state.get("status"), "summary": state.get("summary", ""),
                "history": [record_summary(WorkRecord.model_validate(row)) for row in state.get("history", [])],
                "artifacts": state.get("artifacts", []), "workstreams": state.get("workstreams", []),
                "questions": [item.value for task in snapshot.tasks for item in task.interrupts],
            }, ensure_ascii=False))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
