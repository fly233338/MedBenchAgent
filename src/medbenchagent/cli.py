"""One CLI for the canonical Compilation core and its experiment UI."""
import argparse
import json
from pathlib import Path
from uuid import uuid4

from medbenchagent.compilation.contracts import GoalSpec
from medbenchagent.compilation.run import create_run
from medbenchagent.runtime.graph_runner import execute_compilation, inspect_compilation
from medbenchagent.runtime import sessions


def _parser():
    parser = argparse.ArgumentParser(prog="medd2b")
    sub = parser.add_subparsers(dest="command")
    create = sub.add_parser("create", help="Create an experiment and editable config")
    create.add_argument("experiment")
    create.add_argument("--config", type=Path)
    run = sub.add_parser("run", help="Start a Compilation from explicit real inputs")
    run.add_argument("experiment", nargs="?")
    run.add_argument("--directory", type=Path)
    run.add_argument("--compilation-id")
    run.add_argument("--config", type=Path)
    run.add_argument("--prepare-only", action="store_true", help="Freeze a run for Studio without executing agents")
    run.add_argument("--goal", default="Compile and deliver a traceable benchmark from the supplied data.")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-manifest", type=Path)
    source.add_argument("--dataset-manifest", type=Path)
    source.add_argument("--dataset", help="Registered dataset ID")
    for command in (run,):
        _goal_arguments(command)
    batch = sub.add_parser("batch", help="Run registered datasets independently in one experiment")
    batch.add_argument("experiment")
    batch.add_argument("dataset_ids", nargs="+")
    batch.add_argument("--goal", default="Compile and deliver a traceable benchmark.")
    _goal_arguments(batch)
    resume = sub.add_parser("resume")
    resume.add_argument("--directory", type=Path, required=True)
    resume.add_argument("--answer")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--directory", type=Path, required=True)
    register = sub.add_parser("register")
    register.add_argument("--dataset-id", required=True)
    register.add_argument("--manifest", type=Path, required=True)
    register.add_argument("--kind", choices=("source", "dataset"), default="dataset")
    export = sub.add_parser("export")
    export.add_argument("source", type=Path)
    export.add_argument("destination", type=Path)
    demo = sub.add_parser("demo", help="Create a tiny explicit source fixture; does not call models")
    demo.add_argument("--directory", type=Path, required=True)
    sub.add_parser("tui")
    return parser


def _goal_arguments(parser):
    parser.add_argument("--goal-kind", choices=("profile", "plan", "benchmark"), default="benchmark")
    parser.add_argument("--plan-level", choices=("scenarios", "executable_leaves"))
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--no-limited-delivery", action="store_true")


def goal_options(args):
    return {"goal_spec": GoalSpec(kind=args.goal_kind,
        plan_level=(args.plan_level or "scenarios") if args.goal_kind == "plan" else args.plan_level,
        allow_limited_delivery=not args.no_limited_delivery), "sample_count": args.sample_count}


def _consume(events):
    status = 0
    for event in events:
        print(json.dumps({"run_id": event.run_session_id, "event": event.kind, **event.payload}, ensure_ascii=False))
        if event.kind == "finished" and event.payload.get("goal_result") in {"blocked", "failed"}:
            status = 1
    return status


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command in {None, "tui"}:
            from medbenchagent.tui import MBAApp
            MBAApp().run()
            return
        if args.command == "create":
            print(json.dumps(sessions.create_experiment(args.experiment, template=args.config)))
        elif args.command == "run":
            inputs = {key: getattr(args, key) for key in ("source_manifest", "dataset_manifest") if getattr(args, key)}
            if args.dataset:
                from medbenchagent.data.dataset_catalog import registered_input
                inputs = registered_input(args.dataset)
            if args.experiment:
                if args.directory or args.compilation_id or args.config:
                    raise ValueError("Experiment runs use their experiment directory and config")
                run = sessions.add_run(args.experiment, args.goal, **inputs, **goal_options(args))
                directory = Path(run["directory"])
            else:
                identifier = args.compilation_id or "run-" + uuid4().hex[:12]
                directory = args.directory or Path(".runtime") / identifier
                create_run(directory, compilation_id=identifier, goal=args.goal, config_path=args.config,
                           **inputs, **goal_options(args))
            if args.prepare_only:
                print(directory.resolve())
                return
            raise SystemExit(_consume(execute_compilation(directory)))
        elif args.command == "batch":
            from medbenchagent.data.dataset_catalog import registered_input
            if len(set(args.dataset_ids)) != len(args.dataset_ids):
                raise ValueError("Batch dataset IDs must be distinct")
            status = 0
            for identifier in args.dataset_ids:
                run = sessions.add_run(args.experiment, args.goal, **registered_input(identifier), **goal_options(args))
                status = max(status, _consume(execute_compilation(run["directory"])))
            raise SystemExit(status)
        elif args.command == "resume":
            raise SystemExit(_consume(execute_compilation(args.directory, resume=True, answer=args.answer)))
        elif args.command == "inspect":
            print(json.dumps(inspect_compilation(args.directory), ensure_ascii=False, indent=2))
        elif args.command == "register":
            from medbenchagent.data.dataset_catalog import register_dataset
            print(json.dumps(register_dataset(args.dataset_id, args.manifest, kind=args.kind)))
        elif args.command == "export":
            from medbenchagent.generation.publication import export_benchmark
            print(export_benchmark(args.source, args.destination))
        elif args.command == "demo":
            from medbenchagent.compilation.demo import create_demo_source
            print(create_demo_source(args.directory))
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"{exc}\n")
