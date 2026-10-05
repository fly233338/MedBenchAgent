"""Deterministic Generation stage for a completed Task Plan."""
from medbenchagent.generation.items import generate_task_items, write_task_items
from medbenchagent.runtime.harness import result

def run_generation(context, *, leaf_path=None, output_path=None, allocation=None, page_size=512):
    from pathlib import Path
    import json
    from medbenchagent.data.index import DatasetIndex
    from medbenchagent.data.manifests import SamplingManifest, load_manifest
    from medbenchagent.planning.resolver import ResolverContext
    access = context["dataset_access_manifest"]
    sampling_manifest = None
    if access.get("sampling_manifest_path"):
        sampling_manifest = load_manifest(access["sampling_manifest_path"], SamplingManifest)
    leaves = json.loads(Path(leaf_path or context["leaf_tasks_path"]).read_text(encoding="utf-8"))
    items = generate_task_items(
        list(DatasetIndex(access).iter_cases()),
        leaves.values(),
        context.get("selected_dataset_id", "dataset"),
        allocation=allocation,
        seed=context.get("seed", 0),
        resolver_context=ResolverContext.from_access(access),
        sampling_manifest=sampling_manifest,
    )
    # Callers may pass a pathlib.Path (the Supervisor does this). Results are
    # persisted through the JSON operation log, so keep artifact references
    # JSON-native at this boundary.
    destination = str(output_path or (Path(context["operation_directory"]) / "benchmark_items.jsonl"))
    write_task_items(destination, items)
    task_counts = {}
    for item in items:
        task_counts[item["task_id"]] = task_counts.get(item["task_id"], 0) + 1
    return result(
        "success",
        f"Generated {len(items)} benchmark items",
        raw_output_path=destination,
        item_count=len(items),
        task_counts=task_counts,
        generation_target_counts=dict(allocation or {}),
    )

def compile_generation_agent(*, checkpointer=None):
    raise ValueError("Generation is executed by the Task Space pipeline")

