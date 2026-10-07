"""Deterministic Generation stage for a completed Task Plan."""
from medbenchagent.generation.items import generate_task_items, write_task_items
from medbenchagent.runtime.harness import result

def run_generation(context, *, output_path=None, page_size=512):
    from pathlib import Path
    from medbenchagent.data.index import DatasetIndex
    from medbenchagent.data.manifests import SamplingManifest, load_manifest
    from medbenchagent.planning.resolver import ResolverContext
    from medbenchagent.planning.bir import BIRStore
    store = BIRStore(context["bir_path"])
    dataset = context["dataset_scope"]
    receipt = store.load("dataset_bindings.json")["datasets"][dataset]["access"]
    access = receipt.get("source_access", receipt)
    allocation = store.load("allocation.json")["counts"][dataset]
    sampling_manifest = None
    if access.get("sampling_manifest_path"):
        sampling_manifest = load_manifest(access["sampling_manifest_path"], SamplingManifest)
    plan = store.load(f"datasets/{context['dataset_scope']}/planning/task_plan.json")
    leaves = plan["leaf_tasks"]
    resolver_context = ResolverContext.from_access(access)
    resolver_context.scenarios = plan["scenarios"]
    index = DatasetIndex(access)
    def pages():
        offset = 0
        while True:
            page = index.sample(offset=offset, limit=page_size)
            yield from page["values"]
            if page["complete"]:
                break
            offset = page["next_offset"]

    items = generate_task_items(
        pages(),
        leaves.values(),
        context.get("selected_dataset_id", "dataset"),
        allocation=allocation,
        seed=context.get("seed", 0),
        resolver_context=resolver_context,
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
