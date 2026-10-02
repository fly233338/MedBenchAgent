"""v2 generation boundary."""
from medbenchagent.generation.v2 import generate_task_items, write_task_items
from medbenchagent.runtime.harness import result

def run_v2_generation(context, *, leaf_path=None, output_path=None, page_size=512):
    from pathlib import Path
    import json
    from medbenchagent.data.v5_index import V5DatasetQuery
    access = context["dataset_access_manifest"]
    leaves = json.loads(Path(leaf_path or context.get("leaf_tasks_path") or context["task_leaves_path"]).read_text(encoding="utf-8"))
    items = generate_task_items(list(V5DatasetQuery(access).iter_cases()), leaves.values(), context.get("selected_dataset_id", "dataset"))
    destination = output_path or str(Path(context["operation_directory"]) / "benchmark_items.v2.jsonl")
    write_task_items(destination, items)
    return result("success", f"Generated {len(items)} v2 benchmark items", raw_output_path=destination, item_count=len(items))

def compile_generation_agent(*, checkpointer=None):
    raise ValueError("Legacy generation Agent removed; use the v2 Task Space pipeline")

def run_generation(*args, **kwargs):
    return run_v2_generation(*args, **kwargs)
