"""generation boundary."""
from medbenchagent.generation.items import generate_task_items, write_task_items
from medbenchagent.runtime.harness import result

def run_generation(context, *, leaf_path=None, output_path=None, page_size=512):
    from pathlib import Path
    import json
    from medbenchagent.data.index import DatasetIndex
    access = context["dataset_access_manifest"]
    leaves = json.loads(Path(leaf_path or context.get("leaf_tasks_path") or context["task_leaves_path"]).read_text(encoding="utf-8"))
    items = generate_task_items(
        list(DatasetIndex(access).iter_cases()),
        leaves.values(),
        context.get("selected_dataset_id", "dataset"),
        seed=context.get("seed", 0),
    )
    destination = output_path or str(Path(context["operation_directory"]) / "benchmark_items.jsonl")
    write_task_items(destination, items)
    return result("success", f"Generated {len(items)} benchmark items", raw_output_path=destination, item_count=len(items))

def compile_generation_agent(*, checkpointer=None):
    raise ValueError("Generation is executed by the Task Space pipeline")

