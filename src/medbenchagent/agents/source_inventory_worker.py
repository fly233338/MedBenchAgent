"""One-shot source inventory delegation retained as a compatibility adapter."""

from pathlib import Path

from medbenchagent.agents.coding_worker import CodingTask, OpenHandsCodingWorker
from medbenchagent.runtime.dependencies import publish

coding_worker = OpenHandsCodingWorker


def build_source_inventory(context):
    workspace = Path(context["operation_directory"]) / "workers" / "inventory"
    source = context["active_source"]
    task = CodingTask(
        task_id=f"{context.get('run_id', 'run')}:{source['id']}:inventory",
        kind="source_inventory", source_root=context["source_root"],
        source_manifest=context["source_manifest"], workspace=str(workspace),
        objective="Inventory the complete source population and write a source report.",
        required_outputs=["source_report.json", "image_population.jsonl", "inspect_source.py"],
        model=context.get("model", ""), base_url=context.get("base_url", ""),
        max_iterations=context.get("coding_worker_max_iterations", 40),
        timeout_seconds=context.get("coding_worker_timeout_seconds", 1800),
        conversation_id=context.get("source_inventory_conversation_id") or None,
        run_id=context.get("run_id", ""), dataset_id=context.get("dataset_scope", ""), source_id=source["id"],
    )
    result = coding_worker(context).run(task)
    if result.status != "complete":
        return {"status": result.status, "summary": result.diagnostic or result.question, "updates": {}}
    updates = {
        "source_report_path": result.outputs.get("source_report.json"),
        "source_population_path": result.outputs.get("image_population.jsonl"),
        "source_inspection_script_path": result.outputs.get("inspect_source.py"),
        "source_inventory_conversation_id": result.conversation_id,
        "source_population_reference": result.outputs.get("image_population.jsonl"),
        "source_report_reference": result.outputs.get("source_report.json"),
    }
    if context.get("source_scope"):
        report = updates["source_report_path"]
        population = updates["source_population_path"]
        from medbenchagent.data.source_report import validate_source_report
        checked = validate_source_report(context, report, population,
                                         Path(context["operation_directory"]) / "verified")
        updates["source_fingerprint"] = checked
        publish(context, "SourceFingerprint", {"report": report, "population": population}, ("Source",))
    return {"status": "success", "summary": "Source inventory completed", "updates": updates}


def compile_pipeline(*, checkpointer=None):
    from langgraph.graph import END, START, StateGraph

    def execute(state):
        context = dict(state["context"])
        outcome = build_source_inventory(context)
        context.update(outcome.get("updates", {}))
        return {"context": context, "result": outcome}

    graph = StateGraph(dict)
    graph.add_node("source_inventory", execute)
    graph.add_edge(START, "source_inventory")
    graph.add_edge("source_inventory", END)
    return graph.compile(checkpointer=checkpointer)
