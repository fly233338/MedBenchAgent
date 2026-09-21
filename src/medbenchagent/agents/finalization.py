"""Global tool-using FinalizationAgent, from completed Generations to delivery."""

import json
from pathlib import Path

from medbenchagent.audit import assembly
from medbenchagent.audit.hard import inspect_generation, read_json, read_rows
from medbenchagent.data.dataset_catalog import resolve_path
from medbenchagent.data.sources import digest_file
from medbenchagent.runtime.dependencies import publish, require, read_index, reference_digests
from medbenchagent.runtime.harness import compile_agent, result
from medbenchagent.runtime.construction import FINALIZATION_AGENT
from medbenchagent.runtime.run_export import PACKAGE_FILES, publish_package
from medbenchagent.runtime.tool_contracts import NoArguments, DatasetId, PageArguments


class DatasetArguments(NoArguments):
    dataset_id: DatasetId


class IssueArguments(PageArguments):
    dataset_id: DatasetId | None = None


def _current(context, kind):
    try:
        require(context, kind)
        return True
    except ValueError:
        return False


def _verify(context, *kinds):
    for record in require(context, *kinds).values():
        if reference_digests(record["references"]) != record["content_digests"]:
            raise ValueError(f"Artifact files changed: {record['kind']}")


def _active(context):
    _, allocation = assembly.plans_and_allocation(context)
    return [d for d, counts in allocation["counts"].items() if sum(counts.values())]


def _delivery_result(context, directory):
    root = Path(directory)
    report, manifest = read_json(root / "audit_report.json"), read_json(root / "benchmark_manifest.json")
    return {"delivery_path": str(root), "manifest_path": str(root / "benchmark_manifest.json"),
            "benchmark_id": manifest["benchmark_id"], "raw_source_paths": manifest["raw_source_paths"],
            "audit_artifacts": {"audited": str(root / "benchmark.jsonl"), "report": str(root / "audit_report.json"),
                                "quarantine": str(root / "quarantine.jsonl")},
            "item_count": report["counts"]["final"], "task_counts": {k: v["final"] for k, v in report["tasks"].items()},
            "generation_shortfalls": {k: v["generation_shortfall"] for k, v in report["tasks"].items()},
            "planning_shortfall": report["counts"]["planning_shortfall"], "total_shortfall": report["counts"]["total_shortfall"],
            "delivery_counts": report["counts"]}


def restore_delivery(context):
    """Recover publication that finished before its checkpoint was recorded."""
    root = Path(context["run_directory"]) / "delivery"
    marker = read_json(root / "complete.json")
    if marker["input_versions"] != assembly.input_versions(context) or marker["files"] != list(PACKAGE_FILES):
        raise ValueError("Published delivery belongs to different inputs; start a new run")
    record = read_index(context)["versions"][marker["validation_version"]]
    if record["kind"] != "FinalValidation" or not record["references"].get("valid") or record["references"]["input_versions"] != marker["input_versions"]:
        raise ValueError("Delivery has no matching final validation")
    expected = {Path(path).name: value for path, value in record["content_digests"].items()}
    if set(expected) != set(PACKAGE_FILES) or any(digest_file(root / name) != expected[name] for name in PACKAGE_FILES):
        raise ValueError("Published delivery files changed")
    references = {name.replace(".", "_") + "_path": str(root / name) for name in (*PACKAGE_FILES, "complete.json")}
    versions = publish(context, "Delivery", references, assembly.input_dependencies(context))
    return {**_delivery_result(context, root), **versions}


def initialize(context):
    _verify(context, *assembly.input_dependencies(context))
    if (Path(context["run_directory"]) / "delivery").exists():
        context.update(restore_delivery(context))
    return context


def inspect_inputs(context):
    """Read all generated inputs, quotas and pending dataset checks."""
    return result("success", json.dumps(describe(context), ensure_ascii=False))


def inspect_dataset(context, *, dataset_id):
    """Validate generated samples against Plan, templates, annotation and images; isolate invalid items."""
    if dataset_id not in _active(context):
        raise ValueError("Only positive-allocation datasets can be inspected")
    if _current(context, f"dataset/{dataset_id}/GenerationCheck"):
        raise ValueError("Dataset check is already current")
    dependencies = (f"dataset/{dataset_id}/Generation", f"dataset/{dataset_id}/Plan", f"dataset/{dataset_id}/Templates",
                    f"dataset/{dataset_id}/Dataset", "Allocation", "FinalizationConfig")
    _verify(context, *dependencies)
    entry = context["dataset_assets"][dataset_id]["entry"]
    checked = inspect_generation(dataset_id=dataset_id, raw_path=context["dataset_generations"][dataset_id]["raw_output_path"],
        plan_path=context["dataset_task_plans"][dataset_id], templates_path=context["dataset_templates"][dataset_id],
        dataset_access=context["dataset_assets"][dataset_id]["access"], manifest_path=resolve_path(entry["image_manifest_path"]),
        output_directory=Path(context["operation_directory"]) / dataset_id)
    summary = read_json(checked["report_path"])
    return result("success", json.dumps(summary, ensure_ascii=False),
                  dataset_checks={**context.get("dataset_checks", {}), dataset_id: checked},
                  **publish(context, f"dataset/{dataset_id}/GenerationCheck", checked, dependencies))


def read_issues(context, *, dataset_id=None, offset=0, limit=20, query=""):
    """Page original rejected records and reasons, or assembled package quarantine."""
    if dataset_id is not None:
        require(context, f"dataset/{dataset_id}/GenerationCheck")
        path = context["dataset_checks"][dataset_id]["rejected_path"]
    else:
        require(context, "CandidateAssembly")
        path = Path(context["assembly_directory"]) / "quarantine.jsonl"
    rows = (row for row in read_rows(path) if query.casefold() in json.dumps(row, ensure_ascii=False).casefold())
    page, total = [], 0
    for row in rows:
        if offset <= total < offset + limit:
            page.append(row)
        total += 1
    return result("success", json.dumps({"total": total, "issues": page}, ensure_ascii=False))


def assemble_candidates(context):
    """Namespace, reconcile identities, remove duplicate/conflicting/excess items and account for every loss."""
    dependencies = (*assembly.input_dependencies(context), *(f"dataset/{d}/GenerationCheck" for d in _active(context)))
    _verify(context, *dependencies)
    root = assembly.assemble(context, Path(context["operation_directory"]) / "assembly")
    refs = {"candidate_path": str(root / "candidates.jsonl"), "report_path": str(root / "audit_report.json"),
            "manifest_path": str(root / "benchmark_manifest.json"), "quarantine_path": str(root / "quarantine.jsonl")}
    return result("success", json.dumps(read_json(root / "audit_report.json")["counts"]), assembly_directory=str(root),
                  **publish(context, "CandidateAssembly", refs, dependencies))


def balance_options(context):
    """Balance final task option positions without changing answers or other sample content."""
    _verify(context, "CandidateAssembly")
    root = Path(context["assembly_directory"])
    assembly.balance(root, context.get("seed", 0))
    refs = {name.replace(".", "_") + "_path": str(root / "package" / name) for name in PACKAGE_FILES}
    return result("success", "Final task option positions balanced", **publish(context, "BalancedCandidates", refs, ("CandidateAssembly",)))


def validate_delivery(context):
    """Check final samples, counts, references, answer semantics, distributions and package consistency."""
    _verify(context, *assembly.input_dependencies(context), "CandidateAssembly", "BalancedCandidates")
    root = Path(context["assembly_directory"])
    refs = {name.replace(".", "_") + "_path": str(root / "package" / name) for name in PACKAGE_FILES}
    try:
        summary = assembly.validate_package(context, root)
    except ValueError as exc:
        return result("blocked", str(exc), **publish(context, "FinalValidation", {**refs, "valid": False}, ("BalancedCandidates",)))
    return result("success", json.dumps(summary), **publish(context, "FinalValidation",
                  {**refs, "valid": True, "input_versions": assembly.input_versions(context)}, ("BalancedCandidates",)))


def publish_delivery(context):
    """Publish the validated package atomically. Shortfall alone never prevents delivery."""
    destination = Path(context["run_directory"]) / "delivery"
    if not destination.exists():
        _verify(context, *assembly.input_dependencies(context), "FinalValidation")
        validation = require(context, "FinalValidation")["FinalValidation"]
        if not validation["references"]["valid"]:
            raise ValueError("Final validation did not pass")
        completion = {"schema_version": "benchmark-delivery.v1", "input_versions": assembly.input_versions(context),
                      "validation_version": validation["id"], "files": list(PACKAGE_FILES)}
        publish_package(Path(context["assembly_directory"]) / "package", destination, completion)
    return result("success", "Benchmark package published", **restore_delivery(context))


def finish(context):
    require(context, "Delivery")
    updates = restore_delivery(context)
    counts = updates["delivery_counts"]
    return result("complete", f"Delivered {counts['final']} of {counts['requested']} requested samples; shortfall {counts['total_shortfall']}", **updates)


def available_tools(context):
    if _current(context, "Delivery"):
        return {"finish"}
    tools = {"inspect_inputs"}
    active = _active(context)
    if any(not _current(context, f"dataset/{d}/GenerationCheck") for d in active):
        tools.add("inspect_dataset")
    else:
        tools.add("assemble_candidates")
    if context.get("dataset_checks") or _current(context, "CandidateAssembly"):
        tools.add("read_issues")
    if _current(context, "CandidateAssembly") and not _current(context, "BalancedCandidates"):
        tools.add("balance_options")
    if _current(context, "BalancedCandidates") and not _current(context, "FinalValidation"):
        tools.add("validate_delivery")
    if _current(context, "FinalValidation") and require(context, "FinalValidation")["FinalValidation"]["references"]["valid"]:
        tools.add("publish_delivery")
    return tools


def describe(context):
    return {"datasets": _active(context), "pending_checks": [d for d in _active(context) if not _current(context, f"dataset/{d}/GenerationCheck")],
            "checks": context.get("dataset_checks", {}), "assembly_directory": context.get("assembly_directory"),
            "delivery_counts": context.get("delivery_counts"), "available_actions": sorted(available_tools(context))}


def compile_finalization_agent():
    functions = (inspect_inputs, inspect_dataset, read_issues, assemble_candidates, balance_options, validate_delivery, publish_delivery)
    return compile_agent(FINALIZATION_AGENT,
        "Finalize the complete benchmark across all generated datasets. Inspect each pending dataset; investigate anomalies "
        "with read_issues as needed. Assemble checked candidates, balance final task options, validate and publish the package. "
        "Choose inspection order and explain observed problems; never override hard checks or invent repairs, labels or quotas. "
        "Shortfall is reported, not a reason to block, ask confirmation, regenerate, or change the target. Zero final items cannot be delivered. "
        "Missing/corrupt required inputs must be fixed in a new run. Do not repeat already completed operations. Finish only after publication.",
        {f.__name__: f for f in functions}, describe, finish, available_tools=available_tools, initialize_context=initialize,
        args_schemas={"inspect_dataset": DatasetArguments, "read_issues": IssueArguments})
