"""Global Concept -> Dataset -> Leaf allocation, with deterministic capacity limits."""

import json
from pathlib import Path
from medbenchagent.data.contracts import Contract
from medbenchagent.planning.tree import Weight
from medbenchagent.planning.sample_counts import allocate_leaf_counts
from medbenchagent.runtime.harness import compile_agent, result
from medbenchagent.runtime.dependencies import publish, require
from medbenchagent.runtime.run_artifacts import atomic_write_json


class AllocationIntent(Contract):
    concepts: dict[str, Weight]
    datasets: dict[str, dict[str, Weight]] | None = None
    leaves: dict[str, dict[str, dict[str, Weight]]] | None = None


def allocation_groups(plans):
    groups = {}
    for dataset_id, plan in sorted(plans.items()):
        for leaf_id, leaf in sorted(plan["tasks"].items()):
            concept = plan["concepts"][leaf["concept_id"]]
            key = json.dumps(["core", leaf["concept_id"]] if concept["origin"] == "core"
                             else ["extension", dataset_id, leaf["concept_id"]], separators=(",", ":"))
            group = groups.setdefault(key, {"description": concept["description"], "datasets": {}})
            group["datasets"].setdefault(dataset_id, {})[leaf_id] = leaf
    return groups


def _weights(raw, keys):
    if raw is None:
        return {k: {"weight": 1.0, "reason": "Equal coverage; no explicit preference"} for k in keys}
    if set(raw) != set(keys):
        raise ValueError("Weights must cover exactly all members of this allocation level")
    return raw


def allocate(plans, intent, total):
    """Allocate hierarchically; capacities are independent Leaf capacities, not joint matching."""
    intent = AllocationIntent.model_validate(intent).model_dump()
    groups = allocation_groups(plans)
    if not groups:
        raise ValueError("No executable tasks in any dataset")
    cw = _weights(intent["concepts"], groups)
    for level in ("datasets", "leaves"):
        if intent[level] is not None and set(intent[level]) != set(groups):
            raise ValueError(f"{level} weights must cover exactly all Concepts")
    dw, lw, capacities = {}, {}, {}
    for concept, group in groups.items():
        datasets = group["datasets"]
        dw[concept] = _weights(None if intent["datasets"] is None else intent["datasets"][concept], datasets)
        supplied = None if intent["leaves"] is None else intent["leaves"][concept]
        if supplied is not None and set(supplied) != set(datasets):
            raise ValueError("Leaf weight datasets do not match the Concept")
        lw[concept], capacities[concept] = {}, {}
        for dataset, leaves in datasets.items():
            lw[concept][dataset] = _weights(None if supplied is None else supplied[dataset], leaves)
            for leaf in leaves.values():
                if type(leaf.get("eligible_count")) is not int or leaf["eligible_count"] < 0:
                    raise ValueError("Every Leaf requires exact nonnegative eligible_count")
            capacities[concept][dataset] = sum(leaf["eligible_count"] for key, leaf in leaves.items()
                                               if lw[concept][dataset][key]["weight"] > 0)
    cc = {c: sum(n for d, n in values.items() if dw[c][d]["weight"] > 0) for c, values in capacities.items()}
    first = allocate_leaf_counts({k: v["weight"] for k, v in cw.items()}, cc, total)
    counts = {d: dict.fromkeys(plan["tasks"], 0) for d, plan in plans.items()}
    dataset_counts = {}
    for concept, count in first["counts"].items():
        dataset_counts[concept] = dict.fromkeys(groups[concept]["datasets"], 0)
        if not count:
            continue
        second = allocate_leaf_counts({k: v["weight"] for k, v in dw[concept].items()}, capacities[concept], count)
        dataset_counts[concept] = second["counts"]
        for dataset, amount in second["counts"].items():
            if amount:
                leaves = groups[concept]["datasets"][dataset]
                third = allocate_leaf_counts({k: v["weight"] for k, v in lw[concept][dataset].items()},
                                             {k: v["eligible_count"] for k, v in leaves.items()}, amount)
                counts[dataset].update(third["counts"])
    return {"schema_version": "allocation.v1", "requested_count": total,
            "allocated_count": first["allocated_count"], "planning_shortfall": first["planning_shortfall"],
            "concept_counts": first["counts"], "dataset_counts": dataset_counts, "counts": counts,
            "capacities": {d: {k: v["eligible_count"] for k, v in p["tasks"].items()} for d, p in plans.items()},
            "intent": {"concepts": cw, "datasets": dw, "leaves": lw}}


def _plans(context):
    require(context, "DatasetSelection")
    require(context, *(f"dataset/{d}/Plan" for d in context["selected_dataset_ids"]))
    return {d: json.loads(Path(context["dataset_task_plans"][d]).read_text(encoding="utf-8")) for d in context["selected_dataset_ids"]}


def submit(context, **intent):
    """Submit Concept weights; null lower levels explicitly mean equal coverage."""
    allocation = allocate(_plans(context), intent, context["benchmark_info"]["sample_count"])
    if allocation["allocated_count"] == 0:
        return result("blocked", "No eligible samples with positive weights")
    path = Path(context["operation_directory"]) / "allocation.json"
    atomic_write_json(path, allocation)
    return result("success", "Allocation computed", allocation_path=str(path),
                  **publish(context, "Allocation", {"path": str(path)},
                            ("Requirement", "DatasetSelection", "AllocationConfig", "AllocationDelegation",
                             *(f"dataset/{d}/Plan" for d in context["selected_dataset_ids"]))))


def finish(context):
    require(context, "Allocation")
    return result("complete", "Global allocation ready", artifacts={"allocation": context["allocation_path"]})


def describe(context):
    groups = allocation_groups(_plans(context))
    for group in groups.values():
        for dataset, leaves in group["datasets"].items():
            group["datasets"][dataset] = {
                key: {"description": leaf["description"], "eligible_count": leaf["eligible_count"],
                      "response_format": leaf["protocol"]["response_format"]}
                for key, leaf in leaves.items()
            }
    return {"requirement": context["user_requirement"], "target": context["benchmark_info"]["sample_count"],
            "benchmark_info": context["benchmark_info"], "delegation": context.get("delegation"),
            "groups": groups}


def compile_allocation_agent():
    return compile_agent("AllocationAgent",
                         "Distribute the requested benchmark budget Concept first, then Dataset, then Leaf. "
                         "Submit justified Concept weights based on user importance, never number of leaves. "
                         "Original requirements and benchmark_info take precedence over delegation; never invent preferences "
                         "or treat delegation as permission to introduce new priorities or exclusions. "
                         "Use null datasets/leaves for equal lower-level coverage unless the requirement justifies "
                         "different weights. Zero explicitly excludes a member. Code computes all counts. Finish after submit.",
                         {"submit": submit},
                         describe, finish,
                         args_schemas={"submit": AllocationIntent})
