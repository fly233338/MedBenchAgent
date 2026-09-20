"""Deterministic final benchmark assembly, accounting and package validation."""

from collections import Counter, defaultdict
import json
import os
import shutil
from pathlib import Path

from medbenchagent.audit.hard import read_json, read_rows, write_rows, load_inputs, item_errors
from medbenchagent.audit.answer_balance import balance_items
from medbenchagent.data.dataset_catalog import PATH_FIELDS, resolve_path
from medbenchagent.data.image_assets import load_image_manifest
from medbenchagent.runtime.dependencies import require
from medbenchagent.runtime.run_artifacts import atomic_write_json


def plans_and_allocation(context):
    require(context, "DatasetSelection", "Allocation")
    plans = {d: read_json(context["dataset_task_plans"][d]) for d in context["selected_dataset_ids"]}
    allocation = read_json(context["allocation_path"])
    counts = allocation["counts"]
    if set(counts) != set(plans):
        raise ValueError("Allocation datasets differ from Selection")
    for dataset, plan in plans.items():
        if set(counts[dataset]) != set(plan["tasks"]) or any(type(n) is not int or n < 0 for n in counts[dataset].values()):
            raise ValueError("Allocation must contain exact nonnegative Leaf counts")
    total = sum(sum(group.values()) for group in counts.values())
    if (type(allocation["requested_count"]) is not int or not 0 < total <= allocation["requested_count"]
            or total != allocation["allocated_count"]):
        raise ValueError("Invalid global allocation totals")
    return plans, allocation


def input_dependencies(context):
    plans, allocation = plans_and_allocation(context)
    dependencies = ["DatasetSelection", "Allocation", "FinalizationConfig"]
    for dataset in plans:
        dependencies += [f"dataset/{dataset}/Plan", f"dataset/{dataset}/Templates"]
        if sum(allocation["counts"][dataset].values()):
            dependencies.append(f"dataset/{dataset}/Generation")
    require(context, *dependencies)
    return tuple(dependencies)


def input_versions(context):
    return {key: context["artifact_versions"][key] for key in input_dependencies(context)}


def image_path(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return path.as_posix() if path.drive.casefold() != root.drive.casefold() else os.path.relpath(path, root).replace(os.sep, "/")


def _identity(dataset, leaf, image):
    return json.dumps([dataset, leaf, image], ensure_ascii=False, separators=(",", ":"))


def assemble(context, directory):
    plans, allocation = plans_and_allocation(context)
    destination = Path(context["run_directory"]) / "delivery"
    groups, quarantined, checks = defaultdict(list), [], {}
    tasks, datasets, concepts = {}, {}, {}
    raw_sources = {}
    for dataset, plan in plans.items():
        asset = context["dataset_assets"][dataset]["entry"]
        datasets[dataset] = {**asset, **{key: str(resolve_path(asset[key])) for key in PATH_FIELDS},
                             "task_plan": context["dataset_task_plans"][dataset]}
        for concept_id, concept in plan["concepts"].items():
            key = concept_id if concept["origin"] == "core" else dataset + "::" + concept_id
            concepts.setdefault(key, {k: concept[k] for k in ("description", "category", "origin")})
        for leaf_id, leaf in plan["tasks"].items():
            concept = leaf["concept_id"]
            tasks[dataset + "::" + leaf_id] = {"dataset_id": dataset, "local_leaf_id": leaf_id,
                "concept_id": concept if plan["concepts"][concept]["origin"] == "core" else dataset + "::" + concept,
                "category": leaf["category"], "response_format": leaf["protocol"]["response_format"],
                "metrics": [leaf["protocol"]["metric"]], "allocated_count": allocation["counts"][dataset][leaf_id]}
        if not sum(allocation["counts"][dataset].values()):
            datasets[dataset]["generation_status"] = "skipped_zero_allocation"
            checks[dataset] = {"generated": 0, "accepted": 0, "rejected": 0, "tasks": {},
                               "unassigned": {"generated": 0, "accepted": 0, "rejected": 0}}
            continue
        require(context, f"dataset/{dataset}/GenerationCheck")
        checked = context["dataset_checks"][dataset]
        checks[dataset] = read_json(checked["report_path"])
        raw_sources[dataset] = context["dataset_generations"][dataset]["raw_output_path"]
        quarantined.extend(read_rows(checked["rejected_path"]))
        images = load_image_manifest(resolve_path(asset["image_manifest_path"]))
        for row in read_rows(checked["accepted_path"]):
            item = dict(row["item"])
            leaf_id, image_id = item["task_id"], item["image_id"]
            if item["dataset"] != dataset or leaf_id not in plan["tasks"] or image_id not in images:
                raise ValueError("Checked sample references changed; repeat the dataset check")
            item.update(id=_identity(dataset, leaf_id, image_id), task_id=dataset + "::" + leaf_id,
                        image_path=image_path(images[image_id].png_path, destination))
            groups[item["id"]].append({"item": item, "source": row["source"]})
    unique = []
    for identity in sorted(groups):
        rows = sorted(groups[identity], key=lambda r: (r["source"]["path"], r["source"]["line"]))
        conflict = any(row["item"] != rows[0]["item"] for row in rows[1:])
        if not conflict:
            unique.append(rows[0])
        for row in rows if conflict else rows[1:]:
            quarantined.append({"source": row["source"], "task_id": tasks[row["item"]["task_id"]]["local_leaf_id"],
                                "reasons": ["conflicting_identity" if conflict else "duplicate_identity"]})
    final, final_counts = [], Counter()
    for row in unique:
        key = row["item"]["task_id"]
        if final_counts[key] >= tasks[key]["allocated_count"]:
            quarantined.append({"source": row["source"], "task_id": tasks[key]["local_leaf_id"], "reasons": ["allocation_excess"]})
        else:
            final.append(row["item"])
            final_counts[key] += 1
    dataset_counts, task_counts = {}, {}
    for dataset, plan in plans.items():
        check = checks[dataset]
        for leaf_id in plan["tasks"]:
            key = dataset + "::" + leaf_id
            values = check["tasks"].get(leaf_id, {"generated": 0, "accepted": 0, "rejected": 0})
            allocated = allocation["counts"][dataset][leaf_id]
            task_counts[key] = {"allocated": allocated, "generated": values["generated"], "accepted": values["accepted"],
                "final": final_counts[key], "generation_shortfall": max(0, allocated - values["generated"]),
                "generation_excess": max(0, values["generated"] - allocated), "audit_rejected": values["rejected"],
                "assembly_removed": values["accepted"] - final_counts[key]}
            tasks[key]["sample_count"] = final_counts[key]
        dataset_counts[dataset] = {"allocated": sum(allocation["counts"][dataset].values()),
            "generated": check["generated"], "accepted": check["accepted"],
            "final": sum(final_counts[dataset + "::" + key] for key in plan["tasks"]),
            "audit_rejected": check["rejected"], "unassigned": check["unassigned"]}
        dataset_counts[dataset]["assembly_removed"] = check["accepted"] - dataset_counts[dataset]["final"]
        for key in ("generation_shortfall", "generation_excess"):
            dataset_counts[dataset][key] = sum(task_counts[dataset + "::" + leaf][key] for leaf in plan["tasks"])
    totals = {key: sum(row[key] for row in dataset_counts.values())
              for key in ("allocated", "generated", "accepted", "final", "audit_rejected", "assembly_removed", "generation_shortfall", "generation_excess")}
    totals.update(requested=allocation["requested_count"], planning_shortfall=allocation["requested_count"] - totals["allocated"],
                  total_shortfall=allocation["requested_count"] - totals["final"],
                  unassigned=sum(row["unassigned"]["generated"] for row in dataset_counts.values()))
    report = {"schema_version": "audit-report.v3", "counts": totals, "datasets": dataset_counts, "tasks": task_counts,
              "rejection_reason_counts": dict(Counter(reason for row in quarantined for reason in row["reasons"]))}
    manifest = {"schema_version": "benchmark-manifest.v4", "benchmark_id": "benchmark-" + context["run_id"],
                "run_id": context["run_id"], "benchmark_info": context["benchmark_info"], "datasets": datasets,
                "concepts": concepts, "tasks": tasks, "allocation": allocation, "counts": totals, "raw_source_paths": raw_sources,
                "artifacts": {"audited": {"path": "benchmark.jsonl"}, "report": {"path": "audit_report.json"},
                              "quarantine": {"path": "quarantine.jsonl"}, "complete": {"path": "complete.json"}}}
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    write_rows(root / "candidates.jsonl", final)
    write_rows(root / "quarantine.jsonl", quarantined)
    atomic_write_json(root / "audit_report.json", report)
    atomic_write_json(root / "benchmark_manifest.json", manifest)
    return root


def option_distribution(items):
    by_task, by_size = defaultdict(Counter), defaultdict(Counter)
    for item in items:
        if isinstance(item.get("options"), list):
            n = len(item["options"])
            for key in (by_task[item["task_id"]], by_size[str(n)]):
                for index in range(n):
                    key.setdefault(chr(65 + index), 0)
                key[item["answer"]] += 1
    return {"tasks": {k: dict(v) for k, v in by_task.items()}, "option_counts": {k: dict(v) for k, v in by_size.items()}}


def balance(directory, seed):
    root = Path(directory)
    items = balance_items(list(read_rows(root / "candidates.jsonl")), seed)
    package = root / "package"
    package.mkdir(parents=True, exist_ok=True)
    write_rows(package / "benchmark.jsonl", items)
    for filename in ("quarantine.jsonl", "benchmark_manifest.json"):
        shutil.copyfile(root / filename, package / filename)
    report = read_json(root / "audit_report.json")
    report["option_distribution"] = option_distribution(items)
    atomic_write_json(package / "audit_report.json", report)


def validate_package(context, directory):
    root = Path(directory)
    plans, allocation = plans_and_allocation(context)
    package = root / "package"
    items = list(read_rows(package / "benchmark.jsonl"))
    candidates = {row["id"]: row for row in read_rows(root / "candidates.jsonl")}
    if not items:
        raise ValueError("No deliverable samples remain")
    if len(items) != len(candidates) or {row["id"] for row in items} != set(candidates):
        raise ValueError("Final sample identities differ from assembled candidates")
    tasks, datasets = Counter(), Counter()
    inputs = {}
    for dataset, plan in plans.items():
        if sum(allocation["counts"][dataset].values()):
            entry = context["dataset_assets"][dataset]["entry"]
            inputs[dataset] = (*load_inputs(context["dataset_task_plans"][dataset], context["dataset_templates"][dataset],
                resolve_path(entry["annotation_path"]), resolve_path(entry["image_manifest_path"])),
                resolve_path(entry["image_manifest_path"]).parent, {})
    for item in items:
        original = candidates[item["id"]]
        dataset = item["dataset"]
        if dataset not in inputs:
            raise ValueError("Final sample dataset is not active")
        plan, templates, annotations, images, image_root, errors_cache = inputs[dataset]
        prefix = dataset + "::"
        if not item["task_id"].startswith(prefix):
            raise ValueError("Invalid final task namespace")
        leaf_id = item["task_id"][len(prefix):]
        local = {**item, "task_id": leaf_id}
        errors = item_errors(local, dataset, plan["tasks"], templates, annotations, images, image_root, errors_cache, balanced=True)
        if errors:
            raise ValueError("Final sample failed validation: " + ", ".join(errors))
        if item["id"] != _identity(dataset, leaf_id, item["image_id"]) or item["image_path"] != image_path(images[item["image_id"]].png_path, Path(context["run_directory"]) / "delivery"):
            raise ValueError("Final identity or image path differs from source references")
        if isinstance(item.get("options"), list):
            answer = item["options"][ord(item["answer"]) - 65]
            original_answer = original["options"][ord(original["answer"]) - 65]
            if answer != original_answer or sorted(item["options"]) != sorted(original["options"]):
                raise ValueError("Balancing changed answer semantics")
            if {k: v for k, v in item.items() if k not in {"options", "answer"}} != {k: v for k, v in original.items() if k not in {"options", "answer"}}:
                raise ValueError("Balancing changed sample content")
        elif item != original:
            raise ValueError("Non-letter sample changed during balancing")
        tasks[item["task_id"]] += 1
        datasets[dataset] += 1
        if tasks[item["task_id"]] > allocation["counts"][dataset][leaf_id]:
            raise ValueError("Final task exceeds allocation")
    distribution = option_distribution(items)
    if any(max(counts.values()) - min(counts.values()) > 1 for counts in distribution["tasks"].values()):
        raise ValueError("Final task answer positions are not balanced")
    manifest, report = read_json(package / "benchmark_manifest.json"), read_json(package / "audit_report.json")
    if manifest["counts"] != report["counts"] or report["counts"]["final"] != len(items) or report["option_distribution"] != distribution:
        raise ValueError("Final package totals or distributions differ")
    if manifest["allocation"] != allocation or manifest["benchmark_info"] != context["benchmark_info"]:
        raise ValueError("Package requirement or allocation changed")
    expected_tasks = {dataset + "::" + leaf for dataset, plan in plans.items() for leaf in plan["tasks"]}
    if set(manifest["tasks"]) != expected_tasks or set(report["tasks"]) != expected_tasks or set(manifest["datasets"]) != set(plans):
        raise ValueError("Package task/dataset inventory differs")
    for key in expected_tasks:
        if report["tasks"][key]["final"] != tasks[key] or manifest["tasks"][key]["sample_count"] != tasks[key]:
            raise ValueError("Final task count differs from report")
    for dataset, row in report["datasets"].items():
        if row["final"] != datasets[dataset] or row["generated"] != row["accepted"] + row["audit_rejected"] or row["accepted"] != row["final"] + row["assembly_removed"]:
            raise ValueError("Dataset count reconciliation failed")
    if sum(1 for _ in read_rows(package / "quarantine.jsonl")) != report["counts"]["generated"] - len(items):
        raise ValueError("Quarantine count does not reconcile with generated and final counts")
    if report["counts"]["total_shortfall"] != allocation["requested_count"] - len(items):
        raise ValueError("Incorrect total shortfall")
    return {"item_count": len(items), "task_counts": dict(tasks)}
