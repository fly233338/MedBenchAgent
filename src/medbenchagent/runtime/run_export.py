"""Merge audited samples only; source annotations and images remain dataset-local."""

from collections import Counter
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.data.image_assets import load_image_manifest
from medbenchagent.data.dataset_catalog import PATH_FIELDS, resolve_path


def _image_path(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path.drive.casefold() != root.drive.casefold():
        return path.as_posix()
    return os.path.relpath(path, root).replace(os.sep, "/")


def export_benchmark(context):
    from medbenchagent.runtime.dependencies import require, publish
    require(context, "DatasetSelection", "Allocation")
    root = Path(context["run_directory"])
    allocation = json.loads(Path(context["allocation_path"]).read_text(encoding="utf-8"))
    plans = {d: json.loads(Path(context["dataset_task_plans"][d]).read_text(encoding="utf-8"))
             for d in context["selected_dataset_ids"]}
    tasks, datasets, concepts, dependencies = {}, {}, {}, ["DatasetSelection", "Allocation"]
    audit_totals = {"accepted": 0, "rejected": 0}
    flags = 0
    for dataset, plan in plans.items():
        asset = context["dataset_assets"][dataset]
        datasets[dataset] = {**asset["entry"], **{k: str(resolve_path(asset["entry"][k])) for k in PATH_FIELDS},
                             "task_plan": context["dataset_task_plans"][dataset]}
        for concept_id, concept in plan["concepts"].items():
            key = concept_id if concept["origin"] == "core" else dataset + "::" + concept_id
            concepts.setdefault(key, {"description": concept["description"], "category": concept["category"], "origin": concept["origin"]})
        for leaf_id, leaf in plan["tasks"].items():
            key = dataset + "::" + leaf_id
            tasks[key] = {"dataset_id": dataset, "local_leaf_id": leaf_id,
                          "concept_id": leaf["concept_id"] if plan["concepts"][leaf["concept_id"]]["origin"] == "core" else dataset + "::" + leaf["concept_id"],
                          "category": leaf["category"], "response_format": leaf["protocol"]["response_format"],
                          "metrics": [leaf["protocol"]["metric"]], "sample_count": allocation["counts"][dataset][leaf_id]}
        if sum(allocation["counts"][dataset].values()):
            require(context, f"dataset/{dataset}/Audit")
            dependencies.append(f"dataset/{dataset}/Audit")
            report = json.loads(Path(context["dataset_audits"][dataset]["report"]).read_text(encoding="utf-8"))
            for key in audit_totals:
                audit_totals[key] += report["hard_audit"][key]
            flags += report.get("semantic_audit", {}).get("flagged", 0)
        else:
            datasets[dataset]["generation_status"] = "skipped_zero_allocation"
    counts = {}
    with TemporaryDirectory(prefix=".export-", dir=root) as staging:
        for kind, filename in (("raw", "benchmark.raw.jsonl"), ("audited", "benchmark.jsonl")):
            target = Path(staging) / filename
            temporary = target.with_suffix(".jsonl.tmp")
            seen, totals = set(), Counter()
            try:
                with temporary.open("w", encoding="utf-8", newline="\n") as out:
                    for dataset in sorted(plans):
                        if not sum(allocation["counts"][dataset].values()):
                            continue
                        source = context["dataset_generations"][dataset]["raw_output_path"] if kind == "raw" else context["dataset_audits"][dataset]["audited"]
                        entry = context["dataset_assets"][dataset]["entry"]
                        images = load_image_manifest(resolve_path(entry["image_manifest_path"]))
                        local_count = 0
                        with Path(source).open(encoding="utf-8") as handle:
                            for line in handle:
                                if not line.strip():
                                    continue
                                item = json.loads(line)
                                leaf_id = item["task_id"]
                                key = dataset + "::" + leaf_id
                                if item["dataset"] != dataset or key not in tasks or item["image_id"] not in images:
                                    raise ValueError("Export sample has invalid dataset/task/image references")
                                identity = json.dumps([dataset, leaf_id, item["image_id"]], ensure_ascii=False, separators=(",", ":"))
                                if identity in seen:
                                    raise ValueError("Duplicate sample identity during export")
                                seen.add(identity)
                                item.update(id=identity, task_id=key,
                                            image_path=_image_path(images[item["image_id"]].png_path, root))
                                totals[key] += 1
                                local_count += 1
                                if totals[key] > tasks[key]["sample_count"]:
                                    raise ValueError("Export exceeds allocated Leaf count")
                                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                        if kind == "audited":
                            report = json.loads(Path(context["dataset_audits"][dataset]["report"]).read_text(encoding="utf-8"))
                            if local_count != report["hard_audit"]["accepted"]:
                                raise ValueError("Audited sample count differs from report")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            counts[kind] = totals
        for filename in ("benchmark.raw.jsonl", "benchmark.jsonl"):
            (Path(staging) / filename).replace(root / filename)
    total = sum(counts["audited"].values())
    shortfalls = {key: task["sample_count"] - counts["raw"][key] for key, task in tasks.items()}
    report = {"hard_audit": audit_totals, "semantic_audit": {"flagged": flags},
              "planning_shortfall": allocation["planning_shortfall"], "generation_shortfalls": shortfalls,
              "total_shortfall": allocation["requested_count"] - total,
              "datasets": {d: context["dataset_audits"][d] for d in plans if sum(allocation["counts"][d].values())}}
    atomic_write_json(root / "audit_report.json", report)
    manifest = {"schema_version": "benchmark-manifest.v3", "benchmark_id": "benchmark-" + context["run_id"],
                "run_id": context["run_id"], "benchmark_info": context["benchmark_info"],
                "datasets": datasets, "concepts": concepts, "tasks": tasks, "allocation": allocation,
                "artifacts": {"audited": {"path": "benchmark.jsonl"}, "raw": {"path": "benchmark.raw.jsonl"},
                              "report": {"path": "audit_report.json"}}}
    atomic_write_json(root / "benchmark_manifest.json", manifest)
    versions = publish(context, "Export", {"path": str(root / "benchmark_manifest.json"),
                                          "audited": str(root / "benchmark.jsonl")}, tuple(dependencies))
    return {"manifest_path": str(root / "benchmark_manifest.json"), "benchmark_id": manifest["benchmark_id"],
            "raw_output_path": str(root / "benchmark.raw.jsonl"),
            "audit_artifacts": {"audited": str(root / "benchmark.jsonl"), "report": str(root / "audit_report.json")},
            "item_count": total, "task_counts": dict(counts["audited"]), "generation_shortfalls": shortfalls,
            "planning_shortfall": allocation["planning_shortfall"], "total_shortfall": report["total_shortfall"], **versions}
