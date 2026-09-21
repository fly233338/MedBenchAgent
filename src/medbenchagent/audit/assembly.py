"""Deterministic final benchmark assembly, accounting and package validation."""

from collections import Counter, defaultdict
import json
import os
import shutil
from pathlib import Path
from duckdb import ConstraintException

from medbenchagent.audit.hard import read_json, read_rows, write_rows, load_plan_templates, item_errors
from medbenchagent.audit.answer_balance import balance_items
from medbenchagent.data.dataset_catalog import PATH_FIELDS, resolve_path
from medbenchagent.data.operators import connect
from medbenchagent.audit.storage import ReferencedInputs, batches, query_rows
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
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    with connect(root / "assembly.duckdb", root / "scratch") as db:
        return _assemble(context, root, db)


def _assemble(context, root, db):
    plans, allocation = plans_and_allocation(context)
    destination = Path(context["run_directory"]) / "delivery"
    checks, rejection_reasons = {}, Counter()
    db.execute("CREATE OR REPLACE TABLE candidates (id VARCHAR, payload VARCHAR, source_path VARCHAR, line BIGINT, source VARCHAR)")
    db.execute("CREATE OR REPLACE TABLE quarantine (payload VARCHAR)")
    db.execute("CREATE OR REPLACE TABLE final (id VARCHAR PRIMARY KEY, dataset VARCHAR, task VARCHAR, image VARCHAR, format VARCHAR, payload VARCHAR)")
    db.execute("CREATE INDEX final_image ON final(dataset,image)")

    def quarantine(row):
        rejection_reasons.update(row["reasons"])
        db.execute("INSERT INTO quarantine VALUES (?)", [json.dumps(row, ensure_ascii=False)])

    tasks, datasets, concepts = {}, {}, {}
    raw_sources = {}
    for dataset, plan in plans.items():
        asset = context["dataset_assets"][dataset]["entry"]
        datasets[dataset] = {**asset, **{key: str(resolve_path(asset[key])) for key in PATH_FIELDS},
                             "task_plan": context["dataset_task_plans"][dataset],
                             "scope_resolutions": plan.get("scope_resolutions", [])}
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
        for row in read_rows(checked["rejected_path"]):
            quarantine(row)
        with ReferencedInputs(context["dataset_assets"][dataset]["access"], resolve_path(asset["image_manifest_path"]), root / "inputs" / dataset) as inputs:
            for batch in batches(read_rows(checked["accepted_path"])):
                _, images = inputs.read([row["item"]["image_id"] for row in batch])
                entries = []
                for row in batch:
                    item = dict(row["item"])
                    leaf_id, image_id = item["task_id"], item["image_id"]
                    if item["dataset"] != dataset or leaf_id not in plan["tasks"] or image_id not in images:
                        raise ValueError("Checked sample references changed; repeat the dataset check")
                    item.update(id=_identity(dataset, leaf_id, image_id), task_id=dataset + "::" + leaf_id,
                                image_path=image_path(images[image_id].png_path, destination))
                    entries.append((item["id"], json.dumps(item, sort_keys=True), row["source"]["path"], row["source"]["line"], json.dumps(row["source"])))
                db.executemany("INSERT INTO candidates VALUES (?,?,?,?,?)", entries)
    final_counts = Counter()
    grouped = query_rows(db, """SELECT payload,source,
        count(DISTINCT payload) OVER (PARTITION BY id) AS variants,
        row_number() OVER (PARTITION BY id ORDER BY source_path,line) AS position
        FROM candidates ORDER BY id,source_path,line""")
    for payload, source, variants, position in grouped:
        item, source = json.loads(payload), json.loads(source)
        key, dataset, image = item["task_id"], item["dataset"], item["image_id"]
        fmt = tasks[key]["response_format"]
        reason = "conflicting_identity" if variants > 1 else "duplicate_identity" if position > 1 else None
        if not reason and final_counts[key] >= tasks[key]["allocated_count"]:
            reason = "allocation_excess"
        if not reason:
            total, per_format = db.execute("SELECT count(*),count(*) FILTER (WHERE format=?) FROM final WHERE dataset=? AND image=?",
                                          [fmt, dataset, image]).fetchone()
            if context.get("max_total_tasks_per_image") is not None and total >= context["max_total_tasks_per_image"]:
                reason = "total_image_limit"
            elif context.get("max_tasks_per_image") is not None and per_format >= context["max_tasks_per_image"]:
                reason = "per_format_image_limit"
        if reason:
            quarantine({"source": source, "task_id": tasks[key]["local_leaf_id"], "reasons": [reason]})
        else:
            db.execute("INSERT INTO final VALUES (?,?,?,?,?,?)", [item["id"], dataset, key, image, fmt, payload])
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
              "rejection_reason_counts": dict(rejection_reasons),
              "execution_limits": {"max_tasks_per_image": context.get("max_tasks_per_image"),
                                   "max_total_tasks_per_image": context.get("max_total_tasks_per_image")},
              "capacity_note": "Leaf capacities are independent; shared image limits may cause generation shortfalls."}
    manifest = {"schema_version": "benchmark-manifest.v4", "benchmark_id": "benchmark-" + context["run_id"],
                "run_id": context["run_id"], "benchmark_info": context["benchmark_info"], "datasets": datasets,
                "sample_scope": context.get("sample_scope", []), "execution_limits": report["execution_limits"],
                "concepts": concepts, "tasks": tasks, "allocation": allocation, "counts": totals, "raw_source_paths": raw_sources,
                "artifacts": {"audited": {"path": "benchmark.jsonl"}, "report": {"path": "audit_report.json"},
                              "quarantine": {"path": "quarantine.jsonl"}, "complete": {"path": "complete.json"}}}
    write_rows(root / "candidates.jsonl", (json.loads(payload) for (payload,) in query_rows(db, "SELECT payload FROM final ORDER BY id")))
    write_rows(root / "quarantine.jsonl", (json.loads(payload) for (payload,) in query_rows(db, "SELECT payload FROM quarantine")))
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
    counts = Counter(item["task_id"] for item in read_rows(root / "candidates.jsonl") if isinstance(item.get("options"), list))
    items = balance_items(read_rows(root / "candidates.jsonl"), counts, seed)
    package = root / "package"
    package.mkdir(parents=True, exist_ok=True)
    write_rows(package / "benchmark.jsonl", items)
    for filename in ("quarantine.jsonl", "benchmark_manifest.json"):
        shutil.copyfile(root / filename, package / filename)
    report = read_json(root / "audit_report.json")
    report["option_distribution"] = option_distribution(read_rows(package / "benchmark.jsonl"))
    atomic_write_json(package / "audit_report.json", report)


def validate_package(context, directory):
    root = Path(directory)
    with connect(root / "validation.duckdb", root / "validation-scratch") as db:
        try:
            return _validate_package(context, root, db)
        except ConstraintException as exc:
            raise ValueError("Final candidate and sample identities must be unique") from exc


def _validate_package(context, root, db):
    plans, allocation = plans_and_allocation(context)
    package = root / "package"
    db.execute("CREATE OR REPLACE TABLE originals (id VARCHAR PRIMARY KEY, payload VARCHAR)")
    db.execute("CREATE OR REPLACE TABLE items (id VARCHAR PRIMARY KEY, dataset VARCHAR, image VARCHAR, format VARCHAR, payload VARCHAR)")
    for batch in batches(read_rows(root / "candidates.jsonl")):
        db.executemany("INSERT INTO originals VALUES (?,?)", [(row["id"], json.dumps(row)) for row in batch])
    active = {d for d in plans if sum(allocation["counts"][d].values())}
    for batch in batches(read_rows(package / "benchmark.jsonl")):
        entries = []
        for item in batch:
            dataset = item["dataset"]
            if dataset not in active:
                raise ValueError("Final sample dataset is not active")
            prefix = dataset + "::"
            if not item["task_id"].startswith(prefix) or item["task_id"][len(prefix):] not in plans[dataset]["tasks"]:
                raise ValueError("Invalid final task namespace")
            fmt = plans[dataset]["tasks"][item["task_id"][len(prefix):]]["protocol"]["response_format"]
            entries.append((item["id"], dataset, item["image_id"], fmt, json.dumps(item)))
        db.executemany("INSERT INTO items VALUES (?,?,?,?,?)", entries)
    item_count = db.execute("SELECT count(*) FROM items").fetchone()[0]
    if not item_count:
        raise ValueError("No deliverable samples remain")
    if db.execute("SELECT count(*) FROM items FULL OUTER JOIN originals USING(id) WHERE items.id IS NULL OR originals.id IS NULL").fetchone()[0]:
        raise ValueError("Final sample identities differ from assembled candidates")
    for columns, limit in [("dataset,image", context.get("max_total_tasks_per_image")),
                           ("dataset,image,format", context.get("max_tasks_per_image"))]:
        if limit is not None and db.execute(f"SELECT 1 FROM items GROUP BY {columns} HAVING count(*) > ? LIMIT 1", [limit]).fetchone():
            raise ValueError("Final samples exceed per-image limits")
    tasks, datasets = Counter(), Counter()
    for dataset, plan in plans.items():
        if dataset not in active:
            continue
        from medbenchagent.planning.scope import validate_scope_plan
        validate_scope_plan(plan, context.get("sample_scope", []))
        entry = context["dataset_assets"][dataset]["entry"]
        plan, templates = load_plan_templates(context["dataset_task_plans"][dataset], context["dataset_templates"][dataset])
        manifest_path = resolve_path(entry["image_manifest_path"])
        with ReferencedInputs(context["dataset_assets"][dataset]["access"], manifest_path, root / "validation-inputs" / dataset) as inputs:
            rows = query_rows(db, "SELECT items.payload,originals.payload FROM items JOIN originals USING(id) WHERE dataset=? ORDER BY id", [dataset])
            for raw_batch in batches(rows):
                batch = [(json.loads(item), json.loads(original)) for item, original in raw_batch]
                annotations, images = inputs.read([item["image_id"] for item, _ in batch])
                errors_cache = {}
                for item, original in batch:
                    leaf_id = item["task_id"][len(dataset) + 2:]
                    local = {**item, "task_id": leaf_id}
                    errors = item_errors(local, dataset, plan["tasks"], templates, annotations, images, manifest_path.parent, errors_cache, balanced=True)
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
    distribution = option_distribution(read_rows(package / "benchmark.jsonl"))
    if any(max(counts.values()) - min(counts.values()) > 1 for counts in distribution["tasks"].values()):
        raise ValueError("Final task answer positions are not balanced")
    manifest, report = read_json(package / "benchmark_manifest.json"), read_json(package / "audit_report.json")
    if manifest["counts"] != report["counts"] or report["counts"]["final"] != item_count or report["option_distribution"] != distribution:
        raise ValueError("Final package totals or distributions differ")
    if manifest["allocation"] != allocation or manifest["benchmark_info"] != context["benchmark_info"]:
        raise ValueError("Package requirement or allocation changed")
    limits = {"max_tasks_per_image": context.get("max_tasks_per_image"), "max_total_tasks_per_image": context.get("max_total_tasks_per_image")}
    if manifest.get("execution_limits") != limits or report.get("execution_limits") != limits or manifest.get("sample_scope") != context.get("sample_scope", []):
        raise ValueError("Package scope or execution limits changed")
    expected_tasks = {dataset + "::" + leaf for dataset, plan in plans.items() for leaf in plan["tasks"]}
    if set(manifest["tasks"]) != expected_tasks or set(report["tasks"]) != expected_tasks or set(manifest["datasets"]) != set(plans):
        raise ValueError("Package task/dataset inventory differs")
    for key in expected_tasks:
        if report["tasks"][key]["final"] != tasks[key] or manifest["tasks"][key]["sample_count"] != tasks[key]:
            raise ValueError("Final task count differs from report")
    for dataset, row in report["datasets"].items():
        if row["final"] != datasets[dataset] or row["generated"] != row["accepted"] + row["audit_rejected"] or row["accepted"] != row["final"] + row["assembly_removed"]:
            raise ValueError("Dataset count reconciliation failed")
    if sum(1 for _ in read_rows(package / "quarantine.jsonl")) != report["counts"]["generated"] - item_count:
        raise ValueError("Quarantine count does not reconcile with generated and final counts")
    if report["counts"]["total_shortfall"] != allocation["requested_count"] - item_count:
        raise ValueError("Incorrect total shortfall")
    return {"item_count": item_count, "task_counts": dict(tasks)}
