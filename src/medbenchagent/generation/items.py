"""Deterministic benchmark item generation for Leaf Tasks."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from medbenchagent.contracts import BenchmarkItem
from medbenchagent.planning.resolver import NotApplicable, ResolverContext, encode_closed_label, resolve_task_input
from medbenchagent.planning.sampling import sampling_units
from medbenchagent.planning.task_space import LeafTask, OutputSpec
from medbenchagent.data.manifests import SamplingManifest, SamplingUnit


def _identity(
    dataset_id: str,
    leaf_id: str,
    case_id: str,
    sampling_unit_id: str | None = None,
    view_ids: list[str] | None = None,
    panel_id: str | None = None,
    order: int | None = None,
) -> str:
    payload = json.dumps(
        [dataset_id, leaf_id, case_id, sampling_unit_id, view_ids or [], panel_id, order],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _input_assets(case: dict[str, Any], evidence_ids: list[str]) -> list[str]:
    by_id = {item.get("id"): item for item in case.get("evidence", [])}
    return list(dict.fromkeys(
        str(by_id[evidence_id]["value"])
        for evidence_id in evidence_ids
        if evidence_id in by_id and by_id[evidence_id].get("type") == "asset"
    ))


def _input_bindings(case: dict[str, Any], bindings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {item.get("id"): item for item in case.get("evidence", [])}
    return [
        {
            "role": binding["role"],
            "evidence_ids": list(binding["evidence_ids"]),
            "assets": [
                str(by_id[evidence_id]["value"])
                for evidence_id in binding["evidence_ids"]
                if evidence_id in by_id and by_id[evidence_id].get("type") == "asset"
            ],
        }
        for binding in bindings
    ]


def _template_cycle(seed: int, dataset_id: str, leaf_id: str, count: int) -> list[int]:
    payload = json.dumps([seed, dataset_id, leaf_id], separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    cycle = list(range(count))
    rng.shuffle(cycle)
    return cycle


def generate_task_items(
    cases: Iterable[dict[str, Any]],
    leaves: Iterable[LeafTask | dict[str, Any]],
    dataset_id: str,
    allocation: dict[str, int] | None = None,
    seed: int = 0,
    resolver_context: ResolverContext | None = None,
    sampling_manifest: SamplingManifest | dict[str, Any] | Iterable[SamplingUnit | dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    parsed = [LeafTask.model_validate(leaf) for leaf in leaves]
    items = []
    limits = allocation or {leaf.leaf_task_id: None for leaf in parsed}
    counts = {leaf.leaf_task_id: 0 for leaf in parsed}
    cycles = {
        leaf.leaf_task_id: _template_cycle(seed, dataset_id, leaf.leaf_task_id, len(leaf.question_templates))
        for leaf in parsed
    }
    case_list = list(cases)
    if sampling_manifest is not None:
        from dataclasses import replace
        manifest = (sampling_manifest if isinstance(sampling_manifest, SamplingManifest)
                    else SamplingManifest.model_validate(sampling_manifest) if isinstance(sampling_manifest, dict)
                    else SamplingManifest(dataset_id=dataset_id, revision="current", units=list(sampling_manifest)))
        resolver_context = replace(resolver_context or ResolverContext(),
                                   manifests={**(resolver_context.manifests if resolver_context else {}), "sampling": manifest})
        sampling_manifest = manifest
    for case, sampling_unit in sampling_units(case_list, sampling_manifest):
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            continue
        for leaf in parsed:
            limit = limits.get(leaf.leaf_task_id)
            if limit is not None and counts[leaf.leaf_task_id] >= limit:
                continue
            resolved = resolve_task_input(case, leaf, sampling_unit, resolver_context)
            if isinstance(resolved, NotApplicable):
                continue
            answer = resolved.reference
            if leaf.output_spec == OutputSpec.TEXT and isinstance(answer, dict):
                answer = answer.get("value")
            if leaf.output_spec == OutputSpec.LABEL and leaf.options is not None:
                answer = encode_closed_label(answer.get("value", answer) if isinstance(answer, dict) else answer)
                if answer not in leaf.options:
                    raise ValueError(
                        f"Resolved label {answer!r} is missing from Leaf Task options for {leaf.leaf_task_id}"
                    )
            template_index = cycles[leaf.leaf_task_id][counts[leaf.leaf_task_id] % len(cycles[leaf.leaf_task_id])]
            item = {
                "item_id": _identity(dataset_id, leaf.leaf_task_id, case_id, sampling_unit.sampling_unit_id, sampling_unit.view_ids, sampling_unit.panel_id, sampling_unit.order),
                "dataset_id": dataset_id,
                "scenario_id": leaf.scenario_id,
                "leaf_task_id": leaf.leaf_task_id,
                "case_id": case_id,
                "capability_category": leaf.capability_category.value,
                "task_paradigm": leaf.task_paradigm.value,
                "options": list(leaf.options) if leaf.options is not None else None,
                "input_images": resolved.presentations,
                "input_bindings": _input_bindings(case, resolved.input_bindings),
                "input_assets": _input_assets(case, resolved.input_evidence_ids),
                "input_evidence_ids": resolved.input_evidence_ids,
                "question": leaf.question_templates[template_index],
                "template_index": template_index,
                "template_count": len(leaf.question_templates),
                "output_spec": leaf.output_spec.value,
                "answer": answer,
                "reference_evidence_ids": resolved.reference_evidence_ids,
            }
            if sampling_manifest is not None:
                item.update({
                    "sampling_unit_id": sampling_unit.sampling_unit_id,
                    "split": sampling_unit.split,
                    "view_ids": list(sampling_unit.view_ids),
                    "panel_id": sampling_unit.panel_id,
                    "sampling_order": sampling_unit.order,
                })
            items.append(BenchmarkItem.model_validate(item).model_dump(exclude_unset=True))
            counts[leaf.leaf_task_id] += 1
    return items


def write_task_items(path: str | Path, items: Iterable[dict[str, Any]]) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for item in items:
                item = BenchmarkItem.model_validate(item).model_dump(exclude_unset=True)
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return str(destination)



def run_generation(context, *, output_path=None):
    from pathlib import Path
    from medbenchagent.settings import get_settings
    page_size = get_settings().generation.fetch_batch_size
    from medbenchagent.data.index import DatasetIndex
    from medbenchagent.data.manifests import SamplingManifest, load_manifest
    from medbenchagent.planning.resolver import ResolverContext
    from medbenchagent.planning.bir import BIRStore
    store = BIRStore(context["bir_path"])
    access = store.load("dataset_binding.json")["access"]
    dataset = access["dataset_id"]
    allocation = store.load("allocation.json")["counts"]
    sampling_manifest = None
    if access.get("sampling_manifest_path"):
        sampling_manifest = load_manifest(access["sampling_manifest_path"], SamplingManifest)
    plan = store.load("task_plan.json")
    if plan["phase"] != "complete":
        raise ValueError("Generation requires a complete Task Plan")
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
        dataset,
        allocation=allocation,
        seed=context.get("seed", 0),
        resolver_context=resolver_context,
        sampling_manifest=sampling_manifest,
    )
    # Callers may pass a pathlib.Path (the Supervisor does this). Results are
    # persisted through the JSON operation log, so keep artifact references
    # JSON-native at this boundary.
    destination = str(output_path or (Path(context["experiment_output_directory"]) / "generation" /
                      store.digest("allocation.json").removeprefix("sha256:") / "benchmark_items.jsonl"))
    write_task_items(destination, items)
    task_counts = {}
    for item in items:
        task_counts[item["leaf_task_id"]] = task_counts.get(item["leaf_task_id"], 0) + 1
    from medbenchagent.planning.bir import business_dependencies, digest_file
    from medbenchagent.data.sources import digest_file as content_digest
    store.update("generation.json", {
        "schema_version": "bir-output", "dataset_id": dataset, "status": "complete",
        "benchmark_items_path": destination, "benchmark_items_digest": content_digest(destination),
        "counts": task_counts, "allocation": dict(allocation or {}),
    }, run_id=context.get("run_id", "unknown"), actor="generation",
       dependencies=business_dependencies(context, "allocation.json", "task_plan.json")
                    | {"file:" + str(Path(destination).resolve()): digest_file(destination)})
    from medbenchagent.runtime.harness import result
    return result(
        "success",
        f"Generated {len(items)} benchmark items",
        data={"raw_output_path": destination, "item_count": len(items), "task_counts": task_counts},
    )
