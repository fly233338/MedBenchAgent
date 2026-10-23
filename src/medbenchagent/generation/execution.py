"""Allocation, deterministic batches and candidate materialization over prepared data."""

import hashlib
import json
from pathlib import Path

from medbenchagent.contracts import BenchmarkItem
from medbenchagent.data.dataset import _relative_file, file_digest
from medbenchagent.data.task_data import write_once
from .backends import assemble_item
from .contracts import Allocation, CaseFailure, GenerationBatch, GenerationCandidate


def load_execution(spec_binding, resolve):
    """resolve(binding, kind) supplies a Host-verified path from explicitly bound references."""
    from medbenchagent.data.task_data import TaskDataReader
    from medbenchagent.planning.contracts import TaskPlan
    from .contracts import GenerationSpec, validate_generation_spec

    spec = GenerationSpec.model_validate_json(resolve(spec_binding, "generation_spec").read_bytes())
    plan = TaskPlan.model_validate_json(resolve(spec.task_plan, "task_plan").read_bytes())
    reader = TaskDataReader(resolve(spec.task_data, "task_data"))
    validate_generation_spec(
        spec,
        plan=plan,
        plan_artifact_id=spec.task_plan.artifact_id,
        plan_digest=spec.task_plan.digest,
        reader=reader,
    )
    return spec_binding, spec, plan, reader


def selected_cases(spec, task, reader):
    selected = spec.case_ids or [
        case.case_id for case in reader.cases if task.scenario_id in case.scenario_ids
    ]
    return sorted(
        selected, key=lambda key: hashlib.sha256(f"{spec.seed}/{key}".encode()).hexdigest()
    )[: spec.sample_count]


def allocate_counts(capacities: dict[str, int], requested: int, target):
    if (
        requested < 1
        or not capacities
        or any(type(v) is not int or v < 0 for v in capacities.values())
    ):
        raise ValueError("Positive target and nonnegative capacities are required")
    counts = dict.fromkeys(sorted(capacities), 0)
    remaining = requested
    while remaining:
        active = [key for key in counts if counts[key] < capacities[key]]
        if not active:
            break
        share = max(1, remaining // len(active))
        for key in active:
            amount = min(share, capacities[key] - counts[key], remaining)
            counts[key] += amount
            remaining -= amount
    return Allocation(target=target, requested=requested, counts=counts, capacities=capacities)


def generate_batch(spec, spec_binding, task, reader, case_ids):
    allowed = set(selected_cases(spec, task, reader))
    if len(case_ids) != len(set(case_ids)) or set(case_ids) - allowed:
        raise ValueError("Batch scope is outside the prepared execution specification")
    items, failures = [], []
    for identifier in case_ids:
        try:
            items.append(assemble_item(task, reader, identifier, spec.seed))
        except (ValueError, KeyError) as exc:
            failures.append(
                CaseFailure(case_id=identifier, code="case_assembly_failed", message=str(exc))
            )
    return GenerationBatch(spec=spec_binding, case_ids=case_ids, items=items, failures=failures)


def finalize_candidate(destination, *, batches, specs, allocation, quarantined):
    """batches/specs are (ProductBinding, value) pairs already verified by the Host."""
    spec_map = {ref.artifact_id: (ref, spec) for ref, spec in specs}
    tasks = [spec.task_id for _, spec in specs]
    if len(tasks) != len(set(tasks)) or set(tasks) != set(allocation.counts):
        raise ValueError("Select exactly one execution specification for each allocated task")
    if len(spec_map) != len(specs) or len({ref.artifact_id for ref, _ in batches}) != len(batches):
        raise ValueError("Duplicate specification or batch selections")
    seen, items, failures, attempted = set(), [], [], set()
    for _, batch in batches:
        if (
            batch.spec.artifact_id not in spec_map
            or batch.spec != spec_map[batch.spec.artifact_id][0]
        ):
            raise ValueError("Batch binds a different specification")
        spec = spec_map[batch.spec.artifact_id][1]
        for identifier in batch.case_ids:
            attempt = (spec.task_id, identifier)
            if attempt in attempted:
                raise ValueError(
                    "Overlapping selected batches; select the replacement instead of both attempts"
                )
            attempted.add(attempt)
        for item in batch.items:
            if (
                item.item_id in seen
                or item.leaf_task_id != spec.task_id
                or item.case_id not in batch.case_ids
            ):
                raise ValueError("Duplicate or incorrectly bound generated item")
            seen.add(item.item_id)
            items.append(item)
        failures.extend(batch.failures)
    if set(quarantined) - seen or any(not reason.strip() for reason in quarantined.values()):
        raise ValueError("Quarantine requires known item IDs and explicit reasons")
    kept = [item for item in items if item.item_id not in quarantined]
    per_task = {
        key: {
            "requested": allocation.counts[key],
            "generated": sum(item.leaf_task_id == key for item in items),
            "accepted": sum(item.leaf_task_id == key for item in kept),
        }
        for key in tasks
    }
    if any(value["accepted"] > value["requested"] for value in per_task.values()):
        raise ValueError("Candidate exceeds its task allocation")
    destination = Path(destination)
    item_bytes = ("".join(item.model_dump_json() + "\n" for item in kept)).encode("utf-8")
    quarantine_bytes = (
        "".join(
            json.dumps(
                {"item": item.model_dump(mode="json"), "reason": quarantined[item.item_id]},
                ensure_ascii=False,
            )
            + "\n"
            for item in items
            if item.item_id in quarantined
        )
    ).encode("utf-8")
    write_once(destination / "items.jsonl", item_bytes)
    write_once(destination / "quarantine.jsonl", quarantine_bytes)
    candidate = GenerationCandidate(
        allocation=allocation,
        specs=[ref for ref, _ in specs],
        batches=[ref for ref, _ in batches],
        members={
            "items.jsonl": hashlib.sha256(item_bytes).hexdigest(),
            "quarantine.jsonl": hashlib.sha256(quarantine_bytes).hexdigest(),
        },
        counts={
            "requested": allocation.requested,
            "generated": len(items),
            "accepted": len(kept),
            "quarantined": len(items) - len(kept),
            "shortfall": max(0, allocation.requested - len(kept)),
            "failed_cases": len(failures),
        },
        task_counts=per_task,
        failures=failures,
    )
    path = destination / "candidate.json"
    write_once(path, (candidate.model_dump_json(indent=2) + "\n").encode("utf-8"))
    return path


class CandidateReader:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.manifest = GenerationCandidate.model_validate_json(self.path.read_bytes())
        self._digest = file_digest(self.path)
        self.verify_integrity()
        self.items = [
            BenchmarkItem.model_validate_json(line)
            for line in _relative_file(self.path.parent, self.manifest.items_path)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.quarantine = [
            json.loads(line)
            for line in _relative_file(self.path.parent, self.manifest.quarantine_path)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        if len({item.item_id for item in self.items}) != len(self.items):
            raise ValueError("Duplicate candidate item IDs")
        quarantined = [row["item"]["item_id"] for row in self.quarantine]
        if len(quarantined) != len(set(quarantined)) or set(quarantined) & {
            item.item_id for item in self.items
        }:
            raise ValueError("Quarantine IDs must be unique and disjoint from accepted items")
        counts = self.manifest.counts
        if (
            counts.get("accepted") != len(self.items)
            or counts.get("quarantined") != len(self.quarantine)
            or counts.get("generated") != len(self.items) + len(self.quarantine)
            or counts.get("requested") != self.manifest.allocation.requested
            or counts.get("shortfall") != max(0, counts["requested"] - len(self.items))
        ):
            raise ValueError("Candidate counts do not match its actual contents")

    def verify_integrity(self):
        if file_digest(self.path) != self._digest:
            raise ValueError("Candidate Manifest changed")
        if (
            not {self.manifest.items_path, self.manifest.quarantine_path}
            <= self.manifest.members.keys()
        ):
            raise ValueError("Candidate member coverage is incomplete")
        for name, digest in self.manifest.members.items():
            if file_digest(_relative_file(self.path.parent, name)) != digest:
                raise ValueError(f"Candidate member changed: {name}")
