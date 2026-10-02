"""Run-context adapter for the deterministic v2 benchmark pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from medbenchagent.planning.pipeline_v2 import run_v2_pipeline
from medbenchagent.runtime.harness import result
from medbenchagent.runtime.run_artifacts import atomic_write_json


def run_task_space_pipeline(context, *, requested_count: int | None = None):
    """Resume v2 planning artifacts and publish auditable benchmark outputs."""
    access = context.get("dataset_access_manifest", {})
    if access.get("schema_version") != "annotation.v5":
        raise ValueError("Task Space pipeline requires annotation.v5")
    operation = Path(context["operation_directory"])
    axis_path = context.get("axis_analysis_path")
    leaves_path = context.get("task_leaves_path")
    nodes_path = context.get("task_nodes_path")
    for label, path in (("Axis Analysis", axis_path), ("Task Nodes", nodes_path), ("Task Leaves", leaves_path)):
        if not path or not Path(path).is_file():
            raise ValueError(f"{label} artifact is required")
    axis = json.loads(Path(axis_path).read_text(encoding="utf-8"))
    json.loads(Path(nodes_path).read_text(encoding="utf-8"))
    leaves = json.loads(Path(leaves_path).read_text(encoding="utf-8"))
    from medbenchagent.data.v5_index import V5DatasetQuery
    query = V5DatasetQuery(access)
    cases, offset = [], 0
    while True:
        page = query.sample(offset=offset, limit=50)
        cases.extend(page["values"])
        if page["complete"]:
            break
        offset = page["next_offset"]
    # Reconstruct deterministic constraint facts from the published v5 profile.
    # Evidence names must remain available here; type-only aggregate keys make
    # valid Leaf selectors fail during the final pipeline validation.
    profile = json.loads(Path(access["profile_path"]).read_text(encoding="utf-8"))
    node_context = {"evidence": {}, "evidence_profile": profile.get("evidence", []),
                    "relations": {}, "assets": {}}
    for row in profile.get("evidence", []):
        node_context["evidence"][row["name"]] = row
    assets = profile.get("assets", {})
    per_case = {int(key): value for key, value in assets.get("assets_per_case", {}).items()}
    related = sum(value for size, value in per_case.items() if size >= 2)
    node_context["assets"] = {"volume_assets": assets.get("kinds", {}).get("volume", 0),
                               "related_image_cases": related}
    node_context["relations"] = {
        "related": sum(row.get("has_ref", 0) for row in profile.get("evidence", [])),
        "paired": related, "paired_study": related, "comparable": related,
        "paired_text": sum(row.get("types", {}).get("text", 0) for row in profile.get("evidence", [])),
    }
    output = run_v2_pipeline(dataset={"dataset_id": context.get("selected_dataset_id", "dataset")}, axis_analysis=axis,
                             node_context=node_context, leaves=leaves.values(), cases=cases,
                             requested_count=requested_count or context.get("benchmark_info", {}).get("sample_count", 1))
    paths = {}
    for name, payload in (("task_space_items.json", output["items"]), ("task_space_audit.json", output["audit"]), ("task_space_manifest.json", output["manifest"])):
        path = operation / name
        atomic_write_json(path, payload)
        paths[name] = str(path)
    return result("success", f"v2 pipeline completed with {len(output['items'])} items", **paths,
                  task_space_output=output)

