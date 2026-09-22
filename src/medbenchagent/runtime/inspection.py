"""Private inspection receipts tied to actual dataset/source artifact versions."""

import json
from uuid import uuid4

from medbenchagent.runtime.dependencies import require
from medbenchagent.runtime.harness import result


def inspected(context, operation, arguments, value, dependencies, *, source_id=None, **updates):
    records = require(context, *dependencies)
    reference = str(uuid4())
    receipt = {"dataset_id": context.get("dataset_scope", context.get("selected_dataset_id")),
               "source_id": source_id, "operation": operation, "arguments": arguments,
               "dependencies": {r["kind"]: r["id"] for r in records.values()}, "value": value}
    return result("success", json.dumps({"inspection_reference": reference, "inspection": value}, ensure_ascii=False),
                  inspections={**context.get("inspections", {}), reference: receipt}, **updates)


def inspection(context, reference, *, source_id=None):
    receipt = context.get("inspections", {}).get(reference)
    if (not receipt or receipt["dataset_id"] != context.get("dataset_scope", context.get("selected_dataset_id"))
            or receipt["source_id"] != source_id):
        raise ValueError("Inspection reference is missing or belongs to another dataset/source")
    current = require(context, *receipt["dependencies"])
    if any(current[key]["id"] != version for key, version in receipt["dependencies"].items()):
        raise ValueError("Inspection reference is stale; inspect the current inputs")
    return receipt


def inspection_index(context):
    """Expose reusable receipt identities without copying all inspected data into prompts."""
    return {reference: {key: receipt[key] for key in ("operation", "source_id", "arguments")}
            for reference, receipt in context.get("inspections", {}).items()
            if receipt["dataset_id"] == context.get("dataset_scope", context.get("selected_dataset_id"))
            and all(context.get("artifact_versions", {}).get(key) == version
                    for key, version in receipt["dependencies"].items())}
