"""Fixtures exercising the production compilation stages with real local image assets."""
import json
from pathlib import Path

from image_fixtures import image_context
from medbenchagent.agents.allocation import allocate_benchmark
from medbenchagent.agents.generation import run_generation
from medbenchagent.data.index import prepare_dataset
from medbenchagent.data.models import ImageEntry
from medbenchagent.generation.delivery import audit_candidate
from medbenchagent.planning.access import DatasetAccessReceipt
from medbenchagent.planning.bir import BIRStore, business_dependencies
from medbenchagent.planning.task_discovery import summarize_leaf


def prepared_access(bundle, cases):
    bundle = Path(bundle)
    images = image_context(bundle, cases)
    (bundle / "cases.jsonl").write_text("".join(json.dumps(row) + "\n" for row in cases), encoding="utf-8")
    (bundle / "image_manifest.jsonl").write_text("".join(ImageEntry.model_validate(value).model_dump_json() + "\n"
        for value in images.assets.values()), encoding="utf-8")
    return prepare_dataset(bundle / "cases.jsonl", bundle / "image_manifest.jsonl", bundle / "cache")[1]


def compile_fixture(root, plan, cases, requested=1, access=None):
    root = Path(root)
    dataset = plan.dataset["dataset_id"]
    if access is None:
        access = prepared_access(root / "dataset", cases)
    store = BIRStore.initialize(root / "BIR", "config/bir_template", "test")
    context = {"bir_path": str(store.root), "run_id": "test", "run_directory": str(root / "runtime"),
               "operation_directory": str(root / "runtime/planning"), "dataset_scope": dataset,
               "selected_dataset_ids": [dataset],
               "dataset_access_manifest": access, "experiment_output_directory": str(root / "outputs"),
               "benchmark_info": {"sample_count": requested, "language": "en"}, "seed": 0,
               "dataset_assets": {dataset: {"access": access}}}
    store.update("benchmark_info.json", context["benchmark_info"], run_id="test", actor="fixture")
    store.update("dataset_bindings.json", {"datasets": {dataset: {"access": DatasetAccessReceipt.bind(dataset, access).model_dump(exclude_none=True)}}}, run_id="test", actor="fixture")
    for key, leaf in plan.leaf_tasks.items():
        summary = summarize_leaf(access, leaf, scenarios=plan.scenarios)
        plan.leaf_tasks[key] = leaf.model_copy(update={"eligible_count": summary["eligible_count"]})
    store.update(f"datasets/{dataset}/planning/task_plan.json", plan.model_dump(mode="json"), run_id="test", actor="fixture",
                 dependencies=business_dependencies(context, "benchmark_info.json", "dataset_bindings.json", registry=True))
    context.update(allocate_benchmark(context)["updates"])
    context.update(run_generation(context)["updates"])
    context.update(audit_candidate(context)["updates"])
    return context


def screen(context, *, scenario_id, decision, reason, validated_clinical_targets=None,
           supporting_fields=None, evidence_basis=None):
    from medbenchagent.agents.task_planning import review_candidate_scenarios
    return review_candidate_scenarios(context, scenario_id=scenario_id, review={
        "decision": decision, "reason": reason, "validated_clinical_targets": validated_clinical_targets or [],
        "supporting_fields": supporting_fields or [],
        "evidence_basis": evidence_basis or [{"reference": f"scenario:{scenario_id}", "detail": reason}],
        "exclusion_category": "unsupported" if decision == "exclude" else None,
    })
