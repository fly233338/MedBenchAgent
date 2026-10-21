"""Fixtures exercising the production compilation stages with real local image assets."""
import json
from pathlib import Path

from image_fixtures import image_context
from medbenchagent.planning.allocation import allocate_benchmark
from medbenchagent.generation.items import run_generation
from medbenchagent.data.index import prepare_dataset
from medbenchagent.data.models import ImageEntry
from medbenchagent.generation.delivery import audit_candidate
from medbenchagent.planning.bir import BIRStore, business_dependencies
from medbenchagent.planning.task_discovery import summarize_leaf


def prepared_access(bundle, cases, dataset_id="d", revision="7"):
    bundle = Path(bundle)
    images = image_context(bundle, cases)
    (bundle / "cases.jsonl").write_text("".join(json.dumps(row) + "\n" for row in cases), encoding="utf-8")
    (bundle / "image_manifest.jsonl").write_text("".join(ImageEntry.model_validate(value).model_dump_json() + "\n"
        for value in images.assets.values()), encoding="utf-8")
    return prepare_dataset(bundle / "cases.jsonl", bundle / "image_manifest.jsonl", bundle / "cache", dataset_id=dataset_id, revision=revision)[1]


def registry():
    from medbenchagent.planning.bir import RegistrySnapshot
    return RegistrySnapshot.from_directory(Path(__file__).resolve().parents[1] / "config/bir_template/registry")


def axis_values(axis):
    return registry().axis_values(str(axis))


def default_cases():
    return [{"case_id": "case", "evidence": [
        {"id": "image", "name": "image", "type": "asset", "value": "image.png", "ref": []},
        {"id": "report", "name": "report", "type": "text", "value": "Clear.", "ref": ["image"]},
        {"id": "label", "name": "label", "type": "value", "value": "yes", "ref": []}]}]


def planning_context(root, *, plan=None, cases=None, dataset_id="d", brief=None, access=None, template="config/bir_template"):
    root = Path(root)
    dataset = plan.dataset["dataset_id"] if plan is not None else dataset_id
    if not (root / "BIR").is_dir():
        access = access if access is not None else prepared_access(root / "dataset", cases or default_cases(), dataset)
        store = BIRStore.initialize(root / "BIR", template, "test")
        info = {"sample_count": 1, "language": "en", "user_requirement": f"Compile {dataset}",
                "sample_scope": [], "allocation_preferences": [], **(brief or {})}
        store.update("benchmark_info.json", info, run_id="test", actor="fixture")
        store.update("dataset_binding.json", {"schema_version": "dataset-binding", "access": access}, run_id="test", actor="fixture")
    else:
        store = BIRStore(root / "BIR")
        dataset = store.load("dataset_binding.json")["access"]["dataset_id"]
    context = {"bir_path": str(store.root), "run_id": "test", "run_directory": str(root / "runtime"),
               "operation_directory": str(root / "runtime/planning"), "dataset_scope": dataset,
               "experiment_output_directory": str(root / "outputs"), "seed": 0,
               "construction_events_path": str(root / "runtime/events.jsonl"),
               "question_template_count": 1, "benchmark_input_policy": "image-only", "user_requirement": f"Compile {dataset}"}
    Path(context["operation_directory"]).mkdir(parents=True, exist_ok=True)
    if plan is not None:
        store.update("task_plan.json", plan.model_dump(mode="json"), run_id="test", actor="fixture",
                     dependencies=business_dependencies(context, "benchmark_info.json", "dataset_binding.json", "planning_extensions.json", registry=True))
    return context


def write_plan(root, plan):
    return Path(planning_context(root, plan=plan)["bir_path"]) / "task_plan.json"


def compile_fixture(root, plan, cases, requested=1, access=None):
    context = planning_context(root, cases=cases, dataset_id=plan.dataset["dataset_id"], access=access,
                               brief={"sample_count": requested})
    store = BIRStore(context["bir_path"])
    access = store.load("dataset_binding.json")["access"]
    for key, leaf in plan.leaf_tasks.items():
        summary = summarize_leaf(access, leaf, scenarios=plan.scenarios)
        plan.leaf_tasks[key] = leaf.model_copy(update={"eligible_count": summary["eligible_count"]})
    planning_context(root, plan=plan)
    allocate_benchmark(context)
    run_generation(context)
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
