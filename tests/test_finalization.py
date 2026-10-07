import json
import shutil
from pathlib import Path
from test_preparation_graph import Worker, state
from medbenchagent.preparation.graph import compile_preparation_graph
from medbenchagent.data.bundle_catalog import verify_bundle
from medbenchagent.data.dataset_catalog import register_dataset, load_prepared


def test_catalog_cache_is_derived_and_no_projection_is_required(tmp_path):
    context = state(tmp_path)
    result = compile_preparation_graph(worker=Worker()).invoke(context)
    bundle = Path(context["bundle_root"])
    revision = Path(result["publication"]["revision_path"])
    entry = register_dataset("d", bundle, catalog_path=tmp_path / "catalog.json")
    assert Path(entry["cases_path"]).parent == revision
    before = {p.relative_to(revision).as_posix():p.read_bytes() for p in revision.rglob("*") if p.is_file()}
    shutil.rmtree(bundle / ".derived")
    card, access = load_prepared(entry)
    assert card["case_count"] == 1
    assert Path(access["profile_path"]).is_relative_to(bundle / ".derived")
    Path(access["profile_path"]).write_text("corrupt derived profile", encoding="utf-8")
    assert verify_bundle(bundle)["card"]["case_count"] == 1
    assert before == {p.relative_to(revision).as_posix():p.read_bytes() for p in revision.rglob("*") if p.is_file()}
    (bundle / "cases.jsonl").write_text("bad projection", encoding="utf-8")
    assert verify_bundle(bundle)["ready"]["access"]["cases_path"] == entry["cases_path"]
    assert json.loads((tmp_path / "catalog.json").read_text())["datasets"][0]["dataset_id"] == "d"


def test_minimal_bundle_drives_planning_generation_and_delivery(tmp_path):
    from medbenchagent.agents import task_planning
    from medbenchagent.agents.finalization import finalize_benchmark
    from medbenchagent.planning.task_space import TaskPlanState, TaskScenario
    from medbenchagent.planning.pipeline import run_pipeline
    from medbenchagent.planning.resolver import ResolverContext
    from medbenchagent.generation.items import write_task_items
    from test_task_input import leaf
    context = state(tmp_path)
    prepared = compile_preparation_graph(worker=Worker()).invoke(context)["publication"]
    access = prepared["access_manifest"]
    operation = tmp_path / "planning"
    initialized = task_planning._initialize_planning_context({
        "dataset_scope": "d", "operation_directory": str(operation), "dataset_access_manifest": access})
    overview = task_planning.read_dataset_overview(initialized)
    assert json.loads(overview["updates"]["overview"])["case_count"] == 1
    assert not (operation / "evidence").exists()
    task = leaf().model_copy(update={"options": ["yes", "no"]})
    plan = TaskPlanState(dataset={"dataset_id": "d"}, phase="complete", leaf_tasks={"l": task}, scenarios={
        "s": TaskScenario(scenario_id="s", input_structure="single-image", operation="recognize", clinical_target="finding",
            capability_category="Recognition", leaf_task_ids=["l"], status="completed",
            validated_clinical_targets=[{"target_id": "finding", "name": "finding", "target_type": "finding",
                "subject": "case", "scope": "case", "reference_kind": "direct", "reference_fields": ["label"]}]),
    })
    cases = [json.loads(line) for line in Path(access["cases_path"]).read_text().splitlines()]
    built = run_pipeline(dataset={"dataset_id": "d"}, plan=plan, cases=cases, requested_count=1,
                         resolver_context=ResolverContext.from_access(access))
    assert built["audit"]["accepted"] == 1
    plan.leaf_tasks["l"] = task.model_copy(update={"eligible_count": 1})
    plan_path = tmp_path / "task_plan.json"
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
    raw = tmp_path / "generated.jsonl"
    write_task_items(raw, built["items"])
    allocation = tmp_path / "allocation.json"
    allocation.write_text(json.dumps({"requested_count": 1, "counts": {"d": {"l": 1}}}), encoding="utf-8")
    finalize_benchmark({"selected_dataset_ids": ["d"], "allocation_path": str(allocation),
        "dataset_task_plans": {"d": str(plan_path)}, "dataset_generations": {"d": {"raw_output_path": str(raw)}},
        "dataset_assets": {"d": {"access": access}}, "operation_directory": str(tmp_path), "seed": 0,
    }, output_directory=tmp_path / "delivery")
    delivered = json.loads((tmp_path / "delivery/benchmark.jsonl").read_text())
    assert Path(delivered["input_images"][0]["path"]).read_bytes() == (tmp_path / "raw/raw.jpg").read_bytes()
