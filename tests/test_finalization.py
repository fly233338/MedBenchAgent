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
    from medbenchagent.planning.task_space import TaskPlanState, TaskScenario
    from compilation_fixtures import compile_fixture, planning_context
    from test_task_input import leaf
    context = state(tmp_path)
    prepared = compile_preparation_graph(worker=Worker()).invoke(context)["publication"]
    _, access = load_prepared(prepared["dataset_entry"])
    operation = tmp_path / "planning"
    initialized = task_planning._initialize_planning_context(planning_context(operation, access=access))
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
    compiled = compile_fixture(tmp_path / "compiled", plan, cases, access=access)
    delivered = json.loads((Path(compiled["candidate_path"]) / "benchmark.jsonl").read_text())
    assert Path(delivered["inputs"][0]["value"]["path"]).read_bytes() == (tmp_path / "raw/raw.jpg").read_bytes()
