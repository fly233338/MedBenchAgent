"""Dataset content identity and closed-VQA execution contract checks."""

import json

import pytest

from medbenchagent.compilation.artifacts import resolve_artifact
from medbenchagent.compilation.demo import create_synthetic_inputs
from medbenchagent.data.dataset import FileDatasetReader, snapshot_dataset
from medbenchagent.generation.contracts import GenerationSpec, validate_generation_spec
from medbenchagent.planning.contracts import TaskPlan, SemanticTaskSpec, TaskExclusion
from medbenchagent.planning.semantic import build_scenarios, validate_plan


def feasible_plan(reader, digest):
    manifest = reader.get_manifest()
    dataset = {"dataset_id": manifest.dataset_id, "revision": manifest.revision}
    refs = [{"case_id": "case-a", "evidence_id": "value"}]
    inputs = [{"case_id": "case-a", "evidence_id": "image"}]
    plan = TaskPlan(
        plan_id="p",
        dataset=dataset,
        dataset_digest=digest,
        axis_assessments={
            axis: [{"value": value, "supported": True, "reason": "Evidence", "evidence_refs": refs}]
            for axis, value in (
                ("input_structure", "single-image"),
                ("operation", "recognition"),
                ("clinical_target", "number"),
            )
        },
    )
    build_scenarios(plan)
    scenario = next(iter(plan.scenarios.values()))
    scenario.decision = "keep"
    scenario.reason = "Evidence supports it"
    scenario.evidence_refs = plan.axis_assessments["operation"][0].evidence_refs
    plan.tasks["task"] = SemanticTaskSpec(
        task_id="task",
        dataset=dataset,
        scenario_id=scenario.scenario_id,
        task_family="closed_vqa",
        objective="Read a source label",
        input_requirements=["Image"],
        reference_requirements=["Value"],
        output_paradigm="label",
        evidence_refs=refs,
        input_evidence_refs=inputs,
        reference_evidence_refs=refs,
        feasibility="feasible",
    )
    return validate_plan(plan, reader, publish=True)


def test_frozen_dataset_members_and_portable_snapshot(tmp_path):
    ref = create_synthetic_inputs(tmp_path / "original", "data")[0]
    source = resolve_artifact(tmp_path / "original", ref)
    copied = snapshot_dataset(source, tmp_path / "copy")
    reader = FileDatasetReader(copied)
    mtimes = {path: path.stat().st_mtime_ns for path in copied.parent.rglob("*") if path.is_file()}
    assert snapshot_dataset(source, copied.parent) == copied
    assert all(path.stat().st_mtime_ns == time for path, time in mtimes.items())
    assert reader.get_case("case-a").case_id == "case-a"
    for relative in (
        "cases.jsonl",
        "mapping.json",
        "sources/raw/image-a.png",
        "sources/raw/source.json",
    ):
        path = copied.parent / relative
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with pytest.raises(ValueError, match="content changed"):
            reader.get_case("case-a")
        path.write_bytes(original)
    (source.parent / "source.json").write_text("changed", encoding="utf-8")
    assert reader.get_case("case-a").case_id == "case-a"
    raw = json.loads(copied.read_text())
    raw["members"].pop()
    copied.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="cover exactly"):
        FileDatasetReader(copied)


def test_generation_spec_rejects_stale_bindings_leaks_and_invalid_closed_labels(tmp_path):
    ref = create_synthetic_inputs(tmp_path, "generation")[0]
    reader = FileDatasetReader(resolve_artifact(tmp_path, ref))
    plan = feasible_plan(reader, ref.digest)
    spec = GenerationSpec(
        task_plan={"artifact_id": "plan", "digest": "a" * 64},
        task_id="task",
        task_revision=1,
        dataset=plan.dataset,
        dataset_digest=ref.digest,
        case_ids=["case-a", "case-b"],
        inputs=[{"name": "view", "evidence_type": "asset"}],
        reference_field="value",
        reference_rule="label_map",
        label_map={"2": "low", "5": "high"},
        templates=["Choose the label. {options}"],
        options=["low", "high"],
        sample_count=2,
        seed=0,
    )
    kwargs = dict(plan=plan, plan_artifact_id="plan", plan_digest="a" * 64, reader=reader)
    assert validate_generation_spec(spec, **kwargs) == spec
    for update, error in (
        ({"task_revision": 2}, "revision"),
        ({"dataset_digest": "b" * 64}, "Dataset"),
        ({"reference_field": "view"}, "Reference field"),
        ({"label_map": {"2": "low"}}, "outside the closed options"),
        ({"inputs": [{"name": "value", "evidence_type": "value"}]}, "input evidence"),
    ):
        invalid = GenerationSpec.model_validate({**spec.model_dump(), **update})
        with pytest.raises(ValueError, match=error):
            validate_generation_spec(invalid, **kwargs)
    with pytest.raises(ValueError, match="interpolate"):
        GenerationSpec.model_validate({**spec.model_dump(), "templates": ["Answer is {reference}"]})
    plan.exclusions["task"] = TaskExclusion(
        reason="Excluded", evidence_refs=plan.tasks["task"].evidence_refs
    )
    with pytest.raises(ValueError, match="not executable"):
        validate_generation_spec(spec, **kwargs)
