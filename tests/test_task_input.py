from image_fixtures import image_context
from medbenchagent.data.manifests import SamplingManifest, SamplingUnit
from medbenchagent.planning.resolver import NotApplicable, resolve_task_input
from medbenchagent.planning.task_space import LeafTask
from medbenchagent.generation.items import generate_task_items
from medbenchagent.audit.task_items import audit_task_items


def leaf():
    return LeafTask.model_validate({
        "leaf_task_id": "l", "name": "label", "scenario_id": "s", "clinical_target_id": "finding",
        "question_templates": ["Finding?"], "input": [{"role": "image", "name": "image", "evidence_type": "asset"}],
        "target": "finding", "reference_rule": "read label", "output_spec": "label",
        "task_paradigm": "VQA", "capability_category": "Recognition",
        "resolver": {"inputs": [{"role": "image", "name": "image", "evidence_type": "asset"}],
                     "steps": [{"op": "read", "selector": {"role": "label", "name": "label", "evidence_type": "value"}}],
                     "output_spec": "label", "result_step": "step_0"},
    })


def test_sampling_units_select_views_and_all_consumers_agree(tmp_path):
    cases = [{"case_id": key, "evidence": [
        {"id": "i1", "type": "asset", "name": "image", "value": key + "1", "view_id": "v1"},
        {"id": "i2", "type": "asset", "name": "image", "value": key + "2", "view_id": "v2"},
        {"id": "label", "type": "value", "name": "label", "value": "yes"},
    ]} for key in ("a", "b")]
    manifest = SamplingManifest(dataset_id="d", revision="r", units=[
        SamplingUnit(sampling_unit_id=f"{key}-{view}", case_id=key, split="test", eligible=True,
                     view_ids=[view], ordered_evidence_ids=[evidence], order=order)
        for order, (key, view, evidence) in enumerate([("a", "v2", "i2"), ("a", "v1", "i1"), ("b", "v1", "i1")])
    ])
    context = image_context(tmp_path, cases)
    context.manifests["sampling"] = manifest
    task = leaf()
    capacity = sum(not isinstance(resolve_task_input(next(c for c in cases if c["case_id"] == u.case_id), task, u, context), NotApplicable)
                   for u in manifest.units)
    items = generate_task_items(cases, [task], "d", resolver_context=context, sampling_manifest=manifest)
    assert capacity == len(items) == 3
    assert [row["input_assets"] for row in items] == [["a2"], ["a1"], ["b1"]]
    assert audit_task_items(cases, items, [task], context)["accepted"] == 3
    items[0]["input_images"][0]["path"] = "wrong.png"
    assert audit_task_items(cases, items, [task], context)["rejected"] == 1


def test_duplicate_asset_and_missing_or_corrupt_image_are_ineligible(tmp_path):
    case = {"case_id": "c", "evidence": [
        {"id": "i", "type": "asset", "name": "image", "value": "a"},
        {"id": "label", "type": "value", "name": "label", "value": "yes"},
    ]}
    context = image_context(tmp_path, [case])
    assert not isinstance(resolve_task_input(case, leaf(), None, context), NotApplicable)
    (tmp_path / "a").write_bytes(b"corrupt")
    assert resolve_task_input(case, leaf(), None, context).reason == "invalid_input_image"
    case["evidence"].append({"id": "alias", "type": "asset", "name": "image", "value": "a"})
    assert resolve_task_input(case, leaf(), None, context).reason == "duplicate_input_asset"


def test_volume_view_ids_cannot_duplicate_the_same_image(tmp_path):
    case = {"case_id": "c", "evidence": [
        {"id": "i1", "type": "asset", "name": "image", "value": "volume", "view_id": "v1"},
        {"id": "i2", "type": "asset", "name": "image", "value": "volume", "view_id": "v2"},
        {"id": "label", "type": "value", "name": "label", "value": "yes"},
    ]}
    context = image_context(tmp_path, [case])
    context.assets["volume"].update(kind="volume", format="nifti")
    unit = SamplingUnit(sampling_unit_id="u", case_id="c", split="test", eligible=True,
                        view_ids=["v2", "v1"], ordered_evidence_ids=["i2", "i1"], order=0)
    manifest = SamplingManifest(dataset_id="d", revision="r", units=[unit])
    context.manifests["sampling"] = manifest
    items = generate_task_items([case], [leaf()], "d", resolver_context=context, sampling_manifest=manifest)
    assert items == []
    assert resolve_task_input(case, leaf(), unit, context).reason == "duplicate_input_asset"


def test_hidden_calibration_and_physical_units_are_not_image_inputs(tmp_path):
    case = {"case_id":"c", "evidence":[
        {"id":"i", "type":"asset", "name":"image", "value":"image.png"},
        {"id":"l", "type":"value", "name":"label", "value":"yes", "unit":"mm"}]}
    context = image_context(tmp_path, [case])
    assert resolve_task_input(case, leaf(), None, context).reason == "physical_calibration_not_visible"
    case["evidence"][1].pop("unit")
    context.assets["image.png"]["metadata"]["semantic_role"] = "calibration"
    assert resolve_task_input(case, leaf(), None, context).reason == "reference_only_asset_input"


def test_sampling_evidence_cannot_override_allowed_view(tmp_path):
    case = {"case_id":"c", "evidence":[
        {"id":"i1", "type":"asset", "name":"image", "value":"one.png", "view_id":"v1"},
        {"id":"i2", "type":"asset", "name":"image", "value":"two.png", "view_id":"v2"},
        {"id":"l", "type":"value", "name":"label", "value":"yes"}]}
    context = image_context(tmp_path, [case])
    unit = SamplingUnit(sampling_unit_id="u", case_id="c", split="test", eligible=True,
        view_ids=["v1"], ordered_evidence_ids=["i2"], order=0)
    assert isinstance(resolve_task_input(case, leaf(), unit, context), NotApplicable)
