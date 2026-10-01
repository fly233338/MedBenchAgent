from medbenchagent.planning.node_builder import build_task_nodes
from medbenchagent.planning.resolver import ResolverSpec, resolve
from medbenchagent.planning.task_space import AxisAnalysis, AxisAssessment, EvidenceBasis, OutputSpec, TaskParadigm


def _analysis(**supported):
    def values(axis, names):
        allowed = {"input_structure": ("single-image", "multi-image", "multi-view", "volume", "image-text", "paired-study"),
                   "operation": ("recognize", "localize", "quantify", "compare", "align", "describe"),
                   "clinical_target": ("acquisition", "anatomy", "finding", "attribute", "measurement", "assessment", "diagnosis", "quality")} [axis]
        return [AxisAssessment(value=value, supported=value in names,
                               evidence_basis=[EvidenceBasis(reference="case:c1", detail="observed")], reason="observed") for value in allowed]
    return AxisAnalysis(input_structure=values("input_structure", supported.get("input_structure", [])),
                        operation=values("operation", supported.get("operation", [])),
                        clinical_target=values("clinical_target", supported.get("clinical_target", [])),
                        dataset_revision="1")


def test_axis_registry_and_node_ids_are_deterministic():
    analysis = _analysis(input_structure=["single-image"], operation=["recognize"], clinical_target=["finding"])
    first = build_task_nodes(analysis, {"evidence": {}})
    second = build_task_nodes(analysis, {"evidence": {}})
    assert first.keys() == second.keys()
    assert len(first) == 1
    assert next(iter(first.values())).capability_category == "Recognition"


def test_node_constraints_preserve_empty_nodes_with_reason():
    analysis = _analysis(input_structure=["multi-view"], operation=["recognize"], clinical_target=["finding"])
    nodes = build_task_nodes(analysis, {"assets": {"related_image_cases": 0}, "evidence": {}})
    node = next(iter(nodes.values()))
    assert node.exclusion_reason == "multi-view requires related image assets"


def test_resolver_separates_input_asset_from_reference_text():
    case = {"case_id": "c1", "evidence": [
        {"id": "asset", "type": "asset", "name": "image", "value": "a1", "ref": []},
        {"id": "report", "type": "text", "name": "report", "value": "Clear.", "ref": ["asset"]},
    ]}
    spec = ResolverSpec(
        inputs=[{"role": "image", "name": "image", "evidence_type": "asset"}],
        steps=[
            {"op": "read", "selector": {"role": "image", "name": "image", "evidence_type": "asset"}},
            {"op": "reference_text", "selector": {"role": "reference", "name": "report", "evidence_type": "text"}},
        ], output_spec=OutputSpec.TEXT, result_step="step_1",
    )
    result = resolve(case, spec)
    assert result.reference == "Clear." or result.reference.get("value") == "Clear."
    assert result.input_evidence_ids == ["asset"]
    assert result.reference_evidence_ids == ["report"]


def test_task_paradigm_rejects_incompatible_output():
    try:
        from medbenchagent.planning.task_space import TaskLeaf
        TaskLeaf(leaf_id="l", name="l", node_id="n", question="q", input=[{"role": "x"}],
                 target="t", answer="a", resolver={}, output_spec=OutputSpec.BBOX,
                 task_paradigm=TaskParadigm.VQA, capability_category="Recognition")
    except ValueError as exc:
        assert "incompatible" in str(exc)
    else:
        raise AssertionError("incompatible TaskLeaf was accepted")


def test_v5_planner_axis_and_node_tools_persist_bounded_artifacts(tmp_path, monkeypatch):
    from medbenchagent.agents import task_planning

    context = {
        "operation_directory": str(tmp_path),
        "dataset_access_manifest": {"schema_version": "annotation.v5", "revision": "7"},
    }
    monkeypatch.setattr(task_planning, "require", lambda *_args: None)
    monkeypatch.setattr(task_planning, "_profile", lambda _context: {
        "schema_version": "dataset-profile.v5",
        "assets": {"kinds": {"image": 1}, "assets_per_case": {1: 1}},
        "evidence": [{"name": "label", "types": {"value": 1}, "has_ref": 0}],
    })
    analysis = _analysis(input_structure=["single-image"], operation=["recognize"], clinical_target=["finding"]).model_copy(update={"dataset_revision": "7"}).model_dump()
    saved = task_planning.analyze_axes(context, analysis=analysis)
    context.update(saved["updates"])
    built = task_planning.build_nodes(context)
    assert built["updates"]["task_nodes_path"]
    assert built["updates"]["task_nodes"]


def test_save_task_leaf_compiles_resolver_and_counts_capacity(tmp_path, monkeypatch):
    from medbenchagent.agents import task_planning

    context = {"operation_directory": str(tmp_path),
               "dataset_access_manifest": {"schema_version": "annotation.v5", "revision": "7"}}
    monkeypatch.setattr(task_planning, "require", lambda *_args: None)
    monkeypatch.setattr(task_planning, "_profile", lambda _context: {
        "schema_version": "dataset-profile.v5", "assets": {"kinds": {"image": 1}, "assets_per_case": {1: 1}},
        "evidence": [{"name": "report", "types": {"text": 1}, "has_ref": 1}],
    })
    axis = _analysis(input_structure=["single-image"], operation=["describe"], clinical_target=["finding"]).model_copy(update={"dataset_revision": "7"}).model_dump()
    context.update(task_planning.analyze_axes(context, analysis=axis)["updates"])
    context.update(task_planning.build_nodes(context)["updates"])
    node_id = next(key for key, value in context["task_nodes"].items() if not value["exclusion_reason"])

    class FakeQuery:
        def __init__(self, _access): pass
        def sample(self, **_kwargs):
            return {"values": [{"case_id": "c1", "evidence": [
                {"id": "asset", "type": "asset", "name": "image", "value": "a1", "ref": []},
                {"id": "report", "type": "text", "name": "report", "value": "Clear", "ref": ["asset"]},
            ]}]}
    monkeypatch.setattr("medbenchagent.data.query.DatasetQuery", FakeQuery)
    leaf = {"leaf_id": "leaf-1", "name": "report", "node_id": node_id, "question": "Describe the finding.",
            "input": [{"role": "image", "name": "image", "evidence_type": "asset"}], "target": "finding",
            "answer": "Read the paired report", "resolver": {"inputs": [{"role": "image", "name": "image", "evidence_type": "asset"}],
            "steps": [{"op": "read", "selector": {"role": "image", "name": "image", "evidence_type": "asset"}},
                      {"op": "reference_text", "selector": {"role": "reference", "name": "report", "evidence_type": "text"}}],
            "output_spec": "text", "result_step": "step_1"}, "output_spec": "text", "task_paradigm": "Report Generation",
            "capability_category": "Generation"}
    saved = task_planning.save_task_leaf(context, leaf=leaf)
    assert saved["updates"]["task_leaves"]["leaf-1"]["eligible_count"] == 1
