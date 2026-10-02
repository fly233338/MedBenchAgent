from medbenchagent.agents.task_planning import AxisArguments
from medbenchagent.data.task_facts import build_dataset_facts
from medbenchagent.planning.node_builder import build_task_scenarios
from medbenchagent.planning.task_space import (
    AXIS_VALUES,
    AxisAnalysis,
    AxisValueDecision,
    CapabilityCategory,
    EvidenceBasis,
    TaskAxis,
)


def _analysis(supported):
    def values(axis):
        return [
            AxisValueDecision(
                value=value,
                supported=value in supported.get(axis, []),
                evidence_basis=[EvidenceBasis(reference="overview:v2", detail="observed")],
                reason="observed",
            )
            for value in AXIS_VALUES[TaskAxis(axis)]
        ]

    return AxisAnalysis(
        input_structure=values("input_structure"),
        operation=values("operation"),
        clinical_target=values("clinical_target"),
        dataset_revision="7",
    )


def test_axis_tool_schema_is_nested_and_strict():
    schema = AxisArguments.model_json_schema()
    item = schema["$defs"]["AxisValueDecision"]
    assert item["type"] == "object"
    assert set(item["required"]) == {"value", "supported", "evidence_basis", "reason"}
    assert item["additionalProperties"] is False


def test_breast_like_facts_do_not_count_masks_as_multi_image_or_age_as_index():
    assets = {
        "image": {"kind": "image"},
        "mask": {"kind": "image"},
    }
    case = {
        "case_id": "case1",
        "patient_id": None,
        "evidence": [
            {"id": "image-ev", "type": "asset", "name": "ultrasound_image", "value": "image", "ref": []},
            {"id": "mask-ev", "type": "asset", "name": "mask_tumor", "value": "mask", "ref": []},
            {"id": "region", "type": "region", "name": "mask_tumor_region", "value": [0, 0, 1, 1], "ref": ["mask-ev"]},
            {"id": "age", "type": "value", "name": "age", "value": 57, "ref": []},
        ],
    }
    facts = build_dataset_facts([case], assets, revision="7")
    assert facts["patient_count"] == 0
    assert facts["patient_identity"] == "absent"
    assert facts["assets"]["primary_image_assets"] == 1
    assert facts["assets"]["annotation_image_assets"] == 1
    assert facts["assets"]["multi_image_cases"] == 0
    assert facts["evidence"]["index"] == 0


def test_scenario_category_is_derived_from_operation():
    analysis = _analysis({
        "input_structure": ["single-image"],
        "operation": ["align"],
        "clinical_target": ["finding"],
    })
    scenarios = build_task_scenarios(
        analysis,
        {
            "assets": {"primary_image_assets": 2, "related_image_cases": 1},
            "relations": {"related_image_groups": 1},
            "evidence": {"text": 1},
        },
    )
    active = [value for value in scenarios.values() if not value.exclusion_reason]
    assert active
    assert {value.capability_category for value in active} == {CapabilityCategory.INTEGRATION}
