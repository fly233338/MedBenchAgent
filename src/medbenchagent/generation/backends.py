"""Deterministic task-output backends over prepared Case fields, independent of raw formats."""

import hashlib
import json
import math
from typing import Any

from medbenchagent.contracts import BenchmarkItem
from medbenchagent.data.task_data import TaskDataReader
from medbenchagent.planning.contracts import SemanticTaskSpec


BACKENDS = {
    "vqa": {
        "families": ["closed_vqa", "open_vqa"],
        "outputs": ["label", "labels", "number", "text", "index"],
    },
    "report_generation": {"families": ["report_generation"], "outputs": ["text"]},
    "visual_grounding": {"families": ["visual_grounding"], "outputs": ["bbox"]},
    "detection": {"families": ["detection"], "outputs": ["detections"]},
}


def backend_catalog():
    return [
        {
            "backend_id": key,
            **value,
            "input": "Ordered visible TaskData fields; asset fields may contain ordered asset lists",
            "reference": (
                "region object with asset_id, box, coordinate_system, optional unit"
                if key == "visual_grounding"
                else "list of region objects with an additional label"
                if key == "detection"
                else "Prepared value or text; closed VQA requires explicit options"
            ),
            "raw_format_restriction": None,
        }
        for key, value in BACKENDS.items()
    ]


def validate_leaf(task: SemanticTaskSpec, reader: TaskDataReader):
    assembly = task.assembly
    if assembly is None or task.task_data is None:
        raise ValueError("Leaf requires TaskData and an assembly definition")
    if task.capability_category is None:
        raise ValueError("Planner must assign the Leaf capability category")
    if (
        assembly.backend_id != reader.manifest.backend_id
        or task.scenario_id not in reader.manifest.scenario_ids
    ):
        raise ValueError("Leaf backend or Scenario does not match TaskData")
    capability = BACKENDS[assembly.backend_id]
    if (
        task.task_family not in capability["families"]
        or assembly.output_spec not in capability["outputs"]
    ):
        raise ValueError("Backend does not implement this task family/output")
    if task.task_family == "closed_vqa" and (
        assembly.output_spec not in {"label", "labels"} or not assembly.options
    ):
        raise ValueError("Closed VQA requires label(s) and explicit options")
    if task.output_paradigm != assembly.output_spec:
        raise ValueError("Leaf output paradigm and assembly output differ")
    for case in reader.cases:
        if task.scenario_id not in case.scenario_ids:
            continue
        fields = {field.name: field for field in case.fields}
        needed = {
            *assembly.input_fields,
            *assembly.question_fields.values(),
            assembly.reference_field,
        }
        if needed - fields.keys():
            raise ValueError(
                f"Case {case.case_id} is missing Leaf fields: {sorted(needed - fields.keys())}"
            )
        visible = {*assembly.input_fields, *assembly.question_fields.values()}
        if assembly.reference_field in visible or any(
            fields[name].input_visibility != "model_visible" for name in visible
        ):
            raise ValueError("Input/reference leakage or unknown input visibility")
        if any(fields[name].type == "asset" for name in assembly.question_fields.values()):
            raise ValueError(
                "Question interpolation requires prepared text or values, not asset paths"
            )
    return task


def _region(value: Any, asset_ids: set[str], *, label=False):
    if not isinstance(value, dict):
        raise ValueError("Region references require an object")
    if value.get("asset_id") not in asset_ids:
        raise ValueError("Region must anchor a selected input asset")
    coordinates = value.get("box")
    system = value.get("coordinate_system")
    size = 6 if system == "physical_xyzxyz" else 4
    if system not in {"normalized_xyxy", "pixel_xyxy", "physical_xyzxyz"}:
        raise ValueError("Region requires an explicit supported coordinate system")
    if (
        not isinstance(coordinates, list)
        or len(coordinates) != size
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in coordinates)
    ):
        raise ValueError("Invalid region coordinates")
    dimensions = size // 2
    if any(coordinates[i] >= coordinates[i + dimensions] for i in range(dimensions)):
        raise ValueError("Region minima must precede maxima")
    if system == "normalized_xyxy" and any(not 0 <= x <= 1 for x in coordinates):
        raise ValueError("Normalized coordinates must be in [0, 1]")
    if system == "pixel_xyxy" and any(x < 0 for x in coordinates):
        raise ValueError("Pixel coordinates must be nonnegative")
    if system == "physical_xyzxyz" and not value.get("unit"):
        raise ValueError("Physical coordinates require a unit")
    if label and (not isinstance(value.get("label"), str) or not value["label"].strip()):
        raise ValueError("Detection targets require a label")


def assemble_item(task: SemanticTaskSpec, reader: TaskDataReader, case_id: str, seed: int):
    assembly = task.assembly
    case = reader.get_case(case_id)
    if task.scenario_id not in case.scenario_ids:
        raise ValueError("Case is outside the Leaf Scenario")
    fields = {field.name: field for field in case.fields}
    inputs, assets = [], []
    for name in assembly.input_fields:
        field = fields[name]
        if field.input_visibility != "model_visible" or name == assembly.reference_field:
            raise ValueError("Input/reference leakage")
        if field.type == "asset":
            for identifier in field.value if isinstance(field.value, list) else [field.value]:
                inputs.append(
                    {"kind": "asset", "role": name, "value": reader.asset_value(identifier)}
                )
                assets.append(identifier)
        else:
            inputs.append({"kind": field.type, "role": name, "value": field.value})
    parameters = {
        "task_objective": task.objective,
        "options": json.dumps(assembly.options, ensure_ascii=False),
    }
    if len(assets) != len(set(assets)):
        raise ValueError("A prepared asset cannot fill multiple input slots")
    for variable, name in assembly.question_fields.items():
        field = fields[name]
        if (
            field.input_visibility != "model_visible"
            or name == assembly.reference_field
            or field.type == "asset"
        ):
            raise ValueError("Question template leaks reference data or asset paths")
        parameters[variable] = (
            field.value
            if isinstance(field.value, str)
            else json.dumps(field.value, ensure_ascii=False)
        )
    answer = fields[assembly.reference_field].value
    if assembly.label_map:
        answer = (
            [assembly.label_map[str(value)] for value in answer]
            if isinstance(answer, list)
            else assembly.label_map[str(answer)]
        )
    output = assembly.output_spec
    if output == "label" and (
        not isinstance(answer, str)
        or assembly.options is not None
        and answer not in assembly.options
    ):
        raise ValueError("Reference is outside the declared labels")
    if output == "labels" and (
        not isinstance(answer, list)
        or not all(isinstance(v, str) for v in answer)
        or assembly.options is not None
        and set(answer) - set(assembly.options)
    ):
        raise ValueError("Reference is outside the declared label set")
    if output == "number" and (type(answer) not in (int, float) or not math.isfinite(answer)):
        raise ValueError("Reference must be a finite number")
    if output == "text" and not isinstance(answer, str):
        raise ValueError("Reference must be text")
    if output == "index" and (type(answer) is not int or not 0 <= answer < len(inputs)):
        raise ValueError("Reference index is outside ordered inputs")
    if output == "bbox":
        _region(answer, set(assets))
    if output == "detections":
        if not isinstance(answer, list):
            raise ValueError("Detection reference must be a list")
        for target in answer:
            _region(target, set(assets), label=True)
    identity = {"task": task.model_dump(mode="json"), "case_id": case_id, "seed": seed}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    template_index = int(digest[:16], 16) % len(assembly.question_templates)
    return BenchmarkItem(
        item_id=digest,
        dataset_id=task.dataset.dataset_id,
        scenario_id=task.scenario_id,
        leaf_task_id=task.task_id,
        case_id=case_id,
        capability_category=task.capability_category,
        task_paradigm={
            "visual_grounding": "Visual Grounding",
            "report_generation": "Report Generation",
            "detection": "Detection",
        }.get(task.task_family, "VQA"),
        output_spec=output,
        question=assembly.question_templates[template_index].format_map(parameters),
        template_index=template_index,
        template_count=len(assembly.question_templates),
        options=assembly.options,
        inputs=inputs,
        answer=answer,
        input_assets=assets,
        input_evidence_ids=list(
            dict.fromkeys([*assembly.input_fields, *assembly.question_fields.values()])
        ),
        reference_evidence_ids=[assembly.reference_field],
    )
