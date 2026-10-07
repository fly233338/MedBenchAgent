"""Shared sampling enumeration and local Evidence order."""
from typing import Any, Iterable
from medbenchagent.data.manifests import SamplingManifest, SamplingUnit

def case_for_unit(case: dict[str, Any], unit: SamplingUnit) -> dict[str, Any]:
    """Return a shallow Case view with manifest order applied to Evidence."""
    if not unit.ordered_evidence_ids and not unit.view_ids:
        return case
    evidence = list(case.get("evidence", []))
    evidence_order = {identifier: index for index, identifier in enumerate(unit.ordered_evidence_ids)}
    view_order = {identifier: index for index, identifier in enumerate(unit.view_ids)}
    original = {id(item): index for index, item in enumerate(evidence)}

    def key(item):
        if item.get("id") in evidence_order:
            return (0, evidence_order[item["id"]])
        if item.get("view_id") in view_order:
            return (1, view_order[item["view_id"]])
        return (2, original[id(item)])

    return {**case, "evidence": sorted(evidence, key=key)}


def sampling_units(
    cases: list[dict[str, Any]],
    sampling_manifest: SamplingManifest | dict[str, Any] | Iterable[SamplingUnit | dict[str, Any]] | None,
) -> list[tuple[dict[str, Any], SamplingUnit]]:
    by_case = {case.get("case_id"): case for case in cases if isinstance(case.get("case_id"), str)}
    if sampling_manifest is None:
        return [
            (case, SamplingUnit(
                sampling_unit_id=str(case["case_id"]), case_id=str(case["case_id"]), split="unspecified",
                view_ids=[], ordered_evidence_ids=[], panel_id=None, eligible=True, reason=None, order=index,
            ))
            for index, case in enumerate(cases)
            if isinstance(case.get("case_id"), str) and case.get("case_id")
        ]
    if isinstance(sampling_manifest, SamplingManifest):
        units = sampling_manifest.units
    elif isinstance(sampling_manifest, dict):
        units = SamplingManifest.model_validate(sampling_manifest).units
    else:
        units = [SamplingUnit.model_validate(unit) for unit in sampling_manifest]
    result = []
    for unit in sorted(units, key=lambda item: item.order):
        case = by_case.get(unit.case_id)
        if case is not None and unit.eligible:
            result.append((case_for_unit(case, unit), unit))
    return result




def input_evidence_for_unit(rows, unit):
    if unit is None:
        return rows
    # ordered_evidence_ids may freeze reference collections (e.g. argmax),
    # while view_ids independently define the allowed visual input.
    if unit.view_ids:
        return [row for row in rows if row.get("view_id") in unit.view_ids
                or row["type"] != "asset" and row["id"] in unit.ordered_evidence_ids]
    return [row for row in rows if not unit.ordered_evidence_ids or row["id"] in unit.ordered_evidence_ids]
