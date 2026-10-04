"""Typed deterministic Reference Resolver for annotation cases."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from medbenchagent.planning.task_space import OutputSpec, EvidenceSelector


@dataclass
class ResolverContext:
    """Frozen asset access shared by planning, generation and audit."""

    assets: dict[str, dict[str, Any]] = field(default_factory=dict)
    roots: dict[str, Path] = field(default_factory=dict)
    cache: dict[tuple[Any, ...], Any] = field(default_factory=dict)

    @classmethod
    def from_access(cls, access: dict[str, Any] | None) -> "ResolverContext":
        access = access or {}
        manifest_path = access.get("asset_manifest_path")
        if not manifest_path or not Path(manifest_path).is_file():
            return cls()
        from medbenchagent.data.asset_manifest import iter_asset_manifest

        assets = {
            entry.asset_id: entry.model_dump(mode="json")
            for entry in iter_asset_manifest(manifest_path)
        }
        roots: dict[str, Path] = {}
        bundle_root = access.get("bundle_root")
        if not bundle_root and access.get("cases_path"):
            bundle_root = str(Path(access["cases_path"]).resolve().parent)
        if bundle_root:
            roots["source"] = Path(bundle_root).resolve()
            roots["bundle"] = Path(bundle_root).resolve()
        for key in ("source_root", "bundle_root"):
            if access.get(key):
                roots["source_root" if key == "source_root" else "bundle"] = Path(access[key]).resolve()
        for key in ("source_roots", "asset_roots"):
            values = access.get(key)
            if isinstance(values, dict):
                roots.update({str(name): Path(value).resolve() for name, value in values.items()})
        nested = access.get("source_access")
        if isinstance(nested, dict) and nested is not access:
            nested_context = cls.from_access(nested)
            for asset_id, asset in nested_context.assets.items():
                assets.setdefault(asset_id, asset)
            for name, root in nested_context.roots.items():
                roots.setdefault(name, root)
        return cls(assets=assets, roots=roots)

    def _asset_path(self, asset_id: str) -> tuple[dict[str, Any], Path] | None:
        asset = self.assets.get(asset_id)
        if not asset:
            return None
        source_id = str(asset.get("source_id", "source"))
        candidates = []
        for root in (
            self.roots.get(source_id), self.roots.get("source"),
            self.roots.get("source_root"), self.roots.get("bundle"),
            *self.roots.values(),
        ):
            if root is not None and root not in candidates:
                candidates.append(root)
        relative = Path(str(asset.get("source_path", "")))
        if not str(relative) or relative.is_absolute():
            return None
        for root in candidates:
            path = (root / relative).resolve()
            if path.is_relative_to(root) and path.is_file():
                return asset, path
        return None

    def mask_bbox(
        self,
        asset_id: str,
        *,
        channel: str,
        foreground: list[int],
    ) -> tuple[list[float] | None, str | None]:
        key = ("mask_to_bbox", asset_id, channel, tuple(foreground))
        if key in self.cache:
            return self.cache[key]
        located = self._asset_path(asset_id)
        if located is None:
            value = (None, "missing_asset")
            self.cache[key] = value
            return value
        asset, path = located
        from medbenchagent.data.sources import digest_file

        expected_digest = asset.get("file_digest")
        if expected_digest and digest_file(path) != expected_digest:
            value = (None, "asset_digest_mismatch")
            self.cache[key] = value
            return value
        try:
            from PIL import Image
            import numpy as np

            with Image.open(path) as image:
                image.load()
                rgba = image.convert("RGBA")
                array = np.asarray(rgba)
        except Exception:
            value = (None, "mask_read_error")
            self.cache[key] = value
            return value
        if array.ndim != 3 or array.shape[2] != 4:
            value = (None, "mask_invalid_dimensions")
            self.cache[key] = value
            return value
        expected_width = asset.get("width")
        expected_height = asset.get("height")
        if (expected_width and expected_width != array.shape[1]) or (
            expected_height and expected_height != array.shape[0]
        ):
            value = (None, "asset_dimension_mismatch")
            self.cache[key] = value
            return value
        channel_indexes = {"red": 0, "green": 1, "blue": 2, "alpha": 3}
        if channel == "any":
            selected = np.isin(array, foreground).any(axis=2)
        elif channel in channel_indexes:
            selected = np.isin(array[:, :, channel_indexes[channel]], foreground)
        elif channel == "luminance":
            luminance = np.asarray(Image.fromarray(array, mode="RGBA").convert("L"))
            selected = np.isin(luminance, foreground)
        else:
            value = (None, "mask_channel_invalid")
            self.cache[key] = value
            return value
        ys, xs = np.where(selected)
        if not len(xs):
            value = (None, "mask_empty")
            self.cache[key] = value
            return value
        height, width = array.shape[:2]
        bbox = [
            float(xs.min() / width),
            float(ys.min() / height),
            float((xs.max() + 1) / width),
            float((ys.max() + 1) / height),
        ]
        value = (bbox, None)
        self.cache[key] = value
        return value


class ResolvedTaskItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    input_evidence_ids: list[str]
    input_bindings: list[dict[str, Any]]
    reference: Any
    reference_evidence_ids: list[str]


class NotApplicable(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    reason: str = Field(min_length=1)


class ReadStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    op: Literal["read"] = "read"
    selector: EvidenceSelector


class CollectStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    op: Literal["collect"] = "collect"
    selector: EvidenceSelector


class SelectStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    op: Literal["select"] = "select"
    input: str = Field(min_length=1)
    index: int | None = None
    value: Any = None


class CompareStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    op: Literal["compare"] = "compare"
    left: str = Field(min_length=1)
    right: str = Field(min_length=1)
    mode: Literal["equal", "less", "greater", "difference"] = "equal"


class ComputeStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    op: Literal["compute"] = "compute"
    inputs: list[str] = Field(min_length=1)
    operation: Literal["sum", "mean", "difference", "ratio", "count"]


class LocateStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    op: Literal["locate"] = "locate"
    input: str = Field(min_length=1)


class MaskToBBoxStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    op: Literal["mask_to_bbox"] = "mask_to_bbox"
    input: str = Field(min_length=1)
    channel: Literal["red", "green", "blue", "alpha", "luminance", "any"] = "alpha"
    foreground: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_foreground(self):
        if any(type(value) is not int or value < 0 or value > 255 for value in self.foreground):
            raise ValueError("mask_to_bbox foreground values must be integers in [0,255]")
        if len(set(self.foreground)) != len(self.foreground):
            raise ValueError("mask_to_bbox foreground values must be unique")
        return self


class ReferenceTextStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    op: Literal["reference_text"] = "reference_text"
    selector: EvidenceSelector


class MapLabelStep(BaseModel):
    """Deterministically map an encoded scalar label to an explicit output label."""

    model_config = ConfigDict(extra="forbid", strict=True)
    op: Literal["map_label"] = "map_label"
    input: str = Field(min_length=1)
    mapping: dict[str, str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_mapping(self):
        if any(not key.strip() for key in self.mapping):
            raise ValueError("map_label mapping keys must be non-empty encoded labels")
        if any(not value.strip() for value in self.mapping.values()):
            raise ValueError("map_label mapping values must be non-empty strings")
        return self


ResolverStep = Union[
    ReadStep, CollectStep, SelectStep, CompareStep, ComputeStep,
    LocateStep, MaskToBBoxStep, ReferenceTextStep, MapLabelStep,
]


class ResolverSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inputs: list[EvidenceSelector] = Field(min_length=1)
    steps: list[ResolverStep] = Field(min_length=1)
    output_spec: OutputSpec
    result_step: str = Field(
        min_length=1,
        description="ID of the result step, using positional names step_0, step_1, step_2, ...",
    )

    @model_validator(mode="after")
    def compile(self):
        roles = {s.role for s in self.inputs}
        if len(roles) != len(self.inputs):
            raise ValueError("Input roles must be unique")
        seen_roles = set()
        for selector in self.inputs:
            if selector.evidence_type not in {"asset", "text", "value", "region"} or not selector.name:
                raise ValueError(
                    "Resolver inputs require explicit asset/text/value/region names "
                    "and any input relation must reference an earlier input role"
                )
            if selector.relation is not None and selector.relation not in seen_roles:
                raise ValueError(
                    "Input relation must reference an earlier input role; "
                    "Resolver inputs require explicit asset/text/value/region names"
                )
            seen_roles.add(selector.role)
        seen = set()
        step_types: dict[str, str | None] = {}
        collection_steps: set[str] = set()
        for index, step in enumerate(self.steps):
            dependencies = ([step.input] if isinstance(step, (SelectStep, LocateStep, MaskToBBoxStep, MapLabelStep)) else
                            [step.left, step.right] if isinstance(step, CompareStep) else
                            step.inputs if isinstance(step, ComputeStep) else [])
            if not set(dependencies) <= seen:
                raise ValueError("Resolver steps must reference earlier steps")
            if isinstance(step, SelectStep) and ((step.index is None) == (step.value is None)):
                raise ValueError("Select requires exactly one index or value")
            if isinstance(step, ComputeStep) and step.operation in {"difference", "ratio"} and len(step.inputs) != 2:
                raise ValueError("Binary compute requires two inputs")
            if hasattr(step, "selector") and step.selector.relation and step.selector.relation not in roles:
                raise ValueError("Reference relation must name a model input role")
            key = f"step_{index}"
            if isinstance(step, (ReadStep, CollectStep, ReferenceTextStep)):
                step_types[key] = step.selector.evidence_type
                if isinstance(step, CollectStep):
                    collection_steps.add(key)
            elif isinstance(step, SelectStep):
                step_types[key] = step_types.get(step.input)
            elif isinstance(step, CompareStep):
                step_types[key] = "value"
            elif isinstance(step, ComputeStep):
                step_types[key] = "value"
            elif isinstance(step, LocateStep):
                input_type = step_types.get(step.input)
                if input_type not in {"region", "index"} or step.input in collection_steps:
                    raise ValueError("locate requires a region or index reference")
                step_types[key] = "bbox" if input_type == "region" else "index"
            elif isinstance(step, MaskToBBoxStep):
                if step_types.get(step.input) != "asset" or step.input in collection_steps:
                    raise ValueError("mask_to_bbox requires a preceding asset reference")
                step_types[key] = "bbox"
            elif isinstance(step, MapLabelStep):
                step_types[key] = "label"
            seen.add(key)
        if self.result_step not in seen:
            available = ", ".join(sorted(seen))
            raise ValueError(
                f"Unknown result step {self.result_step!r}; "
                f"result_step must be one of: {available}"
            )
        result_type = step_types.get(self.result_step)
        if self.output_spec == OutputSpec.BBOX and result_type not in {"bbox", "region"}:
            raise ValueError("bbox output requires a region, locate or mask_to_bbox result")
        if self.output_spec == OutputSpec.INDEX and result_type != "index":
            raise ValueError("index output requires an index reference")
        return self

    @classmethod
    def parse(cls, value: Any) -> "ResolverSpec":
        return cls.model_validate(value)


def validate_resolver_policy(spec: ResolverSpec | dict[str, Any], policy: dict[str, Any]) -> ResolverSpec:
    """Apply the experiment Registry policy after structural parsing."""
    parsed = ResolverSpec.parse(spec)
    allowed_steps = set(policy.get("step_operations", []))
    allowed_compute = set(policy.get("compute_operations", []))
    allowed_compare = set(policy.get("compare_modes", []))
    allowed_types = set(policy.get("input_evidence_types", []))
    for selector in parsed.inputs:
        if selector.evidence_type not in allowed_types:
            raise ValueError(f"Resolver input evidence type is disabled by the Registry: {selector.evidence_type}")
    for step in parsed.steps:
        if step.op not in allowed_steps:
            raise ValueError(f"Resolver operation is disabled by the Registry: {step.op}")
        if isinstance(step, ComputeStep) and step.operation not in allowed_compute:
            raise ValueError(f"Resolver compute operation is disabled by the Registry: {step.operation}")
        if isinstance(step, CompareStep) and step.mode not in allowed_compare:
            raise ValueError(f"Resolver compare mode is disabled by the Registry: {step.mode}")
    return parsed


def _evidence(case: dict[str, Any], selector: EvidenceSelector, roles=None) -> list[dict[str, Any]]:
    rows = []
    for item in case.get("evidence", []):
        if selector.name is not None and item.get("name") != selector.name:
            continue
        if selector.evidence_type is not None and item.get("type") != selector.evidence_type:
            continue
        if selector.unit is not None and item.get("unit") != selector.unit:
            continue
        if selector.relation is not None and not set(item.get("ref", [])) & set((roles or {}).get(selector.relation, [])):
            continue
        rows.append(item)
    return rows


def _scalar(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Resolver requires finite numeric evidence")
    return float(value)


def validate_output(value, spec):
    """The same strict output boundary is used by resolution, generation and audit."""
    valid = {
        OutputSpec.LABEL: lambda: type(value) in (str, int, bool) and value != "",
        OutputSpec.LABELS: lambda: isinstance(value, list) and all(type(x) is str and x for x in value) and len(set(value)) == len(value),
        OutputSpec.NUMBER: lambda: type(value) in (int, float) and math.isfinite(value),
        OutputSpec.TEXT: lambda: isinstance(value, str) and bool(value.strip()),
        OutputSpec.BBOX: lambda: isinstance(value, list) and len(value) == 4 and all(type(x) in (int, float) and math.isfinite(x) for x in value) and 0 <= value[0] < value[2] <= 1 and 0 <= value[1] < value[3] <= 1,
        OutputSpec.INDEX: lambda: type(value) is int and value >= 0,
    }
    if not valid[OutputSpec(spec)]():
        raise ValueError("Invalid reference for OutputSpec")
    return value


def encode_closed_label(value: Any) -> str:
    """Encode a scalar Resolver result for a string-valued Closed VQA domain."""
    if isinstance(value, str):
        if not value:
            raise ValueError("Closed VQA labels must be non-empty strings")
        return value
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) in (int, float):
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        if not isinstance(encoded, str):
            raise ValueError("Closed VQA label encoding failed")
        return encoded
    raise ValueError("Closed VQA labels require scalar string, boolean or numeric answers")


def _value(value):
    if isinstance(value, dict):
        return value["value"]
    if isinstance(value, list):
        return [_value(item) for item in value]
    return value


def resolve(
    case: dict[str, Any],
    resolver_spec: ResolverSpec | dict[str, Any],
    *,
    resolver_context: ResolverContext | None = None,
) -> ResolvedTaskItem | NotApplicable:
    spec = ResolverSpec.parse(resolver_spec)
    values: dict[str, Any] = {}
    value_refs: dict[str, set[str]] = {}
    roles: dict[str, list[str]] = {}
    input_bindings: list[dict[str, Any]] = []
    input_ids: list[str] = []
    asset_ids: set[str] = set()
    for selector in spec.inputs:
        rows = _evidence(case, selector, roles)
        ids = [item["id"] for item in rows]
        if not ids:
            return NotApplicable(reason="missing_input_evidence")
        if selector.evidence_type == "asset":
            if asset_ids.intersection(ids):
                return NotApplicable(reason="duplicate_input_asset")
            asset_ids.update(ids)
        roles[selector.role] = ids
        input_bindings.append({"role": selector.role, "evidence_ids": ids})
        input_ids.extend(ids)
    for index, step in enumerate(spec.steps):
        key = f"step_{index}"
        if isinstance(step, (ReadStep, CollectStep, ReferenceTextStep)):
            rows = _evidence(case, step.selector, roles)
            if not rows:
                return NotApplicable(reason=f"missing_{step.op}_evidence")
            if not isinstance(step, CollectStep) and len(rows) != 1:
                return NotApplicable(reason="multiple_matches")
            if isinstance(step, ReferenceTextStep) and any(row["type"] != "text" for row in rows):
                return NotApplicable(reason="invalid_reference_text")
            values[key] = rows if isinstance(step, CollectStep) else rows[0]
            value_refs[key] = {row["id"] for row in rows}
        elif isinstance(step, SelectStep):
            source = values.get(step.input)
            if not isinstance(source, list) or not source:
                return NotApplicable(reason="select_source_missing")
            selected = source[step.index] if step.index is not None and -len(source) <= step.index < len(source) else None
            if step.value is not None:
                matches = [x for x in source if _value(x) == step.value]
                selected = matches[0] if len(matches) == 1 else None
            if selected is None:
                return NotApplicable(reason="select_no_match")
            values[key] = selected
            value_refs[key] = {selected["id"]} if isinstance(selected, dict) and "id" in selected else set(value_refs.get(step.input, set()))
        elif isinstance(step, CompareStep):
            left, right = values.get(step.left), values.get(step.right)
            if left is None or right is None:
                return NotApplicable(reason="compare_input_missing")
            lv, rv = _value(left), _value(right)
            try:
                if step.mode == "equal":
                    values[key] = type(lv) is type(rv) and lv == rv
                elif step.mode == "less":
                    values[key] = _scalar(lv) < _scalar(rv)
                elif step.mode == "greater":
                    values[key] = _scalar(lv) > _scalar(rv)
                else:
                    values[key] = _scalar(lv) - _scalar(rv)
            except (TypeError, ValueError):
                return NotApplicable(reason="compare_invalid_values")
            value_refs[key] = value_refs.get(step.left, set()) | value_refs.get(step.right, set())
        elif isinstance(step, ComputeStep):
            try:
                numbers = [_scalar(_value(values[name])) for name in step.inputs] if step.operation != "count" else []
            except (KeyError, TypeError, ValueError):
                return NotApplicable(reason="compute_invalid_values")
            if step.operation == "sum":
                result = sum(numbers)
            elif step.operation == "mean":
                result = sum(numbers) / len(numbers)
            elif step.operation == "difference":
                result = numbers[0] - numbers[1]
            elif step.operation == "ratio":
                if numbers[1] == 0:
                    return NotApplicable(reason="compute_division_by_zero")
                result = numbers[0] / numbers[1]
            else:
                result = sum(len(values[name]) if isinstance(values[name], list) else 1 for name in step.inputs)
            values[key] = result
            value_refs[key] = set().union(*(value_refs.get(name, set()) for name in step.inputs))
        elif isinstance(step, LocateStep):
            source = values.get(step.input)
            values[key] = source.get("value", source) if source is not None else None
            value_refs[key] = set(value_refs.get(step.input, set()))
        elif isinstance(step, MaskToBBoxStep):
            source = values.get(step.input)
            if not isinstance(source, dict) or source.get("type") != "asset":
                return NotApplicable(reason="mask_to_bbox_input_invalid")
            if resolver_context is None:
                return NotApplicable(reason="missing_asset_context")
            bbox, reason = resolver_context.mask_bbox(
                str(source.get("value")),
                channel=step.channel,
                foreground=step.foreground,
            )
            if bbox is None:
                return NotApplicable(reason=reason or "mask_to_bbox_failed")
            values[key] = bbox
            value_refs[key] = set(value_refs.get(step.input, set()))
        elif isinstance(step, MapLabelStep):
            source = values.get(step.input)
            if source is None:
                return NotApplicable(reason="map_label_input_missing")
            value = _value(source)
            try:
                encoded = encode_closed_label(value)
            except ValueError:
                return NotApplicable(reason="map_label_input_not_scalar")
            if encoded not in step.mapping:
                return NotApplicable(reason="map_label_unmapped")
            values[key] = step.mapping[encoded]
            value_refs[key] = set(value_refs.get(step.input, set()))
        if values.get(key) is None:
            return NotApplicable(reason=f"{step.op}_not_applicable")
    reference = values.get(spec.result_step)
    if reference is None:
        return NotApplicable(reason="result_missing")
    ref_ids = sorted(value_refs.get(spec.result_step, set()))
    if isinstance(reference, dict) and "id" in reference:
        ref_ids = sorted(set(ref_ids) | {reference["id"]})
    if set(input_ids) & set(ref_ids):
        return NotApplicable(reason="input_reference_leakage")
    by_id = {row["id"]: row for row in case["evidence"]}
    def ancestors(identifier, seen=None):
        seen = set() if seen is None else seen
        if identifier in seen:
            return seen
        seen.add(identifier)
        for parent in by_id[identifier].get("ref", []):
            ancestors(parent, seen)
        return seen
    if any(ancestors(identifier) & set(ref_ids) for identifier in input_ids):
        return NotApplicable(reason="input_reference_leakage")
    units = {by_id[identifier].get("unit") for identifier in ref_ids}
    if any(isinstance(step, (ComputeStep, CompareStep)) for step in spec.steps) and len(units) > 1:
        return NotApplicable(reason="incompatible_units")
    try:
        reference = validate_output(_value(reference), spec.output_spec)
    except ValueError:
        return NotApplicable(reason="invalid_output")
    return ResolvedTaskItem(
        input_evidence_ids=input_ids,
        input_bindings=input_bindings,
        reference=reference,
        reference_evidence_ids=ref_ids,
    )

