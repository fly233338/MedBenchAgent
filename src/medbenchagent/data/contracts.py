"""Strict, versioned plans for source-supported data preparation."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(min_length=1)]
Name = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_-]*$")]
Scalar = str | int | float | bool | None


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Source(Contract):
    id: Name
    version: Text | None = None
    kind: Literal["raw", "annotation"] = "raw"
    root: Text
    annotation: str = "annotation.jsonl"


class Read(Contract):
    op: Literal["read"] = "read"
    output: Name
    path: Text
    sheet: str | None = None
    delimiter: str = ","


class Unary(Contract):
    input: Name
    output: Name


class Select(Unary):
    op: Literal["select"] = "select"
    columns: list[Text] = Field(min_length=1)


class Filter(Unary):
    op: Literal["filter"] = "filter"
    field: Text
    values: list[Scalar]
    keep: bool = True
    reason: Text


class Rename(Unary):
    op: Literal["rename"] = "rename"
    columns: dict[str, str]


class Cast(Unary):
    op: Literal["cast"] = "cast"
    field: Text
    dtype: Literal["VARCHAR", "BIGINT", "DOUBLE", "BOOLEAN"]


class ValueMapping(Contract):
    value: Scalar
    target: Scalar


class MapValues(Unary):
    op: Literal["value_mapping", "missing_normalization"]
    field: Text
    mapping: list[ValueMapping] = Field(min_length=1)
    unmapped: Literal["error", "preserve"] = "error"


class Join(Unary):
    op: Literal["join"] = "join"
    right: Name
    left_keys: list[Text] = Field(min_length=1)
    right_keys: list[Text] = Field(min_length=1)
    cardinality: Literal["one_to_one", "many_to_one", "one_to_many"]
    how: Literal["left", "inner"] = "left"
    null_keys: Literal["error", "unmatched"] = "error"
    unmatched: Literal["error", "retain", "exclude"] = "error"

    @model_validator(mode="after")
    def matching_keys(self):
        if len(self.left_keys) != len(self.right_keys):
            raise ValueError("Join key lists must have the same length")
        if self.how == "inner" and self.unmatched == "retain":
            raise ValueError("Inner join cannot retain unmatched rows")
        if self.how == "left" and self.unmatched == "exclude":
            raise ValueError("Use inner join to explicitly exclude unmatched rows")
        return self


class Group(Unary):
    op: Literal["group_findings"] = "group_findings"
    keys: list[Text] = Field(min_length=1)
    fields: dict[str, str]
    target: Text


class BBox(Unary):
    op: Literal["bbox"] = "bbox"
    fields: list[Text] = Field(min_length=4, max_length=4)
    target: Text
    convention: Literal["xyxy", "xywh", "cxcywh", "normalized_xyxy"]
    width_field: str | None = None
    height_field: str | None = None


class Mask(Unary):
    op: Literal["mask_to_bbox"] = "mask_to_bbox"
    field: Text
    target: Text
    foreground: list[int] = Field(min_length=1)
    instances: Literal["single", "by_value"]
    width_field: Text
    height_field: Text


class Namespace(Unary):
    op: Literal["namespace_ids"] = "namespace_ids"
    field: Text
    entity: Literal["patient", "case", "study", "image"]


class Emit(Contract):
    op: Literal["emit"] = "emit"
    input: Name
    # Canonical leaf/object paths -> input columns. Null optional fields are filled by the engine.
    fields: dict[str, str]
    constants: dict[str, Scalar] = Field(default_factory=dict)


Step = Annotated[
    Read
    | Select
    | Filter
    | Rename
    | Cast
    | MapValues
    | Join
    | Group
    | BBox
    | Mask
    | Namespace
    | Emit,
    Field(discriminator="op"),
]


class Population(Contract):
    """An inspectable relation of image identities, using existing relational operators."""
    steps: list[Annotated[Read | Select | Rename | Filter | Join, Field(discriminator="op")]] = Field(min_length=1)
    input: Name
    image_id: Text
    image_path: Text

    @model_validator(mode="after")
    def relations(self):
        names = set()
        for step in self.steps:
            for key in ("input", "right"):
                if hasattr(step, key) and getattr(step, key) not in names:
                    raise ValueError("Population references an unknown input relation")
            if step.output in names:
                raise ValueError("Population relation outputs must be unique")
            names.add(step.output)
        if self.input not in names:
            raise ValueError("Population input must be a declared relation")
        return self


class PreparationEvidence(Contract):
    source: Literal["document", "user"]
    reference: Text
    quote: Text


class Mapping(Contract):
    authoritative_population: Text
    expected_images: int = Field(gt=0)
    entity_meanings: dict[str, str]
    reason: Text
    population_reference: Text | None = None
    inspection_references: list[Text] = Field(default_factory=list)
    semantic_evidence: list[PreparationEvidence] = Field(default_factory=list)
    invariant_fields: list[str] = Field(default_factory=list)


class PreparationPlan(Contract):
    schema_version: Literal["preparation.v2"] = "preparation.v2"
    mapping: Mapping
    steps: list[Step] = Field(default_factory=list)
    fallback_reason: str | None = None

    @model_validator(mode="after")
    def executable(self):
        if self.fallback_reason:
            if self.steps:
                raise ValueError("Fallback and operator steps are mutually exclusive")
        elif not self.steps or not isinstance(self.steps[-1], Emit):
            raise ValueError("Operator plans must end in emit")
        names = set()
        for index, step in enumerate(self.steps):
            if isinstance(step, Emit) and index != len(self.steps) - 1:
                raise ValueError("Emit must be the final step")
            for key in ("input", "right"):
                if hasattr(step, key) and getattr(step, key) not in names:
                    raise ValueError(f"Unknown step input: {getattr(step, key)}")
            if hasattr(step, "output"):
                if step.output in names:
                    raise ValueError("Step outputs must be unique")
                names.add(step.output)
        return self


class FieldAlignment(Contract):
    source_id: Name
    field: Text
    action: Literal["merge", "keep_separate", "exclude"]
    target: str | None = None
    mapping: list[ValueMapping] = Field(default_factory=list)
    scale: float | None = None
    offset: float = 0.0
    reason: Text
    inspection_references: list[Text] = Field(default_factory=list)
    semantic_evidence: list[PreparationEvidence] = Field(default_factory=list)


class HarmonizationPlan(Contract):
    schema_version: Literal["harmonization.v2"] = "harmonization.v2"
    fields: list[FieldAlignment] = Field(default_factory=list)

    @model_validator(mode="after")
    def unambiguous(self):
        import json
        import math

        seen = set()
        for field in self.fields:
            key = (field.source_id, field.field)
            if key in seen:
                raise ValueError("Each source field needs exactly one harmonization decision")
            seen.add(key)
            if field.action == "merge" and not field.target:
                raise ValueError("Merge requires a target")
            if field.action != "merge" and (
                field.target or field.mapping or field.scale is not None
            ):
                raise ValueError("Only merge accepts target/value/unit mappings")
            if (
                not math.isfinite(field.offset)
                or field.scale is not None
                and not math.isfinite(field.scale)
            ):
                raise ValueError("Unit transformations must be finite")
            keys = [json.dumps(m.value, allow_nan=False) for m in field.mapping]
            if len(set(keys)) != len(keys):
                raise ValueError("Harmonization value mappings must be unique")
        return self


class QualityIssue(Contract):
    code: Text
    stage: Text
    source_id: Text
    locator: Text
    image_id: str | None = None
    severity: Literal["warning", "error"] = "error"
    action: Literal["report", "quarantine", "block"]
    message: Text


class Provenance(Contract):
    source_id: Text
    source_version: str | None
    original_ids: dict[str, str | None]
    original_image_path: Text
    record_ref: Text
    plan_ref: Text


class SourcesArguments(Contract):
    sources: list[Source] = Field(min_length=1)
    dataset_id: Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")]
    feedback: str = ""

    @model_validator(mode="after")
    def unique_sources(self):
        if len({s.id for s in self.sources}) != len(self.sources):
            raise ValueError("Source IDs must be unique within a dataset")
        return self
