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
    """Existing low-level operator request; agents use SemanticPreparationSpec and ExecutionPlan."""
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


class SemanticDecision(Contract):
    meaning: Text
    inspection_references: list[Text] = Field(min_length=1)
    semantic_evidence: list[PreparationEvidence] = Field(default_factory=list)
    interpretation: Literal["preserve", "interpreted"] = "interpreted"


class FieldMeaning(SemanticDecision):
    target: Text
    source_field: str | None = None
    source_path: str | None = None
    source_sheet: str | None = None
    source_delimiter: str = ","
    transformation: Literal["preserve", "value_mapping", "custom"] = "preserve"
    mapping: list[ValueMapping] = Field(default_factory=list)
    unmapped: Literal["error", "preserve"] = "error"

    @model_validator(mode="after")
    def field_contract(self):
        if not self.target.startswith(tuple(e + "." for e in ("patient", "case", "study", "image"))):
            raise ValueError("Field target must belong to a canonical entity")
        if self.transformation != "custom" and (not self.source_field or not self.source_path):
            raise ValueError("Direct mappings require source_path and source_field")
        if self.transformation == "value_mapping" and not self.mapping:
            raise ValueError("Value mapping requires explicit values")
        if self.transformation != "value_mapping" and self.mapping:
            raise ValueError("Only value_mapping accepts a mapping")
        if self.transformation != "preserve" and self.interpretation != "interpreted":
            raise ValueError("Transformations require interpreted semantic evidence")
        import json
        keys = [json.dumps(item.value, allow_nan=False) for item in self.mapping]
        if len(keys) != len(set(keys)):
            raise ValueError("Semantic value mappings must be unique")
        return self


class SemanticPreparationSpec(Contract):
    schema_version: Literal["semantic-preparation.v1"] = "semantic-preparation.v1"
    population: SemanticDecision
    population_mode: Literal["independent", "human_review"] = "independent"
    population_reference: Text
    expected_images: int | None = Field(default=None, gt=0)
    identities: dict[str, SemanticDecision]
    relationships: list[SemanticDecision] = Field(default_factory=list)
    fields: list[FieldMeaning] = Field(default_factory=list)
    missing: SemanticDecision
    findings: SemanticDecision
    spatial: SemanticDecision
    invariant_fields: list[Text] = Field(default_factory=list)

    @model_validator(mode="after")
    def identities_and_fields(self):
        if set(self.identities) != {"patient", "case", "study", "image"}:
            raise ValueError("Describe all four identity sources; absent optional identities remain null")
        targets = [f.target for f in self.fields]
        if len(targets) != len(set(targets)):
            raise ValueError("Semantic field targets must be unique")
        if any(path not in targets or "[]" in path or not path.startswith(
                ("patient.annotation.", "case.annotation.", "study.annotation.")) for path in self.invariant_fields):
            raise ValueError("Entity invariants require declared scalar patient/case/study annotation fields")
        return self


class SourceFingerprint(Contract):
    schema_version: Literal["source-fingerprint.v1"] = "source-fingerprint.v1"
    dataset_id: Text
    source: Source
    source_artifact: Text
    files_path: Text
    used_files_path: Text
    inspections: list[dict]
    capabilities: list[Text]


class ExecutionPlan(Contract):
    schema_version: Literal["preparation-execution.v1"] = "preparation-execution.v1"
    executor: Literal["built_in", "generated_python"]
    steps: list[Step] = Field(default_factory=list)
    adapter_reference: str | None = None
    reason: str = ""
    identity_import: bool = False

    @model_validator(mode="after")
    def executable(self):
        if self.executor == "generated_python":
            if self.steps or self.identity_import or not self.adapter_reference:
                raise ValueError("Generated Python requires only an adapter reference")
        elif self.adapter_reference or (bool(self.steps) == self.identity_import):
            raise ValueError("Built-in execution requires steps or identity import")
        if self.steps:
            if not isinstance(self.steps[-1], Emit):
                raise ValueError("Built-in steps must end in emit")
            names = set()
            for index, step in enumerate(self.steps):
                if isinstance(step, Emit) and index != len(self.steps) - 1:
                    raise ValueError("Emit must be the final step")
                for key in ("input", "right"):
                    if hasattr(step, key) and getattr(step, key) not in names:
                        raise ValueError("Step references an unknown input relation")
                if hasattr(step, "output"):
                    if step.output in names:
                        raise ValueError("Step outputs must be unique")
                    names.add(step.output)
            if any(isinstance(step, Namespace) for step in self.steps):
                raise ValueError("Identity namespaces are assigned by trusted normalization")
        return self


class SourceLocator(Contract):
    schema_version: Literal["source-locator.v1"] = "source-locator.v1"
    kind: Literal["table", "json", "jsonl", "mask"]
    path: Text
    row: int | None = Field(default=None, ge=1)
    sheet: str | None = None
    delimiter: str = ","
    pointer: str = ""
    foreground: list[int] = Field(default_factory=list)
    instance: int | None = None

    @model_validator(mode="after")
    def address(self):
        if (self.kind in {"table", "jsonl"}) != (self.row is not None):
            raise ValueError("Table/JSONL locators require a positive logical record number")
        if self.pointer and (self.kind not in {"json", "jsonl"} or not self.pointer.startswith("/")):
            raise ValueError("Only JSON locators accept a JSON Pointer")
        if self.kind != "table" and (self.sheet is not None or self.delimiter != ","):
            raise ValueError("Reader options belong to table locators")
        if (self.kind == "mask") != bool(self.foreground):
            raise ValueError("Mask locators require foreground labels")
        if self.instance is not None and (self.kind != "mask" or self.instance not in self.foreground):
            raise ValueError("A mask instance must be one of its foreground labels")
        return self


class FindingReference(Contract):
    target: Text
    locators: list[SourceLocator] = Field(min_length=1)


class ExtractionProvenance(Contract):
    schema_version: Literal["extraction-provenance.v1"] = "extraction-provenance.v1"
    id: Text
    locators: list[SourceLocator] = Field(min_length=1)
    findings: list[FindingReference] = Field(default_factory=list)
    step: int | str


class JsonPopulation(Contract):
    path: Text
    collection: str = ""
    image_id: Text
    image_path: Text
    field: str | None = None
    values: list[Scalar] = Field(default_factory=list)
    keep: bool = True


class PopulationAnswer(Contract):
    decision: Literal["approve", "reject", "clarify"]
    reason: Text


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


class PreparedDatasetCandidate(Contract):
    """Immutable candidate submitted by a Coding Session."""
    candidate_path: Text
    status: Literal["submitted"] = "submitted"
    session_id: Text | None = None
    source_ids: list[Name] = Field(default_factory=list)
    validation_report: str | None = None


class CodingResult(Contract):
    """Business-facing result of one Coding Session turn."""
    status: Literal["submitted", "needs_input", "blocked"]
    conversation_id: Text
    workspace: Text
    candidate: PreparedDatasetCandidate | None = None
    question: str = ""
    diagnostic: str = ""


class DataProcessingResult(Contract):
    """Stable result exposed after deterministic finalization."""
    status: Literal["complete", "needs_input", "blocked"]
    dataset_id: Name
    dataset_entry: dict[str, object] | None = None
    card_path: str | None = None
    annotation_path: str | None = None
    access_manifest: dict[str, object] | None = None
    validation_report: str | None = None


class SourcesArguments(Contract):
    sources: list[Source] = Field(min_length=1)
    dataset_id: Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")]
    feedback: str = ""

    @model_validator(mode="after")
    def unique_sources(self):
        if len({s.id for s in self.sources}) != len(self.sources):
            raise ValueError("Source IDs must be unique within a dataset")
        return self
