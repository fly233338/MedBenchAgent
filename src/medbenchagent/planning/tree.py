"""Validated two-level planning objects; identities never encode execution behavior."""

from typing import Annotated, Literal, TypedDict

from pydantic import Field, ValidationError, model_validator

from medbenchagent.planning.protocol import Protocol
from medbenchagent.planning.scope import ScopeResolution
from medbenchagent.runtime.tool_contracts import NoArguments, NonEmpty
from medbenchagent.answer_mapping import canonical_answer_scalar


class Evidence(NoArguments):
    pdf: NonEmpty
    page: int = Field(ge=1)
    chunk_id: NonEmpty


class SemanticBasis(NoArguments):
    field: NonEmpty
    interpretation: NonEmpty
    source: Literal["material", "document", "user"]
    reference: NonEmpty
    quote: NonEmpty
    dataset_specific: bool = False


class MappingGrounding(NoArguments):
    field: NonEmpty
    raw_value: str | int | float | bool
    option: NonEmpty
    mode: Literal["direct", "interpreted"]
    interpretation: NonEmpty
    evidence_references: list[NonEmpty] = Field(default_factory=list)


def validate_mapping_grounding(protocol, grounding, basis):
    """Require exact coverage; source meaning is still judged by the Planning review."""
    records = [MappingGrounding.model_validate(item).model_dump() for item in grounding]
    mappings = protocol["answer_contract"].get("value_mapping", [])
    expected = {(canonical_answer_scalar(m["raw_value"]), m["option"]) for m in mappings}
    actual = {(canonical_answer_scalar(m["raw_value"]), m["option"]) for m in records}
    if actual != expected or len(records) != len(expected):
        raise ValueError("Mapping grounding must cover every raw_value-to-option mapping exactly once")
    field = protocol["data_binding"]["answer_field"]
    for item in records:
        if item["field"] != field:
            raise ValueError("Mapping grounding must reference the answer field")
        if item["mode"] == "direct":
            value = item["raw_value"]
            if (not isinstance(value, str) or " ".join(value.split()).casefold() != " ".join(item["option"].split()).casefold()
                    or item["evidence_references"]):
                raise ValueError("Direct mapping requires the same annotation string; encoding and semantic changes need evidence")
        else:
            references = item["evidence_references"]
            eligible = {b["reference"] for b in basis if b["field"] == field
                        and (b["source"] in {"document", "user"} or b.get("dataset_specific"))}
            if not references or not set(references) <= eligible:
                raise ValueError("Interpreted mapping requires dataset-specific semantic evidence")
    return records


class LeafArguments(NoArguments):
    leaf_id: NonEmpty
    concept_id: NonEmpty
    description: NonEmpty
    protocol: Protocol
    evidence: list[Evidence] = Field(default_factory=list)
    semantic_basis: list[SemanticBasis] = Field(default_factory=list)
    mapping_grounding: list[MappingGrounding] = Field(default_factory=list)


class ConceptArguments(NoArguments):
    concept_id: NonEmpty
    status: Literal["complete", "unsupported"]
    reason: NonEmpty
    inspection_references: list[NonEmpty] = Field(default_factory=list)


class NodeArguments(NoArguments):
    node_id: NonEmpty


class FieldValuesArguments(NoArguments):
    field: NonEmpty
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=50)


class ExtensionArguments(NoArguments):
    concept_id: NonEmpty
    description: NonEmpty
    category: NonEmpty
    fields: list[NonEmpty] = Field(min_length=1)
    clinical_rationale: NonEmpty
    requirement_relevance: NonEmpty
    cannot_fit_existing: NonEmpty
    proposed_leaf_types: list[Literal["letter_classification", "bbox_grounding", "free_text"]] = (
        Field(min_length=1)
    )
    evidence: list[Evidence] = Field(default_factory=list)


class Weight(NoArguments):
    weight: float = Field(ge=0, allow_inf_nan=False)
    reason: NonEmpty


class ExtensionDecision(NoArguments):
    accepted: bool
    reason: NonEmpty


class ReviewIssue(NoArguments):
    action: Literal["merge", "remove", "reparent", "revise", "add_leaf"]
    node_ids: list[NonEmpty] = Field(min_length=1)
    reason: NonEmpty


class GlobalReview(NoArguments):
    issues: list[ReviewIssue]


def concepts_complete(tree: dict, *, core_only=False) -> bool:
    concepts = [c for c in tree["concepts"].values() if not core_only or c["origin"] == "core"]
    return bool(concepts) and all(c["status"] in {"complete", "unsupported"} for c in concepts)


NonNegative = Annotated[int, Field(ge=0)]


class Concept(NoArguments):
    description: NonEmpty
    category: NonEmpty
    origin: Literal["core", "extension"]
    status: Literal["pending", "complete", "unsupported"]
    reason: str
    inspection_references: list[NonEmpty] = Field(default_factory=list)
    require: str | None = None


class Leaf(NoArguments):
    concept_id: NonEmpty
    description: NonEmpty
    category: NonEmpty
    protocol: Protocol
    evidence: list[Evidence]
    semantic_basis: list[SemanticBasis] = Field(default_factory=list)
    mapping_grounding: list[MappingGrounding] = Field(default_factory=list)
    eligible_count: NonNegative | None = None


class ExtensionProposal(ExtensionArguments):
    decision: ExtensionDecision | None


class ReviewSnapshot(NoArguments):
    revision: NonNegative | None = None
    issues: list[ReviewIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_review_issues(self):
        if self.revision is not None and "issues" not in self.model_fields_set:
            raise ValueError("A reviewed revision must explicitly contain its issues")
        return self


class Coverage(NoArguments):
    used_fields: list[NonEmpty]
    unused_fields: list[NonEmpty]


class DatasetReference(NoArguments):
    dataset_id: NonEmpty | None


class PlanningDraft(NoArguments):
    benchmark_info: dict[str, object]
    dataset: DatasetReference
    concepts: dict[NonEmpty, Concept]
    tasks: dict[NonEmpty, Leaf]
    proposals: dict[NonEmpty, ExtensionProposal]
    revision: NonNegative
    discovery_complete: bool
    review: ReviewSnapshot
    reworks_used: NonNegative
    correction_open: bool
    retired_ids: list[NonEmpty]
    coverage: Coverage | None = None
    template_feedback_id: str | None = None
    scope_resolutions: list[ScopeResolution] = Field(default_factory=list)
    scope_semantic_basis: list[SemanticBasis] = Field(default_factory=list)


class PlanIssue(TypedDict):
    path: list[str | int]
    message: str


def _parse_draft(tree: object) -> tuple[PlanningDraft | None, list[PlanIssue]]:
    try:
        return PlanningDraft.model_validate(tree), []
    except ValidationError as exc:
        return None, [
            {"path": list(error["loc"]), "message": error["msg"]}
            for error in exc.errors(include_url=False, include_input=False)
        ]


def _reference_issues(draft: PlanningDraft) -> list[PlanIssue]:
    issues: list[PlanIssue] = []
    for key in draft.concepts.keys() & draft.tasks.keys():
        issues.append(
            {"path": ["tasks", key], "message": "Concept and leaf identities must be disjoint"}
        )
    for key in set(draft.retired_ids) & (
        draft.concepts.keys() | draft.tasks.keys() | draft.proposals.keys()
    ):
        issues.append(
            {"path": ["retired_ids"], "message": f"Retired leaf identity cannot be reused: {key}"}
        )
    for key, leaf in draft.tasks.items():
        try:
            validate_mapping_grounding(leaf.protocol.model_dump(), [m.model_dump() for m in leaf.mapping_grounding],
                                      [b.model_dump() for b in leaf.semantic_basis])
        except ValueError as exc:
            issues.append({"path": ["tasks", key, "mapping_grounding"], "message": str(exc)})
        concept = draft.concepts.get(leaf.concept_id)
        if concept is None:
            issues.append(
                {"path": ["tasks", key, "concept_id"], "message": "Leaf must reference a Concept"}
            )
        elif concept.category != leaf.category:
            issues.append(
                {
                    "path": ["tasks", key, "category"],
                    "message": "Leaf category must match its Concept",
                }
            )
    for key, proposal in draft.proposals.items():
        if proposal.concept_id != key:
            issues.append(
                {
                    "path": ["proposals", key, "concept_id"],
                    "message": "Proposal identity must match its key",
                }
            )
        if key in draft.tasks or (key in draft.concepts and draft.concepts[key].origin == "core"):
            issues.append(
                {
                    "path": ["proposals", key],
                    "message": "Proposal identity conflicts with a leaf or Core Concept",
                }
            )
    return issues


def validate_draft(tree: object) -> list[PlanIssue]:
    """Check persisted shape and two-level references without requiring completed planning."""
    draft, issues = _parse_draft(tree)
    return issues if draft is None else _reference_issues(draft)


def validate_plan(tree: object) -> list[PlanIssue]:
    """Return located structural and readiness problems; malformed input never escapes as a KeyError."""
    draft, issues = _parse_draft(tree)
    if draft is None:
        return issues
    issues.extend(_reference_issues(draft))
    if not draft.concepts:
        issues.append({"path": ["concepts"], "message": "Plan requires medical concepts"})
    for key, concept in draft.concepts.items():
        if concept.status == "unsupported" and not concept.inspection_references:
            issues.append({"path": ["concepts", key, "inspection_references"],
                           "message": "Unsupported concepts require inspection references"})
        children = [leaf for leaf in draft.tasks.values() if leaf.concept_id == key]
        if concept.status == "pending":
            issues.append(
                {
                    "path": ["concepts", key, "status"],
                    "message": "Concept requires a completed decision",
                }
            )
        elif (concept.status == "complete") != bool(children):
            issues.append(
                {
                    "path": ["concepts", key, "status"],
                    "message": "Completion status conflicts with leaves",
                }
            )
        if not concept.reason.strip():
            issues.append(
                {
                    "path": ["concepts", key, "reason"],
                    "message": "Concept decision requires a reason",
                }
            )
        proposal = draft.proposals.get(key)
        if concept.origin == "extension" and not (
            proposal and proposal.decision and proposal.decision.accepted
        ):
            issues.append(
                {"path": ["concepts", key, "origin"], "message": "Extension concept lacks approval"}
            )
    for key, proposal in draft.proposals.items():
        if proposal.decision is None:
            issues.append(
                {
                    "path": ["proposals", key, "decision"],
                    "message": "Extension proposal requires review",
                }
            )
        elif proposal.decision.accepted and key not in draft.concepts:
            issues.append(
                {"path": ["proposals", key], "message": "Accepted proposal requires its Concept"}
            )
    if not draft.discovery_complete or draft.coverage is None:
        issues.append(
            {"path": ["discovery_complete"], "message": "Coverage gap discovery is incomplete"}
        )
    if draft.review.revision != draft.revision or draft.review.issues or draft.correction_open:
        issues.append({"path": ["review"], "message": "Current tree has not passed global review"})
    for key, leaf in draft.tasks.items():
        if leaf.eligible_count is None or leaf.eligible_count <= 0:
            issues.append({"path": ["tasks", key, "eligible_count"], "message": "Exact positive capacity is required"})
    return issues
