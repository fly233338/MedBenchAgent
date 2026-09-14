"""Validated two-level planning objects; identities never encode execution behavior."""

from typing import Annotated, Literal, TypedDict

from pydantic import Field, ValidationError, model_validator

from medbenchagent.planning.protocol import Protocol
from medbenchagent.runtime.tool_contracts import NoArguments, NonEmpty


class Evidence(NoArguments):
    pdf: NonEmpty
    page: int = Field(ge=1)
    chunk_id: NonEmpty


class LeafArguments(NoArguments):
    leaf_id: NonEmpty
    concept_id: NonEmpty
    description: NonEmpty
    protocol: Protocol
    evidence: list[Evidence] = Field(default_factory=list)


class ConceptArguments(NoArguments):
    concept_id: NonEmpty
    status: Literal["complete", "unsupported"]
    reason: NonEmpty


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


class WeightsArguments(NoArguments):
    weights: dict[str, Weight]


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
    require: str | None = None


class Leaf(NoArguments):
    concept_id: NonEmpty
    description: NonEmpty
    category: NonEmpty
    protocol: Protocol
    evidence: list[Evidence]
    sample_count: NonNegative | None = None


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


class Allocation(NoArguments):
    revision: NonNegative
    requested_count: Annotated[int, Field(gt=0)]
    allocated_count: NonNegative
    planning_shortfall: NonNegative
    counts: dict[NonEmpty, NonNegative]
    capacities: dict[NonEmpty, NonNegative]
    weights: dict[NonEmpty, Annotated[float, Field(ge=0, allow_inf_nan=False)]]
    weight_reasons: dict[NonEmpty, NonEmpty]


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
    allocation: Allocation | None = None


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
    allocation = draft.allocation
    if allocation is None or allocation.revision != draft.revision:
        issues.append({"path": ["allocation"], "message": "Current allocation is missing"})
    else:
        for field in ("counts", "capacities", "weights", "weight_reasons"):
            if set(getattr(allocation, field)) != set(draft.tasks):
                issues.append(
                    {
                        "path": ["allocation", field],
                        "message": "Allocation must cover exactly every leaf",
                    }
                )
        for key, count in allocation.counts.items():
            leaf = draft.tasks.get(key)
            capacity = allocation.capacities.get(key)
            if leaf is None or capacity is None or count > capacity or leaf.sample_count != count:
                issues.append(
                    {"path": ["allocation", "counts", key], "message": "Invalid leaf allocation"}
                )
        requested = draft.benchmark_info.get("sample_count")
        if (
            type(requested) is not int
            or requested <= 0
            or allocation.requested_count != requested
            or sum(allocation.counts.values()) != allocation.allocated_count
            or allocation.allocated_count + allocation.planning_shortfall != requested
        ):
            issues.append(
                {
                    "path": ["allocation"],
                    "message": "Allocation totals disagree with benchmark request",
                }
            )
        if allocation.allocated_count <= 0:
            issues.append(
                {"path": ["allocation", "allocated_count"], "message": "No eligible samples"}
            )
    return issues
