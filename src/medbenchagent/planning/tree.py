"""Validated two-level planning objects; identities never encode execution behavior."""

from typing import Literal

from pydantic import Field

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


def validate_plan(tree: dict) -> list[str]:
    errors = []
    concepts, leaves = tree["concepts"], tree["tasks"]
    if not concepts_complete(tree):
        errors.append("Every concept must be completed or explicitly unsupported")
    if not tree.get("discovery_complete"):
        errors.append("Coverage gap discovery is incomplete")
    for key, proposal in tree.get("proposals", {}).items():
        if not isinstance(proposal.get("decision"), dict):
            errors.append(f"{key}: extension proposal has not been reviewed")
    for key, concept in concepts.items():
        if concept["origin"] == "extension" and not (
            tree.get("proposals", {}).get(key, {}).get("decision") or {}
        ).get("accepted"):
            errors.append(f"{key}: extension concept lacks approval")
    for key, leaf in leaves.items():
        if leaf["concept_id"] not in concepts:
            errors.append(f"{key}: unknown concept")
        Protocol.model_validate(leaf["protocol"])
    for key, concept in concepts.items():
        children = [v for v in leaves.values() if v["concept_id"] == key]
        if (concept["status"] == "complete") != bool(children):
            errors.append(f"{key}: completion status conflicts with leaves")
    review = tree.get("review", {})
    if review.get("revision") != tree["revision"] or review.get("issues") != []:
        errors.append("Current tree has not passed global review")
    allocation = tree.get("allocation", {})
    if allocation.get("revision") != tree["revision"]:
        errors.append("Current allocation is missing")
    else:
        counts = allocation["counts"]
        if (
            set(counts) != set(leaves)
            or set(allocation["capacities"]) != set(leaves)
            or any(
                type(n) is not int
                or n < 0
                or n > allocation["capacities"][k]
                or leaves[k].get("sample_count") != n
                for k, n in counts.items()
            )
        ):
            errors.append("Invalid leaf allocation")
        if (
            sum(counts.values()) != allocation["allocated_count"]
            or allocation["requested_count"] != tree["benchmark_info"]["sample_count"]
            or allocation["planning_shortfall"] < 0
            or allocation["allocated_count"] + allocation["planning_shortfall"]
            != tree["benchmark_info"]["sample_count"]
        ):
            errors.append("Allocation totals disagree")
        if allocation["allocated_count"] <= 0:
            errors.append("No eligible samples")
    return errors
