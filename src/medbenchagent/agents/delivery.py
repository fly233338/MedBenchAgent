"""Thin deterministic delivery node; publication rules belong to the generation domain."""

import hashlib
import json
from types import SimpleNamespace

from medbenchagent.compilation.artifacts import read_json_artifact, resolve_artifact, register_file
from medbenchagent.compilation.contracts import WorkRequest, WorkResult, WorkIssue
from medbenchagent.compilation.interfaces import AgentSpec
from medbenchagent.generation.contracts import AuditReport, ReviewReport
from medbenchagent.generation.execution import CandidateReader, load_execution
from medbenchagent.generation.publication import publish_delivery
from .handoff import binding, bound_path


def delivery_spec():
    def node(state, runtime):
        request = WorkRequest.model_validate(state["request"])
        directory = runtime.context.directory
        try:

            def one(kind):
                refs = [ref for ref in request.inputs if ref.kind == kind]
                if len(refs) != 1:
                    raise ValueError(f"Delivery requires exactly one selected {kind}")
                return refs[0]

            candidate_ref, audit_ref, review_ref = (
                one("generation_candidate"),
                one("audit_report"),
                one("review_report"),
            )
            candidate = CandidateReader(resolve_artifact(directory, candidate_ref))
            target_ref = one("benchmark_request")
            if (
                binding(target_ref) != candidate.manifest.allocation.target
                or read_json_artifact(directory, target_ref)["sample_count"]
                != candidate.manifest.allocation.requested
            ):
                raise ValueError("Delivery target does not match the current benchmark request")
            audit = AuditReport.model_validate(read_json_artifact(directory, audit_ref))
            review = ReviewReport.model_validate(read_json_artifact(directory, review_ref))
            bound = SimpleNamespace(
                state={}, context=SimpleNamespace(directory=directory, request=request)
            )
            executions = {
                spec.artifact_id: load_execution(spec, lambda p, k: bound_path(bound, p, k))
                for spec in candidate.manifest.specs
            }
            identity = hashlib.sha256(
                json.dumps(
                    [candidate_ref.model_dump(), audit_ref.model_dump(), review_ref.model_dump()],
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            destination = directory / "artifacts" / request.request_id / "delivery" / identity
            reason = (
                "\n".join([request.objective, *request.constraints])
                if "allow_limited_delivery" in request.scope
                else ""
            )
            path = publish_delivery(
                destination,
                candidate=candidate,
                candidate_binding=binding(candidate_ref),
                audit=audit,
                audit_binding=binding(audit_ref),
                review=review,
                review_binding=binding(review_ref),
                executions=executions,
                limitation_reason=reason,
            )
            ref = register_file(
                directory, request, path.relative_to(directory).as_posix(), kind="delivery_manifest"
            )
            result = WorkResult(
                request_id=request.request_id,
                outcome="completed",
                artifacts=[ref],
                summary=f"Portable package: {candidate.manifest.counts['accepted']} items; shortfall {candidate.manifest.counts['shortfall']}",
            )
        except (ValueError, KeyError, OSError) as exc:
            result = WorkResult(
                request_id=request.request_id,
                outcome="failed",
                summary="Delivery was not published",
                issues=[WorkIssue(code="delivery_rejected", message=str(exc))],
            )
        return {"result": result.model_dump(mode="json")}

    return AgentSpec(
        "Deterministic portable delivery. Bind exactly one candidate, matching Audit/Review and all referenced "
        "specs, TaskPlans and TaskData. Empty or blocking results cannot publish. For a nonempty limited delivery, "
        "include allow_limited_delivery in scope and explain the limitation in the objective/constraints.",
        node,
    )
