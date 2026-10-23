"""Assembly-injected handoff policy. Domain validators own product semantics."""

import json
from functools import partial

from .artifacts import read_json_artifact, resolve_artifact
from .contracts import WorkIssue, WorkScope


OPERATIONS = {
    "data_processing": ("profile", "prepare_task_data"),
    "task_planner": ("discover_scenarios", "design_leaves"),
    "generation": ("generate",),
    "review": ("review_plan", "review_candidate"),
    "delivery": ("publish",),
}
REQUIRED = {
    "profile": ({"source_manifest"},),
    "discover_scenarios": ({"dataset_profile", "dataset"},),
    "prepare_task_data": ({"task_plan"}, {"data_requirement"}, {"dataset_profile", "dataset"}),
    "design_leaves": ({"task_plan"}, {"task_data"}, {"dataset_profile", "dataset"}),
    "generate": ({"task_plan"}, {"task_data"}, {"benchmark_request"}),
    "review_plan": ({"task_plan"},),
    "review_candidate": ({"generation_candidate"},),
    "publish": ({"generation_candidate"}, {"audit_report"}, {"review_report"}, {"benchmark_request"}),
}
PRODUCT = {
    "profile": "dataset_profile", "prepare_task_data": "task_data",
    "discover_scenarios": "task_plan", "design_leaves": "task_plan",
    "generate": "generation_candidate", "review_plan": "review_report",
    "review_candidate": "review_report", "publish": "delivery_manifest",
}


def check_request(directory, request):
    if request.operation not in OPERATIONS.get(request.agent_id, ()):
        raise ValueError("Operation is not registered for this role")
    inputs = {r.artifact_id: r for r in request.inputs}
    if len(inputs) != len(request.inputs) or not request.scope:
        raise ValueError("Bind distinct inputs and an explicit scope")
    for ref in inputs.values():
        resolve_artifact(directory, ref)
    kinds = {r.kind for r in inputs.values()}
    for alternatives in REQUIRED[request.operation]:
        if not alternatives & kinds:
            raise ValueError(f"{request.operation} needs an input of kind {sorted(alternatives)}")
    for scope in request.scope:
        ref = inputs.get(scope.artifact_id)
        if ref is None:
            raise ValueError("Scope must belong to a bound input artifact")
        if scope.kind != "artifact":
            if ref.kind != "task_plan":
                raise ValueError("Scenario/task/requirement scope belongs to a TaskPlan")
            value = read_json_artifact(directory, ref)
            collection = {"scenario": "scenarios", "task": "tasks", "requirement": "requirements"}[scope.kind]
            if set(scope.object_ids) - value[collection].keys():
                raise ValueError("Scope contains unknown objects")
    if request.operation in {"discover_scenarios", "design_leaves", "prepare_task_data", "generate"}:
        plans = [r for r in inputs.values() if r.kind == "task_plan"]
        if len(plans) > 1:
            raise ValueError("Select one TaskPlan revision for this operation")
        if plans:
            plan, reader, prepared = planning_inputs(directory, request.inputs, plans[0])
            from medbenchagent.planning.semantic import validate_plan
            validate_plan(plan, reader, task_data=prepared)
            if request.operation == "generate":
                from medbenchagent.planning.semantic import check_operation_complete
                check_operation_complete(plan, "design_leaves", request.scope, task_data=prepared)
    if request.operation == "publish":
        from medbenchagent.generation.contracts import AuditReport, ReviewReport
        from medbenchagent.data.task_data import ProductBinding
        def one(kind):
            matches = [r for r in request.inputs if r.kind == kind]
            if len(matches) != 1:
                raise ValueError(f"Select one {kind}")
            return matches[0]
        candidate, audit_ref, review_ref = one("generation_candidate"), one("audit_report"), one("review_report")
        audit = AuditReport.model_validate(read_json_artifact(directory, audit_ref))
        review = ReviewReport.model_validate(read_json_artifact(directory, review_ref))
        target = ProductBinding(artifact_id=candidate.artifact_id, digest=candidate.digest)
        if audit.target != target or review.target != target or not audit.passed or not review.approved:
            raise ValueError("Publication needs matching passing Audit and approved Review")
        if review.audit != ProductBinding(artifact_id=audit_ref.artifact_id, digest=audit_ref.digest):
            raise ValueError("Review binds a different Audit")


def planning_inputs(directory, inputs, plan_ref):
    from medbenchagent.data.dataset import FileDatasetReader
    from medbenchagent.data.task_data import ProfileReader, TaskDataReader
    from medbenchagent.planning.contracts import TaskPlan
    plan = TaskPlan.model_validate(read_json_artifact(directory, plan_ref))
    base = [r for r in inputs if r.kind in {"dataset", "dataset_profile"} and r.digest == plan.dataset_digest]
    if len(base) != 1:
        raise ValueError("Bind the plan's exact source Profile/Dataset")
    reader = (ProfileReader if base[0].kind == "dataset_profile" else FileDatasetReader)(resolve_artifact(directory, base[0]))
    prepared = {r.artifact_id: TaskDataReader(resolve_artifact(directory, r)) for r in inputs if r.kind == "task_data"}
    return plan, reader, prepared


def check_result(directory, request, result):
    from medbenchagent.data.task_data import ProfileReader, TaskDataReader
    from medbenchagent.generation.execution import CandidateReader
    products = [r for r in result.artifacts if r.kind == PRODUCT[request.operation]]
    for ref in result.artifacts:
        path = resolve_artifact(directory, ref)
        reader = {"dataset_profile": ProfileReader, "task_data": TaskDataReader, "generation_candidate": CandidateReader}.get(ref.kind)
        if reader:
            reader(path)
    if result.outcome != "completed":
        return result
    if not products:
        raise ValueError("Completed operation requires its formal product")
    if request.operation == "prepare_task_data":
        from medbenchagent.planning.contracts import DataRequirement
        from medbenchagent.data.task_data import validate_preparation
        requested = {r.artifact_id: r for r in request.inputs if r.kind == "data_requirement"}
        covered = set()
        for ref in products:
            reader = TaskDataReader(resolve_artifact(directory, ref))
            for key, binding in reader.manifest.requirements.items():
                original = requested.get(binding.artifact_id)
                if original is None or original.digest != binding.digest:
                    raise ValueError("TaskData requirement is not a delegated exact input")
                requirement = DataRequirement.model_validate(read_json_artifact(directory, original))
                validate_preparation(reader, requirement)
                covered.add(key)
        plans = [r for r in request.inputs if r.kind == "task_plan"]
        from medbenchagent.planning.semantic import scoped_objects
        plan, _, _ = planning_inputs(directory, request.inputs, plans[0])
        _, _, required = scoped_objects(plan, request.scope)
        if required - covered:
            raise ValueError(f"Preparation left requirements unresolved: {sorted(required - covered)}")
    if request.agent_id == "task_planner":
        from medbenchagent.planning.semantic import validate_plan, check_operation_complete
        if len(products) != 1:
            raise ValueError("Select one published TaskPlan")
        plan, reader, prepared = planning_inputs(directory, [*request.inputs, *result.artifacts], products[0])
        validate_plan(plan, reader, publish=True, task_data=prepared)
        check_operation_complete(plan, request.operation, request.scope, task_data=prepared)
    if request.agent_id == "review":
        from medbenchagent.generation.contracts import ReviewReport
        kind = "task_plan" if request.operation == "review_plan" else "generation_candidate"
        targets = {r.artifact_id: r.digest for r in request.inputs if r.kind == kind}
        for ref in products:
            report = ReviewReport.model_validate(read_json_artifact(directory, ref))
            if targets.get(report.target.artifact_id) != report.target.digest:
                raise ValueError("Review must bind the delegated target")
            if not report.approved:
                result = result.model_copy(update={"issues": [*result.issues, *[
                    WorkIssue(code=f"review_{f.category}", message=f"{f.suggested_owner}/{f.object_id}: {f.summary}")
                    for f in report.findings if f.blocking
                ]]})
    return result


def check_goal(directory, goal, benchmark_request, decision, artifacts, history):
    selected = [r for r in artifacts if r.artifact_id in decision.evidence_artifact_ids]
    if len(selected) != len(set(decision.evidence_artifact_ids)):
        raise ValueError("Finish references unknown evidence")
    if decision.goal_result in {"blocked", "failed"}:
        no_scenarios = any(ref.kind == "task_plan" and all(s["decision"] == "exclude"
                           for s in read_json_artifact(directory, ref)["scenarios"].values()) for ref in selected)
        if not no_scenarios and not any(row.result.issues or row.result.needs or row.result.outcome in {"failed", "blocked"} for row in history):
            raise ValueError("Report a concrete work blocker before ending without delivery")
        return
    expected = {"profile": "dataset_profile", "plan": "task_plan", "benchmark": "delivery_manifest"}[goal.kind]
    products = [r for r in selected if r.kind == expected]
    if len(products) != 1:
        raise ValueError(f"Goal completion needs one selected {expected}")
    ref = products[0]
    producer = next((row for row in reversed(history) if ref in row.result.artifacts and row.result.outcome == "completed"), None)
    if producer is None:
        raise ValueError("Completion evidence must have passed its operation's result check")
    if goal.kind == "profile":
        from medbenchagent.data.task_data import ProfileReader
        ProfileReader(resolve_artifact(directory, ref))
    elif goal.kind == "plan":
        from medbenchagent.planning.semantic import validate_plan, check_operation_complete
        plan, reader, prepared = planning_inputs(directory, artifacts, ref)
        validate_plan(plan, reader, publish=True, task_data=prepared)
        check_operation_complete(plan, "discover_scenarios" if goal.plan_level == "scenarios" else "design_leaves",
                                 [WorkScope(artifact_id=ref.artifact_id)], task_data=prepared)
    else:
        from medbenchagent.evaluation.loader import load_benchmark_package
        root = resolve_artifact(directory, ref).parent
        load_benchmark_package(root)
        manifest = json.loads((root / "benchmark_manifest.json").read_text(encoding="utf-8"))
        counts = manifest["counts"]
        if not benchmark_request or counts["requested"] != benchmark_request["sample_count"]:
            raise ValueError("Delivery changed the original quantity goal")
        limited = counts["shortfall"] > 0 or counts["quarantined"] > 0
        if limited and (not goal.allow_limited_delivery or decision.goal_result != "partial"):
            raise ValueError("Limited delivery is partial and requires the frozen goal policy")
        if not limited and decision.goal_result != "succeeded":
            raise ValueError("A complete delivery should report succeeded")
    if goal.kind != "benchmark" and decision.goal_result != "succeeded":
        raise ValueError("Profile/plan completion reports succeeded; unfinished work reports blocked")


def readiness(directory, agents, artifacts):
    """Compute concrete suggestions with the same request guards used at dispatch."""
    from .contracts import WorkRequest
    catalog = {r.artifact_id: r for r in artifacts}
    values = {}
    def read(ref):
        if ref.artifact_id not in values:
            values[ref.artifact_id] = read_json_artifact(directory, ref)
        return values[ref.artifact_id]

    def dependencies(value, chosen):
        if isinstance(value, dict):
            if "artifact_id" in value and "digest" in value:
                ref = catalog.get(value["artifact_id"])
                if ref is not None and ref.digest == value["digest"] and ref.artifact_id not in chosen:
                    chosen[ref.artifact_id] = ref
                    dependencies(read(ref), chosen)
            for item in value.values():
                dependencies(item, chosen)
        elif isinstance(value, list):
            for item in value:
                dependencies(item, chosen)

    primary = {"profile": {"source_manifest"}, "discover_scenarios": {"dataset", "dataset_profile"},
               "prepare_task_data": {"task_plan"}, "design_leaves": {"task_plan"}, "generate": {"task_plan"},
               "review_plan": {"task_plan"}, "review_candidate": {"generation_candidate"}, "publish": {"generation_candidate"}}
    ready, blocked = [], []
    for role, spec in agents.items():
        for operation in spec.operations:
            candidates = [r for r in artifacts if r.kind in primary[operation]]
            if not candidates:
                blocked.append({"agent_id": role, "operation": operation, "reason": f"Missing {sorted(primary[operation])}"})
            for target in candidates[-4:]:
                entry = {"agent_id": role, "operation": operation, "scope": [WorkScope(artifact_id=target.artifact_id).model_dump()]}
                try:
                    chosen = {target.artifact_id: target}
                    value = read(target)
                    dependencies(value, chosen)
                    if target.kind == "task_plan":
                        for ref in artifacts:
                            if ref.kind in {"dataset", "dataset_profile"} and ref.digest == value["dataset_digest"]:
                                chosen[ref.artifact_id] = ref
                            if ref.kind == "data_requirement" and read(ref) in value["requirements"].values():
                                chosen[ref.artifact_id] = ref
                            if operation == "design_leaves" and ref.kind == "task_data" and read(ref)["dataset"] == value["dataset"]:
                                chosen[ref.artifact_id] = ref
                                dependencies(read(ref), chosen)
                    if operation in {"generate", "publish"}:
                        chosen.update({r.artifact_id: r for r in artifacts if r.kind == "benchmark_request"})
                    if operation == "publish":
                        for kind, flag in (("review_report", "approved"), ("audit_report", "passed")):
                            matches = [r for r in artifacts if r.kind == kind and read(r).get(flag)
                                       and read(r)["target"] == {"artifact_id": target.artifact_id, "digest": target.digest}]
                            if matches:
                                ref = sorted(matches, key=lambda r: r.artifact_id)[0]
                                chosen[ref.artifact_id] = ref
                                dependencies(read(ref), chosen)
                    # Candidate/spec dependencies include plans whose original evidence is still needed.
                    for ref in list(chosen.values()):
                        if ref.kind == "task_plan":
                            plan = read(ref)
                            chosen.update({r.artifact_id: r for r in artifacts
                                if r.kind in {"dataset", "dataset_profile"} and r.digest == plan["dataset_digest"]})
                    request = WorkRequest(request_id="readiness", compilation_id="readiness", agent_id=role,
                        operation=operation, objective="Check selected dependencies", scope=[WorkScope(artifact_id=target.artifact_id)],
                        inputs=list(chosen.values()))
                    check_request(directory, request)
                    ready.append({**entry, "input_artifact_ids": list(chosen)})
                except (ValueError, KeyError, OSError) as exc:
                    blocked.append({**entry, "reason": str(exc)})
    return {"available_operations": ready, "blocked_reasons": blocked,
            "note": "Recent bounded suggestions, not an implicit latest selection. Inspect older products when useful."}


def attach_checks(spec, directory, role):
    from dataclasses import replace
    return replace(spec, operations=OPERATIONS[role], check_request=partial(check_request, directory),
                   check_result=partial(check_result, directory))
