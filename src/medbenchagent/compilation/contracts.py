"""Small, serializable handoffs shared by all professional agents."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")]
WorkOutcome = Literal["completed", "progress", "needs_input", "blocked", "failed"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ArtifactRef(Contract):
    artifact_id: Identifier
    kind: Text
    path: Text
    digest: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class WorkScope(Contract):
    """Objects in one explicitly delegated input; whole artifacts use kind=artifact."""

    artifact_id: Identifier
    kind: Literal["artifact", "scenario", "task", "requirement"] = "artifact"
    object_ids: list[Text] = Field(default_factory=list)

    @model_validator(mode="after")
    def explicit_objects(self):
        if (self.kind == "artifact") != (not self.object_ids):
            raise ValueError("Use artifact for the whole input, otherwise supply object IDs")

        if len(set(self.object_ids)) != len(self.object_ids):
            raise ValueError("Duplicate scope objects")
        return self


class GoalSpec(Contract):
    kind: Literal["profile", "plan", "benchmark"] = "benchmark"
    plan_level: Literal["scenarios", "executable_leaves"] | None = None
    allow_limited_delivery: bool = True

    @model_validator(mode="after")
    def level(self):
        if (self.kind == "plan") != (self.plan_level is not None):
            raise ValueError("Only plan goals require plan_level")
        return self


class RepairLimits(Contract):
    per_problem: int = Field(default=3, ge=1)
    no_progress: int = Field(default=3, ge=1)
    total: int = Field(default=12, ge=1)


class WorkRequest(Contract):
    request_id: Identifier
    compilation_id: Identifier
    agent_id: Identifier
    operation: Identifier
    workstream_id: Identifier | None = None
    objective: Text
    scope: list[WorkScope] = Field(default_factory=list)
    inputs: list[ArtifactRef] = Field(default_factory=list)
    constraints: list[Text] = Field(default_factory=list)


class WorkIssue(Contract):
    code: Text
    message: Text


class WorkNeed(Contract):
    need_id: Identifier
    kind: Literal["work", "user"]
    summary: Text
    detail_ref: ArtifactRef | None = None


class WorkResult(Contract):
    request_id: Identifier
    outcome: WorkOutcome
    summary: Text
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    issues: list[WorkIssue] = Field(default_factory=list)
    needs: list[WorkNeed] = Field(default_factory=list)


class WorkRecord(Contract):
    """One checkpoint-owned handoff, without SDK messages or file contents."""

    request: WorkRequest
    result: WorkResult

    @model_validator(mode="after")
    def matching_request(self):
        if self.request.request_id != self.result.request_id:
            raise ValueError("Recorded result belongs to another request")
        return self


class Delegate(Contract):
    action: Literal["delegate"] = "delegate"
    agent_id: Identifier
    operation: Identifier
    workstream_id: Identifier | None = None
    objective: Text
    scope: list[WorkScope] = Field(default_factory=list)
    inputs: list[ArtifactRef] = Field(default_factory=list)
    constraints: list[Text] = Field(default_factory=list)


class Workstream(Contract):
    """Host-owned coordination identity; never contains SDK session state."""

    workstream_id: Identifier
    agent_id: Identifier
    objective: Text
    scope: list[WorkScope] = Field(default_factory=list)


class AskUser(Contract):
    action: Literal["ask_user"] = "ask_user"
    question: Text


class Finish(Contract):
    """End a coordination run, without claiming a delivered benchmark."""

    action: Literal["finish"] = "finish"
    summary: Text
    goal_result: Literal["succeeded", "partial", "blocked", "failed"]
    evidence_artifact_ids: list[Identifier] = Field(default_factory=list)


SupervisorDecision = Annotated[Delegate | AskUser | Finish, Field(discriminator="action")]
