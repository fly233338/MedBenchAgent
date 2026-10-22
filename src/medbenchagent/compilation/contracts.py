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


class WorkRequest(Contract):
    request_id: Identifier
    compilation_id: Identifier
    agent_id: Identifier
    workstream_id: Identifier | None = None
    objective: Text
    scope: list[Text] = Field(default_factory=list)
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
    workstream_id: Identifier | None = None
    objective: Text
    scope: list[Text] = Field(default_factory=list)
    inputs: list[ArtifactRef] = Field(default_factory=list)
    constraints: list[Text] = Field(default_factory=list)


class Workstream(Contract):
    """Host-owned coordination identity; never contains SDK session state."""

    workstream_id: Identifier
    agent_id: Identifier
    objective: Text
    scope: list[Text] = Field(default_factory=list)


class AskUser(Contract):
    action: Literal["ask_user"] = "ask_user"
    question: Text


class Finish(Contract):
    """End a coordination run, without claiming a delivered benchmark."""

    action: Literal["finish"] = "finish"
    summary: Text


SupervisorDecision = Annotated[Delegate | AskUser | Finish, Field(discriminator="action")]
