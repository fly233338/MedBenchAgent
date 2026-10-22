"""Small, serializable handoffs shared by all professional agents."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")]


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
    outcome: Literal["completed", "progress", "needs_input", "blocked", "failed"]
    summary: Text
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    issues: list[WorkIssue] = Field(default_factory=list)
    needs: list[WorkNeed] = Field(default_factory=list)


class Delegate(Contract):
    action: Literal["delegate"] = "delegate"
    agent_id: Identifier
    objective: Text
    scope: list[Text] = Field(default_factory=list)
    inputs: list[ArtifactRef] = Field(default_factory=list)
    constraints: list[Text] = Field(default_factory=list)


class AskUser(Contract):
    action: Literal["ask_user"] = "ask_user"
    question: Text


class Finish(Contract):
    """End a coordination run, without claiming a delivered benchmark."""

    action: Literal["finish"] = "finish"
    summary: Text


SupervisorDecision = Annotated[Delegate | AskUser | Finish, Field(discriminator="action")]
