"""Agent-facing interfaces, independent of the graph and concrete implementations."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from .contracts import (
    ArtifactRef,
    GoalSpec,
    RepairLimits,
    SupervisorDecision,
    WorkRecord,
    WorkResult,
    Workstream,
)


class AgentInput(TypedDict):
    request: dict[str, Any]


class AgentOutput(TypedDict):
    result: dict[str, Any]


def record_summary(record: WorkRecord) -> dict[str, str]:
    """A projection for displays and context, never another persisted history."""
    row = {
        "request_id": record.request.request_id,
        "agent_id": record.request.agent_id,
        "operation": record.request.operation,
        "outcome": record.result.outcome,
        "summary": record.result.summary,
    }
    if record.request.workstream_id is not None:
        row["workstream_id"] = record.request.workstream_id
    return row


@dataclass(frozen=True)
class SupervisorView:
    compilation_id: str
    goal: str
    agents: Mapping[str, str]
    artifacts: tuple[ArtifactRef, ...]
    last_result: WorkResult | None
    history: tuple[WorkRecord, ...]
    answers: tuple[str, ...]
    workstreams: tuple[Workstream, ...] = ()
    workstream_agents: tuple[str, ...] = ()
    readiness: dict = field(default_factory=dict)
    feedback: tuple[str, ...] = ()
    goal_spec: GoalSpec = field(default_factory=GoalSpec)

    def validate_workstream(self, agent_id: str, workstream_id: str | None) -> None:
        if workstream_id is None:
            return

        if agent_id not in self.workstream_agents:
            raise ValueError(f"Agent does not use Workstreams: {agent_id}")
        selected = next(
            (item for item in self.workstreams if item.workstream_id == workstream_id),
            None,
        )
        if selected is None:
            raise ValueError(f"Unknown Workstream ID: {workstream_id}")

        if selected.agent_id != agent_id:
            raise ValueError("Workstream belongs to another Agent")

    def workstream_summaries(self) -> list[dict[str, Any]]:
        latest = {
            row.request.workstream_id: record_summary(row)
            for row in self.history
            if row.request.workstream_id is not None
        }
        return [
            {**item.model_dump(mode="json"), "last_result": latest.get(item.workstream_id)}
            for item in self.workstreams
        ]


@dataclass(frozen=True)
class CompilationContext:
    directory: Path
    runtime_directory: Path = field(kw_only=True)
    decide: Callable[..., SupervisorDecision]  # RunnableConfig is injected when accepted.
    goal: str
    goal_spec: GoalSpec = field(default_factory=GoalSpec)
    repair_limits: RepairLimits = field(default_factory=RepairLimits)
    check_goal: Callable | None = None
    readiness: Callable | None = None


@dataclass(frozen=True)
class AgentSpec:
    description: str
    node: Any  # A node callable or compiled subgraph with AgentInput/AgentOutput.
    uses_workstreams: bool = False
    operations: tuple[str, ...] = ()
    check_request: Callable | None = None
    check_result: Callable | None = None
