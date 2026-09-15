"""Planner query arguments, kept separate from the data engine."""

from pydantic import Field
from medbenchagent.data.contracts import Contract, Text
from medbenchagent.planning.protocol import Protocol


class SingleField(Contract):
    field: Text


class Sample(Contract):
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=5, ge=1, le=50)
    source_id: str | None = None


class Pair(Contract):
    left: Text
    right: Text


class Array(Contract):
    path: Text


class Capacity(Contract):
    protocol: Protocol
