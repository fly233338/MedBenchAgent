"""Role-specialized Agent entry points."""

from medbenchagent.agents.audit import compile_audit_agent
from medbenchagent.agents.task_planning import compile_task_planning_agent
from medbenchagent.agents.medical_align import compile_medical_align_agent
from medbenchagent.agents.generation import compile_generation_agent
from medbenchagent.agents.specification import compile_specification_agent

__all__ = [
    "compile_audit_agent",
    "compile_task_planning_agent",
    "compile_medical_align_agent",
    "compile_generation_agent",
    "compile_specification_agent",
]
