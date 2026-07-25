"""Role-specialized Agent entry points."""

from medbenchagent.agents.audit import compile_audit_agent
from medbenchagent.agents.capability import compile_capability_agent
from medbenchagent.agents.clinical_design import compile_clinical_design_agent
from medbenchagent.agents.generation import compile_generation_agent
from medbenchagent.agents.specification import compile_specification_agent

__all__ = [
    "compile_audit_agent",
    "compile_capability_agent",
    "compile_clinical_design_agent",
    "compile_generation_agent",
    "compile_specification_agent",
]
