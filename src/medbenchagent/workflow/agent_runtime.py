"""Bind MedD2B business capabilities to the domain-independent agent executor."""
from medbenchagent.runtime.harness import compile_agent as compile_executor
from medbenchagent.planning.bir import business_snapshot
from medbenchagent.workflow.clarifications import validate_ask_user
from medbenchagent.workflow.interaction import ask_user


def pending_tools(context):
    issue = context.get("global_issue") or {}
    allowed = {"ask_user", "resolve_clarification", "list_evidence", "read_evidence"}
    if issue.get("status") == "answered":
        if issue.get("purpose") == "dataset_selection":
            allowed.update({"process_dataset", "discover_datasets", "select_datasets", "bind_dataset"})
        elif issue.get("purpose") == "planning_extension_selection":
            allowed.update({"discover_planning_extensions", "select_planning_extensions"})
    return allowed


def compile_agent(name, instructions, tools, describe, finish, *, include_inspection_tools=True, **kwargs):
    if name == "SupervisorAgent":
        from medbenchagent.workflow.supervision import compilation_view
        kwargs["after_tool"] = lambda context, outcome: {"view": compilation_view(context)}
    if include_inspection_tools:
        from medbenchagent.workflow.evidence import list_evidence, read_evidence, EvidencePage, EvidenceRead
        tools = {**tools, "list_evidence": list_evidence, "read_evidence": read_evidence}
        kwargs["args_schemas"] = {**kwargs.get("args_schemas", {}),
                                  "list_evidence": EvidencePage, "read_evidence": EvidenceRead}
        kwargs["read_tools"] = kwargs.get("read_tools", frozenset()) | {"list_evidence", "read_evidence"}
    return compile_executor(name, instructions, tools, describe, finish,
                            dependency_snapshot=business_snapshot, ask_user_handler=ask_user,
                            validate_question=validate_ask_user, pending_tools=pending_tools, **kwargs)
