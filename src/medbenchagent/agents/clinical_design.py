"""ClinicalDesignAgent wrapper around the existing RAG-design subgraph."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from medbenchagent.agents.contracts import (
    CLINICAL_DESIGN_AGENT,
    ClinicalDesignAgentInputState,
    ClinicalDesignAgentOutputState,
    ClinicalDesignAgentState,
)
from medbenchagent.rag.graph import compile_rag_designs_subgraph


def compile_clinical_design_agent(*, checkpointer: Any = None):
    """Wrap the unchanged RAG workflow as a role-level Agent."""
    graph = StateGraph(
        ClinicalDesignAgentState,
        input_schema=ClinicalDesignAgentInputState,
        output_schema=ClinicalDesignAgentOutputState,
    )
    graph.add_node("rag_designs", compile_rag_designs_subgraph())
    graph.add_edge(START, "rag_designs")
    graph.add_edge("rag_designs", END)
    return graph.compile(checkpointer=checkpointer, name=CLINICAL_DESIGN_AGENT)
