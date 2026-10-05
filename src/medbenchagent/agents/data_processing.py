"""Dataset processing graph entrypoint.

Source inspection and conversion are performed by the two-stage preparation
subgraph.  This module remains as the small integration surface used by the
Supervisor workflow.
"""

from __future__ import annotations


def compile_data_processing_agent(*, checkpointer=None):
    from medbenchagent.agents.dataset_agent import compile_dataset_agent
    return compile_dataset_agent(checkpointer=checkpointer)
