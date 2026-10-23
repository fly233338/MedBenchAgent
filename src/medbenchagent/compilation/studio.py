"""LangGraph Server factory using the same assembly and host policies."""
from pathlib import Path
from contextlib import ExitStack
import weakref
import os
from filelock import FileLock
from medbenchagent.data.task_data import write_once
from langchain_core.runnables import RunnableConfig
from medbenchagent.compilation.assembly import assemble_compilation
from medbenchagent.compilation.graph import build_compilation_graph
from medbenchagent.compilation.run import RunOptions, implementation_digest
from medbenchagent.agents.data_processing.coding.skills import verify_skills
from medbenchagent.runtime.graph_runner import initial_state


def graph(config: RunnableConfig):
    selected = config.get("configurable", {}).get("directory") or os.getenv("MEDBENCH_STUDIO_RUN")
    if not selected:
        raise ValueError("Set MEDBENCH_STUDIO_RUN to a run created with --prepare-only")
    directory = Path(selected).resolve()
    thread = config.get("configurable", {}).get("thread_id")
    if thread:
        with FileLock(directory / "studio.lock"):
            write_once(directory / "studio-thread.txt", str(thread).encode())
    options = RunOptions.model_validate_json((directory / "run.json").read_bytes())
    if options.implementation_digest != implementation_digest():
        raise ValueError("Incompatible run implementation; create a new run")
    if options.coding_skills is not None:
        verify_skills(directory, options.coding_skills)
    # Server owns checkpoint persistence. Clients live for this graph instance.
    stack = ExitStack()
    try:
        agents, context = stack.enter_context(assemble_compilation(options, directory))
        result = build_compilation_graph(agents, checkpointer=None, default_context=context,
                                         initial=initial_state(directory, options))
    except Exception:
        stack.close()
        raise
    weakref.finalize(result, stack.close)
    return result
