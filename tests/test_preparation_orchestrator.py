from pathlib import Path

from medbenchagent.data.contracts import Source
from medbenchagent.runtime import preparation_orchestrator as orchestrator
from medbenchagent.runtime.dataset_runtime import DatasetAgentResult


class _FakeRuntime:
    def run(self, task):
        output_root = Path(task.workspace) / "outputs"
        for name in task.required_outputs:
            path = output_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n" if name.endswith(".jsonl") else "{}", encoding="utf-8")
        return DatasetAgentResult(
            status="complete", conversation_id=task.conversation_id,
            outputs={name: str(output_root / name) for name in task.required_outputs},
            progress={"iterations": 1},
        )


def test_orchestrator_writes_stage_state_and_candidate(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "source.txt").write_text("source", encoding="utf-8")
    context = {
        "run_id": "run", "dataset_scope": "dataset", "run_directory": str(tmp_path / "run"),
        "bundle_root": str(tmp_path / "bundle"),
        "sources": [Source(id="source", root=str(source)).model_dump()],
        "execution_runtime": {"backend": "local"},
    }
    monkeypatch.setattr(orchestrator, "dataset_runtime", lambda _context: _FakeRuntime())

    result = orchestrator.run_preparation_orchestrator(context)

    worker = tmp_path / "run" / "workers" / "dataset"
    assert result["status"] == "complete"
    assert (worker / "state" / "workflow.json").is_file()
    assert (worker / "state" / "memory.json").is_file()
    assert (worker / "state" / "progress.json").is_file()
    assert (worker / "outputs" / "cases.jsonl").is_file()
