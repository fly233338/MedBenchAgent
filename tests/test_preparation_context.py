import pytest

from medbenchagent.data.contracts import Source
from medbenchagent.data.sources import inventory_source
from medbenchagent.preparation.context import ContextBudgetError, build_stage_context


def _inputs(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "records.txt").write_text("record", encoding="utf-8")
    manifest = tmp_path / "manifest"
    inventory_source(Source(id="source", root=str(source_root)), manifest)
    accepted = tmp_path / "accepted"
    accepted.mkdir()
    mapping = accepted / "mapping.json"
    mapping.write_text("{}", encoding="utf-8")
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "draft.json").write_text("draft", encoding="utf-8")
    return {
        "run_id": "run", "dataset_id": "dataset", "sources": [{"id": "source", "root": str(source_root)}],
        "source_manifests": {"source": str(manifest)},
        "accepted_artifacts": {"mapping.json": {"path": str(mapping), "digest": __import__("hashlib").sha256(mapping.read_bytes()).hexdigest(), "producer": "host", "stage": "understand", "status": "accepted"}},
        "previous_outputs_root": str(previous),
    }


def test_context_writes_complete_prompt_and_keeps_unaccepted_outputs_separate(tmp_path):
    packet, prompt, token_count, key = build_stage_context("prepare", _inputs(tmp_path), tmp_path / "attempt")

    assert "You are the MedBench data preparation coding agent." in prompt
    assert "Current stage: prepare" in prompt
    assert token_count > 0 and key
    assert (tmp_path / "attempt" / "inputs" / "context.json").is_file()
    assert (tmp_path / "attempt" / "inputs" / "prompt.md").is_file()
    assert (tmp_path / "attempt" / "inputs" / "accepted" / "mapping.json").is_file()
    assert (tmp_path / "attempt" / "inputs" / "previous" / "draft.json").is_file()
    assert packet.identity["stage"] == "prepare"


def test_required_context_budget_fails_without_truncating_prompt(tmp_path):
    state = _inputs(tmp_path)
    state["preparation_stage_budgets"] = {"prepare": {"initial_message_tokens": 256}}
    with pytest.raises(ContextBudgetError, match="required_context_too_large"):
        build_stage_context("prepare", state, tmp_path / "attempt")
