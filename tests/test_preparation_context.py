import pytest

from medbenchagent.data.contracts import Source
from medbenchagent.data.sources import inventory_source
from medbenchagent.preparation.context import ContextBudgetError, build_stage_context, verify_previous_reference
from medbenchagent.preparation.contracts import AcceptedHandoff, ArtifactRef, digest_file, digest_payload


def _inputs(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "records.txt").write_text("record", encoding="utf-8")
    manifest = tmp_path / "manifest"
    inventory_source(Source(id="source", root=str(source_root)), manifest)
    accepted = tmp_path / "accepted"
    accepted.mkdir()
    files = {}
    for name, value in {
        "source_report.json": "{}",
        "mapping.json": "{}",
        "image_population.jsonl": "{}\n",
        "inspect_source.py": "print('ok')\n",
    }.items():
        path = accepted / name
        path.write_text(value, encoding="utf-8")
        files[name] = ArtifactRef(name=name, path=str(path), digest=digest_file(path), producer="host", stage="understand", status="accepted")
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "outputs").mkdir()
    (previous / "outputs" / "draft.json").write_text("draft", encoding="utf-8")
    return {
        "run_id": "run", "dataset_id": "dataset", "sources": [{"id": "source", "root": str(source_root)}],
        "source_manifests": {"source": str(manifest)},
        "accepted_handoff": AcceptedHandoff(
            handoff_id="handoff", dataset_id="dataset", source_snapshot_digest="source",
            source_report=files["source_report.json"], mapping=files["mapping.json"],
            population=files["image_population.jsonl"], inspection_script=files["inspect_source.py"],
            validation_receipt_ref=str(tmp_path / "validation.json"), digest=digest_payload(files),
        ).model_dump(mode="json"),
        "previous_attempt_root": str(previous),
    }


def test_context_writes_complete_prompt_and_keeps_unaccepted_outputs_separate(tmp_path):
    packet, prompt, token_count, key = build_stage_context("prepare", _inputs(tmp_path), tmp_path / "attempt")

    assert "You are the MedBench data preparation coding agent." in prompt
    assert "Current stage: prepare" in prompt
    assert token_count > 0 and key
    assert (tmp_path / "attempt" / "inputs" / "context.json").is_file()
    assert (tmp_path / "attempt" / "inputs" / "prompt.md").is_file()
    assert (tmp_path / "attempt" / "inputs" / "accepted" / "mapping.json").is_file()
    assert (tmp_path / "attempt" / "inputs" / "previous_reference.json").is_file()
    assert not (tmp_path / "attempt" / "inputs" / "previous" / "draft.json").exists()
    previous = next(item for item in packet.files if item.role == "previous")
    assert previous.host_path == str((tmp_path / "previous").resolve())
    assert previous.worker_path == str((tmp_path / "previous").resolve())
    assert packet.identity["stage"] == "prepare"


def test_required_context_budget_fails_without_truncating_prompt(tmp_path):
    state = _inputs(tmp_path)
    state["preparation_stage_budgets"] = {"prepare": {"initial_message_tokens": 256}}
    with pytest.raises(ContextBudgetError, match="required_context_too_large"):
        build_stage_context("prepare", state, tmp_path / "attempt")


def test_previous_reference_detects_local_mutation(tmp_path):
    state = _inputs(tmp_path)
    attempt = tmp_path / "attempt"
    build_stage_context("prepare", state, attempt)
    (tmp_path / "previous" / "outputs" / "draft.json").write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="reference changed"):
        verify_previous_reference(attempt)


def test_previous_reference_uses_docker_worker_path(tmp_path):
    state = _inputs(tmp_path)
    state["execution_runtime"] = {"backend": "docker"}

    packet, _, _, _ = build_stage_context("prepare", state, tmp_path / "attempt")

    previous = next(item for item in packet.files if item.role == "previous")
    assert previous.worker_path == "/workspace/previous"
    assert packet.sources[0].root == "/workspace/sources/source"
