from dataclasses import replace
import json
from pathlib import Path
import pytest
from medbenchagent import settings
from medbenchagent.planning.dataset_selection import select_datasets
from medbenchagent.runtime.dependencies import publish
from test_dataset_cards import prepared_dataset


def context(tmp_path, monkeypatch):
    cfg = settings.get_settings()
    monkeypatch.setattr(settings, "_SETTINGS", replace(cfg, paths=replace(cfg.paths, dataset_catalog=tmp_path / "catalog.json")))
    c = {"run_directory": str(tmp_path / "run"), "operation_directory": str(tmp_path / "run/op"), "user_requirement": "Choose relevant data"}
    publish(c, "Requirement", {}, ())
    publish(c, "DataConfig", {}, ())
    return c


def test_explicit_unknown_does_not_fall_back(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    prepared_dataset(tmp_path, "a", monkeypatch)
    monkeypatch.setattr("medbenchagent.planning.dataset_selection.complete_json", lambda *a: {"dataset_ids": ["unknown"]})
    outcome = select_datasets(c)
    assert outcome["status"] == "blocked" and "Unknown dataset" in outcome["summary"]


def test_new_catalog_entry_does_not_enter_existing_run_view(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    prepared_dataset(tmp_path, "a", monkeypatch)
    frozen = json.loads((tmp_path / "catalog.json").read_text(encoding="utf-8"))["datasets"]
    snapshot_path = tmp_path / "run" / "frozen_catalog.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps({"entries": frozen}), encoding="utf-8")
    c["task_catalog_snapshot_path"] = str(snapshot_path)
    prepared_dataset(tmp_path, "b", monkeypatch)
    c["requested_dataset_ids"] = ["a"]
    outcome = select_datasets(c)
    assert outcome["status"] == "success"
    candidates = json.loads(Path(outcome["updates"]["candidate_snapshot_path"]).read_text(encoding="utf-8"))
    assert [entry["dataset_id"] for entry in candidates["catalog_entries"]] == ["a"]


def test_explicit_ids_bypass_model_and_new_preparation_invalidates_selection(tmp_path, monkeypatch):
    from medbenchagent.runtime.dependencies import require
    c = context(tmp_path, monkeypatch)
    prepared_dataset(tmp_path, "a", monkeypatch)
    c["requested_dataset_ids"] = ["a"]
    publish(c, "PreparedCandidates", {"entries": {}}, ())
    monkeypatch.setattr("medbenchagent.planning.dataset_selection.complete_json", lambda *a: pytest.fail("Explicit IDs must bypass LLM"))
    outcome = select_datasets(c)
    c.update(outcome["updates"])
    assert c["selected_dataset_ids"] == ["a"]
    require(c, "DatasetSelection")
    publish(c, "PreparedCandidates", {"entries": {"new": {}}}, ())
    with pytest.raises(ValueError, match="stale"):
        require(c, "DatasetSelection")
    with pytest.raises(ValueError, match="new selection operation"):
        select_datasets(c)


def test_single_and_multiple_selection(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    prepared_dataset(tmp_path, "a", monkeypatch)
    responses = iter([{"dataset_ids": None}, {"selected_dataset_ids": ["a"], "selection_reason": {"a": "Relevant"}}])
    monkeypatch.setattr("medbenchagent.planning.dataset_selection.complete_json", lambda *a: next(responses))
    assert select_datasets(c)["updates"]["selected_dataset_ids"] == ["a"]
    prepared_dataset(tmp_path, "b", monkeypatch)
    c["operation_directory"] = str(tmp_path / "run/second-selection")
    responses = iter([{"dataset_ids": None}, {"selected_dataset_ids": ["a", "b"], "selection_reason": {"a": "Relevant", "b": "Also relevant"}}])
    monkeypatch.setattr("medbenchagent.planning.dataset_selection.complete_json", lambda *a: next(responses))
    assert select_datasets(c)["updates"]["selected_dataset_ids"] == ["a", "b"]


def test_no_related_candidates_blocks_without_path_question(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    for d in ("a", "b"):
        prepared_dataset(tmp_path, d, monkeypatch)
    responses = iter([{"dataset_ids": None}, {"selected_dataset_ids": [], "selection_reason": {}}])
    monkeypatch.setattr("medbenchagent.planning.dataset_selection.complete_json", lambda *a: next(responses))
    assert select_datasets(c)["status"] == "blocked"


def test_only_candidate_must_be_relevant(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    prepared_dataset(tmp_path, "a", monkeypatch)
    responses = iter([{"dataset_ids": None}, {"selected_dataset_ids": [], "selection_reason": {}}])
    monkeypatch.setattr("medbenchagent.planning.dataset_selection.complete_json", lambda *a: next(responses))
    outcome = select_datasets(c)
    assert outcome["status"] == "blocked"
    snapshot = json.loads(Path(outcome["updates"]["candidate_snapshot_path"]).read_text(encoding="utf-8"))
    assert snapshot["selection_method"] == "model"


def test_snapshot_preserves_rejected_candidates_and_replays_without_live_cards(tmp_path, monkeypatch):
    from medbenchagent.data.dataset_catalog import refresh_card
    from medbenchagent.data.sources import digest_file
    from medbenchagent.runtime.dependencies import read_index
    c = context(tmp_path, monkeypatch)
    entries = {d: prepared_dataset(tmp_path, d, monkeypatch) for d in ("a", "b")}
    catalog = tmp_path / "catalog.json"
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    payload["datasets"].append({**entries["a"], "dataset_id": "disabled", "available": False})
    payload["datasets"].append({**entries["a"], "dataset_id": "unprepared", "annotation_path": str(tmp_path / "missing/annotation.jsonl")})
    catalog.write_text(json.dumps(payload), encoding="utf-8")
    responses = iter([{"dataset_ids": None}, {"selected_dataset_ids": ["a"], "selection_reason": {"a": "Relevant"}}])
    prompts = []
    def model(system, prompt):
        prompts.append(json.loads(prompt))
        return next(responses)
    monkeypatch.setattr("medbenchagent.planning.dataset_selection.complete_json", model)
    outcome = select_datasets(c)
    document_path = Path(outcome["updates"]["selection_path"])
    before = document_path.read_bytes()
    saved = json.loads(before)
    assert saved["candidate_order"] == ["a", "b"]
    assert saved["name_resolution_input"] == prompts[0]
    assert saved["selection_input"] == prompts[1]
    assert saved["selection_method"] == "model"
    assert set(saved["excluded_candidates"]) == {"disabled", "unprepared"}
    assert saved["candidates"]["b"]["card_digest"] == digest_file(entries["b"]["dataset_card_path"])
    version = c["artifact_versions"]["DatasetSelection"]
    assert "DatasetCandidates" in read_index(c)["versions"][version]["dependencies"]
    monkeypatch.setattr("medbenchagent.data.dataset_card.complete_json", lambda *a: {"description": "Updated intrinsic description"})
    refresh_card("b", catalog_path=catalog)
    # Selected and rejected original Cards are no longer needed for this operation.
    for entry in entries.values():
        Path(entry["dataset_card_path"]).unlink()
    monkeypatch.setattr("medbenchagent.planning.dataset_selection.complete_json", lambda *a: pytest.fail("Must replay frozen decision"))
    assert select_datasets(c)["updates"]["selected_dataset_ids"] == ["a"]
    assert document_path.read_bytes() == before
    c["operation_directory"] = str(tmp_path / "run/new-selection")
    monkeypatch.setattr("medbenchagent.planning.dataset_selection.complete_json", lambda *a: {"dataset_ids": ["b"]})
    updated = select_datasets(c)
    new_doc = json.loads(Path(updated["updates"]["selection_path"]).read_text(encoding="utf-8"))
    assert new_doc["candidates"]["b"]["summary"]["description"] == "Updated intrinsic description"
    assert document_path.read_bytes() == before


def test_model_failure_reuses_candidate_snapshot_on_retry(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    for dataset in ("a", "b"):
        prepared_dataset(tmp_path, dataset, monkeypatch)
    def failing(system, prompt):
        if "Extract only" in system:
            return {"dataset_ids": None}
        raise ValueError("Temporary model failure")
    monkeypatch.setattr("medbenchagent.planning.dataset_selection.complete_json", failing)
    with pytest.raises(ValueError, match="Temporary"):
        select_datasets(c)
    snapshot = Path(c["operation_directory"]) / "dataset_candidates.json"
    before = snapshot.read_bytes()
    (tmp_path / "catalog.json").write_text('{"datasets":[]}', encoding="utf-8")
    def retry(system, prompt):
        assert "Extract only" not in system
        assert [v["dataset_id"] for v in json.loads(prompt)["cards"]] == ["a", "b"]
        return {"selected_dataset_ids": ["a"], "selection_reason": {"a": "Relevant"}}
    monkeypatch.setattr("medbenchagent.planning.dataset_selection.complete_json", retry)
    assert select_datasets(c)["updates"]["selected_dataset_ids"] == ["a"]
    assert snapshot.read_bytes() == before


@pytest.mark.parametrize("explicit", [True, False])
def test_missing_prepared_requests_separate_preparation_and_replays(tmp_path, monkeypatch, explicit):
    c = context(tmp_path, monkeypatch)
    cfg = settings.get_settings()
    monkeypatch.setattr(settings, "_SETTINGS", replace(cfg, dataset_sources=({"dataset_id": "raw", "name": "Raw",
                        "description": "Matching population", "sources": [{"id": "source", "root": str(tmp_path), "kind": "raw"}]},)))
    if explicit:
        c["requested_dataset_ids"] = ["raw"]
        monkeypatch.setattr("medbenchagent.planning.dataset_selection.complete_json", lambda *a: pytest.fail("Explicit ID needs no model"))
    else:
        monkeypatch.setattr("medbenchagent.planning.dataset_selection.complete_json",
                            lambda *a: {"dataset_ids": None})
    outcome = select_datasets(c)
    assert outcome["status"] == "blocked" and "Request dataset preparation" in outcome["summary"]
    monkeypatch.setattr("medbenchagent.planning.dataset_selection.complete_json",
                        lambda *a: pytest.fail("Replay uses frozen candidates"))
    assert select_datasets(c)["status"] == "blocked"


def test_missing_source_path_blocks_without_asking_user(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    cfg = settings.get_settings()
    monkeypatch.setattr(settings, "_SETTINGS", replace(cfg, dataset_sources=({"dataset_id": "raw", "name": "Raw",
                        "description": "Population", "sources": [{"id": "source", "root": str(tmp_path / "missing"), "kind": "raw"}]},)))
    c["requested_dataset_ids"] = ["raw"]
    outcome = select_datasets(c)
    assert outcome["status"] == "blocked" and "Request dataset preparation" in outcome["summary"]
    assert "global_issue" not in outcome["updates"]
