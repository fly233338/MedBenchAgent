from dataclasses import replace
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
    with pytest.raises(ValueError, match="Unknown dataset"):
        select_datasets(c)


def test_single_and_multiple_selection(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    prepared_dataset(tmp_path, "a", monkeypatch)
    monkeypatch.setattr("medbenchagent.planning.dataset_selection.complete_json", lambda *a: {"dataset_ids": None})
    assert select_datasets(c)["updates"]["selected_dataset_ids"] == ["a"]
    prepared_dataset(tmp_path, "b", monkeypatch)
    responses = iter([{"dataset_ids": None}, {"selected_dataset_ids": ["a", "b"], "selection_reason": {"a": "Relevant", "b": "Also relevant"}}])
    monkeypatch.setattr("medbenchagent.planning.dataset_selection.complete_json", lambda *a: next(responses))
    assert select_datasets(c)["updates"]["selected_dataset_ids"] == ["a", "b"]


def test_no_related_candidates_requests_clarification(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    for d in ("a", "b"):
        prepared_dataset(tmp_path, d, monkeypatch)
    responses = iter([{"dataset_ids": None}, {"selected_dataset_ids": [], "selection_reason": {}}])
    monkeypatch.setattr("medbenchagent.planning.dataset_selection.complete_json", lambda *a: next(responses))
    assert select_datasets(c)["status"] == "needs_input"
