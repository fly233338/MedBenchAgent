import json

import pytest

from medbenchagent.data.dataset_catalog import normalize_dataset_id
from medbenchagent.runtime import source_discovery
from medbenchagent.workflow.graph import SUPERVISOR_INSTRUCTIONS, automatic_supervisor_action, available_stages


def test_dynamic_dataset_id_uses_directory_name(tmp_path):
    root = tmp_path / "BrEaST Dataset"
    root.mkdir()

    assert source_discovery.dynamic_dataset_id(f'Prepare "{root}"') == "breast-dataset"
    assert source_discovery.dynamic_dataset_id("Prepare BrEaST") is None


def test_normalize_dataset_id_lowercases_names():
    assert normalize_dataset_id("BrEaST") == "breast"


def test_discover_source_uses_explicit_path_without_llm(tmp_path, monkeypatch):
    root = tmp_path / "BrEaST"
    root.mkdir()
    (root / "image.png").write_bytes(b"pixels")

    def fail(*_args, **_kwargs):
        raise AssertionError("explicit paths must not invoke source-discovery LLM")

    monkeypatch.setattr(source_discovery, "complete_validated_json", fail)
    entry = source_discovery.discover_source(
        {"run_directory": str(tmp_path / "run"), "user_requirement": f"Prepare {root}"},
        "breast",
    )

    assert entry["dataset_id"] == "breast"
    assert entry["sources"][0]["root"] == str(root.resolve())
    decision = json.loads((tmp_path / "run" / "discovery" / "sources" / "breast" / "source_decision.json").read_text())
    assert decision["entry"] == entry


def test_discover_source_does_not_cache_conflicting_registration(tmp_path, monkeypatch):
    root = tmp_path / "BrEaST"
    root.mkdir()
    (root / "image.png").write_bytes(b"pixels")

    def fail(*_args, **_kwargs):
        raise ValueError("Conflicting dataset_sources entry: breast")

    monkeypatch.setattr("medbenchagent.runtime.sessions.upsert_dataset_source", fail)
    with pytest.raises(ValueError, match="Conflicting dataset_sources"):
        source_discovery.discover_source(
            {"run_directory": str(tmp_path / "run"), "experiments_root": str(tmp_path),
             "experiment_name": "pilot", "user_requirement": f"Prepare {root}"},
            "breast",
        )
    assert not (tmp_path / "run" / "discovery" / "sources" / "breast" / "source_decision.json").exists()


def test_supervisor_remains_first_decision_for_explicit_path(tmp_path):
    root = tmp_path / "BrEaST"
    root.mkdir()
    (root / "image.png").write_bytes(b"pixels")
    context = {"run_directory": str(tmp_path / "run"), "user_requirement": f"Prepare {root}"}

    assert automatic_supervisor_action(context) is None
    assert {"ask_user", "prepare_data"} <= available_stages(context)
    assert "DataProcessingAgent" in SUPERVISOR_INSTRUCTIONS
    assert "Do not ask" in SUPERVISOR_INSTRUCTIONS
    assert "discover_datasets" in SUPERVISOR_INSTRUCTIONS
    assert "请" not in SUPERVISOR_INSTRUCTIONS


def test_source_discovery_still_reports_unusable_path(tmp_path):
    root = tmp_path / "BrEaST"
    root.mkdir()

    error = source_discovery.explicit_source_error(f"Prepare {root}")
    assert error is not None
    assert "empty" in error


def test_supervisor_requires_candidate_discovery_before_selection(tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    (control / "artifacts.json").write_text('{"versions": {}}', encoding="utf-8")
    context = {"brief_ready": True, "artifact_versions": {}, "user_requirement": "Build a benchmark",
               "run_directory": str(tmp_path)}

    stages = available_stages(context)

    assert "discover_datasets" in stages
    assert "select_datasets" not in stages
