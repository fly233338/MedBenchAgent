import json
from dataclasses import replace
from pathlib import Path
import pytest

import medbenchagent.planning.benchmark_info as benchmark_info
from medbenchagent.planning.benchmark_info import (
    merge_info,
    missing_fields,
    write_working_blueprint,
)


def grounded(values, quote="Build"):
    values = dict(values)
    values.setdefault("sample_scope", [])
    values["evidence"] = [
        {"field": field, "value": value, "quote": quote}
        for field in (*benchmark_info.REQUIRED_FIELDS, *benchmark_info.OPTIONAL_FIELDS, "max_total_tasks_per_image")
        for value in (values.get(field, []) if field in {"diseases", "modalities"} else [values.get(field)])
        if value is not None and value != ""
    ]
    return values


@pytest.mark.parametrize("language", [None, "English", "French", "", "fr"])
def test_unsupported_language_needs_clarification(language):
    assert missing_fields({"clinical_scope": "Any", "sample_count": 1, "language": language}) == ["language"]


def test_modality_union_and_evidence_contract(monkeypatch):
    answer = grounded({"clinical_scope": "Imaging", "modalities": ["CT", "MRI"],
                       "language": "en", "sample_count": 4,
                       "sample_scope": [{"id": "m", "kind": "modality", "description": "CT or MRI",
                                         "quote": "CT or MRI", "modalities": ["CT", "MRI"]}]},
                      "CT or MRI")
    monkeypatch.setattr(benchmark_info, "complete_json", lambda *a: answer)
    assert len(benchmark_info.extract_benchmark_info("CT or MRI")["sample_scope"]) == 1
    answer["sample_scope"][0]["modalities"] = ["CT"]
    with pytest.raises(ValueError, match="all explicit modalities"):
        benchmark_info.extract_benchmark_info("CT or MRI")
    answer.pop("sample_scope")
    with pytest.raises(ValueError, match="sample_scope"):
        benchmark_info.extract_benchmark_info("CT or MRI")


@pytest.mark.parametrize("evidence", [[], [{"field": "clinical_scope", "value": "Imaging", "quote": "invented"}]])
def test_missing_or_fabricated_extraction_evidence(monkeypatch, evidence):
    monkeypatch.setattr(benchmark_info, "complete_json", lambda *a: {
        "clinical_scope": "Imaging", "sample_scope": [], "evidence": evidence})
    with pytest.raises(ValueError, match="evidence"):
        benchmark_info.extract_benchmark_info("Build")


def test_merge_normalizes_count_and_reports_missing_fields() -> None:
    info = merge_info({"clinical_scope": "breast disease"}, {"sample_count": "2.5k", "language": "en"})
    assert info["sample_count"] == 2_500
    assert missing_fields(info) == []


def test_multi_disease_scope_does_not_invent_focus():
    info = merge_info({}, {"clinical_scope": "Rare diseases", "diseases": ["A", "B"],
                           "modalities": ["CT", "MRI"], "sample_count": "10k", "language": "en"})
    assert not missing_fields(info)
    assert info["evaluation_focus"] is None
    assert info["diseases"] == ["A", "B"] and info["modalities"] == ["CT", "MRI"]
    assert benchmark_info.parse_sample_count(True) is None
    assert benchmark_info.parse_sample_count("1.5") is None


def test_extract_uses_benchmark_level_schema(monkeypatch) -> None:
    def fake_complete_json(system_prompt: str, user_prompt: str):
        return grounded({
            "diseases": ["breast cancer"],
            "modalities": ["mammography"],
            "clinical_scope": "diagnosis benchmark",
            "evaluation_focus": None,
            "language": "en",
            "sample_count": 20,
            "sample_scope": [{"id": "m", "kind": "modality", "description": "Mammography", "quote": "钼靶摄影", "modalities": ["mammography"]}],
        }, "创建一个英文乳腺癌钼靶摄影 benchmark，目标20题")

    monkeypatch.setattr(benchmark_info, "complete_json", fake_complete_json)
    result = benchmark_info.extract_benchmark_info("创建一个英文乳腺癌钼靶摄影 benchmark，目标20题")

    assert result["diseases"] == ["breast cancer"]
    assert result["sample_count"] == 20
    assert result["modalities"] == ["mammography"]
    assert result["evaluation_focus"] is None
    assert not missing_fields(result)


def test_working_blueprint_replaces_info_without_changing_template(tmp_path: Path) -> None:
    template = tmp_path / "template.json"
    template.write_text(
        json.dumps({"benchmark_info": {"clinical_scope": "str"}, "tasks": {"diagnosis": {}}}), encoding="utf-8"
    )
    output_path = Path(
        write_working_blueprint({"diseases": ["breast cancer"], "sample_count": 20}, tmp_path / "runs", template)
    )

    assert json.loads(template.read_text(encoding="utf-8"))["benchmark_info"] == {"clinical_scope": "str"}
    assert json.loads(output_path.read_text(encoding="utf-8"))["benchmark_info"] == {
        "diseases": ["breast cancer"],
        "sample_count": 20,
    }


@pytest.mark.parametrize("value,expected,source,missing", [(None, 17, "configuration", []), (8, 8, "user", []),
                                                          (0, 0, "user", ["sample_count"]),
                                                          ("", "", "user", ["sample_count"]),
                                                          ("many", "many", "user", ["sample_count"])])
def test_specify_defaults_only_absent_count(tmp_path, monkeypatch, value, expected, source, missing):
    from medbenchagent import settings
    from medbenchagent.workflow.graph import specify
    cfg = settings.get_settings()
    monkeypatch.setattr(settings, "_SETTINGS", replace(cfg, benchmark=replace(cfg.benchmark, default_sample_count=17)))
    monkeypatch.setattr(benchmark_info, "complete_json", lambda *a: grounded({"clinical_scope": "Disease", "language": "en", "sample_count": value}))
    c = {"run_directory": str(tmp_path), "operation_directory": str(tmp_path / "op"), "user_requirement": "Build"}
    updates = specify(c)["updates"]
    assert updates["benchmark_info"]["sample_count"] == expected
    assert updates["sample_count_source"] == source
    assert updates["brief_missing_fields"] == missing


def test_missing_language_gates_repeated_specify_until_answer(tmp_path, monkeypatch):
    from medbenchagent.workflow.graph import specify, available_stages
    monkeypatch.setattr(benchmark_info, "complete_json", lambda *a: grounded({"clinical_scope": "Disease", "language": None}))
    c = {"run_directory": str(tmp_path), "operation_directory": str(tmp_path / "op"), "user_requirement": "Build",
         "run_intent": {"intent": "build_benchmark", "prepare_dataset_ids": []}}
    c.update(specify(c)["updates"])
    assert c["brief_missing_fields"] == ["language"]
    tools = available_stages(c)
    assert "ask_user" in tools and "list_datasets" in tools
    assert not tools & {"specify", "select_datasets", "plan", "prepare_data"}
    with pytest.raises(ValueError, match="already been extracted"):
        specify(c)
    c["user_requirement"] += "\nEnglish"
    assert "specify" in available_stages(c)


@pytest.mark.parametrize("limit,expected", [(None, 3), (1, 1)])
def test_scope_and_explicit_total_limit_are_execution_requirements(tmp_path, monkeypatch, limit, expected):
    from medbenchagent.workflow.graph import specify
    from medbenchagent.runtime.dependencies import require
    answer = {"clinical_scope": "Breast", "language": "en", "sample_count": 6, "modalities": ["US"],
              "sample_scope": [{"id": "us", "kind": "modality", "description": "US only", "quote": "US", "modalities": ["US"]}],
              "max_total_tasks_per_image": limit}
    monkeypatch.setattr(benchmark_info, "complete_json", lambda *a: grounded(answer, "US pathology benchmark"))
    c = {"run_directory": str(tmp_path), "operation_directory": str(tmp_path / "op"),
         "user_requirement": "US pathology benchmark", "configured_total_tasks_per_image": 3}
    updates = specify(c)["updates"]
    c.update(updates)
    assert c["max_total_tasks_per_image"] == expected
    assert "sample_scope" not in c["benchmark_info"]
    assert require(c, "GenerationConfig")["GenerationConfig"]["references"]["max_total_tasks_per_image"] == expected
    assert require(c, "Requirement")["Requirement"]["references"]["sample_scope"] == answer["sample_scope"]
    c["user_requirement"] += " corrected"
    answer["sample_scope"] = []
    with pytest.raises(ValueError, match="modalities"):
        specify(c)
