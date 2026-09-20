import json
from pathlib import Path

import medbenchagent.planning.benchmark_info as benchmark_info
from medbenchagent.planning.benchmark_info import (
    merge_info,
    missing_fields,
    write_working_blueprint,
)


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
        return {
            "diseases": ["breast cancer"],
            "modalities": ["mammography"],
            "clinical_scope": "diagnosis benchmark",
            "evaluation_focus": None,
            "language": "en",
            "sample_count": 20,
        }

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
