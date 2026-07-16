import json
from pathlib import Path

import medbenchagent.benchmark_info as benchmark_info
from medbenchagent.benchmark_info import (
    extract_benchmark_field,
    merge_info,
    missing_fields,
    parse_sample_count,
    write_working_blueprint,
)


def test_sample_count_accepts_scale_expressions() -> None:
    assert parse_sample_count("10k") == 10_000
    assert parse_sample_count("2.5k") == 2_500
    assert parse_sample_count("0") is None


def test_merge_normalizes_count_and_reports_missing_fields() -> None:
    info = merge_info({"disease": "breast cancer"}, {"sample_count": "2.5k", "language": "en"})
    assert info["sample_count"] == 2_500
    assert missing_fields(info) == ["modality", "description"]


def test_extract_uses_only_five_field_schema(monkeypatch) -> None:
    captured = {}

    def fake_complete_json(system_prompt: str, user_prompt: str):
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        return {
            "disease": "breast cancer",
            "modality": "mammography",
            "description": "diagnosis benchmark",
            "language": "en",
            "sample_count": 20,
        }

    monkeypatch.setattr(benchmark_info, "complete_json", fake_complete_json)
    result = benchmark_info.extract_benchmark_info("创建一个英文乳腺癌钼靶摄影 benchmark，目标20题")

    assert result["disease"] == "breast cancer"
    assert result["sample_count"] == 20
    assert "Blueprint:" not in captured["user"]
    assert '"disease"' in captured["system"]


def test_extract_field_requests_and_returns_only_the_missing_field(monkeypatch) -> None:
    captured = {}

    def fake_complete_json(system_prompt: str, user_prompt: str):
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        return {"disease": "breast cancer"}

    monkeypatch.setattr(benchmark_info, "complete_json", fake_complete_json)

    assert extract_benchmark_field("disease", "乳腺癌") == "breast cancer"
    assert '"disease": value' in captured["system"]
    assert "User answer for disease" in captured["user"]


def test_working_blueprint_replaces_info_without_changing_template(tmp_path: Path) -> None:
    template = tmp_path / "template.json"
    template.write_text(
        json.dumps({"benchmark_info": {"disease": "str"}, "tasks": {"diagnosis": {}}}), encoding="utf-8"
    )
    output_path = Path(
        write_working_blueprint({"disease": "breast cancer", "sample_count": 20}, tmp_path / "runs", template)
    )

    assert json.loads(template.read_text(encoding="utf-8"))["benchmark_info"] == {"disease": "str"}
    assert json.loads(output_path.read_text(encoding="utf-8"))["benchmark_info"] == {
        "disease": "breast cancer",
        "sample_count": 20,
    }
