from medbenchagent.benchmark_info import merge_info, missing_fields, parse_sample_count


def test_sample_count_accepts_scale_expressions() -> None:
    assert parse_sample_count("10k") == 10_000
    assert parse_sample_count("2.5k") == 2_500
    assert parse_sample_count("0") is None


def test_merge_normalizes_count_and_reports_missing_fields() -> None:
    info = merge_info({"disease": "breast cancer"}, {"sample_count": "2.5k", "language": "en"})
    assert info["sample_count"] == 2_500
    assert missing_fields(info) == ["modality", "description"]
