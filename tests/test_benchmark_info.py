import pytest

from medbenchagent.planning.benchmark_info import _validate_extraction


def _brief(evidence):
    return {
        "clinical_scope": "thoracic imaging",
        "modalities": ["Computed Tomography"],
        "language": "en",
        "sample_count": "20",
        "sample_scope": [{
            "id": "scope-1", "kind": "modality", "description": "CT cases",
            "quote": "CT cases", "modalities": ["Computed Tomography"], "operator": "in", "logic": "any",
        }],
        "allocation_preferences": [],
        "dataset_mentions": ["Example CT"],
        "intent_notes": ["Focus on image interpretation"],
        "evidence": evidence,
    }


def test_evidence_validates_source_quotes_independently_of_normalized_values():
    result = _validate_extraction(_brief([
        {"field": "clinical_scope", "quote": "thoracic imaging", "input_index": 0},
        {"field": "modalities", "quote": "CT cases", "input_index": 0},
        {"field": "language", "quote": "English benchmark", "input_index": 0},
        {"field": "sample_count", "quote": "20 questions", "input_index": 0},
        {"field": "dataset_mentions", "quote": "Example CT", "input_index": 0},
        {"field": "intent_notes", "quote": "image interpretation", "input_index": 0},
    ]), ["Build a 20 questions English benchmark for thoracic imaging CT cases using Example CT; focus on image interpretation"])

    assert result["modalities"] == ["Computed Tomography"]
    assert result["sample_count"] == 20


def test_evidence_rejects_quotes_not_present_in_user_inputs():
    with pytest.raises(ValueError, match="quote must occur"):
        _validate_extraction(_brief([
            {"field": "clinical_scope", "quote": "abdominal imaging", "input_index": 0},
            {"field": "modalities", "quote": "CT cases", "input_index": 0},
            {"field": "language", "quote": "English benchmark", "input_index": 0},
            {"field": "sample_count", "quote": "20 questions", "input_index": 0},
            {"field": "dataset_mentions", "quote": "Example CT", "input_index": 0},
            {"field": "intent_notes", "quote": "image interpretation", "input_index": 0},
        ]), ["Build a 20 questions English benchmark for thoracic imaging CT cases using Example CT; focus on image interpretation"])


def test_missing_modality_scope_is_derived_from_source_quote():
    result = _validate_extraction({
        "clinical_scope": "breast cancer", "modalities": ["ultrasound"], "language": "en",
        "sample_count": 20, "dataset_mentions": ["breast-us"], "intent_notes": [],
        "sample_scope": [], "allocation_preferences": [], "evidence": [
            {"field": "clinical_scope", "quote": "breast cancer", "input_index": 0},
            {"field": "modalities", "quote": "breast-us", "input_index": 0},
            {"field": "language", "quote": "English", "input_index": 0},
            {"field": "sample_count", "quote": "20", "input_index": 0},
            {"field": "dataset_mentions", "quote": "breast-us", "input_index": 0},
        ],
    }, ["Build a 20 question English benchmark for breast cancer using breast-us"])
    assert result["sample_scope"][0]["modalities"] == ["ultrasound"]


def test_unquoted_intent_notes_are_dropped():
    result = _validate_extraction(_brief([
        {"field": "clinical_scope", "quote": "thoracic imaging", "input_index": 0},
        {"field": "modalities", "quote": "CT cases", "input_index": 0},
        {"field": "language", "quote": "English benchmark", "input_index": 0},
        {"field": "sample_count", "quote": "20 questions", "input_index": 0},
        {"field": "dataset_mentions", "quote": "Example CT", "input_index": 0},
    ]), ["Build a 20 questions English benchmark for thoracic imaging CT cases using Example CT"])
    assert result["intent_notes"] == []
