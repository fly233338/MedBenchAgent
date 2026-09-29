import asyncio
import json

from medbenchagent.data.validation import AnnotationValidationError, validate_v4_record
from medbenchagent.runtime.preparation_session import CodingSession


def test_coding_session_resume_and_cancel(tmp_path):
    calls = []

    def runner(payload):
        calls.append(payload)
        return {"status": "submitted", "candidate": str(tmp_path / "output")}

    session = CodingSession(tmp_path, "conversation-1", runner)
    first = asyncio.run(session.run("prepare"))
    second = asyncio.run(session.continue_with("fix validation"))
    asyncio.run(session.cancel())
    third = asyncio.run(session.continue_with("resume"))
    assert first.status == second.status == "submitted"
    assert third.status == "blocked"
    assert [call["conversation_id"] for call in calls] == ["conversation-1", "conversation-1"]
    assert json.loads((tmp_path / "state" / "session_state.json").read_text())["status"] == "blocked"


def test_v4_finding_contract():
    record = {
        "patient": {"patient_id": "p", "annotation": {}},
        "case": {"case_id": "c", "annotation": {}},
        "study": {"study_id": "s", "annotation": {}},
        "image": {"image_id": "i", "image_path": "images/i.png", "modality": "CT",
                  "anatomy": None, "annotation": {"findings": [{"finding_id": "f1",
                  "status": "known", "bbox": [0, 0, 2, 2]}]}},
        "provenance": {"source_id": "src", "source_version": None, "original_ids": {},
                       "original_image_path": "i.png", "record_ref": "row:1", "plan_ref": "plan"},
    }
    assert validate_v4_record(record)["image"]["image_id"] == "i"
    record["image"]["annotation"]["findings"][0]["status"] = "invalid"
    try:
        validate_v4_record(record)
    except AnnotationValidationError:
        pass
    else:
        raise AssertionError("invalid finding status must be rejected")
