import hashlib
import json
from pathlib import Path

from PIL import Image

from medbenchagent.audit import _answer_evidence_matches, audit_run
from medbenchagent.generation.context import evidence_sha256
from medbenchagent.llm import ModelUnavailable
from medbenchagent.run_artifacts import blueprint_sha256
from medbenchagent.semantic_audit import semantic_audit_run
from medbenchagent.workflow.graph import _generation_summary_node


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _initial_report(accepted: int, rejected: int = 0) -> dict:
    return {
        "schema_version": "audit-report.v1",
        "input": accepted + rejected,
        "hard_audit": {
            "accepted": accepted,
            "rejected": rejected,
            "reason_counts": {},
            "duplicate_statistics": {},
        },
        "semantic_audit": {
            "eligible": accepted,
            "sampled": 0,
            "completed": 0,
            "passed": 0,
            "flagged": 0,
            "not_run": 0,
            "not_sampled": accepted,
            "violation_counts": {},
        },
    }


def _jsonl_ids(path: Path) -> set[str]:
    return {
        str(json.loads(line)["id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    }


def _jsonl_items(path: Path) -> set[str]:
    return {
        json.dumps(json.loads(line), sort_keys=True, separators=(",", ":"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    }


def _assert_report_invariants(report: dict, audited_path: Path, flagged_path: Path) -> None:
    hard = report["hard_audit"]
    semantic = report["semantic_audit"]
    assert report["schema_version"] == "audit-report.v1"
    assert report["input"] == hard["accepted"] + hard["rejected"]
    assert semantic["eligible"] == hard["accepted"]
    assert semantic["eligible"] == semantic["sampled"] + semantic["not_sampled"]
    assert semantic["sampled"] == semantic["completed"] + semantic["not_run"]
    assert semantic["completed"] == semantic["passed"] + semantic["flagged"]
    assert len(_jsonl_ids(audited_path)) == hard["accepted"]
    assert _jsonl_items(flagged_path) <= _jsonl_items(audited_path)
    assert len(_jsonl_ids(flagged_path)) == semantic["flagged"]


def test_audit_keeps_repeated_questions_and_rejects_item_duplicates(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "images" / "a.png"
    image_path.parent.mkdir()
    image = Image.new("L", (2, 2))
    image.putdata([0, 1, 2, 3])
    image.save(image_path)
    blueprint = {
        "dataset": {"dataset_id": "set"},
        "tasks": {
            "view": {
                "response_format": "letter",
                "supporting_fields": ["image.annotation.view"],
                "answer_field": "image.annotation.view",
                "answer_value_mapping": [
                    {"raw_value": "A", "option": "A"},
                    {"raw_value": "B", "option": "B"},
                ],
            }
        },
    }
    blueprint_path = tmp_path / "blueprint.json"
    blueprint_path.write_text(json.dumps(blueprint), encoding="utf-8")
    annotation = tmp_path / "annotation.jsonl"
    annotation.write_text(
        json.dumps(
            {
                "patient": {"patient_id": None, "annotation": {}},
                "case": {"case_id": None, "annotation": {}},
                "study": {"study_id": None, "annotation": {}},
                "image": {
                    "image_id": "image",
                    "image_path": "images/a.dicom",
                    "modality": "Mammography",
                    "anatomy": "Breast",
                    "annotation": {"view": "A"},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    item = {
        "id": "one",
        "dataset": "set",
        "task_id": "view",
        "image_id": "image",
        "question": "Which view?",
        "options": ["A", "B"],
        "answer": "A",
    }
    raw, provenance, events, manifest = (
        tmp_path / name
        for name in ("raw.jsonl", "provenance.jsonl", "events.jsonl", "image_manifest.jsonl")
    )
    exact_duplicate = dict(item)
    _write_jsonl(raw, [item, {**item, "id": "two"}, {**item, "question": "Other?"}, exact_duplicate])
    source = {
        "dataset_id": "set",
        "task_id": "view",
        "image_id": "image",
        "supporting_fields": ["image.annotation.view"],
        "source_row": 1,
        "answer_evidence": {
            "path": "image.annotation.view",
            "indices": [],
            "evidence_sha256": evidence_sha256("image.annotation.view", [], "A"),
        },
        "generation_call_ids": [],
        "review_call_ids": [],
        "blueprint_hash": blueprint_sha256(blueprint_path),
    }
    _write_jsonl(
        provenance,
        [
            {"item_id": "one", **source},
            {"item_id": "two", **source},
            {"item_id": "one", **source},
        ],
    )
    events.touch()
    _generation_summary_node(
        {
            "construction_events_path": str(events),
            "run_id": "run",
            "selected_dataset_id": "set",
            "task_counts": {"view": 4},
            "generation_shortfalls": {"view": 0},
        }
    )
    _write_jsonl(
        manifest,
        [
            {
                "image_id": "image",
                "png_path": "images/a.png",
                "width": 2,
                "height": 2,
                "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            }
        ],
    )

    original_read_bytes = Path.read_bytes
    blueprint_reads = 0

    def guarded_read_bytes(path: Path) -> bytes:
        nonlocal blueprint_reads
        if path == image_path:
            raise AssertionError("read_bytes")
        if path == blueprint_path:
            blueprint_reads += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(Image.Image, "getdata", lambda *_args: (_ for _ in ()).throw(AssertionError("getdata")))
    result = audit_run(
        raw_path=raw,
        blueprint_path=blueprint_path,
        provenance_path=provenance,
        events_path=events,
        manifest_path=manifest,
        annotation_path=annotation,
        annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),
        output_directory=tmp_path,
        run_id="run",
    )

    assert result["item_count"] == 2
    assert blueprint_reads == 1
    assert len((Path(result["rejected_path"])).read_text(encoding="utf-8").splitlines()) == 2
    report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
    assert report["hard_audit"]["duplicate_statistics"] == {
        "duplicate_item_id_count": 2,
        "exact_item_duplicate_count": 1,
        "repeated_question_count": 2,
        "question_repetition_rate": 0.5,
    }
    _assert_report_invariants(
        report,
        Path(result["audited_path"]),
        Path(result["flagged_path"]),
    )
    assert "retention" not in report
    event_types = [
        json.loads(line)["event_type"]
        for line in events.read_text(encoding="utf-8").splitlines()
    ]
    assert event_types[0] == "generation_summary"
    first_audit_item = next(
        index
        for index, event_type in enumerate(event_types)
        if event_type in {"audit_accept", "audit_reject"}
    )
    assert event_types.index("generation_summary") < first_audit_item

    tampered = tmp_path / "tampered"
    tampered.mkdir()
    bad_source = {
        **source,
        "answer_evidence": source["answer_evidence"] | {"evidence_sha256": "0" * 64},
    }
    _write_jsonl(
        provenance,
        [
            {"item_id": "one", **bad_source},
            {"item_id": "two", **bad_source},
        ],
    )
    tampered_result = audit_run(
        raw_path=raw,
        blueprint_path=blueprint_path,
        provenance_path=provenance,
        events_path=events,
        manifest_path=manifest,
        annotation_path=annotation,
        annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),
        output_directory=tampered,
        run_id="run",
    )
    tampered_report = json.loads(
        Path(tampered_result["report_path"]).read_text(encoding="utf-8")
    )
    assert tampered_result["item_count"] == 0
    assert blueprint_reads == 2
    assert tampered_report["hard_audit"]["reason_counts"]["answer_evidence"] == 4
    _assert_report_invariants(
        tampered_report,
        Path(tampered_result["audited_path"]),
        Path(tampered_result["flagged_path"]),
    )

    malformed = tmp_path / "malformed-calls"
    malformed.mkdir()
    call_items = [{**item, "id": item_id} for item_id in ("bad-shape", "unknown", "valid")]
    _write_jsonl(raw, call_items)
    _write_jsonl(
        provenance,
        [
            {
                "item_id": "bad-shape",
                **source,
                "generation_call_ids": "not-a-list",
            },
            {
                "item_id": "unknown",
                **source,
                "generation_call_ids": ["missing-call"],
            },
            {"item_id": "valid", **source},
        ],
    )
    malformed_result = audit_run(
        raw_path=raw,
        blueprint_path=blueprint_path,
        provenance_path=provenance,
        events_path=events,
        manifest_path=manifest,
        annotation_path=annotation,
        annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),
        output_directory=malformed,
        run_id="run",
    )
    malformed_report = json.loads(
        Path(malformed_result["report_path"]).read_text(encoding="utf-8")
    )
    assert malformed_result["item_count"] == 1
    assert malformed_report["hard_audit"]["reason_counts"]["provenance_calls"] == 2
    assert blueprint_reads == 3


def test_audit_accepts_readable_color_images_and_rejects_constant_thumbnails(tmp_path: Path) -> None:
    color_path = tmp_path / "color.png"
    color = Image.new("RGB", (2, 2), "white")
    color.putpixel((0, 0), (0, 0, 0))
    color.save(color_path)
    black_path = tmp_path / "black.png"
    Image.new("RGB", (2, 2), "black").save(black_path)

    from medbenchagent.audit import _image_errors

    assert _image_errors(
        {
            "png_path": color_path.name,
            "width": 2,
            "height": 2,
            "sha256": hashlib.sha256(color_path.read_bytes()).hexdigest(),
        },
        tmp_path,
    ) == []
    assert _image_errors(
        {
            "png_path": black_path.name,
            "width": 2,
            "height": 2,
            "sha256": hashlib.sha256(black_path.read_bytes()).hexdigest(),
        },
        tmp_path,
    ) == ["image_pixels"]


def test_invalid_image_dimensions_skip_bbox_bounds_without_type_error(tmp_path: Path) -> None:
    from medbenchagent.audit import _contract, _image_dimensions, _image_errors

    image_path = tmp_path / "image.png"
    image = Image.new("L", (2, 2))
    image.putdata([0, 1, 2, 3])
    image.save(image_path)
    entry = {
        "png_path": image_path.name,
        "width": "2",
        "height": 2,
        "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
    }

    assert _image_errors(entry, tmp_path) == ["image_dimensions"]
    assert _image_dimensions(entry) is None
    assert _contract(
        {"question": "Where?", "answer": [0, 0, 2, 2]},
        {"response_format": "bbox"},
        None,
    ) == []


def test_letter_contract_rejects_unhashable_options() -> None:
    from medbenchagent.audit import _contract

    assert _contract(
        {"question": "Which view?", "options": [["CC"], "MLO"], "answer": "A"},
        {"response_format": "letter"},
        (1, 1),
    ) == ["letter_answer"]


def test_answer_evidence_replays_free_text_and_bbox_exactly() -> None:
    record = {
        "patient": {"patient_id": None, "annotation": {}},
        "case": {"case_id": None, "annotation": {}},
        "study": {"study_id": None, "annotation": {}},
        "image": {
            "image_id": "image",
            "image_path": "images/a.dicom",
            "modality": "Mammography",
            "anatomy": "Breast",
            "annotation": {
                "report": "Normal study.",
                "findings": [{"bbox": [1, 2, 8, 9]}],
            },
        },
    }
    report_path = "image.annotation.report"
    report_source = {
        "answer_evidence": {
            "path": report_path,
            "indices": [],
            "evidence_sha256": evidence_sha256(report_path, [], "Normal study."),
        }
    }
    report_task = {"response_format": "free_text"}
    assert _answer_evidence_matches(
        {"answer": "Normal study."},
        report_task,
        report_source,
        record,
        [report_path],
    )
    assert not _answer_evidence_matches(
        {"answer": "Normal"},
        report_task,
        report_source,
        record,
        [report_path],
    )
    assert not _answer_evidence_matches(
        {"answer": "Normal study."},
        report_task,
        {
            "answer_evidence": report_source["answer_evidence"]
            | {"evidence_sha256": "0" * 64}
        },
        record,
        [report_path],
    )

    bbox_path = "image.annotation.findings[*].bbox[*]"
    bbox_source = {
        "answer_evidence": [
            {
                "path": bbox_path,
                "indices": [0, coordinate],
                "evidence_sha256": evidence_sha256(
                    bbox_path,
                    [0, coordinate],
                    value,
                ),
            }
            for coordinate, value in enumerate([1, 2, 8, 9])
        ]
    }
    bbox_task = {"response_format": "bbox"}
    assert _answer_evidence_matches(
        {"answer": [1.0, 2.0, 8.0, 9.0]},
        bbox_task,
        bbox_source,
        record,
        [bbox_path],
    )
    assert not _answer_evidence_matches(
        {"answer": [1.0, 2.0, 7.0, 9.0]},
        bbox_task,
        bbox_source,
        record,
        [bbox_path],
    )
    bad_coordinates = [dict(coordinate) for coordinate in bbox_source["answer_evidence"]]
    bad_coordinates[2]["indices"] = [0, 3]
    assert not _answer_evidence_matches(
        {"answer": [1.0, 2.0, 8.0, 9.0]},
        bbox_task,
        {"answer_evidence": bad_coordinates},
        record,
        [bbox_path],
    )

    changed_record = json.loads(json.dumps(record))
    changed_record["image"]["annotation"]["findings"][0]["bbox"][2] = 7
    assert not _answer_evidence_matches(
        {"answer": [1.0, 2.0, 8.0, 9.0]},
        bbox_task,
        bbox_source,
        changed_record,
        [bbox_path],
    )


def test_answer_evidence_replays_structured_letter_mapping() -> None:
    answer_path = "image.annotation.finding.shape"
    record = {
        "image": {
            "annotation": {"finding": {"shape": "irregular"}},
        }
    }
    source = {
        "answer_evidence": {
            "path": answer_path,
            "indices": [],
            "evidence_sha256": evidence_sha256(answer_path, [], "irregular"),
        }
    }
    task = {
        "response_format": "letter",
        "answer_field": answer_path,
        "answer_value_mapping": [
            {"raw_value": "oval", "option": "Oval"},
            {"raw_value": "irregular", "option": "Irregular"},
        ],
    }
    item = {"options": ["Oval", "Irregular"], "answer": "B"}

    assert _answer_evidence_matches(item, task, source, record, [answer_path])
    assert not _answer_evidence_matches(
        item | {"answer": "A"}, task, source, record, [answer_path]
    )
    assert not _answer_evidence_matches(
        item,
        task | {"answer_field": "image.annotation.finding.margin"},
        source,
        record,
        [answer_path],
    )


def test_semantic_audit_samples_once_per_task_and_flags_without_rejecting(
    tmp_path: Path, monkeypatch
) -> None:
    audited = [
        {
            "id": f"view-{index}",
            "task_id": "view",
            "dataset": "set",
            "question": f"Which view is shown in variation {index}?",
            "options": ["CC", "MLO"],
            "answer": "A",
        }
        for index in range(12)
    ] + [
        {
            "id": f"report-{index}",
            "task_id": "report",
            "dataset": "set",
            "question": f"Provide report {index}.",
            "options": None,
            "answer": "No acute finding.",
        }
        for index in range(2)
    ]
    audited_path = tmp_path / "audited.jsonl"
    flagged_path = tmp_path / "flagged.jsonl"
    report_path = tmp_path / "audit_report.json"
    blueprint_path = tmp_path / "blueprint.json"
    events_path = tmp_path / "events.jsonl"
    _write_jsonl(audited_path, audited)
    _write_jsonl(flagged_path, [])
    report_path.write_text(
        json.dumps(_initial_report(len(audited))),
        encoding="utf-8",
    )
    blueprint_path.write_text(
        json.dumps(
            {
                "tasks": {
                    "view": {
                        "description": "Identify the view.",
                        "require": "Use the image.",
                        "question_templates": ["Which view is shown?"],
                        "medical_task_description": "View recognition.",
                    },
                    "report": {
                        "description": "Generate a report.",
                        "require": "Use the image.",
                        "question_templates": ["Provide a report."],
                        "medical_task_description": "Report generation.",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    events_path.touch()
    calls: list[dict] = []

    def fake(_system: str, user: str) -> dict:
        payload = json.loads(user)
        assert "annotation" not in user
        calls.append(payload)
        return {
            "items": [
                {
                    "item_id": item["item_id"],
                    "decision": "flag" if index == 0 else "pass",
                    "violations": ["answer_leakage"] if index == 0 else [],
                }
                for index, item in enumerate(payload["items"])
            ]
        }

    monkeypatch.setattr("medbenchagent.semantic_audit.complete_json", fake)
    result = semantic_audit_run(
        audited_path=audited_path,
        flagged_path=flagged_path,
        report_path=report_path,
        blueprint_path=blueprint_path,
        events_path=events_path,
        output_directory=tmp_path,
        run_id="run",
        dataset_id="set",
        seed=7,
    )

    assert len(calls) == 2
    assert sorted(len(call["items"]) for call in calls) == [2, 10]
    records = [
        json.loads(line)
        for line in Path(result["semantic_records_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 12
    assert sum(record["decision"] == "flag" for record in records) == 2
    assert len(flagged_path.read_text(encoding="utf-8").splitlines()) == 2
    assert len(audited_path.read_text(encoding="utf-8").splitlines()) == 14
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["semantic_audit"] == {
        "eligible": 14,
        "sampled": 12,
        "completed": 12,
        "passed": 10,
        "flagged": 2,
        "not_run": 0,
        "not_sampled": 2,
        "violation_counts": {"answer_leakage": 2},
    }
    _assert_report_invariants(report, audited_path, flagged_path)


def test_semantic_audit_invalid_or_unavailable_is_fail_open(tmp_path: Path, monkeypatch) -> None:
    audited_path = tmp_path / "audited.jsonl"
    flagged_path = tmp_path / "flagged.jsonl"
    report_path = tmp_path / "audit_report.json"
    blueprint_path = tmp_path / "blueprint.json"
    events_path = tmp_path / "events.jsonl"
    items = [
        {
            "id": task_id,
            "task_id": task_id,
            "dataset": "set",
            "question": "Question?",
            "options": ["yes", "no"],
            "answer": "A",
        }
        for task_id in ("first", "second")
    ]
    _write_jsonl(audited_path, items)
    _write_jsonl(flagged_path, [])
    report_path.write_text(
        json.dumps(_initial_report(2)),
        encoding="utf-8",
    )
    blueprint_path.write_text(
        json.dumps(
            {
                "tasks": {
                    task_id: {
                        "description": "Task.",
                        "require": "Use the image.",
                        "question_templates": ["Question?"],
                    }
                    for task_id in ("first", "second")
                }
            }
        ),
        encoding="utf-8",
    )
    events_path.touch()
    responses = iter(
        [
            {"items": []},
            ModelUnavailable("offline"),
        ]
    )

    def fake(_system: str, _user: str) -> dict:
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr("medbenchagent.semantic_audit.complete_json", fake)
    result = semantic_audit_run(
        audited_path=audited_path,
        flagged_path=flagged_path,
        report_path=report_path,
        blueprint_path=blueprint_path,
        events_path=events_path,
        output_directory=tmp_path,
        run_id="run",
        dataset_id="set",
        seed=0,
    )

    records = [
        json.loads(line)
        for line in Path(result["semantic_records_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert {record["audit_status"] for record in records} == {
        "invalid_response",
        "model_unavailable",
    }
    assert flagged_path.read_text(encoding="utf-8") == ""
    assert len(audited_path.read_text(encoding="utf-8").splitlines()) == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["semantic_audit"]["not_run"] == 2
    assert report["semantic_audit"]["not_sampled"] == 0
    _assert_report_invariants(report, audited_path, flagged_path)
