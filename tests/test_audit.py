import hashlib
import json
from pathlib import Path

from PIL import Image

from medbenchagent.audit import audit_run
from medbenchagent.run_artifacts import blueprint_sha256


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_audit_uses_manifest_and_rejects_duplicate_question(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "a.png"
    image_path.parent.mkdir()
    image = Image.new("L", (2, 2))
    image.putdata([0, 1, 2, 3])
    image.save(image_path)
    blueprint = {
        "dataset": {"dataset_id": "set"},
        "tasks": {"view": {"response_format": "letter", "supporting_fields": ["annotation.view"]}},
    }
    blueprint_path = tmp_path / "blueprint.json"
    blueprint_path.write_text(json.dumps(blueprint), encoding="utf-8")
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
    _write_jsonl(raw, [item, {**item, "id": "two"}])
    source = {
        "dataset_id": "set",
        "task_id": "view",
        "image_id": "image",
        "supporting_fields": ["annotation.view"],
        "generation_call_ids": [],
        "review_call_ids": [],
        "blueprint_hash": blueprint_sha256(blueprint_path),
    }
    _write_jsonl(provenance, [{"item_id": "one", **source}, {"item_id": "two", **source}])
    events.touch()
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

    result = audit_run(
        raw_path=raw,
        blueprint_path=blueprint_path,
        provenance_path=provenance,
        events_path=events,
        manifest_path=manifest,
        output_directory=tmp_path,
        run_id="run",
    )

    assert result["item_count"] == 1
    assert len((Path(result["rejected_path"])).read_text(encoding="utf-8").splitlines()) == 1
