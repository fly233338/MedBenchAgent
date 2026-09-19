import json
from pathlib import Path
import pytest
from PIL import Image
from medbenchagent.data.dataset_card import build_card, validate_semantics
from medbenchagent.data.dataset_catalog import register_dataset, load_prepared, load_dataset_catalog
from medbenchagent.runtime.dependencies import snapshot_dataset
from test_generation import _record


def prepared_dataset(root, dataset_id, monkeypatch):
    source = root / dataset_id / "source"
    source.mkdir(parents=True)
    rows, manifest = [], []
    for index, label in enumerate(["benign", "malignant"]):
        row = _record(index, {"status": label})
        row["image"]["image_path"] = f"{index}.png"
        rows.append(row)
        image = Image.new("L", (4, 4))
        image.putpixel((0, 0), 250 - index * 40)
        image.save(source / f"{index}.png")
        manifest.append({"image_id": f"img-{index}", "png_path": f"{index}.png", "width": 4, "height": 4})
    (source / "annotation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (source / "image_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest), encoding="utf-8")
    # Shared internal Source IDs deliberately collide across independent datasets.
    annotation = snapshot_dataset(source / "annotation.jsonl", root / dataset_id / "prepared", dataset_id="shared")
    monkeypatch.setattr("medbenchagent.data.dataset_card.complete_json", lambda *args: {
        "description": "Annotated status dataset", "field_roles": {"image.annotation.status": "Status"}})
    build_card(dataset_id, annotation.parent)
    return register_dataset(dataset_id, annotation.parent, catalog_path=root / "catalog.json")


def test_card_registry_reuse_and_tampering(tmp_path, monkeypatch):
    entry = prepared_dataset(tmp_path, "a", monkeypatch)
    card, access = load_prepared(entry)
    assert card.total_images == 2 and card.annotation_digest == access["annotation_digest"]
    monkeypatch.setattr("medbenchagent.data.dataset_card.complete_json", lambda *a: pytest.fail("Card should be reused"))
    assert build_card("a", Path(entry["annotation_path"]).parent) == card
    payload = json.loads(Path(entry["dataset_card_path"]).read_text())
    payload["total_images"] = 99
    Path(entry["dataset_card_path"]).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="facts changed"):
        register_dataset("a", Path(entry["annotation_path"]).parent, catalog_path=tmp_path / "catalog.json")
    assert load_dataset_catalog(tmp_path / "catalog.json") == [entry]


def test_unknown_semantic_path_rejected():
    with pytest.raises(ValueError, match="annotation paths"):
        validate_semantics({"description": "Example", "field_roles": {"invented": "Label"}},
                           {"fields": {}, "structures": []}, {})
