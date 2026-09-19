import json
from pathlib import Path
import pytest
from PIL import Image
from medbenchagent.data.dataset_card import build_card, validate_semantics
from medbenchagent.data.dataset_catalog import register_dataset, load_prepared, load_dataset_catalog, refresh_card
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
                           {"fields": {}, "structures": []})


def test_blueprint_changes_do_not_invalidate_or_rebuild_cards(tmp_path, monkeypatch):
    from dataclasses import replace
    from medbenchagent import settings
    entry = prepared_dataset(tmp_path, "a", monkeypatch)
    original, _ = load_prepared(entry)
    template = tmp_path / "changed_blueprint.json"
    template.write_text('{"concepts":{"entirely_new":{}}}', encoding="utf-8")
    cfg = settings.get_settings()
    monkeypatch.setattr(settings, "_SETTINGS", replace(cfg, paths=replace(cfg.paths, blueprint_template=template)))
    monkeypatch.setattr("medbenchagent.data.dataset_card.complete_json", lambda *a: pytest.fail("No semantic rebuild needed"))
    monkeypatch.setattr("medbenchagent.data.dataset.prepare_annotation_dataset", lambda *a, **k: pytest.fail("No preparation needed"))
    assert load_prepared(entry)[0] == original
    assert build_card("a", Path(entry["annotation_path"]).parent) == original
    assert "blueprint_digest" not in original.model_dump()
    assert "capability_hints" not in original.semantics.model_dump()


def test_refresh_migrates_v1_without_old_goal_or_reprocessing(tmp_path, monkeypatch):
    entry = prepared_dataset(tmp_path, "a", monkeypatch)
    bundle = Path(entry["annotation_path"]).parent
    old_path = Path(entry["dataset_card_path"])
    payload = json.loads(old_path.read_text(encoding="utf-8"))
    payload.update(schema_version="dataset-card.v1", blueprint_digest="obsolete-blueprint")
    payload.pop("preparation_metadata")
    payload["name"] = "OLD_GOAL_BIASED_NAME"
    payload["semantics"] = {"description": "OLD_GOAL_BIASED_DESCRIPTION", "field_roles": {},
                            "capability_hints": [{"concept_id": "old-goal", "fields": [], "reason": "old-goal"}]}
    old_path.write_text(json.dumps(payload), encoding="utf-8")
    original_files = {p: p.read_bytes() for p in bundle.rglob("*") if p.is_file()}
    def semantic_model(system, prompt):
        assert "OLD_GOAL" not in prompt and "old-goal" not in prompt and "blueprint" not in prompt
        assert json.loads(prompt)["preparation_metadata"]["dataset_description"] is None
        return {"description": "Facts from Profile only", "field_roles": {"image.annotation.status": "Status"}}
    monkeypatch.setattr("medbenchagent.data.dataset_card.complete_json", semantic_model)
    monkeypatch.setattr("medbenchagent.data.dataset.prepare_annotation_dataset", lambda *a, **k: pytest.fail("Do not rebuild Profile"))
    monkeypatch.setattr("medbenchagent.data.execution.run", lambda *a, **k: pytest.fail("Do not process images"))
    with pytest.raises(ValueError, match="refresh-card"):
        load_prepared(entry)
    updated = refresh_card("a", catalog_path=tmp_path / "catalog.json")
    assert updated["dataset_card_path"] != str(old_path)
    assert all(p.read_bytes() == data for p, data in original_files.items())
    card, _ = load_prepared(updated)
    assert card.schema_version == "dataset-card.v2" and card.name == "a"
    assert card.semantics.description == "Facts from Profile only"
    assert load_dataset_catalog(tmp_path / "catalog.json") == [updated]


def test_failed_refresh_keeps_registered_revision(tmp_path, monkeypatch):
    entry = prepared_dataset(tmp_path, "a", monkeypatch)
    monkeypatch.setattr("medbenchagent.data.dataset_card.complete_json", lambda *a: {
        "description": "Fabricated", "field_roles": {"nonexistent": "Label"}})
    with pytest.raises(ValueError, match="annotation paths"):
        refresh_card("a", catalog_path=tmp_path / "catalog.json")
    assert load_dataset_catalog(tmp_path / "catalog.json") == [entry]
    assert load_prepared(entry)[0].semantics.description == "Annotated status dataset"
