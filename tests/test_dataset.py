from pathlib import Path

import duckdb

from medbenchagent.dataset import build_access_layer, discover_dataset, infer_schema, plan_task_capabilities


def test_inventory_access_layer_and_candidate_views(tmp_path: Path) -> None:
    data_path = tmp_path / "cases.csv"
    data_path.write_text(
        "case_id,image_path,diagnosis\n1,image-1.png,benign\n2,image-2.png,malignant\n",
        encoding="utf-8",
    )
    inventory = discover_dataset(tmp_path)
    schema_map = infer_schema(inventory)
    assert schema_map["unknowns"] == []
    manifest = build_access_layer(inventory, schema_map, tmp_path / "runs")
    blueprint = {"tasks": {"diagnosis": {"response_format": "letter"}}}
    plans = plan_task_capabilities(blueprint, manifest, schema_map)
    assert plans[0]["decision"] == "retain"
    assert plans[0]["eligible_case_count"] == 2
    connection = duckdb.connect(manifest["duckdb_path"])
    try:
        assert connection.execute('SELECT COUNT(*) FROM "task_candidates_diagnosis"').fetchone()[0] == 2
        assert connection.execute("SELECT _source_row FROM cases ORDER BY _source_row").fetchall() == [(2,), (3,)]
    finally:
        connection.close()


def test_access_layer_joins_a_unique_secondary_table(tmp_path: Path) -> None:
    (tmp_path / "images.csv").write_text("case_id,image_path\n1,a.png\n2,b.png\n", encoding="utf-8")
    (tmp_path / "labels.csv").write_text("case_id,diagnosis\n1,benign\n2,malignant\n", encoding="utf-8")
    inventory = discover_dataset(tmp_path)
    schema_map = infer_schema(inventory)
    assert len(schema_map["joins"]) == 1
    manifest = build_access_layer(inventory, schema_map, tmp_path / "runs")
    connection = duckdb.connect(manifest["duckdb_path"])
    try:
        assert connection.execute('SELECT "table_1_diagnosis" FROM cases ORDER BY case_id').fetchall() == [
            ("benign",),
            ("malignant",),
        ]
    finally:
        connection.close()
