import csv
import pytest
from breast_fixtures import create_breast_source
from medbenchagent.data.tabular import inspect_table
from medbenchagent.data.sources import inspect_source


def test_table_statistics_and_bounded_source_inspection(tmp_path):
    book = create_breast_source(tmp_path / "raw")
    table = inspect_table(book, id_columns=["CaseID"], preview_rows=1)
    assert table["rows_scanned"] == 4 and table["complete"]
    assert next(c for c in table["columns"] if c["name"] == "Pathology")["missing"] == 1
    assert len(table["preview"]) == 1
    assert not inspect_table(book, max_rows=2)["complete"]
    source = inspect_source(book.parent, limit=2)
    assert len(source["entries"]) == 2 and source["next_offset"] == 2
    assert all(len(row["digest"]) == 64 for row in source["entries"])
    with pytest.raises(ValueError):
        inspect_source(book.parent, relative="..")
    csv_path = tmp_path / "values.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        csv.writer(output).writerows([["id", "label"], [1, "A"], [1, ""], ["", "B"]])
    table = inspect_table(csv_path, id_columns=["id"])
    assert table["duplicate_id_rows"] == 1 and table["missing_id_rows"] == 1
    with pytest.raises(ValueError):
        inspect_table(csv_path, id_columns=["absent"])
