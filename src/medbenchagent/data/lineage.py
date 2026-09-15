"""Validate source-row references without retaining the provenance index in memory."""

import json
from pathlib import Path

from medbenchagent.data.operators import connect
from medbenchagent.data.validation import source_image_path


def validate_lineage(annotation, provenance, root):
    """Both operators and fallback must account for every emitted record."""
    with connect() as db:
        db.execute("CREATE TABLE refs (id VARCHAR PRIMARY KEY, used BOOLEAN DEFAULT false)")
        with Path(provenance).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if set(row) != {"id", "locators", "step"} or not isinstance(row["id"], str):
                    raise ValueError("Invalid preparation provenance entry")
                if not isinstance(row["locators"], list) or not row["locators"]:
                    raise ValueError("Provenance requires actual input record locators")
                for locator in row["locators"]:
                    if not isinstance(locator, str) or ":" not in locator:
                        raise ValueError("Locator must be relative-file[:sheet]:positive-row")
                    file, number = locator.rsplit(":", 1)
                    if not number.isdigit() or int(number) < 1:
                        raise ValueError("Source row locators must be positive")
                    source_image_path(root, file.split("#sheet=", 1)[0])
                db.execute("INSERT INTO refs (id) VALUES (?)", [row["id"]])
        count = 0
        with Path(annotation).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if set(row) != {"record", "record_ref"}:
                    raise ValueError("Preparation rows require record and record_ref")
                found = db.execute(
                    "SELECT used FROM refs WHERE id=?", [row["record_ref"]]
                ).fetchone()
                if found is None or found[0]:
                    raise ValueError("Missing or reused preparation provenance reference")
                db.execute("UPDATE refs SET used=true WHERE id=?", [row["record_ref"]])
                count += 1
        if db.execute("SELECT count(*) FROM refs WHERE NOT used").fetchone()[0]:
            raise ValueError("Unreferenced preparation provenance entries")
        return count
