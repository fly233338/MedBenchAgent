"""Stream referenced image inspection; expose bounded summaries to the agent."""

import json
from pathlib import Path

from medbenchagent.data.image_conversion import inspect_image, UnsupportedMedicalImage
from medbenchagent.data.operators import connect
from medbenchagent.data.validation import source_image_path, validate_record


def inspect_population(root, annotation, output, *, wrapped=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {"images": 0, "invalid": 0, "needs_display": [], "unresolved": 0}
    with (
        connect(scratch=output / "scratch") as db,
        Path(annotation).open(encoding="utf-8") as source,
        (output / "image_inventory.jsonl").open("w", encoding="utf-8") as target,
    ):
        db.execute("CREATE TABLE manifest (id VARCHAR PRIMARY KEY, path VARCHAR)")
        manifest = Path(root) / "image_manifest.jsonl"
        if not wrapped and manifest.is_file():
            with manifest.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    db.execute(
                        "INSERT INTO manifest VALUES (?,?)", [row["image_id"], row["png_path"]]
                    )
        for number, line in enumerate(source, 1):
            summary["images"] += 1
            key = f"invalid:{number}"
            try:
                payload = json.loads(line)
                row = payload["record"] if wrapped else payload
                validate_record(row, number)
                key = row["image"]["image_id"]
                entry = db.execute("SELECT path FROM manifest WHERE id=?", [key]).fetchone()
                path = entry[0] if entry else row["image"]["image_path"]
                info = inspect_image(source_image_path(root, path))
                if not info["copy"] and not info["preserve_supported"]:
                    summary["unresolved"] += 1
                    if len(summary["needs_display"]) < 20:
                        summary["needs_display"].append({"image_id": key, **info})
            except (ValueError, OSError, EOFError, UnsupportedMedicalImage) as exc:
                info = {"error": str(exc)}
                summary["invalid"] += 1
            target.write(json.dumps({"image_id": key, **info}, ensure_ascii=False) + "\n")
    return summary


def inventory_entries(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                key = item.pop("image_id")
                yield key, item
    else:
        yield from json.loads(path.read_text(encoding="utf-8")).items()
