"""Resolve source objects and validate candidate provenance with disk-backed indexes."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from medbenchagent.data.operators import connect
from medbenchagent.data.contracts import ExtractionProvenance
from medbenchagent.data.locators import LocatorReader, finding_objects, pointer
from medbenchagent.data.population import identity


def validate_mask_geometry(record, provenance, reader, root):
    locators = [*provenance.locators, *(loc for finding in provenance.findings for loc in finding.locators)]
    sizes = {(reader.resolve(loc)["height"], reader.resolve(loc)["width"]) for loc in locators if loc.kind == "mask"}
    if not sizes:
        return
    from medbenchagent.data.image_conversion import _pixels, UnsupportedMedicalImage
    from medbenchagent.data.validation import source_image_path
    import numpy as np
    try:
        pixels = np.asarray(_pixels(source_image_path(root, record["image"]["image_path"]))[0])
    except (ValueError, OSError, EOFError, UnsupportedMedicalImage):
        return  # The image quality stage accounts for missing/unsupported/unreadable images.
    if sizes != {pixels.shape[:2]}:
        raise ValueError("Source mask geometry differs from the referenced image; resizing is forbidden")


def _targets(value, field, address=""):
    parts = field.split(".")
    if not isinstance(value, dict) or parts[0].removesuffix("[]") not in value:
        return
    name = parts[0].removesuffix("[]")
    child = value[name]
    path = address + "/" + name.replace("~", "~0").replace("/", "~1")
    if parts[0].endswith("[]"):
        if not isinstance(child, list):
            return
        for index, item in enumerate(child):
            if len(parts) == 1:
                yield path + "/" + str(index), item
            else:
                yield from _targets(item, ".".join(parts[1:]), path + "/" + str(index))
    elif len(parts) == 1:
        yield path, child
    else:
        yield from _targets(child, ".".join(parts[1:]), path)


def _equal(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def validate_bindings(record, provenance, reader, spec):
    associations = {item.target: item.locators for item in provenance.findings}
    actual = {path for entity in ("patient", "case", "study", "image")
              if isinstance(record.get(entity), dict)
              for path, _ in finding_objects(record[entity].get("annotation", {}), f"/{entity}/annotation")}
    if len(associations) != len(provenance.findings) or set(associations) != actual:
        raise ValueError("Every annotation array object requires its own finding provenance")
    for locators in associations.values():
        for locator in locators:
            if not any(
                (parent.kind, parent.path, parent.row, parent.sheet, parent.delimiter) ==
                (locator.kind, locator.path, locator.row, locator.sheet, locator.delimiter)
                and (locator.pointer == parent.pointer or locator.pointer.startswith(parent.pointer + "/"))
                for parent in provenance.locators
            ):
                raise ValueError("Finding locator must belong to one of its image record's source objects")
            reader.resolve(locator)
    if spec is None:
        return
    # Invalid canonical records remain accounted for and are quarantined whole by normalization.
    from medbenchagent.data.validation import validate_record
    try:
        validate_record(record)
    except ValueError:
        return
    if spec.fields:
        declared = {f.target for f in spec.fields}
        def check_fields(value, path=""):
            if value is None or isinstance(value, dict) and not value:
                return
            if path in declared:
                return
            if isinstance(value, dict):
                for key, item in value.items():
                    check_fields(item, path + "." + key if path else key)
            elif isinstance(value, list) and any(isinstance(item, dict) for item in value):
                for item in value:
                    check_fields(item, path + "[]")
            else:
                raise ValueError(f"Candidate emits an undeclared semantic field: {path}")
        check_fields(record)
    for field in spec.fields:
        if field.transformation == "custom":
            continue
        targets = list(_targets(record, field.target))
        if not targets and "[]" not in field.target:
            raise ValueError(f"Candidate omits declared field: {field.target}")
        for address, emitted in targets:
            parents = [p for p in associations if address == p or address.startswith(p + "/")]
            locators = associations[max(parents, key=len)] if parents else provenance.locators
            values = []
            for locator in locators:
                if locator.path != field.source_path:
                    continue
                if locator.kind == "table" and (locator.sheet != field.source_sheet or locator.delimiter != field.source_delimiter):
                    continue
                source = reader.resolve(locator)
                if field.source_field.startswith("/"):
                    raw = pointer(source, field.source_field)
                elif isinstance(source, dict) and field.source_field in source:
                    raw = source[field.source_field]
                else:
                    continue
                expected = raw
                if field.transformation == "value_mapping":
                    matches = [m for m in field.mapping if _equal(m.value, raw)]
                    if len(matches) > 1 or not matches and field.unmapped == "error":
                        raise ValueError(f"Ambiguous/unmapped source value: {field.target}")
                    expected = matches[0].target if matches else raw
                values.append(expected)
            if not values or any(not _equal(value, emitted) for value in values):
                raise ValueError(f"Candidate field differs from source binding: {field.target} at {address}")


def validate_lineage(annotation, provenance, root, *, spec=None, scratch=None):
    """Validate existence and direct bindings, not arbitrary Python transformation semantics."""
    scratch = Path(scratch or Path(annotation).parent / "scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="lineage-", dir=scratch) as temporary, connect(Path(temporary) / "index.duckdb", scratch=temporary) as db:
        db.execute("CREATE TABLE refs (id VARCHAR PRIMARY KEY, payload VARCHAR, used BOOLEAN DEFAULT false)")
        db.execute("CREATE TABLE images (id VARCHAR PRIMARY KEY, path VARCHAR UNIQUE)")
        reader = LocatorReader(root, db)
        with Path(provenance).open(encoding="utf-8") as handle:
            for line in handle:
                row = ExtractionProvenance.model_validate_json(line)
                for locator in row.locators:
                    reader.resolve(locator)
                if db.execute("SELECT 1 FROM refs WHERE id=?", [row.id]).fetchone():
                    raise ValueError("Duplicate preparation provenance reference")
                db.execute("INSERT INTO refs (id,payload) VALUES (?,?)", [row.id, row.model_dump_json()])
        count = 0
        with Path(annotation).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != {"record", "record_ref"} or not isinstance(row["record_ref"], str):
                    raise ValueError("Preparation rows require record and record_ref")
                record = row["record"]
                if not isinstance(record, dict) or not isinstance(record.get("image"), dict):
                    raise ValueError("Candidate must have an identifiable image")
                if "provenance" in record:
                    raise ValueError("Candidate cannot assign trusted provenance")
                key, path = identity(root, record["image"].get("image_id"), record["image"].get("image_path"))
                if db.execute("SELECT 1 FROM images WHERE id=? OR path=?", [key, path]).fetchone():
                    raise ValueError("Duplicate candidate image identity/path")
                db.execute("INSERT INTO images VALUES (?,?)", [key, path])
                found = db.execute(
                    "SELECT used,payload FROM refs WHERE id=?", [row["record_ref"]]
                ).fetchone()
                if found is None or found[0]:
                    raise ValueError("Missing or reused preparation provenance reference")
                provenance_row = ExtractionProvenance.model_validate_json(found[1])
                validate_bindings(record, provenance_row, reader, spec)
                validate_mask_geometry(record, provenance_row, reader, root)
                db.execute("UPDATE refs SET used=true WHERE id=?", [row["record_ref"]])
                count += 1
        if not count or db.execute("SELECT count(*) FROM refs WHERE NOT used").fetchone()[0]:
            raise ValueError("Empty candidates or unreferenced preparation provenance entries")
        return count
