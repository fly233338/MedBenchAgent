"""Normalize sources and publish a quality-controlled, provenance-preserving bundle."""

import copy
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from medbenchagent.data.contracts import HarmonizationPlan, QualityIssue, Source
from medbenchagent.data.image_assets import validate_image_asset
from medbenchagent.data.image_conversion import (
    _pixels,
    _pixel_info,
    convert_image,
    UnsupportedMedicalImage,
)
from medbenchagent.data.operators import connect
from medbenchagent.data.readers import ReaderUnavailable
from medbenchagent.data.sources import attach_provenance, digest_file
from medbenchagent.data.validation import validate_record, validate_bboxes, source_image_path
from medbenchagent.runtime.run_artifacts import atomic_write_json


def json_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def issue(
    handle, source, locator, code, message, image_id=None, stage="normalize", action="quarantine"
):
    json_line(
        handle,
        QualityIssue(
            code=code,
            stage=stage,
            source_id=source,
            locator=str(locator),
            image_id=image_id,
            action=action,
            severity="warning" if action == "report" else "error",
            message=str(message),
        ).model_dump(),
    )


def pixel_digest(pixels):
    array = np.asarray(pixels)
    # Palette indices alone are not colors. Include palette/transparency metadata.
    metadata = {
        "shape": array.shape,
        "dtype": array.dtype.str,
        "mode": getattr(pixels, "mode", None),
    }
    if getattr(pixels, "mode", None) == "P":
        metadata["palette"] = pixels.getpalette()
        metadata["transparency"] = repr(pixels.info.get("transparency"))
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def normalize_source(
    root, annotation, source: Source, output, *, policies=None, plan_ref="import", wrapped=False
):
    """Validate entire image records; only materialize annotation-referenced image assets."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    policies = policies or {}
    counts = Counter()
    with (
        connect(scratch=output / "scratch") as db,
        Path(annotation).open(encoding="utf-8") as input_file,
        (output / "annotation.jsonl").open("w", encoding="utf-8") as records,
        (output / "image_manifest.jsonl").open("w", encoding="utf-8") as manifest,
        (output / "assets.jsonl").open("w", encoding="utf-8") as assets,
        (output / "quarantine.jsonl").open("w", encoding="utf-8") as quarantine,
        (output / "provenance.jsonl").open("w", encoding="utf-8") as provenance,
    ):
        db.execute("CREATE TABLE ids (id VARCHAR PRIMARY KEY, path VARCHAR UNIQUE)")
        # Standard manifests are authoritative for the location of existing converted assets.
        db.execute(
            "CREATE TABLE input_manifest (id VARCHAR PRIMARY KEY, path VARCHAR, width BIGINT, height BIGINT)"
        )
        original_manifest = Path(root) / "image_manifest.jsonl"
        if source.kind == "annotation" and original_manifest.is_file():
            with original_manifest.open(encoding="utf-8") as handle:
                for line in handle:
                    item = json.loads(line)
                    if set(item) != {"image_id", "png_path", "width", "height"}:
                        raise ValueError("Invalid input manifest contract")
                    db.execute(
                        "INSERT INTO input_manifest VALUES (?,?,?,?)",
                        [item["image_id"], item["png_path"], item["width"], item["height"]],
                    )
        for number, line in enumerate(input_file, 1):
            counts["input"] += 1
            image_id = None
            locator = f"{source.annotation}:{number}"
            relative = None
            try:
                payload = json.loads(line)
                if wrapped and (
                    not isinstance(payload, dict) or set(payload) != {"record", "record_ref"}
                ):
                    raise ValueError("Invalid wrapped preparation record")
                record = payload["record"] if wrapped else payload
                locator = payload["record_ref"] if wrapped else locator
                validate_record(record, number)
                image_id = record["image"]["image_id"]
                image = record["image"]
                if db.execute(
                    "SELECT count(*) FROM ids WHERE id=? OR path=?", [image_id, image["image_path"]]
                ).fetchone()[0]:
                    # Duplicate keys are ambiguous population construction, not a locally bad image.
                    raise DuplicateIdentity(
                        "Repeated image identity/path; group raw rows before emit"
                    )
                db.execute("INSERT INTO ids VALUES (?,?)", [image_id, image["image_path"]])
                input_entry = db.execute(
                    "SELECT path,width,height FROM input_manifest WHERE id=?", [image_id]
                ).fetchone()
                if (
                    source.kind == "annotation"
                    and original_manifest.is_file()
                    and input_entry is None
                ):
                    raise ValueError("Image is absent from input manifest")
                source_relative = input_entry[0] if input_entry else image["image_path"]
                path = source_image_path(root, source_relative)
                decoded = _pixels(path)
                info = _pixel_info(*decoded)
                if input_entry and (info["width"], info["height"]) != input_entry[1:]:
                    raise ValueError("Input manifest dimensions differ from decoded image")
                for entity in ("patient", "case", "study", "image"):
                    validate_bboxes(
                        record[entity]["annotation"], width=info["width"], height=info["height"]
                    )
                policy = policies.get(image_id, {"encoding": "preserve"})
                if (
                    image_id not in policies
                    and not info["copy"]
                    and not info["preserve_supported"]
                    and info["window"]
                ):
                    policy = {
                        "encoding": "window",
                        "low": info["window"][0],
                        "high": info["window"][1],
                        "invert": info["invert"],
                    }
                if (
                    not info["copy"]
                    and not info["preserve_supported"]
                    and policy["encoding"] == "preserve"
                ):
                    raise DisplayDecisionRequired(
                        f"{image_id}: explicit display policy is required"
                    )
                if info["copy"] and policy["encoding"] != "preserve":
                    raise DisplayDecisionRequired("PNG/JPEG must retain source bytes")
                extension = ".jpg" if info["format"] == "raster:jpeg" else ".png"
                relative = f"images/{number:012d}{extension}"
                target = output / relative
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if info["copy"]:
                        shutil.copyfile(path, target)
                    else:
                        converted, _, _ = convert_image(path, policy, decoded=decoded)
                        converted.save(target, format="PNG", compress_level=9)
                except OSError as exc:
                    raise OutputWriteError(
                        f"Cannot materialize image output {relative}: {exc}"
                    ) from exc
                entry = {
                    "image_id": image_id,
                    "png_path": relative,
                    "width": info["width"],
                    "height": info["height"],
                }
                errors = validate_image_asset(
                    entry, output, supported_formats=frozenset({"PNG", "JPEG"})
                )
                if errors:
                    raise ValueError(", ".join(errors))
                prior = record.get("provenance")
                record = attach_provenance(record, source, f"provenance.jsonl#{number}", plan_ref)
                record["image"]["image_path"] = relative
                entry["image_id"] = record["image"]["image_id"]
                json_line(
                    provenance,
                    {
                        "id": number,
                        "record_ref": locator,
                        "upstream": prior,
                        "original_image_path": source_relative,
                        "conversion": policy,
                    },
                )
                json_line(records, record)
                json_line(manifest, entry)
                json_line(
                    assets,
                    {
                        "image_id": entry["image_id"],
                        "file_hash": digest_file(path),
                        "pixel_hash": pixel_digest(decoded[0]),
                        "source_path": source_relative,
                    },
                )
                counts["accepted"] += 1
            except (DisplayDecisionRequired, DuplicateIdentity, ReaderUnavailable):
                raise
            except (ValueError, OSError, EOFError, UnsupportedMedicalImage) as exc:
                if relative:
                    (output / relative).unlink(missing_ok=True)
                issue(quarantine, source.id, locator, "invalid_image_record", str(exc), image_id)
                counts["quarantined"] += 1
    if not counts["accepted"]:
        raise ValueError("No usable records remain in source")
    atomic_write_json(output / "quality_report.json", dict(counts))
    return dict(counts)


class DisplayDecisionRequired(ValueError):
    pass


class DuplicateIdentity(ValueError):
    pass


class OutputWriteError(RuntimeError):
    """Output/environment failures must not be interpreted as bad source records."""


def _locations(value, parts, indices=()):
    if not parts:
        yield value, indices
        return
    if not isinstance(value, dict):
        return
    key = parts[0].removesuffix("[*]")
    if key not in value:
        return
    nested = value[key]
    if parts[0].endswith("[*]"):
        if isinstance(nested, list):
            for index, item in enumerate(nested):
                yield from _locations(item, parts[1:], (*indices, index))
    else:
        yield from _locations(nested, parts[1:], indices)


def _put(value, parts, indices, new):
    key = parts[0].removesuffix("[*]")
    if parts[0].endswith("[*]"):
        if not indices:
            raise ValueError("Harmonization cannot invent array entities")
        items = value.setdefault(key, [])
        while len(items) <= indices[0]:
            items.append({})
        if len(parts) == 1:
            items[indices[0]] = new
        else:
            _put(items[indices[0]], parts[1:], indices[1:], new)
    elif len(parts) == 1:
        if key in value and value[key] != new:
            raise ValueError("Harmonization writes conflicting target values")
        value[key] = new
    else:
        _put(value.setdefault(key, {}), parts[1:], indices, new)


def align_record(record, plan, *, composite, observed):
    original = copy.deepcopy(record)
    source_id = record["provenance"]["source_id"]
    if composite:
        for entity in ("patient", "case", "study", "image"):
            annotations = record[entity]["annotation"]
            record[entity]["annotation"] = (
                {"sources": {source_id: annotations}} if annotations else {}
            )
    for index, field in enumerate(plan.fields):
        if field.source_id != source_id:
            continue
        if not any(
            field.field.startswith(e + ".annotation.")
            for e in ("patient", "case", "study", "image")
        ):
            raise ValueError("Harmonization fields must belong to annotations")
        locations = list(_locations(original, field.field.split(".")))
        if not locations:
            continue
        observed.add((index, "field"))
        if field.action == "keep_separate":
            continue
        if field.action == "merge":
            if not field.target or field.target.split(".")[0] != field.field.split(".")[0]:
                raise ValueError("Harmonization cannot change entity scope")
            if ".annotation." not in field.target or field.target.count("[*]") != field.field.count(
                "[*]"
            ):
                raise ValueError("Harmonization must preserve entity array depth")
            if not composite:
                _remove(record, field.field.split("."))
            for value, indices in locations:
                mapped = value
                if field.mapping:
                    matches = [
                        (i, m)
                        for i, m in enumerate(field.mapping)
                        if type(m.value) is type(value) and m.value == value
                    ]
                    if len(matches) != 1:
                        raise ValueError(f"Unmapped/ambiguous value in {field.field}: {value!r}")
                    observed.add((index, matches[0][0]))
                    mapped = matches[0][1].target
                if field.scale is not None:
                    if type(mapped) not in (float, int):
                        raise ValueError("Unit conversion requires numeric values")
                    mapped = mapped * field.scale + field.offset
                # Identity targets on a single source intentionally replace their original leaf.
                _put(record, field.target.split("."), indices, mapped)
        source_path = field.field.split(".")
        if composite:
            source_path = [*source_path[:2], "sources", source_id, *source_path[2:]]
        if composite or field.action == "exclude":
            _remove(record, source_path)
    return record


def _remove(value, parts):
    if not isinstance(value, dict):
        return
    key = parts[0].removesuffix("[*]")
    if key not in value:
        return
    if len(parts) == 1:
        value.pop(key)
    elif parts[0].endswith("[*]"):
        for item in value[key]:
            _remove(item, parts[1:])
    else:
        _remove(value[key], parts[1:])
        if value[key] == {} and key != "annotation":
            value.pop(key)


def publish_bundle(
    bundle, normalized, plan: HarmonizationPlan, *, invariants=None, verify_inputs=None
):
    """Inputs already live inside the bundle; publishing never recopies their image assets."""
    bundle = Path(bundle).resolve()
    bundle.mkdir(parents=True, exist_ok=True)
    if (bundle / "ready.json").exists():
        raise ValueError("Bundle is immutable; select a new output version")
    observed = set()
    counts = Counter()
    with (
        connect(bundle / "publication.duckdb", bundle / "scratch") as db,
        (bundle / "quarantine.jsonl").open("w", encoding="utf-8") as quarantine,
        (bundle / "dedup.jsonl").open("w", encoding="utf-8") as dedup,
        (bundle / "entity_quality.jsonl").open("w", encoding="utf-8") as entity_report,
    ):
        db.execute("DROP TABLE IF EXISTS candidates")
        db.execute("DROP TABLE IF EXISTS entity_values")
        db.execute(
            "CREATE TABLE candidates (id VARCHAR PRIMARY KEY, record VARCHAR, manifest VARCHAR, "
            "pixel_hash VARCHAR, file_hash VARCHAR, signature VARCHAR, source_id VARCHAR)"
        )
        db.execute(
            "CREATE TABLE entity_values (source_id VARCHAR, entity_id VARCHAR, path VARCHAR, value VARCHAR)"
        )
        for source_id, folder in sorted(normalized.items()):
            folder = Path(folder).resolve()
            if not folder.is_relative_to(bundle):
                raise ValueError("Normalized assets must reside beneath the bundle root")
            prefix = folder.relative_to(bundle).as_posix()
            quality = json.loads((folder / "quality_report.json").read_text(encoding="utf-8"))
            counts["input"] += quality["input"]
            counts["quarantined"] += quality.get("quarantined", 0)
            with (folder / "quarantine.jsonl").open(encoding="utf-8") as handle:
                shutil.copyfileobj(handle, quarantine)
            with (
                (folder / "annotation.jsonl").open(encoding="utf-8") as records,
                (folder / "image_manifest.jsonl").open(encoding="utf-8") as manifests,
                (folder / "assets.jsonl").open(encoding="utf-8") as assets,
            ):
                for line, entry_line, asset_line in zip(records, manifests, assets, strict=True):
                    record, entry, asset = (
                        json.loads(line),
                        json.loads(entry_line),
                        json.loads(asset_line),
                    )
                    original_record = copy.deepcopy(record)
                    record = align_record(
                        record, plan, composite=len(normalized) > 1, observed=observed
                    )
                    validate_record(record)
                    image_id = record["image"]["image_id"]
                    if not image_id == entry["image_id"] == asset["image_id"]:
                        raise ValueError("Normalized manifests do not agree")
                    for entity in ("patient", "case", "study", "image"):
                        validate_bboxes(
                            record[entity]["annotation"],
                            width=entry["width"],
                            height=entry["height"],
                        )
                    entry["png_path"] = prefix + "/" + entry["png_path"]
                    record["image"]["image_path"] = entry["png_path"]
                    record["provenance"]["record_ref"] = (
                        prefix + "/" + record["provenance"]["record_ref"]
                    )
                    if record["provenance"]["plan_ref"] == "preparation.json":
                        record["provenance"]["plan_ref"] = prefix + "/preparation.json"
                    signature = {e: record[e] for e in ("patient", "case", "study")}
                    signature["image"] = {
                        k: v
                        for k, v in record["image"].items()
                        if k not in {"image_id", "image_path"}
                    }
                    db.execute(
                        "INSERT INTO candidates VALUES (?,?,?,?,?,?,?)",
                        [
                            image_id,
                            json.dumps(record, sort_keys=True),
                            json.dumps(entry),
                            asset["pixel_hash"],
                            asset["file_hash"],
                            json.dumps(signature, sort_keys=True),
                            source_id,
                        ],
                    )
                    for entity in ("patient", "case", "study"):
                        identifier = original_record[entity][entity + "_id"]
                        if identifier is not None:
                            for path, value in _scalar_paths(
                                original_record[entity]["annotation"], entity + ".annotation"
                            ):
                                db.execute(
                                    "INSERT INTO entity_values VALUES (?,?,?,?)",
                                    [
                                        source_id,
                                        identifier,
                                        path,
                                        json.dumps(value, sort_keys=True),
                                    ],
                                )
        for index, field in enumerate(plan.fields):
            if (index, "field") not in observed or any(
                (index, i) not in observed for i in range(len(field.mapping))
            ):
                raise ValueError(f"Harmonization references absent field/value: {field.field}")
        entity_cursor = db.cursor()
        entity_cursor.execute(
            "SELECT source_id,entity_id,path FROM entity_values GROUP BY ALL HAVING count(DISTINCT value)>1"
        )
        try:
            while rows := entity_cursor.fetchmany(512):
                for source_id, identifier, path in rows:
                    declared = path in (invariants or {}).get(source_id, [])
                    issue(
                        entity_report,
                        source_id,
                        identifier,
                        "entity_inconsistency",
                        path,
                        stage="quality",
                        action="block" if declared else "report",
                    )
                    if declared:
                        raise ValueError(f"Declared entity invariant violated: {path}")
        finally:
            entity_cursor.close()
        db.execute(
            "CREATE OR REPLACE TABLE duplicate_groups AS SELECT pixel_hash,min(id) representative,"
            "count(*) n,count(DISTINCT signature)>1 has_conflict FROM candidates GROUP BY pixel_hash"
        )
        cursor = db.cursor()
        cursor.execute(
            "SELECT c.*,g.representative,g.n,g.has_conflict FROM candidates c JOIN duplicate_groups g "
            "USING(pixel_hash) ORDER BY c.id"
        )
        try:
            with (
                (bundle / "annotation.jsonl").open("w", encoding="utf-8") as records,
                (bundle / "image_manifest.jsonl").open("w", encoding="utf-8") as manifests,
            ):
                while rows := cursor.fetchmany(512):
                    for (
                        image_id,
                        payload,
                        manifest,
                        pixels,
                        file_hash,
                        _,
                        source,
                        representative,
                        n,
                        conflict,
                    ) in rows:
                        if conflict:
                            issue(
                                quarantine,
                                source,
                                image_id,
                                "duplicate_conflict",
                                "Exact pixels with conflicting annotations/identity",
                                image_id,
                                stage="dedup",
                            )
                            counts["quarantined"] += 1
                        elif image_id == representative:
                            records.write(payload + "\n")
                            manifests.write(manifest + "\n")
                            counts["accepted"] += 1
                        else:
                            counts["duplicates_removed"] += 1
                        if n > 1:
                            json_line(
                                dedup,
                                {
                                    "image_id": image_id,
                                    "representative": representative,
                                    "pixel_hash": pixels,
                                    "file_hash": file_hash,
                                    "conflict": conflict,
                                    "provenance": json.loads(payload)["provenance"],
                                },
                            )
        finally:
            cursor.close()
    if not counts["accepted"]:
        raise ValueError("No usable records remain after quality/dedup")
    if counts["input"] != counts["accepted"] + counts["quarantined"] + counts["duplicates_removed"]:
        raise ValueError("Dataset population accounting failed")
    atomic_write_json(bundle / "harmonization.json", plan.model_dump())
    atomic_write_json(bundle / "quality_report.json", dict(counts))
    from medbenchagent.data.dataset import prepare_annotation_dataset

    profile, access = prepare_annotation_dataset(bundle / "annotation.jsonl", bundle / "cache")
    if verify_inputs:
        verify_inputs()
    atomic_write_json(
        bundle / "ready.json",
        {"schema_version": "dataset-bundle.v1", "access": access, "sources": sorted(normalized)},
    )
    return profile, access


def _scalar_paths(value, path):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield from _scalar_paths(nested, path + "." + key)
    elif not isinstance(value, list) and value is not None:
        yield path, value
