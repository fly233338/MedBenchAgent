"""Preparation evidence, canonical import and portable preparation archives."""

import json
import shutil
from pathlib import Path

from medbenchagent.data.contracts import SourceLocator, ExtractionProvenance
from medbenchagent.data.locators import finding_objects
from medbenchagent.data.operators import connect
from medbenchagent.data.population import identity
from medbenchagent.data.sources import digest_file
from medbenchagent.data.validation import source_image_path
from medbenchagent.runtime.run_artifacts import atomic_write_json


CAPABILITIES = ["relational_population", "json_object_population", "table_locators", "json_pointer_locators", "2d_mask_locators"]


def import_annotation(root, source, output):
    """Read canonical input without interpreting labels or claiming earlier completeness."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = Path(root) / "image_manifest.jsonl"
    with (connect(scratch=output / "scratch") as db,
          source_image_path(root, source.annotation).open(encoding="utf-8") as source_file,
          (output / "annotation.jsonl").open("w", encoding="utf-8") as annotation,
          (output / "provenance.jsonl").open("w", encoding="utf-8") as provenance,
          (output / "population.jsonl").open("w", encoding="utf-8") as population):
        db.execute("CREATE TABLE manifest (id VARCHAR PRIMARY KEY, path VARCHAR)")
        db.execute("CREATE TABLE images (id VARCHAR PRIMARY KEY, path VARCHAR UNIQUE)")
        if manifest.is_file():
            with manifest.open(encoding="utf-8") as handle:
                for line in handle:
                    item = json.loads(line)
                    db.execute("INSERT INTO manifest VALUES (?,?)", [item["image_id"], item["png_path"]])
        count = 0
        for number, line in enumerate(source_file, 1):
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("image"), dict):
                raise ValueError("Imported records must have an identifiable image")
            image = record["image"]
            entry = db.execute("SELECT path FROM manifest WHERE id=?", [image.get("image_id")]).fetchone()
            # The normalizer still quarantines absent manifest entries and bad dimensions.
            if entry:
                image["image_path"] = entry[0]
            key, path = identity(root, image.get("image_id"), image.get("image_path"))
            if db.execute("SELECT 1 FROM images WHERE id=? OR path=?", [key, path]).fetchone():
                raise ValueError("Duplicate imported identity/path")
            db.execute("INSERT INTO images VALUES (?,?)", [key, path])
            record.pop("provenance", None)
            locator = SourceLocator(kind="jsonl", path=source.annotation, row=number)
            ref = f"record:{number}"
            findings = [{"target": target, "locators": [SourceLocator(kind="jsonl", path=source.annotation,
                         row=number, pointer=target).model_dump()]}
                        for target, _ in finding_objects(record)]
            annotation.write(json.dumps({"record": record, "record_ref": ref}, ensure_ascii=False) + "\n")
            provenance.write(ExtractionProvenance(id=ref, locators=[locator], findings=findings,
                                                  step="identity_import").model_dump_json() + "\n")
            population.write(json.dumps({"source_id": source.id, "image_id": key, "image_path": path}) + "\n")
            count += 1
    if not count:
        raise ValueError("Imported annotation contains no images")
    return {"unique_images": count, "coverage": "full", "authority": "input annotation only"}


def verify_preparation_archive(access):
    path, expected = access.get("preparation_path"), access.get("preparation_digest")
    if path is None and expected is None:
        return "legacy_unknown"
    if not path or not expected or not Path(path).is_file() or digest_file(path) != expected:
        raise ValueError("Prepared preparation index changed or is missing")
    index = json.loads(Path(path).read_text(encoding="utf-8"))
    if index.get("schema_version") != "preparation-archive.v1" or not index.get("sources"):
        raise ValueError("Invalid preparation archive")
    root = Path(path).parent
    for source in index["sources"].values():
        if not isinstance(source, dict) or not isinstance(source.get("files"), dict) or not source["files"]:
            raise ValueError("Preparation archive requires bound source artifacts")
        required = {"spec.json", "fingerprint.json", "execution_plan.json", "population.jsonl",
                    "population_decision.json", "validation.json", "upstream_provenance.jsonl"}
        if not required <= {Path(name).name for name in source["files"]}:
            raise ValueError("Preparation archive is incomplete")
        for relative, digest in source["files"].items():
            file = source_image_path(root, relative)
            if digest_file(file) != digest:
                raise ValueError(f"Preparation artifact changed: {relative}")
    return "bound"


def archive_source(context):
    """Copy controller-authored artifacts, evidence and full population into the bundle."""
    from medbenchagent.runtime.inspection import inspection
    from medbenchagent.data.contracts import SemanticPreparationSpec, SourceFingerprint
    spec = SemanticPreparationSpec.model_validate_json(Path(context["semantic_spec_path"]).read_text(encoding="utf-8"))
    fingerprint = SourceFingerprint.model_validate_json(Path(context["fingerprint_path"]).read_text(encoding="utf-8"))
    source_id = context["source_scope"]
    folder = Path(context["bundle_root"]) / "preparation" / source_id / Path(context["operation_directory"]).name
    folder.mkdir(parents=True, exist_ok=True)
    files = {"spec.json": context["semantic_spec_path"], "fingerprint.json": context["fingerprint_path"],
             "validation.json": context["preparation_validation_path"], "population_decision.json": context["population_decision_path"],
             "population.jsonl": context["accepted_population_path"], "execution_runtime.json": str(Path(context["bundle_root"]) / "execution_runtime.json"),
             "upstream_provenance.jsonl": str(Path(context["prepared_annotation_path"]).parent / "provenance.jsonl")}
    execution = json.loads(Path(context["execution_plan_path"]).read_text(encoding="utf-8"))
    execution.update(spec_path="spec.json", fingerprint_path="fingerprint.json")
    if context.get("annotation_script_path") and execution["plan"]["executor"] == "generated_python":
        files["adapter.py"] = context["annotation_script_path"]
        execution["adapter_path"] = "adapter.py"
    if context.get("source_report_path"):
        files["source_report.json"] = context["source_report_path"]
    if context.get("source_inspection_script_path"):
        files["inspect_source.py"] = context["source_inspection_script_path"]
    files["files.jsonl"] = fingerprint.files_path
    files["used_files.jsonl"] = fingerprint.used_files_path
    for name, path in files.items():
        shutil.copyfile(path, folder / name)
    fingerprint.files_path = "files.jsonl"
    fingerprint.used_files_path = "used_files.jsonl"
    atomic_write_json(folder / "fingerprint.json", fingerprint.model_dump())
    atomic_write_json(folder / "execution_plan.json", execution)
    candidate = json.loads(Path(context["candidate_path"]).read_text(encoding="utf-8"))
    atomic_write_json(folder / "candidate_outputs.json", {"outputs": candidate["outputs"]})
    decisions = [spec.population, *spec.identities.values(), *spec.relationships, *spec.fields,
                 spec.missing, spec.findings, spec.spatial]
    refs = {r for d in decisions for r in d.inspection_references}
    refs.update(e.reference for d in decisions for e in d.semantic_evidence)
    refs.update(item["reference"] for item in fingerprint.inspections)
    refs.add(spec.population_reference)
    decision = json.loads(Path(context["population_decision_path"]).read_text(encoding="utf-8"))
    if decision.get("answer_reference"):
        refs.add(decision["answer_reference"])
    for ref in sorted(refs):
        receipt = inspection(context, ref, source_id=source_id)
        atomic_write_json(folder / "evidence" / (ref + ".json"), receipt)
    relative = folder.relative_to(Path(context["bundle_root"]))
    return {"plan_ref": (relative / "execution_plan.json").as_posix(),
            "files": {p.relative_to(Path(context["bundle_root"]) / "preparation").as_posix(): digest_file(p)
                      for p in sorted(folder.rglob("*")) if p.is_file()},
            "assurance": decision["assurance"]}


def write_preparation_index(bundle, sources):
    path = Path(bundle) / "preparation" / "index.json"
    value = {"schema_version": "preparation-archive.v1", "sources": sources}
    if (Path(bundle) / "ready.json").exists():
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("Published preparation archive is immutable")
    else:
        atomic_write_json(path, value)
    return {"preparation_path": str(path), "preparation_digest": digest_file(path)}
