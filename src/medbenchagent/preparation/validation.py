"""Read-only stage checks and one final cross-file validation."""
import json
import shutil
from pathlib import Path
import os
import uuid
from filelock import FileLock
from medbenchagent.data.sources import digest_path
from .images import mapped_images

from medbenchagent.data.models import CaseRecord, validate_case
from medbenchagent.data.manifests import iter_image_manifest, SamplingManifest, SplitManifest
from medbenchagent.data.asset_adapters import verify_image
from .contracts import (Stage, STAGE_OUTPUTS, PreparationMapping, ImageConversionPlan,
                        AcceptedHandoff, ArtifactRef, Diagnostic, ValidationResult, digest_file, digest_payload)


def read_cases(path):
    cases = [CaseRecord.model_validate_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Cases must be nonempty with unique IDs")
    return cases


def _load_inputs(state):
    handoff = AcceptedHandoff.model_validate(state["accepted_handoff"], strict=False)
    for ref in (handoff.mapping, handoff.image_conversion_plan):
        if digest_file(ref.path) != ref.digest:
            raise ValueError("Accepted semantics changed")
    return (PreparationMapping.model_validate_json(Path(handoff.mapping.path).read_text(encoding="utf-8")),
            ImageConversionPlan.model_validate_json(Path(handoff.image_conversion_plan.path).read_text(encoding="utf-8")))


def validate_stage(root, stage, state):
    root, stage = Path(root), Stage(stage)
    try:
        for name in STAGE_OUTPUTS[stage]:
            if not (root / name).is_file():
                raise ValueError(f"Missing output: {name}")
        if stage is Stage.UNDERSTAND:
            mapping = PreparationMapping.model_validate_json((root / "mapping.json").read_text(encoding="utf-8"))
            plan = ImageConversionPlan.model_validate_json((root / "image_conversion_plan.json").read_text(encoding="utf-8"))
            ids = [r.rule_id for r in mapping.image_rules]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate image rule ID")
            for rule in [*mapping.image_rules, *mapping.image_overrides.values()]:
                if rule.source_id not in state["source_roots"]:
                    raise ValueError("Unknown image source")
                if rule.action == "convert" and rule.policy_id not in plan.policies:
                    raise ValueError("Missing conversion policy")
                if rule.action == "convert" and plan.policies[rule.policy_id].mode in {"slices", "montage"}:
                    policy = plan.policies[rule.policy_id]
                    if policy.axis is None or not policy.indices:
                        raise ValueError("Volume policy requires explicit axis and indices")
                if rule.action == "direct" and rule.source_format.lower() not in {"jpeg", "jpg", "png", "webp", "gif"}:
                    raise ValueError("Nonordinary input must convert to PNG")
            from .accounting import records
            sum(1 for _ in records(mapping, state["source_roots"]))
            if not mapped_images(mapping, plan, state):
                raise ValueError("Image rules match no sources")
        else:
            mapping, plan = _load_inputs(state)
            mapped = mapped_images(mapping, plan, state)
            if stage is Stage.PREPARE_TEXT:
                cases = read_cases(root / "cases.jsonl")
                from .accounting import validate_accounting
                validate_accounting(root.parent / "accounting.json", mapping, state["source_roots"], len(cases))
                for case in cases:
                    validate_case(case)
                    for evidence in case.evidence:
                        if evidence.type != "asset":
                            continue
                        if evidence.value not in mapped:
                            raise ValueError("Case locator does not match accepted image rules")
                        rule, source, _, _ = mapped[evidence.value]
                        if rule.action == "reference_only":
                            if evidence.mask is None or evidence.mask.source_digest != digest_path(source):
                                raise ValueError("Reference mask requires matching embedded geometry")
                        elif evidence.mask is not None:
                            raise ValueError("Mask requires reference-only rule")
                        elif rule.action == "direct":
                            verify_image(source)
            else:
                for ref in state["text_artifacts"].values():
                    if digest_file(ref["path"]) != ref["digest"]:
                        raise ValueError("Accepted Text changed")
                cases = read_cases(Path(state["text_root"]) / "cases.jsonl")
                needed = {e.value for c in cases for e in c.evidence if e.type == "asset" and mapped[e.value][0].action == "convert"}
                images = {row.path: row for row in iter_image_manifest(root / "image_manifest.jsonl")}
                if set(images) != needed:
                    raise ValueError("Converted manifest differs from required converted locators")
                for locator, image in images.items():
                    rule, source, relative, _ = mapped[locator]
                    if (image.source_id, image.source_path, image.source_digest) != (rule.source_id, relative, digest_path(source)):
                        raise ValueError("Converted source identity or digest mismatch")
                    if Path(locator).suffix != ".png":
                        raise ValueError("Converted target must use .png")
                    info = verify_image(root / "images" / Path(locator).name, image.file_digest)
                    if (info["width"], info["height"], info["format"].lower()) != (image.width, image.height, image.format.lower()) or info["format"] != "PNG":
                        raise ValueError("Converted image must be a real PNG with matching metadata")
        return ValidationResult(status="accepted", stage=stage)
    except (ValueError, KeyError, OSError) as exc:
        return ValidationResult(status="rejected", stage=stage, issues=[Diagnostic(stage=stage, code="invalid_output", message=str(exc))])


def accept_mapping(root, state):
    root = Path(root)
    refs = {name: ArtifactRef(name=filename, path=str(root / filename), digest=digest_file(root / filename))
            for name, filename in (("mapping", "mapping.json"), ("image_conversion_plan", "image_conversion_plan.json"))}
    return AcceptedHandoff(dataset_id=state["dataset_id"], source_snapshot_digest=state["source_snapshot_digest"],
        **refs, digest=digest_payload({name: ref.digest for name, ref in refs.items()}))


def bundle_digest(root):
    root = Path(root)
    return digest_payload({p.relative_to(root).as_posix(): digest_file(p) for p in sorted(root.rglob("*")) if p.is_file()})


def validate_bundle(root):
    root = Path(root)
    cases = read_cases(root / "cases.jsonl")
    from medbenchagent.data.manifests import dataset_images
    images = dataset_images(root / "cases.jsonl", root / "image_manifest.jsonl")
    expected = {"cases.jsonl", "image_manifest.jsonl", "metadata.json"}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError("Dataset contains non-dataset files")
    for case in cases:
        validate_case(case, images)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    ids = {case.case_id for case in cases}
    for name, model, field in (("sampling", SamplingManifest, "units"), ("split", SplitManifest, "records")):
        if name in metadata:
            manifest = model.model_validate(metadata[name])
            if any(row.case_id not in ids for row in getattr(manifest, field)):
                raise ValueError(f"Unknown Case in {name}")
    return bundle_digest(root)


def package_dataset(state):
    """Assemble immutable stage results; publication never repairs them."""
    text = Path(state["text_root"])
    outputs = Path(state["attempt_workspace"]) / "work/outputs"
    root = text.parent / "package"
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(text / "cases.jsonl", root / "cases.jsonl")
    mapping, plan = _load_inputs(state)
    mapped = mapped_images(mapping, plan, state)
    cases = read_cases(text / "cases.jsonl")
    needed = {e.value for c in cases for e in c.evidence if e.type == "asset" and e.mask is None}
    converted = {p for p in needed if mapped[p][0].action == "convert"}
    if converted:
        shutil.copy2(outputs / "image_manifest.jsonl", root / "image_manifest.jsonl")
        for image in iter_image_manifest(root / "image_manifest.jsonl"):
            destination = Path(image.path)
            if destination.parent.resolve() != (Path(state["bundle_root"]) / "images").resolve():
                raise ValueError("Converted image outside fixed dataset images directory")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(str(destination) + ".lock"):
                if destination.exists():
                    verify_image(destination, image.file_digest)
                else:
                    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
                    try:
                        shutil.copyfile(outputs / "images" / destination.name, temporary)
                        verify_image(temporary, image.file_digest)
                        os.replace(temporary, destination)
                    finally:
                        temporary.unlink(missing_ok=True)
    else:
        (root / "image_manifest.jsonl").write_text("", encoding="utf-8")
    direct = {p: digest_file(mapped[p][1]) for p in needed - converted}
    metadata = {"dataset_id": state["dataset_id"], "field_semantics": mapping.field_semantics,
                "source_roots": state["source_roots"], "direct_images_digest": digest_payload(direct),
                "fields": [rule.model_dump(mode="json") for rule in mapping.fields if rule.disposition == "medical_evidence"]}
    if (text / "metadata.json").is_file():
        extra = json.loads((text / "metadata.json").read_text(encoding="utf-8"))
        metadata.update({name: extra[name] for name in ("sampling", "split") if name in extra})
    (root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    digest = validate_bundle(root)
    return root, digest
