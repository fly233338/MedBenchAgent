"""Read-only stage checks and one final cross-file validation."""
import json
import shutil
from pathlib import Path

from medbenchagent.data.models import CaseRecord, validate_case
from medbenchagent.data.manifests import iter_image_manifest, SamplingManifest, SplitManifest
from medbenchagent.data.asset_adapters import contained_path, verify_image
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
            paths = set()
            for target, source in mapping.images.items():
                canonical = contained_path(root, target).relative_to(root.resolve()).as_posix()
                if canonical != target or (not source.reference_only and not target.startswith("images/")):
                    raise ValueError("Image locators must be canonical relative paths under images/")
                if canonical.casefold() in paths:
                    raise ValueError("Case-insensitive image target collision")
                paths.add(canonical.casefold())
                if source.policy_id not in plan.policies:
                    raise ValueError("Image mapping references missing conversion policy")
                locator = contained_path(state["source_roots"][source.source_id], source.source_path)
                if not locator.exists():
                    raise ValueError(f"Missing mapped source: {source.source_path}")
        elif stage is Stage.PREPARE_TEXT:
            mapping, _ = _load_inputs(state)
            cases = read_cases(root / "cases.jsonl")
            referenced = {e.value for c in cases for e in c.evidence if e.type == "asset"}
            if not referenced <= mapping.images.keys():
                raise ValueError("Case image locator is absent from accepted mapping")
            required = {path for path, source in mapping.images.items() if source.required}
            if not required <= referenced:
                raise ValueError("Mapped source coverage is incomplete")
            for case in cases:
                for evidence in case.evidence:
                    if evidence.type == "asset" and mapping.images[evidence.value].reference_only and evidence.mask is None:
                        raise ValueError("Reference mask features must be attached directly to Case Evidence")
                    if evidence.mask is not None:
                        if not mapping.images[evidence.value].reference_only:
                            raise ValueError("Mask feature must match reference-only mapping")
                        from medbenchagent.data.sources import digest_path
                        source = mapping.images[evidence.value]
                        if evidence.mask.source_digest != digest_path(contained_path(state["source_roots"][source.source_id], source.source_path)):
                            raise ValueError("Mask feature source_digest mismatch")
        else:
            mapping, plan = _load_inputs(state)
            for ref in state["text_artifacts"].values():
                if digest_file(ref["path"]) != ref["digest"]:
                    raise ValueError("Accepted Text changed")
            images = {row.path: row for row in iter_image_manifest(root / "image_manifest.jsonl")}
            cases = read_cases(Path(state["text_root"]) / "cases.jsonl")
            needed = {e.value for c in cases for e in c.evidence if e.type == "asset" and e.mask is None}
            if set(images) != needed:
                raise ValueError("Image manifest differs from Case locators")
            for path, image in images.items():
                source = mapping.images[path]
                if (source.source_id, source.source_path) != (image.source_id, image.source_path):
                    raise ValueError("Image source differs from accepted mapping")
                info = verify_image(contained_path(root, path), image.file_digest)
                if (info["width"], info["height"], info["format"].lower()) != (image.width, image.height, image.format.lower()):
                    raise ValueError("Image metadata differs from decoded output")
                from medbenchagent.data.sources import digest_path
                if digest_path(contained_path(state["source_roots"][source.source_id], source.source_path)) != image.source_digest:
                    raise ValueError("Image source digest changed")
                ordinary = source.source_format.lower() in {"png", "jpg", "jpeg", "webp", "gif"}
                if not ordinary and info["format"] != "PNG":
                    raise ValueError("Medical displays must be PNG")
                if ordinary and not any(plan.policies[source.policy_id].get(k) for k in ("window", "crop", "resize", "indices")):
                    if image.file_digest != image.source_digest:
                        raise ValueError("Untransformed ordinary images must preserve source bytes")
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
    images = {image.path: image for image in iter_image_manifest(root / "image_manifest.jsonl")}
    needed = {e.value for c in cases for e in c.evidence if e.type == "asset" and e.mask is None}
    if needed != set(images):
        raise ValueError("Case/image manifest coverage mismatch")
    expected = {"cases.jsonl", "image_manifest.jsonl", "metadata.json", *needed}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError("Dataset contains unregistered or non-dataset files")
    for case in cases:
        validate_case(case, images)
    for image in images.values():
        verify_image(contained_path(root, image.path), image.file_digest)
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
    images = Path(state["attempt_workspace"]) / "work/outputs"
    root = Path(state["attempt_workspace"]) / "work/package"
    root.mkdir(parents=True, exist_ok=True)
    # A retry owns a fresh attempt directory; no stale files are merged.
    for name in ("cases.jsonl",):
        shutil.copy2(text / name, root / name)
    shutil.copy2(images / "image_manifest.jsonl", root / "image_manifest.jsonl")
    for image in iter_image_manifest(images / "image_manifest.jsonl"):
        destination = contained_path(root, image.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(contained_path(images, image.path), destination)
    mapping, _ = _load_inputs(state)
    metadata = {"dataset_id": state["dataset_id"], "field_semantics": mapping.field_semantics,
                "fields": [rule.model_dump(mode="json") for rule in mapping.fields if rule.disposition == "medical_evidence"]}
    if (text / "metadata.json").is_file():
        extra = json.loads((text / "metadata.json").read_text(encoding="utf-8"))
        metadata.update({name: extra[name] for name in ("sampling", "split") if name in extra})
    (root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    digest = validate_bundle(root)
    return root, digest
