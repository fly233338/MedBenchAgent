"""Stage validation and deterministic assembly; no semantic decisions."""

import json
import shutil
from pathlib import Path

from medbenchagent.data.models import CaseRecord, AssetEntry, MaskFeatureSet
from medbenchagent.data.manifests import PresentationRecord
from medbenchagent.data.presentation import target_path, contained_path, verify_image
from .contracts import (
    AcceptedHandoff, ImageConversionPlan, ImageRequest, PopulationOutcome,
    Stage, STAGE_SPECS, Submission, ValidationIssue, ValidationResult, digest_file,
)


def rows(path, model):
    return [model.model_validate_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_stage_outputs(root, stage, state):
    root, stage = Path(root), Stage(stage)
    try:
        for name in STAGE_SPECS[stage.value].required_outputs:
            if not (root / name).is_file():
                raise ValueError(f"Missing stage output: {name}")
        submission = Submission.model_validate_json((root / "submission.json").read_text(encoding="utf-8"))
        if submission.requested_outcome != "submitted":
            raise ValueError("Stage must submit completed outputs")
        handoff = AcceptedHandoff.model_validate(state["accepted_handoff"], strict=False)
        for artifact in (handoff.mapping, handoff.population, handoff.image_conversion_plan):
            if digest_file(artifact.path) != artifact.digest:
                raise ValueError(f"Accepted artifact changed: {artifact.name}")
        plan = ImageConversionPlan.model_validate_json(Path(handoff.image_conversion_plan.path).read_text(encoding="utf-8"))
        text_root = root if stage is Stage.PREPARE_TEXT else Path(state["text_root"])
        if stage is Stage.PREPARE_IMAGE:
            for artifact in state["text_artifacts"].values():
                if digest_file(artifact["path"]) != artifact["digest"]:
                    raise ValueError("Accepted Text output changed")
        cases = rows(text_root / "cases.jsonl", CaseRecord)
        case_map = {case.case_id: case for case in cases}
        if not cases or len(case_map) != len(cases):
            raise ValueError("Cases must be nonempty with unique IDs")
        outcomes = rows(text_root / "provenance.jsonl", PopulationOutcome)
        identities = [(row.source_id, row.source_unit_id) for row in outcomes]
        population = { (row["source_id"], row["source_unit_id"])
            for row in map(json.loads, Path(handoff.population.path).read_text(encoding="utf-8").splitlines()) if row }
        retained = {key for row in outcomes if row.status == "retained" for key in row.case_ids}
        if len(set(identities)) != len(identities) or set(identities) != population or retained != set(case_map):
            raise ValueError("Text population and Case provenance do not reconcile")
        requests = rows(text_root / "image_requests.jsonl", ImageRequest)
        paths, request_keys = {}, set()
        for request in requests:
            if request.case_id not in case_map:
                raise ValueError("Image request references unknown Case")
            if request.asset_id not in {e.value for e in case_map[request.case_id].evidence if e.type == "asset"}:
                raise ValueError("Image request references unknown Case asset")
            policy = plan.policies[request.policy_id]
            expected = target_path(request.source_id, request.asset_id, request.view_id, policy, request.source_format)
            if request.target_path != expected:
                raise ValueError("Image target differs from accepted stable path rule")
            contained_path(root, request.target_path)
            source = contained_path(state["source_roots"][request.source_id], request.source_path)
            if not source.exists():
                raise ValueError("Image request source is missing")
            identity = (request.source_id, request.asset_id, request.view_id, request.policy_id)
            folded = request.target_path.casefold()
            if folded in paths and paths[folded] != identity:
                raise ValueError("Image target collision")
            paths[folded] = identity
            key = (request.case_id, request.asset_id, request.view_id, request.order)
            if key in request_keys:
                raise ValueError("Duplicate image request")
            request_keys.add(key)
        requested = {(r.case_id, r.asset_id) for r in requests}
        if any((c.case_id, e.value) not in requested for c in cases for e in c.evidence if e.type == "asset"):
            raise ValueError("Every Case asset requires an Image request")
        if stage is Stage.PREPARE_IMAGE:
            assets = rows(root / "asset_manifest.jsonl", AssetEntry)
            asset_map = {asset.asset_id: asset for asset in assets}
            if len(asset_map) != len(assets):
                raise ValueError("Duplicate Asset IDs")
            if set(asset_map) != {r.asset_id for r in requests}:
                raise ValueError("Asset set differs from Text requests")
            for request in requests:
                asset = asset_map[request.asset_id]
                if (asset.source_id, asset.source_path, asset.format.lower()) != (request.source_id, request.source_path, request.source_format.lower()):
                    raise ValueError("Image stage changed source Asset identity")
                if request.reference_only and asset.metadata.get("input_visibility") != "reference_only":
                    raise ValueError("Reference-only request must retain hidden Asset visibility")
            presentations = rows(root / "presentation_manifest.jsonl", PresentationRecord)
            by_key = {(r.case_id, r.asset_id, r.view_id, r.order): r for r in presentations}
            visible = [r for r in requests if not r.reference_only]
            if len(by_key) != len(presentations) or set(by_key) != {
                (r.case_id, r.asset_id, r.view_id, r.order) for r in visible
            }:
                raise ValueError("Presentation set differs from Text requests")
            for request in visible:
                record = by_key[(request.case_id, request.asset_id, request.view_id, request.order)]
                if record.path != request.target_path or record.input_digest != asset_map[record.asset_id].file_digest:
                    raise ValueError("Presentation identity/digest differs from request")
                info = verify_image(contained_path(root, record.path), record.output_digest)
                policy = plan.policies[request.policy_id]
                if request.source_format.lower() in {"png", "jpg", "jpeg", "webp", "gif"} and not any(
                    policy.get(key) for key in ("window", "resize", "crop", "indices")
                ) and record.output_digest != asset_map[record.asset_id].file_digest:
                    raise ValueError("Native untransformed image bytes must be preserved")
                if request.source_format.lower() not in {"png", "jpg", "jpeg", "webp", "gif"} and info["format"] != "PNG":
                    raise ValueError("Medical source must produce PNG")
            for line in (root / "mask_features.jsonl").read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                case = case_map[row["case_id"]]
                evidence = next(e for e in case.evidence if e.id == row["evidence_id"])
                feature = MaskFeatureSet.model_validate(row["mask"])
                if feature.source_digest != asset_map[evidence.value].file_digest:
                    raise ValueError("Mask feature digest differs from Asset")
        return ValidationResult(status="accepted", stage=stage)
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        return ValidationResult(status="rejected", stage=stage, issues=[ValidationIssue(
            stage=stage, issue_id=f"{stage.value}:invalid", code="candidate_invalid", message=str(exc),
        )])


def assemble_candidate(state):
    """Copy accepted stage outputs into a fresh final candidate and attach measured features."""
    text, images = Path(state["text_root"]), Path(state["attempt_workspace"]) / "work" / "outputs"
    workspace = Path(state["attempt_workspace"]).resolve()
    destination = (workspace / "work" / "package").resolve()
    if not destination.is_relative_to(workspace):
        raise ValueError("Package path escapes attempt workspace")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for source in (text, images):
        for name in ("cases.jsonl", "provenance.jsonl", "asset_manifest.jsonl", "presentation_manifest.jsonl",
                     "sampling_manifest.jsonl", "split_manifest.jsonl"):
            path = source / name
            if path.is_file():
                shutil.copy2(path, destination / name)
    if (images / "images").is_dir():
        shutil.copytree(images / "images", destination / "images")
    handoff = AcceptedHandoff.model_validate(state["accepted_handoff"], strict=False)
    for artifact in (handoff.mapping, handoff.image_conversion_plan):
        shutil.copy2(artifact.path, destination / artifact.name)
    cases = rows(destination / "cases.jsonl", CaseRecord)
    by_case = {case.case_id: case for case in cases}
    for line in (images / "mask_features.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            evidence = next(e for e in by_case[row["case_id"]].evidence if e.id == row["evidence_id"])
            evidence.mask = MaskFeatureSet.model_validate(row["mask"])
    (destination / "cases.jsonl").write_text("".join(c.model_dump_json() + "\n" for c in cases), encoding="utf-8")
    return destination
