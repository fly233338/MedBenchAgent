"""Deterministic image rules shared by converters and host validation."""
from fnmatch import fnmatchcase
from pathlib import Path

from medbenchagent.data.asset_adapters import contained_path
from medbenchagent.data.sources import digest_path
from .contracts import digest_payload


def image_rule(mapping, source_id, source_path):
    override = mapping.image_overrides.get(f"{source_id}:{source_path}")
    matches = [override] if override else [r for r in mapping.image_rules
        if r.source_id == source_id and fnmatchcase(source_path, r.pattern)]
    if len(matches) != 1 or matches[0].source_id != source_id:
        raise ValueError(f"Image must match exactly one rule: {source_id}:{source_path}")
    return matches[0]


def resolve_image(mapping, plan, source_roots, bundle_root, source_id, source_path, *, index=None,
                  locator_roots=None):
    """Read via mounted roots, but serialize host locators, including in Docker."""
    rule = image_rule(mapping, source_id, source_path)
    source = contained_path(source_roots[source_id], source_path)
    if not source.exists():
        raise ValueError(f"Missing image source: {source}")
    if rule.action != "convert":
        root = (locator_roots or source_roots)[source_id]
        return str(Path(root) / source_path).replace("\\", "/")
    policy = plan.policies[rule.policy_id]
    if policy.mode == "slices" and index not in policy.indices:
        raise ValueError("A slice target requires an accepted index")
    if policy.mode != "slices" and index is not None:
        raise ValueError("Slice index is only valid for slices mode")
    identity = digest_payload([source_id, source_path, digest_path(source), policy.model_dump(mode="json"), index])
    return str(Path(bundle_root) / "images" / f"{identity}.png").replace("\\", "/")


def mapped_images(mapping, plan, state):
    """Expand rules in deterministic host code, never in an LLM response."""
    result = {}
    folded = {}
    for source_id, root in state["source_roots"].items():
        for source in sorted(Path(root).rglob("*")):
            relative = source.relative_to(root).as_posix()
            rules = [r for r in mapping.image_rules if r.source_id == source_id and fnmatchcase(relative, r.pattern)]
            if not rules and f"{source_id}:{relative}" not in mapping.image_overrides:
                continue
            rule = image_rule(mapping, source_id, relative)
            if source.is_dir() != (rule.source_format == "dicom_series"):
                continue
            policy = plan.policies.get(rule.policy_id)
            indices = policy.indices if rule.action == "convert" and policy.mode == "slices" else [None]
            for index in indices:
                target = resolve_image(mapping, plan, state["source_roots"], state["bundle_root"],
                    source_id, relative, index=index, locator_roots=state.get("locator_roots"))
                identity = (source_id, relative, index)
                if target.casefold() in folded and folded[target.casefold()] != identity:
                    raise ValueError("Case-insensitive image locator collision")
                folded[target.casefold()] = identity
                result[target] = (rule, source, relative, index)
    return result


def materialize_mapped_image(mapping, plan, state, source_id, source_path, output_root, *, index=None):
    """Produce one converted candidate and its measured record using accepted identities."""
    from medbenchagent.data.models import AssetEntry, ImageEntry
    from medbenchagent.data.asset_adapters import materialize_image, adapter_for
    rule = image_rule(mapping, source_id, source_path)
    if rule.action != "convert":
        raise ValueError("Only convert rules may materialize files")
    locator = resolve_image(mapping, plan, state["source_roots"], state["bundle_root"],
        source_id, source_path, index=index, locator_roots=state.get("locator_roots"))
    root = state["source_roots"][source_id]
    source_digest = digest_path(contained_path(root, source_path))
    asset = AssetEntry(asset_id=f"{source_id}:{source_path}", source_id=source_id, source_path=source_path,
        format=rule.source_format, file_digest=source_digest,
        kind="volume" if rule.source_format in {"nifti", "nrrd", "itk", "dicom_series"} else "image")
    policy = plan.policies[rule.policy_id]
    spec = policy.model_copy(update={"indices": [index]}) if index is not None else policy
    info = materialize_image(asset, spec, Path(output_root) / "images" / Path(locator).name, source_root=root)
    geometry = adapter_for(asset).inspect(asset, source_root=root) if asset.kind == "volume" else {}
    # Medical backends may expose numpy values; keep manifest metadata JSON-native.
    import json
    geometry = json.loads(json.dumps(geometry, default=lambda value: value.tolist()))
    return ImageEntry(path=locator, file_digest=info["output_digest"], format="png", width=info["width"], height=info["height"],
        source_id=source_id, source_path=source_path, source_digest=source_digest, kind=asset.kind,
        metadata={"geometry": geometry, "conversion": policy.model_dump(mode="json"), "slice_index": index})
