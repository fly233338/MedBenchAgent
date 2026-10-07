"""Shared identities, paths and integrity checks for model-visible images."""

from pathlib import Path, PurePosixPath
import hashlib
import json

from PIL import Image

from .sources import digest_file

SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "GIF"}


def target_path(source_id, asset_id, view_id, policy, source_format):
    """Compute the same target before and after conversion, independent of Case."""
    native = source_format.lower() in {"jpg", "jpeg", "png", "webp", "gif"}
    preserve = native and not any(policy.get(key) for key in ("window", "resize", "crop", "indices"))
    suffix = source_format.lower() if preserve else "png"
    identity = json.dumps([source_id, asset_id, view_id, policy], sort_keys=True, separators=(",", ":"))
    return f"images/{hashlib.sha256(identity.encode()).hexdigest()}.{suffix}"


def contained_path(root, relative):
    value = PurePosixPath(relative.replace("\\", "/"))
    if value.is_absolute() or ".." in value.parts or any(":" in part for part in value.parts):
        raise ValueError(f"Unsafe presentation path: {relative}")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Presentation escapes root: {relative}")
    return path


def verify_image(path, expected_digest=None):
    path = Path(path)
    if expected_digest and digest_file(path) != expected_digest:
        raise ValueError(f"Presentation digest mismatch: {path}")
    with Image.open(path) as image:
        image.load()
        if image.format not in SUPPORTED_IMAGE_FORMATS or getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"Unsupported model image: {path}")
        suffixes = {"JPEG": {".jpg", ".jpeg"}, "PNG": {".png"}, "WEBP": {".webp"}, "GIF": {".gif"}}
        if path.suffix.lower() not in suffixes[image.format]:
            raise ValueError(f"Image extension disagrees with encoding: {path}")
        return {"format": image.format, "width": image.width, "height": image.height}


def materialize_presentation(asset, spec, destination, *, source_root):
    """Preserve ordinary native images; render medical sources using shared adapters."""
    import shutil
    from .asset_adapters import adapter_for

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = contained_path(source_root, asset.source_path)
    native = asset.format.lower() in {"jpg", "jpeg", "png", "webp", "gif"}
    if native and spec.mode == "native" and not any((spec.window, spec.resize, spec.crop)):
        verify_image(source, asset.file_digest)
        shutil.copyfile(source, destination)
    else:
        if destination.suffix.lower() != ".png":
            raise ValueError("Derived medical presentations require PNG")
        adapter_for(asset).present_to_path(asset, spec, destination, source_root=source_root)
    return {**verify_image(destination), "output_digest": digest_file(destination)}
