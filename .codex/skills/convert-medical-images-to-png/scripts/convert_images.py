# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26,<3",
#   "pillow>=11,<13",
#   "pydicom>=3,<4",
#   "pylibjpeg>=2,<3",
#   "pylibjpeg-libjpeg>=2,<3",
#   "pylibjpeg-openjpeg>=2,<3",
#   "nibabel>=5,<6",
#   "SimpleITK>=2.4,<3",
# ]
# ///
"""Convert supported 2D medical images into deterministic PNG files."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

WSI_SUFFIXES = {".bif", ".mrxs", ".ndpi", ".scn", ".svs"}
NIFTI_SUFFIXES = {".nii"}
ITK_SUFFIXES = {".mha", ".mhd", ".nhdr", ".nrrd"}
RASTER_FORMATS = {"BMP", "JPEG", "PNG", "TIFF", "WEBP"}
MANIFEST_FIELDS = (
    "source_relative_path",
    "source_filename",
    "source_sha256",
    "detected_format",
    "status",
    "png_relative_path",
    "png_filename",
    "width",
    "height",
    "output_mode",
    "normalization",
    "output_sha256",
    "error",
)


class ConversionError(RuntimeError):
    """Raised when recognized input cannot be converted safely."""


class UnsupportedMedicalImage(ConversionError):
    """Raised when input is recognized but intentionally outside v1."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slug(value: str, limit: int = 80) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    rendered = "".join(char if char.isalnum() or char in "-_" else "_" for char in normalized)
    rendered = re.sub(r"_+", "_", rendered).strip("_.-")
    return (rendered or "image")[:limit]


def output_relative_path(source_relative_path: str) -> str:
    source = PurePosixPath(source_relative_path)
    stem = source.name
    if stem.lower().endswith(".nii.gz"):
        stem = stem[:-7]
    else:
        stem = PurePosixPath(stem).stem
    path_hash = hashlib.sha256(source_relative_path.encode("utf-8")).hexdigest()[:10]
    return f"images/{_slug(stem)}__{path_hash}.png"


def _read_header(path: Path, size: int = 560) -> bytes:
    with path.open("rb") as source:
        prefix = source.read(2)
        source.seek(0)
        if prefix == b"\x1f\x8b":
            with gzip.GzipFile(fileobj=source) as compressed:
                return compressed.read(size)
        return source.read(size)


def _looks_like_nifti(path: Path) -> bool:
    try:
        header = _read_header(path)
    except (OSError, EOFError):
        return False
    if len(header) >= 348:
        sizes = {struct.unpack("<I", header[:4])[0], struct.unpack(">I", header[:4])[0]}
        if 348 in sizes and header[344:348] in {b"n+1\x00", b"ni1\x00"}:
            return True
    if len(header) >= 12:
        sizes = {struct.unpack("<I", header[:4])[0], struct.unpack(">I", header[:4])[0]}
        if 540 in sizes and header[4:8] in {b"n+2\x00", b"ni2\x00"}:
            return True
    return False


def _looks_like_itk(path: Path) -> bool:
    try:
        header = _read_header(path, 256)
    except (OSError, EOFError):
        return False
    return header.startswith(b"NRRD000") or b"ObjectType = Image" in header


def _looks_like_dicom(path: Path) -> bool:
    try:
        from pydicom import dcmread
        from pydicom.misc import is_dicom

        if is_dicom(path):
            return True
        dataset = dcmread(
            path,
            stop_before_pixels=True,
            force=True,
            specific_tags=["Rows", "Columns", "SOPClassUID", "PhotometricInterpretation"],
        )
        return bool(
            getattr(dataset, "Rows", None)
            and getattr(dataset, "Columns", None)
            and getattr(dataset, "SOPClassUID", None)
        )
    except Exception:
        return False


def detect_format(path: Path) -> str | None:
    lower_name = path.name.lower()
    if path.suffix.lower() in WSI_SUFFIXES:
        raise UnsupportedMedicalImage(f"whole-slide image is outside v1: {path.name}")
    if lower_name.endswith(".nii.gz") or path.suffix.lower() in NIFTI_SUFFIXES:
        return "nifti"
    if path.suffix.lower() in ITK_SUFFIXES:
        return "itk"
    if _looks_like_nifti(path):
        return "nifti"
    if _looks_like_itk(path):
        return "itk"
    if _looks_like_dicom(path):
        return "dicom"
    try:
        from PIL import Image

        with Image.open(path) as image:
            if image.format in RASTER_FORMATS:
                return f"raster:{image.format.lower()}"
    except Exception:
        return None
    return None


def _normalize_numeric(array: Any) -> tuple[Any, str]:
    import numpy as np

    values = np.asarray(array, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        raise ConversionError("image contains no finite pixel values")
    finite_values = values[finite]
    low, high = np.percentile(finite_values, [0.5, 99.5])
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8), "constant_to_zero"
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    scaled[~finite] = 0.0
    return np.rint(scaled * 255.0).astype(np.uint8), "percentile_0.5_99.5"


def _array_to_rgb(array: Any) -> tuple[Any, str]:
    import numpy as np
    from PIL import Image

    values = np.asarray(array)
    if values.ndim == 2:
        gray, normalization = _normalize_numeric(values)
        return Image.fromarray(gray, mode="L"), normalization
    if values.ndim == 3 and values.shape[-1] in {3, 4}:
        if values.dtype != np.uint8:
            values, normalization = _normalize_numeric(values)
        else:
            normalization = "preserved_uint8_color"
        image = Image.fromarray(values)
        if values.shape[-1] == 4:
            background = Image.new("RGBA", image.size, (0, 0, 0, 255))
            image = Image.alpha_composite(background, image.convert("RGBA"))
            normalization += "+alpha_on_black"
        return image.convert("RGB"), normalization
    raise ConversionError(f"expected 2D grayscale or HxWx3/4 color pixels, got {values.shape}")


def _convert_raster(path: Path) -> tuple[Any, str]:
    import numpy as np
    from PIL import Image

    with Image.open(path) as source:
        if getattr(source, "n_frames", 1) != 1:
            raise UnsupportedMedicalImage(f"multi-page raster is outside v1: {path.name}")
        if source.format == "TIFF" and source.tag_v2.get(330):
            raise UnsupportedMedicalImage(f"pyramidal TIFF is outside v1: {path.name}")
        source.load()
        if source.mode in {"I", "I;16", "I;16B", "I;16L", "F"}:
            return _array_to_rgb(np.asarray(source))
        if source.mode in {"RGBA", "LA"} or (
            source.mode == "P" and "transparency" in source.info
        ):
            foreground = source.convert("RGBA")
            background = Image.new("RGBA", foreground.size, (0, 0, 0, 255))
            return Image.alpha_composite(background, foreground).convert("RGB"), "alpha_on_black"
        return source.convert("RGB"), "preserved_display_values"


def _convert_dicom(path: Path) -> tuple[Any, str]:
    import numpy as np
    from pydicom import dcmread
    from pydicom.pixels import (
        apply_color_lut,
        apply_modality_lut,
        apply_voi_lut,
        convert_color_space,
    )

    dataset = dcmread(path)
    frames = int(getattr(dataset, "NumberOfFrames", 1) or 1)
    if frames != 1:
        raise UnsupportedMedicalImage(f"multi-frame DICOM is outside v1: {path.name}")
    if hasattr(dataset, "pixel_array_options"):
        dataset.pixel_array_options(raw=True)
    try:
        pixels = dataset.pixel_array
    except Exception as exc:
        syntax = getattr(getattr(dataset, "file_meta", None), "TransferSyntaxUID", "unknown")
        raise ConversionError(f"cannot decode DICOM transfer syntax {syntax}: {exc}") from exc

    photometric = str(getattr(dataset, "PhotometricInterpretation", "")).upper()
    if photometric == "PALETTE COLOR":
        return _array_to_rgb(apply_color_lut(pixels, dataset))[0], "dicom_palette_color_lut"
    if photometric in {"RGB", "YBR_FULL", "YBR_FULL_422"}:
        color = np.asarray(pixels)
        if photometric.startswith("YBR"):
            color = convert_color_space(color, photometric, "RGB")
            normalization = "dicom_ybr_to_rgb"
        else:
            normalization = "dicom_rgb"
        image, numeric = _array_to_rgb(color)
        return image, f"{normalization}+{numeric}"
    if np.asarray(pixels).ndim != 2:
        raise ConversionError(f"unsupported DICOM pixel shape: {np.asarray(pixels).shape}")

    transformed = apply_modality_lut(pixels, dataset)
    steps = ["dicom_modality_lut_or_rescale"]
    has_voi = hasattr(dataset, "VOILUTSequence") or (
        hasattr(dataset, "WindowCenter") and hasattr(dataset, "WindowWidth")
    )
    if has_voi:
        transformed = apply_voi_lut(transformed, dataset, index=0)
        steps.append("voi_lut_or_window_0")
    gray, normalization = _normalize_numeric(transformed)
    steps.append(normalization)
    inverse = photometric == "MONOCHROME1" or str(
        getattr(dataset, "PresentationLUTShape", "")
    ).upper() == "INVERSE"
    if inverse:
        gray = 255 - gray
        steps.append("inverse")
    from PIL import Image

    return Image.fromarray(gray, mode="L"), "+".join(steps)


def _convert_nifti(path: Path) -> tuple[Any, str]:
    import nibabel as nib
    import numpy as np

    image = nib.load(path)
    data = image.get_fdata(dtype=np.float32, caching="unchanged")
    squeezed = np.squeeze(data)
    if squeezed.ndim != 2:
        raise UnsupportedMedicalImage(
            f"NIfTI data must squeeze to 2D, got {tuple(data.shape)}: {path.name}"
        )
    converted, normalization = _array_to_rgb(squeezed)
    return converted, f"nifti_scaled+{normalization}"


def _convert_itk(path: Path) -> tuple[Any, str]:
    import numpy as np
    import SimpleITK as sitk

    image = sitk.ReadImage(str(path))
    data = sitk.GetArrayFromImage(image)
    squeezed = np.squeeze(data)
    if squeezed.ndim != 2:
        raise UnsupportedMedicalImage(
            f"ITK image must squeeze to 2D, got {tuple(data.shape)}: {path.name}"
        )
    converted, normalization = _array_to_rgb(squeezed)
    return converted, f"itk_scaled+{normalization}"


def _resize(image: Any, max_edge: int | None) -> tuple[Any, str]:
    if max_edge is None or max(image.size) <= max_edge:
        return image, ""
    from PIL import Image

    width, height = image.size
    scale = max_edge / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS), f"+resized_max_edge_{max_edge}"


def convert_one(path: Path, detected_format: str, output_path: Path, max_edge: int | None) -> dict[str, Any]:
    handlers = {
        "dicom": _convert_dicom,
        "itk": _convert_itk,
        "nifti": _convert_nifti,
    }
    if detected_format.startswith("raster:"):
        image, normalization = _convert_raster(path)
    else:
        image, normalization = handlers[detected_format](path)
    image, resize_step = _resize(image, max_edge)
    normalization += resize_step
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", compress_level=9)
    return {
        "width": image.width,
        "height": image.height,
        "output_mode": image.mode,
        "normalization": normalization,
        "output_sha256": sha256_file(output_path),
    }


def _source_files(input_path: Path, output_path: Path) -> tuple[Path, list[Path]]:
    if input_path.is_file():
        return input_path.parent, [input_path]
    if not input_path.is_dir():
        raise ConversionError(f"input does not exist: {input_path}")
    files: list[Path] = []
    for path in input_path.rglob("*"):
        if not path.is_file():
            continue
        try:
            path.resolve().relative_to(output_path)
            continue
        except ValueError:
            pass
        files.append(path)
    files.sort(key=lambda path: path.relative_to(input_path).as_posix().casefold())
    return input_path, files


def _write_outputs(staging: Path, rows: list[dict[str, Any]], report: dict[str, Any]) -> None:
    manifest_path = staging / "image_conversion_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            )
    with (staging / "image_name_mapping.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (staging / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _trees_equal(left: Path, right: Path) -> bool:
    left_files = sorted(path.relative_to(left) for path in left.rglob("*") if path.is_file())
    right_files = sorted(path.relative_to(right) for path in right.rglob("*") if path.is_file())
    if left_files != right_files:
        return False
    return all(sha256_file(left / rel) == sha256_file(right / rel) for rel in left_files)


def convert(input_path: Path, output_path: Path, max_edge: int | None = None) -> dict[str, Any]:
    source = input_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    if max_edge is not None and max_edge <= 0:
        raise ConversionError("--max-edge must be a positive integer")
    if output == source or output == Path(output.anchor):
        raise ConversionError("output must be a separate, non-root directory")
    if output.exists() and not output.is_dir():
        raise ConversionError(f"output exists and is not a directory: {output}")

    source_root, files = _source_files(source, output)
    if not files:
        raise ConversionError("input contains no files")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    rows: list[dict[str, Any]] = []
    fatal_errors: list[str] = []
    try:
        for path in files:
            source_relative = path.relative_to(source_root).as_posix()
            row: dict[str, Any] = {
                "source_relative_path": source_relative,
                "source_filename": path.name,
                "source_sha256": sha256_file(path),
                "detected_format": None,
                "status": "skipped_unsupported",
                "png_relative_path": None,
                "png_filename": None,
                "width": None,
                "height": None,
                "output_mode": None,
                "normalization": None,
                "output_sha256": None,
                "error": "unrecognized file",
            }
            try:
                detected = detect_format(path)
                row["detected_format"] = detected
                if detected is None:
                    rows.append(row)
                    continue
                png_relative = output_relative_path(source_relative)
                png_path = staging.joinpath(*PurePosixPath(png_relative).parts)
                details = convert_one(path, detected, png_path, max_edge)
                row.update(details)
                row.update(
                    {
                        "status": "converted",
                        "png_relative_path": png_relative,
                        "png_filename": PurePosixPath(png_relative).name,
                        "error": None,
                    }
                )
            except (ConversionError, UnsupportedMedicalImage) as exc:
                row["error"] = str(exc)
                fatal_errors.append(f"{source_relative}: {exc}")
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                fatal_errors.append(f"{source_relative}: {type(exc).__name__}: {exc}")
            rows.append(row)

        if fatal_errors:
            raise ConversionError("conversion preflight failed:\n- " + "\n- ".join(fatal_errors))
        converted_count = sum(row["status"] == "converted" for row in rows)
        if not converted_count:
            raise ConversionError("no supported 2D medical images were found")
        counts = Counter(str(row["status"]) for row in rows)
        report = {
            "input": str(source),
            "source_file_count": len(rows),
            "converted_count": converted_count,
            "skipped_unsupported_count": counts.get("skipped_unsupported", 0),
            "max_edge": max_edge,
            "valid": True,
        }
        _write_outputs(staging, rows, report)

        if output.exists() and any(output.iterdir()):
            if _trees_equal(staging, output):
                return {**report, "output": str(output), "reused_identical_output": True}
            raise ConversionError(
                f"output directory is non-empty and differs from this run: {output}"
            )
        if output.exists():
            output.rmdir()
        os.replace(staging, output)
        return {**report, "output": str(output), "reused_identical_output": False}
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, dest="input_path")
    parser.add_argument("--output", required=True, type=Path, dest="output_path")
    parser.add_argument("--max-edge", type=int)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = convert(args.input_path, args.output_path, args.max_edge)
    except ConversionError as exc:
        print(f"conversion failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
