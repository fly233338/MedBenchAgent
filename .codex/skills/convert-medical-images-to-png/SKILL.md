---
name: convert-medical-images-to-png
description: Convert supported two-dimensional medical image files into deterministic 8-bit PNGs plus machine-readable and CSV mappings. Use when Codex must prepare raster images, single-frame DICOM, 2D NIfTI, NRRD, MHA, or MHD files for VLM input; discover formats by content; apply medical display transforms; reject volumes, multi-frame files, or whole-slide images; or produce image_conversion_manifest.jsonl for $normalize-medical-image-dataset.
---

# Convert Medical Images to PNG

Convert one source file or directory without modifying it. Use the bundled script for all pixel conversion; do not rewrite conversion code per dataset.

## Boundaries

- Accept only two-dimensional images. Allow singleton dimensions that squeeze to 2D.
- Save scalar two-dimensional inputs as single-channel 8-bit grayscale (`L`) PNGs; retain RGB only for genuinely color inputs.
- Reject multi-frame DICOM, true volumes, WSI, and pyramidal or multi-page TIFF before publishing output.
- Preserve source width and height unless the user explicitly requests `--max-edge`.
- Treat the PNG as a display representation, not a diagnostic or quantitative substitute for the source.
- Do not infer findings or choose a medically preferred slice.

Read [references/format-support.md](references/format-support.md) when diagnosing an unsupported format or extending a handler.

## Workflow

1. Resolve a source file or recursively inventory a source directory.
2. Choose a separate, absent or empty output directory.
3. Run:

```powershell
uv run --script .codex\skills\convert-medical-images-to-png\scripts\convert_images.py `
  --input <file-or-directory> `
  --output <output-directory> `
  [--max-edge <pixels>]
```

4. Inspect `conversion_report.json`. Treat any preflight or conversion error as failure.
5. Use `image_name_mapping.csv` for human review and `image_conversion_manifest.jsonl` for downstream joins.
6. If normalization follows, pass the manifest and conversion output root to `$normalize-medical-image-dataset`.

## Output Contract

Produce only after all recognized supported images convert successfully:

- `images/**/*.png`
- `image_conversion_manifest.jsonl`
- `image_name_mapping.csv`
- `conversion_report.json`

Keep manifest rows in source-relative-path order. Use `status: converted` for outputs and `status: skipped_unsupported` only for unrelated files that are not recognized medical images. Record source and PNG paths, names, format, dimensions, normalization, hashes, and error text.

## Completion Criteria

Finish only when the source is unchanged, every converted file opens as an 8-bit PNG, manifest paths resolve under the output root, output names are collision-free, and a same-input rerun yields identical PNG and manifest bytes.
