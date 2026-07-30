---
name: prepare-medical-images
description: Copy referenced PNG images without pixel changes, or convert non-PNG images to user-approved PNG encodings, and write an exact image_manifest.jsonl. Use after annotation preparation when a dataset-specific conversion script and benchmark image assets must be created.
---

# Prepare Medical Images

Keep source images and `annotation.jsonl` read-only. Do not reinterpret annotations, discover additional images, or use fuzzy path matching.

## Inputs

Obtain the confirmed `annotation.jsonl`, source-image root, and an image output directory inside the annotation artifact directory.

Read `references/medical_image_conversion.py` in this skill directory before exploring formats or writing conversion code. Treat it as a reference implementation: copy only the handlers needed by the supplied images into the generated script. Resolve skill resources relative to this `SKILL.md`; do not assume a repository layout or installation path.

## Workflow

### 1. Explore the referenced images

Inspect annotation rows in order. Require unique, non-empty `image.image_id` values and unique, non-escaping relative `image.image_path` values. Resolve each path exactly beneath the source root and inspect only those files.

Determine formats by content and identify missing, unsupported, multi-frame, volumetric, whole-slide, multi-page, or pyramidal inputs. Do not infer findings or select a slice. If required Python packages are unavailable, report them and let the user prepare the environment.

### 2. Review the conversion plan

Present a concise summary of:

- Referenced image count, detected formats, and unsupported or unresolved files.
- Exact source and output roots and the deterministic PNG naming rule.
- PNG copy or non-PNG conversion policy, format-specific display transforms where applicable, and the annotation-to-PNG mapping.
- The fixed spatial policy: original dimensions and no resize, crop, rotation, flip, or other spatial transform.

If a referenced source is PNG, do not perform any image operation regardless of its pixel mode. Ask whether to reference it unchanged in place or copy its bytes unchanged into the image output directory. Allow in-place use only when the file is beneath the artifact directory and has a safe artifact-relative `png_path`; otherwise require copying. Do not re-encode it in either case. For every non-PNG source, present the available conversion encodings and require the user to choose and approve one before writing conversion code or producing output.

Do not write the dataset-specific script or convert images until the user explicitly approves the plan. Apply feedback to the plan, show the changed decisions, and request approval again. Stop rather than silently choosing a conversion encoding for any non-PNG source.

### 3. Write and run the converter

Write `<artifact_dir>/scripts/prepare_images.py` with explicit annotation, source-root, image-output, and manifest paths. Copy the required pixel-conversion functions from the reference file and implement the dataset-specific orchestration in this generated script.

Process only annotation-referenced files and preserve annotation order. Ensure one PNG per `image.image_id`; create converted or copied outputs through a staging directory, publish them only after the full preparation succeeds, and refuse to overwrite a different non-empty output.

Write `image_manifest.jsonl` beside `annotation.jsonl` with exactly one row per annotation:

```json
{
  "image_id": "image-001",
  "png_path": "images/image-001.png",
  "width": 1024,
  "height": 1024,
  "sha256": "64 lowercase hexadecimal characters"
}
```

Keep `png_path` relative to the manifest directory. The converted PNG files and `image_manifest.jsonl` are the only dataset outputs; the generated script is retained only for reproducibility.

### 4. Validate

Verify that every manifest row matches its annotation `image.image_id`, resolves to a unique PNG beneath the artifact directory, and has correct dimensions and SHA-256. Open every PNG and require format `PNG`, positive dimensions, and non-constant pixels. For copied PNG sources, verify byte-identical output; for converted non-PNG sources, verify the user-approved output encoding.

Run the generated script again with unchanged inputs. Finish only when it produces byte-identical PNG and manifest bytes, no unreferenced source image was converted, and all source files remain unchanged.
