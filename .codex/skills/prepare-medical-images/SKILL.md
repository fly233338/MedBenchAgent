---
name: prepare-medical-images
description: Prepare annotation-referenced 2D image assets, copying PNG/JPEG unchanged and converting other formats to supported PNG encodings without spatial changes.
---

# Prepare Medical Images

Keep source images and annotation.jsonl read-only. Use exact annotation references; do not discover extra images or infer labels. Read [the conversion implementation](references/medical_image_conversion.py) before selecting encodings. It is independently usable; the project runtime uses medbenchagent.data.image_conversion and never imports a skill directory.

Inspect referenced images by content, checking geometry, frames, pixel mode and bit depth. Reject volumes, multiframe, pyramids and whole slides; never select slices or regions. Use inspect_image to distinguish copyable PNG/JPEG, representable pixels and required display decisions.

Record the format counts, original dimensions and per-image encoding overrides in image_plan.json. Copy PNG/JPEG bytes unchanged regardless of their pixel mode. Other supported 2D formats default to PNG with preserved pixel values, bit depth and color. No resize, crop, rotation, flip or automatic percentile normalization. If a display window is necessary, use a unique documented source window or ask the user; never silently select among multiple windows or discard color/bit depth. Bind user decisions to this plan version. Do not request routine approval for unambiguous copying or lossless encodings.

Write a deterministic prepare_images.py using only required handlers from the reference. Use write_image(source_path, output_path, policy); policies select preserve or window with explicit low/high/invert. Process annotation order and only referenced images. Write results into a new version directory and publish only when complete.

The image_manifest.jsonl beside annotation.jsonl must contain exactly image_id, png_path, width, height; png_path is a unique safe relative path (the historical key also covers JPEG). One row per annotation, with unchanged spatial dimensions. Validate readability, encoding, nonconstant pixels, exact image population, copied source byte equality and selected conversion policy. Run twice with unchanged inputs and compare output bytes directly; do not add content fingerprints. Preserve the source inventory, structured plan, scripts and one validation.json; avoid duplicate reports.
