---
name: medical-image
description: Inspect medical image metadata and perform explicit reproducible image or volume presentations.
---

Use `medbenchagent.data.sources.inspect_source` for bounded source listings, detected formats and digests.
Use `medbenchagent.data.readers.DicomReader().read_metadata(path)` for DICOM headers without pixel decoding.
Use `medbenchagent.data.asset_adapters.adapter_for(asset).inspect(asset, source_root=...)` for native
raster, DICOM, DICOM series, NIfTI and NRRD metadata. AssetEntry carries the exact source locator and digest.

When a derived image is required, use `materialize_image(asset, presentation, destination, source_root=...)`
with an explicit PresentationSpec. Record source digest, slice axis/index, window, crop, resize and output
digest; do not silently reduce a volume or guess a multiframe policy. Keep original assets available.
Use DICOM identities or explicit source mappings to establish relationships. A filename or shared
directory does not establish a clinical association. See references/coordinates.md for spatial evidence.
