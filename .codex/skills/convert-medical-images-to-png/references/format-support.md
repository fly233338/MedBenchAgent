# Format Support

## Accepted in v1

| Family | Accepted | Display conversion |
| --- | --- | --- |
| Raster | PNG, JPEG, BMP, WebP, single-page non-pyramidal TIFF | Preserve RGB; normalize high-bit-depth grayscale; remove alpha on black |
| DICOM | One frame with 2D grayscale, RGB, YBR, or palette pixels | Apply modality transform, VOI/windowing when present, MONOCHROME1 inversion, and color conversion |
| NIfTI | NIfTI-1/2 whose data squeezes to exactly 2D | Apply stored scaling and deterministic percentile normalization |
| ITK | NRRD/NHDR and MHA/MHD whose data squeezes to exactly 2D | Deterministic percentile normalization |

Compressed DICOM decoding relies on the pydicom pixel pipeline plus pylibjpeg handlers. Report the transfer-syntax failure instead of silently treating compressed bytes as pixels.

## Rejected in v1

- `NumberOfFrames > 1` DICOM.
- Arrays with more than two non-singleton dimensions.
- SVS, NDPI, MRXS, SCN, BIF, and other whole-slide containers.
- Multi-page or SubIFD/pyramidal TIFF.
- Video, PDF, vector graphics, DICOM structured reports, segmentations without displayable pixels, and proprietary formats unsupported by the declared libraries.

Add a future format through one detection/conversion handler that returns the same image and metadata contract. Do not weaken the 2D preflight or output manifest rules when adding a handler.
