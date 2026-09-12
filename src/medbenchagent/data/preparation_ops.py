"""Fixed source inspection and preparation validation, executed in the data container."""

import json
from pathlib import Path

from medbenchagent.data.validation import load_annotation_jsonl, source_image_path, validate_dataset
from medbenchagent.data.image_assets import load_image_manifest, validate_image_asset
from medbenchagent.data.image_conversion import inspect_image


def read_table(root, path, sheet=None):
    import pandas as pd

    path = source_image_path(root, path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    if path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")


def inspect_table(root, path, sheet=None):
    import pandas as pd

    resolved = source_image_path(root, path)
    sheets = (
        pd.ExcelFile(resolved).sheet_names if resolved.suffix.lower() in {".xlsx", ".xls"} else []
    )
    table = read_table(root, path, sheet)
    return {
        "path": path,
        "sheet": sheet,
        "sheets": sheets,
        "rows": len(table),
        "columns": {
            str(key): {
                "type": str(table[key].dtype),
                "nulls": int(table[key].isna().sum()),
                "unique": int(table[key].nunique()),
            }
            for key in table.columns
        },
    }


def inspect_join(root, left, right, left_key, right_key, left_sheet=None, right_sheet=None):
    a, b = (
        read_table(root, left, left_sheet)[left_key],
        read_table(root, right, right_sheet)[right_key],
    )
    return {
        "left": left,
        "right": right,
        "left_key": left_key,
        "right_key": right_key,
        "left_sheet": left_sheet,
        "right_sheet": right_sheet,
        "left_duplicates": int(a.dropna().duplicated().sum()),
        "right_duplicates": int(b.dropna().duplicated().sum()),
        "left_missing": int((~a.isin(b.dropna())).sum()),
        "right_missing": int((~b.isin(a.dropna())).sum()),
        "matched_left": int(a.isin(b.dropna()).sum()),
        "matched_right": int(b.isin(a.dropna()).sum()),
    }


def inspect_images(root, annotation):
    rows = load_annotation_jsonl(annotation)
    return {
        row["image"]["image_id"]: inspect_image(source_image_path(root, row["image"]["image_path"]))
        for row in rows
    }


def validate_outputs(root, annotation, output):
    rows = load_annotation_jsonl(annotation)
    images = load_image_manifest(output / "image_manifest.jsonl")
    validate_dataset(rows, images)
    expected_files = {entry.png_path for entry in images.values()}
    actual_files = {
        path.resolve()
        for path in output.rglob("*")
        if path.is_file() and path.name not in {"image_manifest.jsonl", "result.json"}
    }
    if actual_files != expected_files:
        raise ValueError("Image stage produced missing or unreferenced output files")
    for row in rows:
        source = source_image_path(root, row["image"]["image_path"])
        entry = images[row["image"]["image_id"]]
        info = inspect_image(source)
        if (entry.width, entry.height) != (info["width"], info["height"]):
            raise ValueError("Image geometry changed")
        errors = validate_image_asset(
            {"png_path": entry.relative_png_path, "width": entry.width, "height": entry.height},
            output,
            supported_formats=frozenset({"PNG", "JPEG"}),
        )
        if errors:
            raise ValueError(f"{entry.image_id}: {errors}")
        if info["copy"] and source.read_bytes() != entry.png_path.read_bytes():
            raise ValueError("PNG/JPEG source bytes must be preserved")
    return {"images": len(images), "geometry": "valid", "population": "exact"}


def main():
    import sys

    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    operation = request.pop("operation")
    root = Path("/source")
    if operation == "table":
        result = inspect_table(root, **request)
    elif operation == "join":
        result = inspect_join(root, **request)
    elif operation == "inspect_images":
        result = inspect_images(root, Path("/annotation/annotation.jsonl"))
    elif operation == "validate_images":
        result = validate_outputs(root, Path("/annotation/annotation.jsonl"), Path("/work"))
    else:
        raise ValueError("Unknown fixed preparation operation")
    Path("/work/result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
