"""Small deterministic inspection worker used by the current preparation runtime."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _path(root: Path, value: str) -> Path:
    candidate = Path(value)
    resolved = (root / candidate).resolve()
    if candidate.is_absolute() or not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f"Source file is unavailable or escapes source root: {value}")
    return resolved


def _read_table(path: Path, *, sheet=None, delimiter=","):
    import pandas as pd
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet or 0)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    return pd.read_csv(path, delimiter=delimiter)


def inspect_table(root: Path, request: dict) -> dict:
    frame = _read_table(_path(root, request["path"]), sheet=request.get("sheet"), delimiter=request.get("delimiter", ","))
    return {"operation": "table", "path": request["path"], "rows": int(len(frame)),
            "columns": [str(column) for column in frame.columns],
            "nulls": {str(column): int(frame[column].isna().sum()) for column in frame.columns},
            "dtypes": {str(column): str(frame[column].dtype) for column in frame.columns}}


def inspect_join(root: Path, request: dict) -> dict:
    left = _read_table(_path(root, request["left"]), delimiter=request.get("delimiter", ","))
    right = _read_table(_path(root, request["right"]), delimiter=request.get("delimiter", ","))
    left_keys = request.get("left_keys") or request.get("on")
    right_keys = request.get("right_keys") or request.get("on")
    if not left_keys or not right_keys or len(left_keys) != len(right_keys):
        raise ValueError("Join inspection requires equally sized left_keys and right_keys")
    merged = left.merge(right, left_on=left_keys, right_on=right_keys, how="left", indicator=True)
    return {"operation": "join", "left_rows": int(len(left)), "right_rows": int(len(right)),
            "matched_rows": int((merged["_merge"] == "both").sum()),
            "unmatched_left_rows": int((merged["_merge"] == "left_only").sum()),
            "left_duplicate_keys": int(left.duplicated(left_keys).sum()),
            "right_duplicate_keys": int(right.duplicated(right_keys).sum())}


def main(argv=None) -> int:
    request = json.loads(Path((argv or sys.argv)[1]).read_text(encoding="utf-8"))
    root = Path(os.environ.get("MEDBENCH_SOURCE_ROOT", ".")).resolve()
    operation = request.get("operation")
    if operation == "table":
        result = inspect_table(root, request)
    elif operation == "join":
        result = inspect_join(root, request)
    else:
        raise ValueError(f"Unsupported preparation inspection operation: {operation}")
    output = Path(os.environ["MEDBENCH_WORK_DIR"]) / "result.json"
    output.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
