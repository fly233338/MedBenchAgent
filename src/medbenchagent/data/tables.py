"""Tabular source readers used by preparation accounting."""
from pathlib import Path


def read_excel(path: Path, *, sheet: str | int | None = None):
    import pandas as pd
    return pd.read_excel(path, sheet_name=0 if sheet is None else sheet)
