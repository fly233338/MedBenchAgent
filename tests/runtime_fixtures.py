"""Explicit disjoint runtime locations for small offline test Runs."""

from pathlib import Path


def runtime_for(directory):
    directory = Path(directory).resolve()
    path = directory.parent / (directory.name + "-runtime")
    (path / "locks").mkdir(parents=True, exist_ok=True)
    return path
