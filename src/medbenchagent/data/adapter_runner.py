"""Run and verify a frozen preparation adapter outside the coding loop."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REQUIRED_OUTPUTS = ("annotation.jsonl", "annotation_fields.json", "image_manifest.jsonl",
                    "provenance.jsonl", "transforms.jsonl", "reconciliation.json", "quarantine.jsonl")


def run_adapter(*, script: Path, plan: Path, sources: Path, output: Path, selection: Path | None = None,
                timeout: int = 3600) -> dict:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Adapter output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(script), "--plan", str(plan), "--sources", str(sources), "--output", str(output)]
    if selection is not None:
        command += ["--selection", str(selection)]
    completed = subprocess.run(command, cwd=script.parent.parent, capture_output=True, text=True,
                               encoding="utf-8", timeout=timeout, check=False)
    if completed.returncode:
        raise RuntimeError((completed.stdout + "\n" + completed.stderr)[-8000:])
    missing = [name for name in REQUIRED_OUTPUTS if not (output / name).is_file()]
    if not (output / "images").is_dir():
        missing.append("images/")
    if missing:
        raise ValueError("Adapter output missing: " + ", ".join(missing))
    reconciliation = json.loads((output / "reconciliation.json").read_text(encoding="utf-8"))
    if reconciliation.get("selected", 1) == 0:
        raise ValueError("Adapter produced an empty candidate")
    return {"output": str(output), "reconciliation": reconciliation, "command": command}
