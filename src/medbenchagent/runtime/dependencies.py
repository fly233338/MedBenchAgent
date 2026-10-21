"""Frozen execution configuration and resume checks; no business state."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.runtime.construction import record_event

CONTRACT_VERSION = "agent-runtime"



def code_identity():
    root = Path(__file__).resolve().parents[3]

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True, encoding="utf-8"
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "source_diff": git("diff", "HEAD", "--", "src", "pyproject.toml", "uv.lock"),
        "untracked_sources": {
            name: (root / name).read_text(encoding="utf-8")
            for name in git("ls-files", "--others", "--exclude-standard", "src").splitlines()
        },
    }


def verify_resume(context):
    path = Path(context["run_directory"]) / "control" / "contract.json"
    if not path.is_file():
        raise ValueError("This checkpoint uses an obsolete runtime; start a new run")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("contract") != CONTRACT_VERSION:
        raise ValueError("Code or runtime contract changed; start a new run")
    current_code = code_identity()
    saved_code = saved["code_at_start"]
    code_changed = saved_code != current_code
    event_path = context.get("construction_events_path")
    if event_path:
        record_event(
            event_path,
            "resume_compatibility",
            run_id=context.get("run_id"),
            code_changed=code_changed,
            start_commit=saved_code.get("commit") if isinstance(saved_code, dict) else None,
            current_commit=current_code.get("commit"),
            source_diff=bool(current_code.get("source_diff")),
            contract=CONTRACT_VERSION,
        )
    frozen = json.loads(Path(context["effective_config_path"]).read_text(encoding="utf-8"))
    if frozen.get("execution_runtime") != context.get("execution_runtime"):
        raise ValueError("Frozen execution runtime changed; start a new run")
    from medbenchagent.runtime.run_artifacts import verify_catalog_snapshot

    catalog_snapshot = context.get("task_catalog_snapshot_path")
    catalog_digest = context.get("task_catalog_snapshot_digest")
    if not catalog_snapshot or not catalog_digest:
        raise ValueError("Frozen task catalog is unavailable; this run cannot resume")
    verify_catalog_snapshot(catalog_snapshot, catalog_digest)
    from medbenchagent.data.sources import verify_source

    for source in Path(context["run_directory"]).rglob("source_manifests/*/source.json"):
        verify_source(source.parent)
    from medbenchagent.data.dataset_catalog import load_prepared
    from medbenchagent.settings import run_settings

    from medbenchagent.planning.bir import BIRStore
    binding = BIRStore(context["bir_path"]).current("dataset_binding.json")
    entry = binding["entry"] if binding else context.get("prepared_entry")
    if entry:
        with run_settings(context["effective_config_path"]):
            load_prepared(entry)


def initialize_dependencies(context):
    atomic_write_json(
        Path(context["run_directory"]) / "control" / "contract.json",
        {"contract": CONTRACT_VERSION, "code_at_start": code_identity()},
    )

