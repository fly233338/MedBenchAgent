"""Atomic commit of an already validated immutable dataset; no repair or business validation."""
import os
import shutil
from pathlib import Path

from medbenchagent.runtime.run_artifacts import atomic_write_json
from .validation import bundle_digest


class DatasetPublisher:
    def __init__(self, bundle_root):
        self.root = Path(bundle_root).resolve()

    def publish(self, staging, context, validation):
        staging = Path(staging).resolve()
        digest = validation.candidate_digest
        if validation.status != "accepted" or not digest or bundle_digest(staging) != digest:
            raise ValueError("Publication requires an unchanged validated staging dataset")
        revision = digest[:32]
        target = self.root / "revisions" / revision
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            # Copy on the destination filesystem; only rename makes the dataset visible.
            temporary = self.root / ".staging" / revision
            if temporary.exists():
                if not temporary.resolve().is_relative_to(self.root / ".staging"):
                    raise ValueError("Unsafe staging path")
                shutil.rmtree(temporary)
            shutil.copytree(staging, temporary)
            if bundle_digest(temporary) != digest:
                raise ValueError("Staging changed while copying")
            os.replace(temporary, target)
        elif bundle_digest(target) != digest:
            raise ValueError("Committed revision is corrupt")
        receipt = {"dataset_id": context["dataset_id"], "revision": revision,
                   "revision_path": str(target), "digest": digest}
        atomic_write_json(self.root / ".commits" / f"{revision}.json", receipt)
        atomic_write_json(self.root / "current.json", receipt)
        return {"status": "complete", "bundle": str(self.root), **receipt}
