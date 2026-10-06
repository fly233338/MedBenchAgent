"""Atomic publication of validated prepared datasets."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from medbenchagent.data.index import prepare_dataset
from medbenchagent.data.preparation_validation import validate_candidate, write_validation_report
from medbenchagent.data.sources import digest_file
from medbenchagent.runtime.run_artifacts import atomic_write_json

from .contracts import digest_payload
from .validation import validate_prepare_candidate


class DatasetPublisher:
    """Build an immutable revision and atomically advance the current pointer."""

    def __init__(self, bundle_root: str | Path):
        self.bundle_root = Path(bundle_root).resolve()
        self.revisions_root = self.bundle_root / "revisions"
        self.staging_root = self.bundle_root / ".staging"
        self.commit_root = self.bundle_root / "commits"

    def publish(self, candidate: str | Path, context: dict[str, Any]) -> dict[str, Any]:
        candidate = Path(candidate).resolve()
        if not candidate.is_dir():
            raise ValueError(f"Candidate directory is missing: {candidate}")
        if context.get("canonical_validation"):
            canonical = validate_prepare_candidate(candidate, state=context, population_path=context.get("preparation_population_path"))
            if canonical.status != "accepted":
                raise ValueError("Canonical candidate validation failed: " + "; ".join(item.message for item in canonical.issues))
        report = validate_candidate(
            candidate,
            source_root=context.get("source_root"),
            source_roots=context.get("source_roots"),
            population_path=context.get("preparation_population_path"),
            expected_mapping_digest=context.get("preparation_mapping_digest"),
        )
        write_validation_report(candidate, report)
        if report.get("systemic_failure") or not report.get("accepted_cases"):
            raise ValueError("Prepared candidate cannot be published")
        dataset_id = str(context.get("dataset_id") or context.get("dataset_scope") or self.bundle_root.name)
        revision = self._revision(candidate, dataset_id, report)
        transaction_id = f"publish-{revision}"
        self.revisions_root.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self.commit_root.mkdir(parents=True, exist_ok=True)
        staging = self.staging_root / transaction_id
        revision_target = self.revisions_root / revision
        if revision_target.exists():
            self._advance_current(revision_target, dataset_id, revision, transaction_id)
            return self._result(revision_target, dataset_id, revision, transaction_id, report)
        staging.mkdir(parents=True, exist_ok=True)
        try:
            self._copy_candidate(candidate, staging)
            self._filter_invalid_cases(staging, report)
            atomic_write_json(staging / "preparation_report.json", report)
            (staging / "quarantine.jsonl").write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in report.get("failures", []) if item.get("kind") == "hard_case_failure"),
                encoding="utf-8", newline="\n",
            )
            (staging / "optional_evidence_dropped.jsonl").write_text("", encoding="utf-8", newline="\n")
            profile, access = prepare_dataset(
                staging / "cases.jsonl", staging / "asset_manifest.jsonl", staging / "cache",
                revision=revision, source_root=context.get("source_root"), source_roots=context.get("source_roots"),
            )
            atomic_write_json(staging / "dataset_profile.json", profile)
            for key, value in list(access.items()):
                if key.endswith("_path") and isinstance(value, str):
                    try:
                        relative = Path(value).resolve().relative_to(staging.resolve())
                    except ValueError:
                        continue
                    access[key] = str(revision_target / relative)
            card = {
                "schema_version": "dataset-card", "dataset_id": dataset_id, "name": dataset_id,
                "description": "annotation dataset", "annotation_digest": digest_file(staging / "cases.jsonl"),
                "asset_manifest_digest": digest_file(staging / "asset_manifest.jsonl"),
                "profile_digest": digest_file(staging / "dataset_profile.json"),
                "case_count": profile["case_count"], "patient_count": profile["patient_count"],
                "evidence_profile": profile["evidence"], "asset_profile": profile["assets"],
            }
            atomic_write_json(staging / "dataset_card.json", card)
            access = {
                **access,
                "cases_path": str(revision_target / "cases.jsonl"),
                "asset_manifest_path": str(revision_target / "asset_manifest.jsonl"),
                "profile_path": str(revision_target / "dataset_profile.json"),
                "dataset_card_path": str(revision_target / "dataset_card.json"),
            }
            atomic_write_json(staging / "ready.json", {
                "schema_version": "dataset-bundle", "annotation_schema": "annotation",
                "revision": revision, "access": access,
            })
            self._write_final_manifest(staging)
            manifest_digest = digest_file(staging / "output_manifest.json")
            atomic_write_json(staging / "complete.json", {
                "status": "complete", "revision": revision, "manifest_digest": manifest_digest,
            })
            os.replace(staging, revision_target)
            commit = {
                "transaction_id": transaction_id, "dataset_id": dataset_id, "revision": revision,
                "status": "committed", "revision_path": str(revision_target),
                "manifest_digest": manifest_digest,
            }
            atomic_write_json(self.commit_root / f"{transaction_id}.json", commit)
            self._advance_current(revision_target, dataset_id, revision, transaction_id, manifest_digest=manifest_digest)
            return self._result(revision_target, dataset_id, revision, transaction_id, report)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise

    def recover(self) -> None:
        if not self.commit_root.is_dir():
            return
        for commit in self.commit_root.glob("*.json"):
            try:
                payload = json.loads(commit.read_text(encoding="utf-8"))
                revision = Path(payload["revision_path"])
                if payload.get("status") == "committed" and revision.is_dir() and not (self.bundle_root / "current.json").is_file():
                    atomic_write_json(self.bundle_root / "current.json", {
                        "dataset_id": payload["dataset_id"], "revision": payload["revision"],
                        "revision_path": str(revision), "transaction_id": payload["transaction_id"],
                        "manifest_digest": payload["manifest_digest"],
                    })
            except (OSError, ValueError, KeyError):
                continue

    @staticmethod
    def _copy_candidate(candidate: Path, target: Path) -> None:
        for source in sorted(candidate.rglob("*")):
            if not source.is_file() or source.name in {"complete.json", "preparation_report.json", "quarantine.jsonl", "optional_evidence_dropped.jsonl"}:
                continue
            destination = target / source.relative_to(candidate)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    @staticmethod
    def _filter_invalid_cases(root: Path, report: dict[str, Any]) -> None:
        """Materialize only valid Cases and the asset/provenance closure."""
        invalid_lines = {
            int(item["line"])
            for item in report.get("failures", [])
            if item.get("kind") == "hard_case_failure" and str(item.get("line", "")).isdigit()
        }
        cases_path = root / "cases.jsonl"
        if not invalid_lines or not cases_path.is_file():
            return
        valid_lines = []
        case_ids: set[str] = set()
        asset_ids: set[str] = set()
        for number, line in enumerate(cases_path.read_text(encoding="utf-8").splitlines(), 1):
            if number in invalid_lines or not line.strip():
                continue
            payload = json.loads(line)
            valid_lines.append(line)
            if payload.get("case_id"):
                case_ids.add(str(payload["case_id"]))
            for evidence in payload.get("evidence", []):
                if evidence.get("type") == "asset" and evidence.get("value"):
                    asset_ids.add(str(evidence["value"]))
        cases_path.write_text("\n".join(valid_lines) + ("\n" if valid_lines else ""), encoding="utf-8", newline="\n")
        assets_path = root / "asset_manifest.jsonl"
        if assets_path.is_file():
            kept = []
            for line in assets_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                if str(payload.get("asset_id")) in asset_ids:
                    kept.append(line)
            assets_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8", newline="\n")
        provenance_path = root / "provenance.jsonl"
        if provenance_path.is_file():
            kept = []
            for line in provenance_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not payload.get("case_id") or str(payload.get("case_id")) in case_ids:
                    kept.append(line)
            provenance_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8", newline="\n")

    @staticmethod
    def _revision(candidate: Path, dataset_id: str, report: dict[str, Any]) -> str:
        values = {
            "dataset_id": dataset_id, "report": report,
            "files": {
                path.relative_to(candidate).as_posix(): digest_file(path)
                for path in sorted(candidate.rglob("*"))
                if path.is_file() and path.name not in {"complete.json", "preparation_report.json"}
            },
        }
        return digest_payload(values)[:32]

    @staticmethod
    def _manifest_digest(root: Path) -> str:
        values = {
            path.relative_to(root).as_posix(): digest_file(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name not in {"output_manifest.json", "submission.json", "complete.json"}
        }
        return digest_payload(values)

    @staticmethod
    def _write_final_manifest(root: Path) -> None:
        files = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name in {"output_manifest.json", "submission.json", "complete.json"} or "cache" in path.relative_to(root).parts:
                continue
            files.append({"path": path.relative_to(root).as_posix(), "kind": "published", "digest": digest_file(path), "producer": "host"})
        mapping = root / "mapping.json"
        population = root / "image_population.jsonl"
        atomic_write_json(root / "output_manifest.json", {
            "files": files, "dependencies": [],
            "mapping_digest": digest_file(mapping) if mapping.is_file() else "",
            "population_digest": digest_file(population) if population.is_file() else "",
        })

    def _materialize_current_projection(self, revision: Path) -> None:
        for source in sorted(revision.rglob("*")):
            if not source.is_file() or "cache" in source.relative_to(revision).parts:
                continue
            target = self.bundle_root / source.relative_to(revision)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.resolve() != source.resolve():
                shutil.copy2(source, target)
        ready = json.loads((revision / "ready.json").read_text(encoding="utf-8"))
        projected = dict(ready)
        projected["access"] = dict(ready["access"])
        for key, value in list(projected["access"].items()):
            if key.endswith("_path") and isinstance(value, str):
                try:
                    relative = Path(value).resolve().relative_to(revision.resolve())
                except ValueError:
                    continue
                projected["access"][key] = str(self.bundle_root / relative)
        atomic_write_json(self.bundle_root / "ready.json", projected)

    def _advance_current(self, revision: Path, dataset_id: str, revision_id: str, transaction_id: str, *, manifest_digest: str | None = None) -> None:
        digest = manifest_digest or digest_file(revision / "output_manifest.json")
        atomic_write_json(self.bundle_root / "current.json", {
            "dataset_id": dataset_id, "revision": revision_id, "revision_path": str(revision),
            "transaction_id": transaction_id, "manifest_digest": digest,
        })
        self._materialize_current_projection(revision)

    @staticmethod
    def _result(revision: Path, dataset_id: str, revision_id: str, transaction_id: str, report: dict[str, Any]) -> dict[str, Any]:
        ready = json.loads((revision / "ready.json").read_text(encoding="utf-8"))
        return {
            "status": "complete", "bundle": str(revision.parent.parent), "revision": revision_id,
            "revision_path": str(revision), "transaction_id": transaction_id,
            "cases_path": str(revision / "cases.jsonl"), "access_manifest": ready["access"],
            "report": report, "card_path": str(revision / "dataset_card.json"),
            "dataset_entry": {
                "dataset_id": dataset_id, "name": dataset_id, "description": "annotation dataset",
                "available": True, "cases_path": str(revision / "cases.jsonl"),
                "asset_manifest_path": str(revision / "asset_manifest.jsonl"),
                "profile_path": str(revision / "dataset_profile.json"),
                "dataset_card_path": str(revision / "dataset_card.json"),
            },
        }


def publish_prepared_dataset(candidate: str | Path, context: dict[str, Any]) -> dict[str, Any]:
    return DatasetPublisher(context["bundle_root"]).publish(candidate, context)
