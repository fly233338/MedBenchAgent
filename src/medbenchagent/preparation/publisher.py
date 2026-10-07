"""Atomic publication of an already validated prepared candidate."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from medbenchagent.data.index import prepare_dataset
from medbenchagent.data.preparation_validation import validate_candidate
from medbenchagent.data.sources import digest_file
from medbenchagent.runtime.run_artifacts import atomic_write_json

from .contracts import DigestEntry, ValidationResult, digest_payload


class DatasetPublisher:
    """The only owner of immutable prepared-bundle publication."""

    def __init__(self, bundle_root: str | Path):
        self.bundle_root = Path(bundle_root).resolve()
        self.revisions_root = self.bundle_root / "revisions"
        self.staging_root = self.bundle_root / ".staging"
        self.commit_root = self.bundle_root / "commits"

    def publish(
        self,
        candidate: str | Path,
        context: dict[str, Any],
        validation: ValidationResult | dict[str, Any],
    ) -> dict[str, Any]:
        candidate = Path(candidate).resolve()
        if not candidate.is_dir():
            raise ValueError(f"Candidate directory is missing: {candidate}")
        result = validation if isinstance(validation, ValidationResult) else ValidationResult.model_validate(validation)
        if result.status != "accepted" or not result.report:
            raise ValueError("DatasetPublisher requires an accepted prepare validation receipt")
        candidate_ledger = self._build_ledger(candidate, result.digest_ledger)
        candidate_digest = self._digest_from_ledger(candidate_ledger)
        if result.candidate_digest and result.candidate_digest != candidate_digest:
            raise ValueError("Candidate changed after prepare validation")
        report = dict(result.report)
        if report.get("systemic_failure") or not report.get("accepted_cases"):
            raise ValueError("Prepared candidate cannot be published")
        dataset_id = str(context.get("dataset_id") or context.get("dataset_scope") or self.bundle_root.name)
        revision = self._revision(candidate, dataset_id, report, candidate_ledger)
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
            copied_digests = self._copy_candidate(candidate, staging, candidate_ledger)
            self._filter_invalid_cases(staging, report)
            for name in ("cases.jsonl", "asset_manifest.jsonl", "provenance.jsonl",
                         "presentation_manifest.jsonl", "sampling_manifest.jsonl", "split_manifest.jsonl"):
                copied_digests.pop(name, None)
            self._write_final_manifest(staging, copied_digests)
            final_report = validate_candidate(
                staging, source_root=context.get("source_root"), source_roots=context.get("source_roots"),
                population_path=(context.get("accepted_handoff") or {}).get("population", {}).get("path"),
            )
            if final_report["status"] != "complete" or not final_report["accepted_cases"]:
                raise ValueError(f"Final dataset consistency validation failed: {final_report['failures']}")
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
            annotation_digest = copied_digests.get("cases.jsonl")
            if annotation_digest is None:
                annotation_digest = digest_file(staging / "cases.jsonl")
            asset_manifest_digest = copied_digests.get("asset_manifest.jsonl")
            if asset_manifest_digest is None:
                asset_manifest_digest = digest_file(staging / "asset_manifest.jsonl")
            card = {
                "schema_version": "dataset-card", "dataset_id": dataset_id, "name": dataset_id,
                "description": "annotation dataset",
                "annotation_digest": annotation_digest,
                "asset_manifest_digest": asset_manifest_digest,
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
            self._write_final_manifest(staging, copied_digests)
            manifest_digest = digest_file(staging / "output_manifest.json")
            atomic_write_json(staging / "complete.json", {
                "status": "complete", "revision": revision, "manifest_digest": manifest_digest,
            })
            os.replace(staging, revision_target)
            atomic_write_json(self.commit_root / f"{transaction_id}.json", {
                "transaction_id": transaction_id, "dataset_id": dataset_id, "revision": revision,
                "status": "committed", "revision_path": str(revision_target),
                "manifest_digest": manifest_digest,
            })
            self._advance_current(revision_target, dataset_id, revision, transaction_id, manifest_digest=manifest_digest)
            return self._result(revision_target, dataset_id, revision, transaction_id, report)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise

    @staticmethod
    def _build_ledger(root: Path, cached=None) -> dict[str, DigestEntry]:
        root = Path(root)
        cached = cached or {}
        ledger: dict[str, DigestEntry] = {}
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            stat = path.stat()
            previous = cached.get(relative)
            if isinstance(previous, dict):
                previous = DigestEntry.model_validate(previous, strict=False)
            if (isinstance(previous, DigestEntry)
                    and previous.size_bytes == stat.st_size
                    and previous.mtime_ns == stat.st_mtime_ns):
                ledger[relative] = previous
            else:
                ledger[relative] = DigestEntry(
                    digest=digest_file(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                )
        return ledger

    @staticmethod
    def _digest_from_ledger(ledger: dict[str, DigestEntry]) -> str:
        return digest_payload({name: item.digest for name, item in ledger.items()})

    @classmethod
    def _candidate_digest(cls, root: Path, ledger=None) -> str:
        return cls._digest_from_ledger(cls._build_ledger(Path(root), ledger))

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
    def _copy_candidate(candidate: Path, target: Path, ledger=None) -> dict[str, str]:
        copied: dict[str, str] = {}
        for source in sorted(candidate.rglob("*")):
            if not source.is_file() or source.name in {"complete.json", "preparation_report.json", "quarantine.jsonl", "optional_evidence_dropped.jsonl"}:
                continue
            relative = source.relative_to(candidate).as_posix()
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            entry = (ledger or {}).get(relative)
            copied[relative] = entry.digest if isinstance(entry, DigestEntry) else digest_file(source)
        return copied

    @staticmethod
    def _filter_invalid_cases(root: Path, report: dict[str, Any]) -> None:
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
                if line.strip() and str(json.loads(line).get("asset_id")) in asset_ids:
                    kept.append(line)
            assets_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8", newline="\n")
        provenance_path = root / "provenance.jsonl"
        if provenance_path.is_file():
            kept = []
            for line in provenance_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    payload = json.loads(line)
                    if payload["status"] == "retained":
                        payload["case_ids"] = [key for key in payload["case_ids"] if key in case_ids]
                        if not payload["case_ids"]:
                            payload.update(status="quarantined", reason="All referenced Cases were quarantined")
                    kept.append(json.dumps(payload, ensure_ascii=False))
            provenance_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8", newline="\n")

        from medbenchagent.data.manifests import PresentationRecord, SamplingManifest, SplitManifest, load_manifest
        for name, model, field in (
            ("presentation", PresentationRecord, None),
            ("sampling", SamplingManifest, "units"),
            ("split", SplitManifest, "records"),
        ):
            path = root / f"{name}_manifest.jsonl"
            if not path.is_file():
                continue
            parsed = load_manifest(path, model)
            if field:
                rows = [row for row in getattr(parsed, field) if row.case_id in case_ids]
                parsed = parsed.model_copy(update={field: rows})
                path.write_text(parsed.model_dump_json() + "\n", encoding="utf-8")
            else:
                rows = parsed if isinstance(parsed, list) else [parsed]
                rows = [row for row in rows if row.case_id in case_ids and row.asset_id in asset_ids]
                if rows:
                    path.write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")
                else:
                    path.unlink()

    @classmethod
    def _revision(cls, candidate: Path, dataset_id: str, report: dict[str, Any], ledger=None) -> str:
        ledger = ledger or cls._build_ledger(candidate)
        return digest_payload({
            "dataset_id": dataset_id, "report": report,
            "files": {
                name: entry.digest for name, entry in ledger.items()
                if Path(name).name not in {"complete.json", "preparation_report.json"}
            },
        })[:32]

    @staticmethod
    def _write_final_manifest(root: Path, digest_hints: dict[str, str] | None = None) -> None:
        digest_hints = digest_hints or {}
        files = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name in {"output_manifest.json", "submission.json", "complete.json"} or "cache" in path.relative_to(root).parts:
                continue
            relative = path.relative_to(root).as_posix()
            digest = digest_hints.get(relative)
            if digest is None:
                digest = digest_file(path)
            files.append({"path": relative, "kind": "published",
                          "digest": digest, "producer": "host"})
        mapping = root / "mapping.json"
        population = root / "image_population.jsonl"
        mapping_digest = digest_hints.get("mapping.json")
        if mapping_digest is None and mapping.is_file():
            mapping_digest = digest_file(mapping)
        population_digest = digest_hints.get("image_population.jsonl")
        if population_digest is None and population.is_file():
            population_digest = digest_file(population)
        atomic_write_json(root / "output_manifest.json", {
            "files": files, "dependencies": [],
            "mapping_digest": mapping_digest or "",
            "population_digest": population_digest or "",
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
