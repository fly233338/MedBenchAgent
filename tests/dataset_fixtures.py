import base64
import hashlib
from pathlib import Path
from medbenchagent.data.dataset import DatasetManifest, seal_manifest
from medbenchagent.data.models import AssetEntry, CaseRecord, CaseRelation, EntityRef, Evidence, EvidenceRef, SourceRef
from medbenchagent.compilation.artifacts import write_artifact, write_json_artifact
from medbenchagent.compilation.contracts import ArtifactRef, WorkRequest

def create_synthetic_inputs(
    directory: Path, compilation_id: str, *, initial_evidence: bool = False
) -> list[ArtifactRef]:
    request = WorkRequest(
        compilation_id=compilation_id, request_id=f"{compilation_id}-input",
        agent_id="fixture", operation="bind", objective="Create a tiny synthetic Case dataset",
    )
    # Two distinct Asset identities can have identical content.
    pixels = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aRZk"
        "AAAAASUVORK5CYII="
    )
    write_json_artifact(
        directory, request, "dataset/source.json", kind="source",
        value={"case-a": 2, "case-b": 5},
    )
    cases, assets = [], []
    for suffix, value in (("a", 2), ("b", 5)):
        case_id = f"case-{suffix}"
        source = SourceRef(source_id="raw", path="source.json", record=case_id)
        write_artifact(
            directory, request, f"dataset/image-{suffix}.png", kind="image", content=pixels,
        )
        assets.append(AssetEntry(
            asset_id=f"asset-{suffix}", kind="image", format="png", source_id="raw",
            source_path=f"image-{suffix}.png", file_digest=hashlib.sha256(pixels).hexdigest(),
            width=1, height=1,
        ))
        cases.append(CaseRecord(
            case_id=case_id, patient_id="synthetic-patient",
            evidence=[
                Evidence(
                    id="image", name="view", type="asset", value=f"asset-{suffix}",
                    view_id=f"view-{suffix}", study_id="synthetic-study",
                    input_visibility="model_visible",
                ),
                Evidence(
                    id="value", name="value", type="value", value=value,
                    entity=EntityRef(kind="sample", id=case_id),
                    input_visibility="model_visible", source_refs=[source],
                ),
            ],
            relations=[CaseRelation(
                subject=EvidenceRef(case_id=case_id, evidence_id="value"), predicate="describes",
                object=EvidenceRef(case_id=case_id, evidence_id="image"), source_refs=[source],
            )],
        ))
    for name, rows in (("cases.jsonl", cases), ("assets.jsonl", assets)):
        write_artifact(
            directory, request, f"dataset/{name}", kind="records",
            content=("\n".join(row.model_dump_json() for row in rows) + "\n").encode("utf-8"),
        )
    write_json_artifact(
        directory, request, "dataset/mapping.json", kind="mapping",
        value={"case_unit": "synthetic sample", "fields": {"value": "synthetic input number"}},
    )
    manifest = DatasetManifest(
        dataset_id="synthetic", revision="fixture", cases_path="cases.jsonl",
        assets_path="assets.jsonl", sources=[{"source_id": "raw", "root": "."}],
        mapping_path="mapping.json",
    )
    manifest = seal_manifest(directory / "artifacts" / request.request_id / "dataset", manifest)
    dataset = write_json_artifact(
        directory, request, "dataset/manifest.json", kind="dataset", value=manifest.model_dump(),
    )
    inputs = [dataset]
    return inputs
