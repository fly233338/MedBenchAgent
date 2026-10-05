import json
from pathlib import Path

from medbenchagent.data.contracts import Source
from medbenchagent.data.sources import inventory_source
from medbenchagent.preparation.contracts import StageResult
from medbenchagent.preparation.graph import run_preparation_graph


class _StageWorker:
    def __init__(self, *, blocking=False):
        self.blocking = blocking

    def run(self, request):
        output_root = Path(request.workspace) / "outputs"
        output_root.mkdir(parents=True, exist_ok=True)
        if request.stage.value == "understand":
            questions = []
            if self.blocking:
                questions = [{
                    "issue_id": "question-1", "stage": "understand", "source_ids": ["source"],
                    "fields": ["value"], "scope": {"table": "records"},
                    "question": "What does this source value mean?", "evidence_refs": ["source:records"],
                    "impact": "The conversion would otherwise invent a mapping.", "blocking": True,
                }]
            values = {
                "source_report.json": {"sources": [{"id": "source"}], "tables": [], "documents": [],
                                       "checks": [], "relationships": [], "population": {"count": 1}, "limitations": []},
                "image_population.jsonl": '{"unit_id":"unit-1"}\n',
                "mapping.json": {"population_policy": {}, "case_identity": {}, "grouping": {}, "joins": [],
                                 "fields": [], "asset_rules": [], "field_semantics": {},
                                 "resolved_decision_refs": [], "limitations": []},
                "semantic_questions.json": {"questions": questions},
                "inspect_source.py": "print('ok')\n",
                "submission.json": {"requested_outcome": "submitted"},
            }
        else:
            mapping = request.input_artifacts["mapping.json"].digest
            population = request.input_artifacts["image_population.jsonl"].digest
            values = {
                "scripts/prepare.py": "print('prepare')\n",
                "cases.jsonl": "{}\n",
                "asset_manifest.jsonl": "{}\n",
                "provenance.jsonl": '{"source_id":"source","record_ref":"unit-1","case_id":"case-1"}\n',
                "conversion_config.json": {}, "grouping_policy.json": {},
                "execution_report.json": {"tests": [], "full_execution": {}, "population_accounting": {},
                                          "mapping_rule_coverage": {}, "limitations": []},
                "output_manifest.json": {
                    "mapping_digest": mapping, "population_digest": population,
                    "files": [{"path": name} for name in (
                        "scripts/prepare.py", "cases.jsonl", "asset_manifest.jsonl", "provenance.jsonl",
                        "conversion_config.json", "grouping_policy.json", "execution_report.json",
                    )],
                    "dependencies": [],
                },
                "submission.json": {"requested_outcome": "submitted"},
            }
        outputs = {}
        for name, value in values.items():
            path = output_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")
            outputs[name] = str(path)
        return StageResult(stage=request.stage, attempt=request.attempt, conversation_id=request.conversation_id,
                           execution_key=request.execution_key, status="submitted", outputs=outputs,
                           input_digest=request.input_digest, token_count=request.prompt_token_count)


def _state(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "records.txt").write_text("record", encoding="utf-8")
    manifest = tmp_path / "manifest"
    inventory_source(Source(id="source", root=str(source_root)), manifest)
    return {
        "run_id": "run", "dataset_id": "dataset", "run_directory": str(tmp_path / "run"),
        "sources": [Source(id="source", root=str(source_root)).model_dump()],
        "source_manifests": {"source": str(manifest)},
    }


def test_two_stage_graph_preserves_accepted_handoff_and_attempt_layout(tmp_path):
    result = run_preparation_graph(_state(tmp_path), worker=_StageWorker())

    assert result["status"] == "complete"
    candidate = Path(result["preparation_candidate"]["candidate_path"])
    assert (candidate / "mapping.json").is_file()
    assert (candidate / "image_population.jsonl").is_file()
    assert (candidate.parent.parent.parent / "understand" / "1" / "inputs" / "prompt.md").is_file()
    assert (candidate.parent.parent.parent / "prepare" / "1" / "inputs" / "context.json").is_file()
    assert not any(path.name == "memory" + ".json" for path in candidate.parents[1].rglob("*"))


def test_blocking_semantic_question_stops_before_prepare(tmp_path):
    result = run_preparation_graph(_state(tmp_path), worker=_StageWorker(blocking=True))

    assert result["status"] == "needs_input"
    assert result["stage"] == "understand"
    assert result["pending_questions"][0]["issue_id"] == "question-1"
    assert not (tmp_path / "run" / "preparation" / "dataset" / "attempts" / "prepare").exists()
