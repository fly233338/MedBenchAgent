"""Single OpenHands worker protocol adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from medbenchagent.data.sources import digest_file

from .contracts import (
    CONTROL_MAPPING_REVISION_REQUIRED,
    Diagnostic,
    Stage,
    StageRequest,
    StageResult,
    Submission,
)


class OpenHandsWorker:
    """Run one stage and normalize its submission into a StageResult."""

    def __init__(self, runtime: Any | None = None):
        self.runtime = runtime

    @staticmethod
    def _submission(request: StageRequest) -> tuple[Submission | None, str]:
        path = Path(request.workspace) / "outputs" / "submission.json"
        if not path.is_file():
            return None, "Worker did not write outputs/submission.json"
        try:
            submission = Submission.model_validate_json(path.read_text(encoding="utf-8"))
            output_root = Path(request.workspace) / "outputs"
            unsafe = [name for name in submission.outputs if Path(name).is_absolute() or ".." in Path(name).parts]
            if unsafe:
                return None, f"Submission references paths outside outputs: {unsafe[:10]}"
            missing = [name for name in submission.outputs if not (output_root / name).is_file()]
            if missing:
                return None, f"Submission references missing outputs: {missing[:10]}"
            if submission.requested_outcome == "submitted":
                required = [name for name in request.required_outputs if name != "submission.json"]
                absent = [name for name in required if not (output_root / name).is_file()]
                if absent:
                    return None, f"Submitted outcome is missing required outputs: {absent}"
            return submission, ""
        except Exception as exc:
            return None, f"Invalid outputs/submission.json: {exc}"

    def run(self, request: StageRequest) -> StageResult:
        from medbenchagent.runtime.dataset_runtime import dataset_runtime

        runtime = self.runtime or dataset_runtime({"execution_runtime": None})
        try:
            result = runtime.run(request)
        except Exception as exc:
            diagnostic = Diagnostic(stage=request.stage, code="worker_execution_failed", category="environment", blocking=True, message=str(exc))
            stage_result = StageResult(
                stage=request.stage, run_id=request.run_id, dataset_id=request.dataset_id,
                attempt=request.attempt, attempt_id=request.attempt_id,
                conversation_id=request.conversation_id, execution_key=request.execution_key,
                status="failed", diagnostics=[diagnostic], diagnostic=str(exc),
                termination_reason="worker_error", input_digest=request.input_digest,
            )
            self._write_result(request, stage_result)
            return stage_result

        submission, submission_error = self._submission(request)
        status = result.status
        diagnostics: list[Diagnostic] = []
        control = result.control
        if result.diagnostic:
            diagnostics.append(Diagnostic(
                stage=request.stage, code="worker",
                category="environment" if status == "failed" else "unknown",
                blocking=status != "submitted", message=result.diagnostic,
            ))
        if submission_error:
            diagnostics.append(Diagnostic(stage=request.stage, code="submission_invalid", category="contract", blocking=True, message=submission_error))
            if status == "submitted":
                status = "failed"
        elif submission and submission.requested_outcome != "submitted" and status == "submitted":
            status = submission.requested_outcome
        if submission:
            for code in submission.diagnostic_codes:
                diagnostics.append(Diagnostic(
                    stage=request.stage, code=code,
                    category="mapping" if code == CONTROL_MAPPING_REVISION_REQUIRED else "unknown",
                    blocking=True, message=f"Worker reported diagnostic code: {code}",
                ))
                if code == CONTROL_MAPPING_REVISION_REQUIRED:
                    if request.stage is not Stage.UNDERSTAND:
                        control = CONTROL_MAPPING_REVISION_REQUIRED
                    else:
                        diagnostics.append(Diagnostic(
                            stage=request.stage, code="invalid_control_stage",
                            category="contract", blocking=True,
                            message=(f"{CONTROL_MAPPING_REVISION_REQUIRED} is only valid for prepare"),
                        ))
                        control = None
                        if status == "submitted":
                            status = "blocked"
        outputs = {name: str(Path(path).resolve()) for name, path in result.outputs.items()}
        stage_result = StageResult(
            stage=request.stage, run_id=request.run_id, dataset_id=request.dataset_id,
            attempt=request.attempt, attempt_id=request.attempt_id,
            conversation_id=request.conversation_id, execution_key=request.execution_key,
            status=status, control=control, outputs=outputs,
            question_refs=submission.question_refs if submission else [],
            diagnostics=diagnostics,
            diagnostic=(submission.summary if submission else result.diagnostic),
            termination_reason=result.termination_reason or ("completed" if status == "submitted" else "worker_error"),
            progress=result.progress,
            submission_path=str((Path(request.workspace) / "outputs" / "submission.json").resolve()) if submission else None,
            input_digest=request.input_digest,
            prompt_digest=digest_file(request.prompt_path) if Path(request.prompt_path).is_file() else "",
            context_digest=digest_file(request.context_path) if Path(request.context_path).is_file() else "",
            token_count=request.prompt_token_count,
        )
        self._write_result(request, stage_result)
        return stage_result

    @staticmethod
    def _write_result(request: StageRequest, result: StageResult) -> None:
        path = Path(request.workspace) / "state" / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
