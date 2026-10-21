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
    StageControl,
)


class OpenHandsWorker:
    """Return candidate outputs and optional exceptional control for Host validation."""

    def __init__(self, runtime: Any | None = None):
        self.runtime = runtime

    @staticmethod
    def _control(request: StageRequest) -> tuple[StageControl | None, str]:
        path = Path(request.workspace) / "work" / "control.json"
        if not path.is_file():
            return None, ""
        try:
            signal = StageControl.model_validate_json(path.read_text(encoding="utf-8"))
            if any(question.stage != request.stage for question in signal.questions):
                return None, "Control questions must belong to the current stage"
            if signal.requested_outcome == "needs_input" and not signal.questions:
                return None, "needs_input control requires scoped questions"
            return signal, ""
        except Exception as exc:
            return None, f"Invalid work/control.json: {exc}"

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

        signal, signal_error = self._control(request)
        status = result.status
        diagnostics: list[Diagnostic] = []
        control = result.control
        if result.diagnostic:
            diagnostics.append(Diagnostic(
                stage=request.stage, code="worker",
                category=("contract" if result.termination_reason == "output_invalid" else
                          "environment" if status == "failed" else "unknown"),
                blocking=status != "submitted", message=result.diagnostic,
            ))
        if signal_error:
            diagnostics.append(Diagnostic(stage=request.stage, code="control_invalid", category="contract", blocking=True, message=signal_error))
            if status not in {"failed", "cancelled"}:
                status = "failed"
        elif signal and status not in {"failed", "cancelled"}:
            status = signal.requested_outcome
        if status == "submitted":
            missing = [name for name in request.required_outputs
                       if name not in result.outputs or not (Path(request.workspace) / "work/outputs" / name).is_file()]
            if missing:
                status = "failed"
                diagnostics.append(Diagnostic(stage=request.stage, code="output_missing", category="contract",
                                              blocking=True, message=f"Missing required outputs: {missing}"))
        if signal:
            for code in signal.diagnostic_codes:
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
                            message=(f"{CONTROL_MAPPING_REVISION_REQUIRED} is only valid for Text or Image"),
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
            question_refs=signal.question_refs if signal else [],
            questions=signal.questions if signal else [],
            diagnostics=diagnostics,
            diagnostic=(signal.summary if signal else signal_error or result.diagnostic),
            termination_reason=result.termination_reason or ("completed" if status == "submitted" else "worker_error"),
            progress=result.progress,
            control_path=str((Path(request.workspace) / "work" / "control.json").resolve()) if signal else None,
            input_digest=request.input_digest,
            prompt_digest=digest_file(request.prompt_path) if Path(request.prompt_path).is_file() else "",
            context_digest=digest_file(request.context_path) if Path(request.context_path).is_file() else "",
            token_count=request.prompt_token_count,
        )
        self._write_result(request, stage_result)
        return stage_result

    @staticmethod
    def _write_result(request: StageRequest, result: StageResult) -> None:
        path = Path(request.workspace) / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
