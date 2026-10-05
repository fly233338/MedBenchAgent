"""Host owned data preparation contracts and stage graph.

The preparation package deliberately keeps the model facing worker small: the
host selects immutable inputs, builds a stage context, invokes one OpenHands
worker and validates the returned files.  The files in this package are also
used by the runtime adapters so there is one canonical stage vocabulary.
"""

from .contracts import (
    ATTEMPT_MODES,
    STAGE_SPECS,
    AttemptContext,
    ArtifactDependency,
    ArtifactRef,
    ContextPacket,
    Diagnostic,
    OutputFile,
    OutputManifest,
    PreparationMapping,
    SourceReport,
    ExecutionReport,
    SemanticQuestion,
    Stage,
    StageBudget,
    StageRequest,
    StageResult,
    Submission,
    UserDecision,
)

__all__ = [
    "ATTEMPT_MODES",
    "STAGE_SPECS",
    "AttemptContext",
    "ArtifactDependency",
    "ArtifactRef",
    "ContextPacket",
    "Diagnostic",
    "OutputFile",
    "OutputManifest",
    "PreparationMapping",
    "SourceReport",
    "ExecutionReport",
    "SemanticQuestion",
    "Stage",
    "StageBudget",
    "StageRequest",
    "StageResult",
    "Submission",
    "UserDecision",
]
