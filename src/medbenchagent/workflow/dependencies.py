"""Construction artifact ownership and upstream business dependencies."""

DATASET_KINDS = frozenset(
    {
        "Dataset",
        "DatasetProfile",
        "DatasetCard",
        "DatasetCandidates",
        "DatasetSelection",
        "PlanningDraft",
        "Plan",
        "PlanningDelegation",
        "Materials",
        "Templates",
        "Generation",
        "GenerationCheck",
        "SourceSelection",
        "Documentation",
        "Allocation",
        "Delivery",
        "PreparedCandidates",
    }
)


DEPENDENCIES = {
    "Dataset": ("DataConfig",),
    "Plan": ("Requirement", "Dataset", "Materials", "PlanningConfig"),
    "Templates": ("Plan", "TemplateConfig"),
    # Task Plan now owns the reviewed question templates and resolver
    # contract.  Annotation generation therefore has no separate Templates
    # artifact dependency.
    "Generation": ("Dataset", "Plan", "GenerationConfig", "Allocation"),
}


def dependencies_for(kind):
    return DEPENDENCIES.get(kind, ())
