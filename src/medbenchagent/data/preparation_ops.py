"""Fixed source inspection and preparation validation, executed in the data container."""

import json
import os
from pathlib import Path

from medbenchagent.data.validation import source_image_path


def inspect_table(root, path, sheet=None, delimiter=","):
    from medbenchagent.data.operators import inspect_table as inspect
    return inspect(root, path, sheet, delimiter)


def inspect_join(root, left, right, left_keys, right_keys, cardinality, left_sheet=None, right_sheet=None,
                 left_delimiter=",", right_delimiter=",", how="left", null_keys="error", unmatched="error"):
    from medbenchagent.data.operators import connect, read_relation, join_stats, join_policy_errors
    from medbenchagent.data.contracts import Join
    policy = Join(input="a", output="joined", right="b", left_keys=left_keys, right_keys=right_keys,
                  cardinality=cardinality, how=how, null_keys=null_keys, unmatched=unmatched)
    with connect(scratch=Path(os.environ["MEDBENCH_DATA_SCRATCH"])) as db:
        read_relation(db, root, left, "a", sheet=left_sheet, delimiter=left_delimiter)
        read_relation(db, root, right, "b", sheet=right_sheet, delimiter=right_delimiter)
        stats = join_stats(db, "a", "b", left_keys, right_keys)
        return {**stats, "violations": join_policy_errors(stats, policy), "policy": policy.model_dump()}


def main():
    import sys

    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    operation = request.pop("operation")
    root = Path(os.environ["MEDBENCH_SOURCE_ROOT"])
    if operation == "table":
        result = inspect_table(root, **request)
    elif operation == "values":
        from medbenchagent.data.operators import inspect_values
        result = inspect_values(root, **request)
    elif operation == "join":
        result = inspect_join(root, **request)
    elif operation == "image_population":
        from medbenchagent.data.contracts import Source
        from medbenchagent.data.operators import inspect_image_population
        result = inspect_image_population(root, Source.model_validate(request["source"]),
                                          request["population"], Path(os.environ["MEDBENCH_WORK_DIR"]))
    elif operation == "execute_plan":
        from medbenchagent.data.contracts import Source, PreparationPlan
        from medbenchagent.data.operators import execute_plan
        result = execute_plan(root, Source.model_validate(request["source"]),
                              PreparationPlan.model_validate(request["plan"]), Path(os.environ["MEDBENCH_WORK_DIR"]))
    elif operation == "normalize":
        from medbenchagent.data.contracts import Source
        from medbenchagent.data.bundle import normalize_source
        result = normalize_source(root, Path(os.environ["MEDBENCH_ANNOTATION_PATH"]),
                                  Source.model_validate(request.pop("source")), Path(os.environ["MEDBENCH_WORK_DIR"]), **request)
    elif operation == "metadata":
        from medbenchagent.data.readers import inspect_metadata
        result = inspect_metadata(source_image_path(root, request["path"]))
    elif operation == "inspect_population":
        from medbenchagent.data.image_inventory import inspect_population
        result = inspect_population(root, Path(os.environ["MEDBENCH_ANNOTATION_PATH"]), Path(os.environ["MEDBENCH_WORK_DIR"]), **request)
    else:
        raise ValueError("Unknown fixed preparation operation")
    (Path(os.environ["MEDBENCH_WORK_DIR"]) / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
