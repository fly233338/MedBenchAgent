"""Fixed source inspection and preparation validation, executed in the data container."""

import json
from pathlib import Path

from medbenchagent.data.validation import source_image_path


def inspect_table(root, path, sheet=None):
    from medbenchagent.data.operators import inspect_table as inspect
    return inspect(root, path, sheet)


def inspect_join(root, left, right, left_key, right_key, left_sheet=None, right_sheet=None):
    from medbenchagent.data.operators import connect, read_relation, join_stats
    with connect(scratch=Path("/work/scratch")) as db:
        read_relation(db, root, left, "a", sheet=left_sheet)
        read_relation(db, root, right, "b", sheet=right_sheet)
        return join_stats(db, "a", "b", [left_key], [right_key])


def main():
    import sys

    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    operation = request.pop("operation")
    root = Path("/source")
    if operation == "table":
        result = inspect_table(root, **request)
    elif operation == "values":
        from medbenchagent.data.operators import inspect_values
        result = inspect_values(root, **request)
    elif operation == "join":
        result = inspect_join(root, **request)
    elif operation == "execute_plan":
        from medbenchagent.data.contracts import Source, PreparationPlan
        from medbenchagent.data.operators import execute_plan
        result = execute_plan(root, Source.model_validate(request["source"]),
                              PreparationPlan.model_validate(request["plan"]), Path("/work"))
    elif operation == "normalize":
        from medbenchagent.data.contracts import Source
        from medbenchagent.data.bundle import normalize_source
        result = normalize_source(root, Path("/annotation/annotation.jsonl"),
                                  Source.model_validate(request.pop("source")), Path("/work"), **request)
    elif operation == "metadata":
        from medbenchagent.data.readers import inspect_metadata
        result = inspect_metadata(source_image_path(root, request["path"]))
    elif operation == "inspect_population":
        from medbenchagent.data.image_inventory import inspect_population
        result = inspect_population(root, Path("/annotation/annotation.jsonl"), Path("/work"), **request)
    else:
        raise ValueError("Unknown fixed preparation operation")
    Path("/work/result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
