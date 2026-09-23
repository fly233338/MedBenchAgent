"""Shared array ancestry for observation and target/box resolution."""


def entity_path(path, *, coordinate=False):
    parts = path.split(".")
    arrays = tuple(".".join(parts[:index + 1]) for index, part in enumerate(parts)
                   if part.endswith("[*]") and not (coordinate and index == len(parts) - 1))
    return arrays or (path.rsplit(".", 1)[0],)


def same_entity(left_path, left_indices, right_path, right_indices, *, right_coordinate=False):
    scalar_list = left_path.endswith("[*]")
    return (entity_path(left_path, coordinate=scalar_list) == entity_path(right_path, coordinate=right_coordinate)
            and tuple(left_indices[:-1] if scalar_list else left_indices) == tuple(right_indices))
