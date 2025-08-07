"""Utils for handling nested dict."""

from typing import Any, Callable, TypeVar


Tree = TypeVar("Tree", bound=dict)


def tree_filter(
    data: Tree,
    criteria_fn: Callable[[Any], bool] = lambda x: x is not None,
) -> Tree:
    """Keep only leaves for which criteria is True.

    Filters out None leaves if criteria is not specified.
    """
    output: Tree = {}  # type: ignore[reportAssignType]
    for k, v in data.items():
        if isinstance(v, dict):
            output[k] = tree_filter(v, criteria_fn=criteria_fn)
        elif criteria_fn(v):
            output[k] = v

    return output
