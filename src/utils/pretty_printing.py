"""Pretty-Print Utils."""

import json
from typing import Any

import pydantic


def _serializer(item: Any) -> dict[str, Any] | str:
    """Serialize using heuristics."""
    if isinstance(item, pydantic.BaseModel):
        return item.model_dump()

    return str(item)


def pretty_print(data: Any) -> None:
    """Print nested items with indentations."""
    print(json.dumps(data, indent=2, default=_serializer))
