"""Helpers for errors when optional dependency extras are not installed."""

from __future__ import annotations


EXTRA_DATA = "data"
EXTRA_NEWS = "news"
EXTRA_GRADIO = "gradio"
EXTRA_WEAVIATE = "weaviate"
EXTRA_OBSERVABILITY = "observability"
EXTRA_CODE_INTERPRETER = "code-interpreter"


def install_hint(extra: str) -> str:
    """Human-readable pip/uv install line for the given extra."""
    return f"pip install 'aieng-agents[{extra}]'"


def raise_missing_optional(
    extra: str,
    *,
    missing: str | None = None,
    from_exc: BaseException | None = None,
) -> None:
    """Raise ``ImportError`` naming the extra and how to install it."""
    suffix = f" (missing module: {missing!r})" if missing else ""
    raise ImportError(
        f"This feature requires the '{extra}' optional dependency{suffix}. "
        f"Install with: {install_hint(extra)}"
    ) from from_exc
