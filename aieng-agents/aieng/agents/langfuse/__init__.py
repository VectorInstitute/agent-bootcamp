"""Utilities for Langfuse integration.

Uses lazy loading to import the Langfuse client and submodules only when needed,
while raising clear ``ImportError`` messages for missing extra dependencies.
"""

import importlib
from types import ModuleType
from typing import Any

from aieng.agents._optional_extras import EXTRA_OBSERVABILITY, raise_missing_optional


def _import_langfuse_submodule(name: str) -> ModuleType:
    """Import ``aieng.agents.langfuse.<name>`` or raise a clear ``ImportError``."""
    fqmn = f"aieng.agents.langfuse.{name}"
    try:
        return importlib.import_module(fqmn)
    except ModuleNotFoundError as exc:
        raise_missing_optional(
            EXTRA_OBSERVABILITY, missing=getattr(exc, "name", None), from_exc=exc
        )


def _get_langfuse_client() -> Any:
    """Return the shared Langfuse client (loads ``shared_client`` on demand)."""
    shared = _import_langfuse_submodule("shared_client")
    return shared._manager.client


def __getattr__(name: str) -> Any:
    """PEP 562: resolve public names without importing observability deps up front."""
    if name == "langfuse_client":
        value = _get_langfuse_client()
        globals()[name] = value
        return value

    if name == "setup_langfuse_tracer":
        mod = _import_langfuse_submodule("oai_sdk_setup")
        value = mod.setup_langfuse_tracer
        globals()[name] = value
        return value

    if name == "set_up_langfuse_otlp_env_vars":
        mod = _import_langfuse_submodule("otlp_env_setup")
        value = mod.set_up_langfuse_otlp_env_vars
        globals()[name] = value
        return value

    if name == "flush_langfuse":
        mod = _import_langfuse_submodule("shared_client")
        value = mod.flush_langfuse
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "flush_langfuse",
    "langfuse_client",
    "set_up_langfuse_otlp_env_vars",
    "setup_langfuse_tracer",
]
