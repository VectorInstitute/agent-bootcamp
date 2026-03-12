"""Utilities for Langfuse integration."""

from typing import TYPE_CHECKING

from aieng.agents.langfuse.oai_sdk_setup import setup_langfuse_tracer
from aieng.agents.langfuse.otlp_env_setup import set_up_langfuse_otlp_env_vars
from aieng.agents.langfuse.shared_client import flush_langfuse


if TYPE_CHECKING:
    from langfuse import Langfuse

langfuse_client: "Langfuse"  # noqa: F822 -- lazily initialized via __getattr__


def __getattr__(name: str) -> "Langfuse":
    if name == "langfuse_client":
        from aieng.agents.langfuse.shared_client import _manager  # noqa: PLC0415

        return _manager.client
    raise AttributeError(f"module has no attribute '{name}'")


__all__ = [
    "flush_langfuse",
    "langfuse_client",
    "set_up_langfuse_otlp_env_vars",
    "setup_langfuse_tracer",
]
