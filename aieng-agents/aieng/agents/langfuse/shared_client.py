"""Shared instance of langfuse client."""

from functools import cached_property

from aieng.agents.env_vars import Configs
from langfuse import Langfuse
from rich.progress import Progress, SpinnerColumn, TextColumn


class _LangfuseClientManager:
    @cached_property
    def config(self) -> Configs:
        return Configs()

    @cached_property
    def client(self) -> Langfuse:
        return Langfuse(
            public_key=self.config.langfuse_public_key,
            secret_key=self.config.langfuse_secret_key,
        )


_manager = _LangfuseClientManager()
langfuse_client: Langfuse  # noqa: F822 -- lazily initialized via __getattr__


def __getattr__(name: str) -> Langfuse:
    """Module-level lazy loading for backward compatibility."""
    if name == "langfuse_client":
        return _manager.client
    raise AttributeError(f"module has no attribute '{name}'")


def flush_langfuse(client: Langfuse | None = None) -> None:
    """Flush shared LangFuse Client. Rich Progress included."""
    if client is None:
        client = _manager.client

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Finalizing Langfuse annotations...", total=None)
        client.flush()


__all__ = ["flush_langfuse", "langfuse_client"]
