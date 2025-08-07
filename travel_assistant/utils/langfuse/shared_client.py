"""Shared instance of langfuse client."""

from os import getenv

from langfuse import Langfuse
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..env_vars import Configs


__all__ = ["langfuse_client"]


config = Configs.from_env_var()
assert getenv("LANGFUSE_PUBLIC_KEY") is not None
langfuse_client = Langfuse(
    public_key=config.langfuse_public_key, secret_key=config.langfuse_secret_key
)


def flush_langfuse(client: "Langfuse | None" = None):
    """Flush shared LangFuse Client. Rich Progress included."""
    if client is None:
        client = langfuse_client

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Finalizing Langfuse annotations...", total=None)
        langfuse_client.flush()
