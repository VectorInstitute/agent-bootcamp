"""Shared instance of langfuse client."""

from aieng.agents.env_vars import Configs
from langfuse import Langfuse
from rich.progress import Progress, SpinnerColumn, TextColumn


__all__ = ["flush_langfuse", "langfuse_client"]


config = Configs()
langfuse_client = Langfuse(
    public_key=config.langfuse_public_key, secret_key=config.langfuse_secret_key
)


def flush_langfuse(client: "Langfuse | None" = None) -> None:
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
