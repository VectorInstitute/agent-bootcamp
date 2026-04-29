"""Async client lifecycle manager for Gradio applications.

Provides idempotent initialization and proper cleanup of async clients
like Weaviate and OpenAI to prevent event loop conflicts during Gradio's
hot-reload process.
"""

from typing import TYPE_CHECKING, Any

from aieng.agents.env_vars import Configs
from openai import AsyncOpenAI


if TYPE_CHECKING:
    from aieng.agents.tools.weaviate_kb import AsyncWeaviateKnowledgeBase
    from weaviate.client import WeaviateAsyncClient


class MissingOptionalDependencyError(ImportError):
    """Raised when an optional integration dependency is unavailable."""


def _load_weaviate_integrations() -> tuple[type["AsyncWeaviateKnowledgeBase"], Any]:
    """Load Weaviate integration symbols lazily with clear install guidance."""
    try:
        from aieng.agents.tools.weaviate_kb import (  # noqa: PLC0415
            AsyncWeaviateKnowledgeBase,
            get_weaviate_async_client,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - import-time branch
        raise MissingOptionalDependencyError(
            "Weaviate integration requires optional dependencies. "
            "Install with `pip install 'aieng-agents[weaviate]'`."
        ) from exc

    return AsyncWeaviateKnowledgeBase, get_weaviate_async_client


class AsyncClientManager:
    """Manages async client lifecycle with lazy initialization and cleanup.

    This class ensures clients are created only once and properly closed,
    preventing ResourceWarning errors from unclosed event loops.

    Parameters
    ----------
    configs: Configs | None, optional, default=None
        Configuration object for client setup. If None, a new ``Configs()`` is created.

    Examples
    --------
    >>> manager = AsyncClientManager()
    >>> # Access clients (created on first access)
    >>> weaviate = manager.weaviate_client
    >>> kb = manager.knowledgebase
    >>> openai = manager.openai_client
    >>> # In finally block or cleanup
    >>> await manager.close()
    """

    def __init__(self, configs: Configs | None = None) -> None:
        """Initialize manager with optional configs."""
        self._configs = configs
        self._weaviate_client: WeaviateAsyncClient | None = None
        self._openai_client = None
        self._knowledgebase: AsyncWeaviateKnowledgeBase | None = None
        self._initialized = False

    @property
    def configs(self) -> Configs:
        """Get or create configs instance."""
        if self._configs is None:
            self._configs = Configs()  # pyright: ignore[reportCallIssue]
        return self._configs

    @property
    def openai_client(self) -> AsyncOpenAI:
        """Get or create OpenAI client."""
        if self._openai_client is None:
            self._openai_client = AsyncOpenAI()
            self._initialized = True
        return self._openai_client

    @property
    def weaviate_client(self) -> "WeaviateAsyncClient":
        """Get or create Weaviate client."""
        if self._weaviate_client is None:
            _, get_weaviate_async_client = _load_weaviate_integrations()
            self._weaviate_client = get_weaviate_async_client(self.configs)
            self._initialized = True
        return self._weaviate_client

    @property
    def knowledgebase(self) -> "AsyncWeaviateKnowledgeBase":
        """Get or create knowledge base instance."""
        if self._knowledgebase is None:
            async_weaviate_knowledge_base, _ = _load_weaviate_integrations()
            self._knowledgebase = async_weaviate_knowledge_base(
                self.weaviate_client,
                collection_name=self.configs.weaviate_collection_name,
            )
            self._initialized = True
        return self._knowledgebase

    async def close(self) -> None:
        """Close all initialized async clients."""
        if self._weaviate_client is not None:
            await self._weaviate_client.close()
            self._weaviate_client = None

        if self._openai_client is not None:
            await self._openai_client.close()
            self._openai_client = None

        self._knowledgebase = None
        self._initialized = False

    def is_initialized(self) -> bool:
        """Check if any clients have been initialized."""
        return self._initialized
