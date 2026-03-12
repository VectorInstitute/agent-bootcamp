"""Test cases for Weaviate integration."""

from typing import Any, AsyncGenerator

import pytest
import pytest_asyncio
from aieng.agents import Configs, pretty_print
from aieng.agents.tools.weaviate_kb import (
    AsyncWeaviateKnowledgeBase,
    get_weaviate_async_client,
)
from dotenv import load_dotenv


load_dotenv(verbose=True)


@pytest.fixture()
def configs() -> Any:
    """Load env var configs for testing."""
    return Configs()


@pytest_asyncio.fixture()
async def weaviate_kb(configs) -> AsyncGenerator[Any, Any]:
    """Weaviate knowledgebase for testing."""
    async_client = get_weaviate_async_client(configs)

    yield AsyncWeaviateKnowledgeBase(
        async_client=async_client, collection_name=configs.weaviate_collection_name
    )

    await async_client.close()


@pytest.mark.asyncio
async def test_weaviate_kb(weaviate_kb: AsyncWeaviateKnowledgeBase) -> None:
    """Test weaviate knowledgebase integration."""
    responses = await weaviate_kb.search_knowledgebase("What is Toronto known for?")
    assert len(responses) > 0
    pretty_print(responses)
