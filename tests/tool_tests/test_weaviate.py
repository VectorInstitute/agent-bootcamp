"""Test cases for Weaviate integration."""

import pytest
import pytest_asyncio
import weaviate

from src.utils import AsyncWeaviateKnowledgeBase, Configs, pretty_print


@pytest.fixture()
def configs():
    """Load env var configs for testing."""
    return Configs.from_env_var()


@pytest_asyncio.fixture()
async def weaviate_kb(configs):
    """Weaviate knowledgebase for testing."""
    async_client = weaviate.use_async_with_local(
        host=configs.weaviate_host,
        port=configs.weaviate_port,
        grpc_port=configs.weaviate_grpc_port,
    )

    yield AsyncWeaviateKnowledgeBase(
        async_client=async_client, collection_name="enwiki_20250520_50k"
    )

    await async_client.close()


@pytest.mark.asyncio
async def test_weaviate_kb(weaviate_kb: AsyncWeaviateKnowledgeBase):
    """Test weaviate knowledgebase integration."""
    responses = await weaviate_kb.search_knowledgebase("What is Toronto known for?")
    assert len(responses) > 0
    pretty_print(responses)
