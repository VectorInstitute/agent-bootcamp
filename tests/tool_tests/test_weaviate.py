"""Test cases for Weaviate integration."""

import pytest
import pytest_asyncio
import weaviate
from weaviate.classes.init import Auth

from src.utils import AsyncWeaviateKnowledgeBase, Configs, pretty_print


@pytest.fixture()
def configs():
    """Load env var configs for testing."""
    return Configs.from_env_var()


@pytest_asyncio.fixture()
async def weaviate_kb(configs):
    """Weaviate knowledgebase for testing."""
    async_client = weaviate.use_async_with_weaviate_cloud(
        cluster_url=configs.weaviate_url,
        auth_credentials=Auth.api_key(configs.weaviate_api_key),
    )

    yield AsyncWeaviateKnowledgeBase(
        async_client=async_client, collection_name="enwiki_20250520_dry_run"
    )

    await async_client.close()


@pytest.mark.asyncio
async def test_weaviate_kb(weaviate_kb: AsyncWeaviateKnowledgeBase):
    """Test weaviate knowledgebase integration."""
    responses = await weaviate_kb.search_knowledgebase("What is Toronto known for?")
    assert len(responses) > 0
    pretty_print(responses)
