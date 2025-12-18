"""Test cases for Weaviate integration."""

import pytest
import pytest_asyncio
from dotenv import load_dotenv

from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    get_weaviate_async_client,
    pretty_print,
)


load_dotenv(verbose=True)


@pytest.fixture()
def configs():
    """Load env var configs for testing."""
    return Configs()


@pytest_asyncio.fixture()
async def weaviate_kb(configs):
    """Weaviate knowledgebase for testing."""
    async_client = get_weaviate_async_client(
        http_host=configs.weaviate_http_host,
        http_port=configs.weaviate_http_port,
        http_secure=configs.weaviate_http_secure,
        grpc_host=configs.weaviate_grpc_host,
        grpc_port=configs.weaviate_grpc_port,
        grpc_secure=configs.weaviate_grpc_secure,
        api_key=configs.weaviate_api_key,
    )

    yield AsyncWeaviateKnowledgeBase(
        async_client=async_client, collection_name=configs.weaviate_collection_name
    )

    await async_client.close()


@pytest.mark.asyncio
async def test_weaviate_kb(weaviate_kb: AsyncWeaviateKnowledgeBase):
    """Test weaviate knowledgebase integration."""
    responses = await weaviate_kb.search_knowledgebase("What is Toronto known for?")
    assert len(responses) > 0
    pretty_print(responses)
