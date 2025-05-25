"""Test cases for ElasticSearch integration."""

import pytest
import pytest_asyncio
from elasticsearch import AsyncElasticsearch

from src.utils import AsyncESKnowledgeBase, Configs, pretty_print


@pytest.fixture()
def configs():
    """Load env var configs for testing."""
    return Configs.from_env_var()


@pytest_asyncio.fixture()
async def es_knowledgebase(configs):
    """ElasticSearch knowledgebase for testing."""
    async_es = AsyncElasticsearch(configs.es_host, api_key=configs.es_api_key)

    yield AsyncESKnowledgeBase(
        async_es_client=async_es,
        es_collection_name="enwiki-20250501",
    )

    await async_es.close()


@pytest.mark.asyncio
async def test_es_knowledgebase(es_knowledgebase: AsyncESKnowledgeBase):
    """Test es knowledgebase integration."""
    responses = await es_knowledgebase.search_knowledgebase("Apple TV remote")
    assert len(responses) > 0
    pretty_print(responses)
