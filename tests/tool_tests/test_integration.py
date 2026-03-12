"""Test cases for Weaviate integration."""

import json
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from aieng.agents import Configs, pretty_print
from aieng.agents.langfuse import set_up_langfuse_otlp_env_vars
from aieng.agents.tools import (
    AsyncWeaviateKnowledgeBase,
    GeminiGroundingWithGoogleSearch,
    get_weaviate_async_client,
)
from dotenv import load_dotenv
from langfuse import get_client
from openai import AsyncOpenAI


load_dotenv(verbose=True)


@pytest.fixture()
def configs():
    """Load env var configs for testing."""
    return Configs()


@pytest_asyncio.fixture()
async def weaviate_kb(
    configs: Configs,
) -> AsyncGenerator[AsyncWeaviateKnowledgeBase, None]:
    """Weaviate knowledgebase for testing."""
    async_client = get_weaviate_async_client(configs)

    yield AsyncWeaviateKnowledgeBase(
        async_client=async_client, collection_name=configs.weaviate_collection_name
    )

    await async_client.close()


def test_vectorizer(weaviate_kb: AsyncWeaviateKnowledgeBase) -> None:
    """Test vectorizer integration."""
    vector = weaviate_kb._vectorize("What is Toronto known for?")
    assert vector is not None
    assert len(vector) > 0
    print(f"Vector ({len(vector)} dimensions): {vector[:10]}...")


@pytest.mark.asyncio
async def test_weaviate_kb(weaviate_kb: AsyncWeaviateKnowledgeBase) -> None:
    """Test weaviate knowledgebase integration."""
    responses = await weaviate_kb.search_knowledgebase("What is Toronto known for?")
    assert len(responses) > 0
    pretty_print(responses)


@pytest.mark.asyncio
async def test_weaviate_kb_tool_and_llm(
    configs: Configs,
    weaviate_kb: AsyncWeaviateKnowledgeBase,
) -> None:
    """Test weaviate knowledgebase tool integration and LLM API."""
    query = "What is Toronto known for?"
    search_results = await weaviate_kb.search_knowledgebase(query)
    assert len(search_results) > 0

    client = AsyncOpenAI()
    messages = [
        {
            "role": "system",
            "content": (
                "Answer the question using the provided information from a knowledge base."
            ),
        },
        {
            "role": "user",
            "content": f"{query}\n\n {
                json.dumps([_result.model_dump() for _result in search_results])
            }",
        },
    ]
    response = await client.chat.completions.create(
        model=configs.default_worker_model, messages=messages
    )
    message = response.choices[0].message
    assert message.role == "assistant"
    messages.append(message.model_dump())
    pretty_print(messages)


def test_langfuse() -> None:
    """Test LangFuse integration."""
    set_up_langfuse_otlp_env_vars()
    langfuse_client = get_client()

    assert langfuse_client.auth_check()


@pytest.mark.asyncio
async def test_web_search_with_gemini_grounding(configs: Configs) -> None:
    """Test Gemini grounding with Google Search integration."""
    # Skip test if the environment variable is not set
    # We do this because these are optional env vars and not everyone
    # running the tests may have them set.
    if not (configs.web_search_base_url and configs.web_search_api_key):
        pytest.skip("WEB_SEARCH_BASE_URL and WEB_SEARCH_API_KEY not set in env vars")

    tool_cls = GeminiGroundingWithGoogleSearch()
    response = await tool_cls.get_web_search_grounded_response(
        "How does the annual growth in the 50th-percentile income "
        "in the US compare with that in Canada?"
    )

    pretty_print(response.text_with_citations)
    assert response.text_with_citations
