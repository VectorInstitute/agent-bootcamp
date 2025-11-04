"""Test cases for Weaviate integration."""

import json
import sys
from pathlib import Path
from typing import AsyncGenerator


# Add project root to path to allow imports from src
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from langfuse import get_client  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from src.utils import (  # noqa: E402
    AsyncWeaviateKnowledgeBase,
    Configs,
    get_weaviate_async_client,
    pretty_print,
)
from src.utils.langfuse.otlp_env_setup import (  # noqa: E402
    set_up_langfuse_otlp_env_vars,
)


load_dotenv(verbose=True)


@pytest.fixture()
def configs():
    """Load env var configs for testing."""
    return Configs.from_env_var()


@pytest_asyncio.fixture()
async def weaviate_kb(
    configs: Configs,
) -> AsyncGenerator[AsyncWeaviateKnowledgeBase, None]:
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
        async_client=async_client, collection_name="enwiki_20250520"
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
        model="gemini-2.5-flash-lite-preview-06-17", messages=messages
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
