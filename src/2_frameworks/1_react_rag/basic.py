"""Non-Interactive Example of OpenAI Agent SDK for Knowledge Retrieval."""

import asyncio
import logging

from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner, function_tool
from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    get_weaviate_async_client,
    pretty_print,
)


load_dotenv(verbose=True)

AGENT_LLM_NAME = "gemini-2.5-flash"
no_tracing_config = RunConfig(tracing_disabled=True)

INSTRUCTIONS = """\
Answer the question using the search tool. \
Always plan your actions before invoking the tool. \
Be sure to mention the sources. \
If the search did not return intended results, try again. \
Do not make up information. You must use the search tool \
for all facts that might change over time.
"""


async def _main(query: str):
    configs = Configs.from_env_var()
    async_weaviate_client = get_weaviate_async_client(
        http_host=configs.weaviate_http_host,
        http_port=configs.weaviate_http_port,
        http_secure=configs.weaviate_http_secure,
        grpc_host=configs.weaviate_grpc_host,
        grpc_port=configs.weaviate_grpc_port,
        grpc_secure=configs.weaviate_grpc_secure,
        api_key=configs.weaviate_api_key,
    )
    async_knowledgebase = AsyncWeaviateKnowledgeBase(
        async_weaviate_client,
        collection_name="enwiki_20250520",
    )

    async_openai_client = AsyncOpenAI()

    wikipedia_agent = Agent(
        name="Wikipedia Agent",
        instructions=INSTRUCTIONS,
        tools=[function_tool(async_knowledgebase.search_knowledgebase)],
        model=OpenAIChatCompletionsModel(
            model=AGENT_LLM_NAME, openai_client=async_openai_client
        ),
    )

    response = await Runner.run(
        wikipedia_agent,
        input=query,
        run_config=no_tracing_config,
    )

    pretty_print(response.raw_responses)
    pretty_print(response.new_items)
    pretty_print(response.final_output)

    await async_weaviate_client.close()
    await async_openai_client.close()


if __name__ == "__main__":
    query = (
        "At which university did the SVP Software Engineering"
        " at Apple (as of June 2025) earn their engineering degree?"
    )

    logging.basicConfig(level=logging.INFO)
    asyncio.run(_main(query))
