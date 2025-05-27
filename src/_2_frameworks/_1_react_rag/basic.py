"""Non-Interactive Example of OpenAI Agent SDK for Knowledge Retrieval."""

import asyncio
import logging

from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool
from elasticsearch import AsyncElasticsearch
from openai import AsyncOpenAI

from src.utils import AsyncESKnowledgeBase, Configs, pretty_print


INSTRUCTIONS = """\
Answer the question using the search tool. \
Be sure to mention the sources. \
If the search did not return intended results, try again. \
Do not make up information. \
"""


async def _main(query: str):
    configs = Configs.from_env_var()
    async_es_client = AsyncElasticsearch(configs.es_host, api_key=configs.es_api_key)
    async_openai_client = AsyncOpenAI()
    async_knowledgebase = AsyncESKnowledgeBase(
        async_es_client,
        es_collection_name="enwiki-20250501",
    )

    wikipedia_agent = Agent(
        name="Wikipedia Agent",
        instructions=INSTRUCTIONS,
        tools=[function_tool(async_knowledgebase.search_knowledgebase)],
        model=OpenAIChatCompletionsModel(
            model="gpt-4o-mini", openai_client=async_openai_client
        ),
    )

    response = await Runner.run(wikipedia_agent, input=query)
    pretty_print(response.final_output)
    pretty_print(response.raw_responses)
    pretty_print(response.new_items)

    await async_es_client.close()


if __name__ == "__main__":
    query = "Does the Apple TV Remote use IR?"

    logging.basicConfig(level=logging.INFO)
    asyncio.run(_main(query))
