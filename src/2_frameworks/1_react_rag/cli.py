"""Non-Interactive Example of OpenAI Agent SDK for Knowledge Retrieval."""

import asyncio
import logging

from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    function_tool,
)
from aieng.agents import pretty_print
from aieng.agents.client_manager import AsyncClientManager
from aieng.agents.prompts import REACT_INSTRUCTIONS
from dotenv import load_dotenv


async def _main(query: str) -> None:
    try:
        wikipedia_agent = Agent(
            name="Wikipedia Agent",
            instructions=REACT_INSTRUCTIONS,
            tools=[function_tool(client_manager.knowledgebase.search_knowledgebase)],
            model=OpenAIChatCompletionsModel(
                model=client_manager.configs.default_worker_model,
                openai_client=client_manager.openai_client,
            ),
        )

        response = await Runner.run(
            wikipedia_agent, input=query, run_config=no_tracing_config
        )

        for item in response.new_items:
            pretty_print(item.raw_item)
            print()

        pretty_print(response.final_output)

        # Uncomment the following for a basic "streaming" example

        # from src.utils import oai_agent_stream_to_gradio_messages
        # result_stream = Runner.run_streamed(
        #     wikipedia_agent, input=query, run_config=no_tracing_config
        # )
        # async for event in result_stream.stream_events():
        #     event_parsed = oai_agent_stream_to_gradio_messages(event)
        #     if len(event_parsed) > 0:
        #         pretty_print(event_parsed)
    finally:
        # Ensure clients are closed on exit
        await client_manager.close()


if __name__ == "__main__":
    load_dotenv(verbose=True)

    logging.basicConfig(level=logging.INFO)

    no_tracing_config = RunConfig(tracing_disabled=True)

    # Initialize client manager
    # This class initializes the OpenAI and Weaviate async clients, as well as the
    # Weaviate knowledge base tool. The initialization is done once when the clients
    # are first accessed, and the clients are reused for subsequent calls.
    client_manager = AsyncClientManager()

    query = (
        "At which university did the SVP Software Engineering"
        " at Apple (as of June 2025) earn their engineering degree?"
    )

    asyncio.run(_main(query))
