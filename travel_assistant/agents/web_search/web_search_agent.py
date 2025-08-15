"""Run the web search agent using DuckDuckGo, Tavily, and SerpAPI."""

import asyncio
import os

from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    function_tool,
)
from dotenv import load_dotenv
from openai import AsyncOpenAI
from web_search_api import AsyncWebSearchClient


load_dotenv()


async def _main(query: str) -> None:
    web_client = AsyncWebSearchClient()
    web_search_tool = function_tool(web_client.search)

    async_openai_client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    model_name = os.environ.get("AGENT_LLM_NAME", "gpt-4o-mini")
    model = OpenAIChatCompletionsModel(
        model=model_name,
        openai_client=async_openai_client,
    )

    web_agent = Agent(
        name="Web Search Agent",
        instructions=(
            #"You search the web using DuckDuckGo, Tavily, and SerpAPI in that order."
            "You search the hotel based on city, check in and check out dates, as well as price range using DuckDuckGo, Tavily, and SerpAPI in that order" 
        ),
        tools=[web_search_tool],
        model=model,
    )

    response = await Runner.run(
        web_agent,
        input=query,
        run_config=RunConfig(model=model, tracing_disabled=True),
    )

    for item in response.new_items:
        print(item.raw_item)
        print()

    print(response.final_output)

    await async_openai_client.close()
    await web_client.close()


if __name__ == "__main__":
    asyncio.run(_main("Provide list of hotels in Toronto between Sep 1 and Sep 3 and price is under 300 CAD per night"))
