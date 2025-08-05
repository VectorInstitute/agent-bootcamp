import asyncio
import os
from dotenv import load_dotenv

from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    function_tool,
)
from openai import AsyncOpenAI

from flight_api import RapidApiClient  # Assuming your class is in flight_api.py

load_dotenv()

# Initialize RapidAPI client and agent
async def _main(query: str):
    rapidapi_client = RapidApiClient(
        api_key=os.environ.get("RAPIDAPI_KEY"),
        host="google-flights2.p.rapidapi.com"
    )

    # Wrap the airport search method as a tool
    flight_search_tool = function_tool(rapidapi_client.search_flight)

    async_openai_client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    model = OpenAIChatCompletionsModel(
        model=os.environ.get("AGENT_LLM_NAME"),
        openai_client=async_openai_client,
    )

    flight_agent = Agent(
        name="Flight Search Agent",
        instructions="You are a travel assistant that helps users find fligh information using RapidAPI google flight API get price graph method. if user provide city name instead of id, first find the corresponding id for origin and destination and then call the api",
        tools=[flight_search_tool],
        model=model,
    )

    response = await Runner.run(
        flight_agent,
        input=query,
        run_config=RunConfig(
            model=model,
            tracing_disabled=True
        )
    )

    for item in response.new_items:
        print(item.raw_item)
        print()

    print(response.final_output)

    await async_openai_client.close()

if __name__ == "__main__":
    query = "Find flights prices and information from Montreal to Vancouver"
    asyncio.run(_main(query))
