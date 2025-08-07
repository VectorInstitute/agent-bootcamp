from hotel_api import AsyncAmadeusClient
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    function_tool,
)
import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI
import os

load_dotenv()

# Initialize Hotel API client
async def _main(query: str):
  async_amadeus_client = AsyncAmadeusClient(api_key=os.environ.get("AMADEUS_API_KEY"),
                                              api_secret=os.environ.get("AMADEUS_API_SECRET"))

  # Define the tool for the agent
  hotel_search_tool = function_tool(async_amadeus_client.search_hotel)

  async_openai_client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
  )

  model = OpenAIChatCompletionsModel(
    model=os.environ.get("AGENT_LLM_NAME"),
    openai_client=async_openai_client,
  )

  hotel_agent = Agent(
      name="Amadeus Agent",
      instructions="""
        You are an assistant that search for hotels using city code. Find city code based on the city name provided and search for hotels in the city code.
        Example: find all hotels in ICN. ICN refers to Incheon, we need to find all hotels in Incheon. 
        
        """,
      tools=[hotel_search_tool],
      model=model,
  )

  response = await Runner.run(
      hotel_agent,
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
  query = (   
     "find all hotels in ICN"
  )
  asyncio.run(_main(query))
