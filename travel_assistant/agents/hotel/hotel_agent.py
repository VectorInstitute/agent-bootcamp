from hotel_api import AsyncPredictHQClient
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

# Initialize PredictHQ API client
async def _main(query: str):
  async_predicthq_client = AsyncPredictHQClient(api_key=APIKEY)

  # Define the tool for the agent
  hotel_search_tool = function_tool(async_predicthq_client.search_hotel)

  async_openai_client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
  )

  model = OpenAIChatCompletionsModel(
    model=os.environ.get("AGENT_LLM_NAME"),
    openai_client=async_openai_client,
  )

  predicthq_agent = Agent(
      name="PredictHQ Agent",
      instructions="You are an assistant that helps plan around real-world events using PredictHQ.",
      tools=[hotel_search_tool],
      model=model,
  )

  response = await Runner.run(
      predicthq_agent,
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

  # await async_predicthq_client.close()
  await async_openai_client.close()


if __name__ == "__main__":
  # query = (
  #     "I'm planning a trip to Toronto Saturday 9th of august 2025." 
  #     "Can you check if there are any major music events or concerts happening there?"
  # )
  query = (
      "Can you find any hotel in London UK with city code LON under 200 pound from september 1 2025 to september 10 2025 for hotel." 
      # "Can you find any hotel under 200 cad?"
  )
  # query = "get me all taylor swift events"

  asyncio.run(_main(query))

