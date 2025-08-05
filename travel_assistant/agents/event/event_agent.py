from event_api import AsyncPredictHQClient
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

# Initialize PredictHQ API client
predicthq_token = "SU7lc8v5aB_YS9DnqT2AW61cZ-OUrZ0rTRZ9_TW8"
OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
openai_key= "AIzaSyC7QLT8uZdVW6yhkPA6ORKUjqeI_8xld-4"

AGENT_LLM_NAME = "gemini-2.5-flash"
async def _main(query: str):
  async_predicthq_client = AsyncPredictHQClient(api_key=predicthq_token)

  # Define the tool for the agent
  event_search_tool = function_tool(async_predicthq_client.search_events)

  async_openai_client = AsyncOpenAI(
    api_key=openai_key,
    base_url=OPENAI_BASE_URL,
  )

  predicthq_agent = Agent(
      name="PredictHQ Agent",
      instructions="You are an assistant that helps plan around real-world events using PredictHQ.",
      tools=[event_search_tool],
      model=OpenAIChatCompletionsModel(
          model=AGENT_LLM_NAME,
          openai_client=async_openai_client,
      ),
  )

  model = OpenAIChatCompletionsModel(
    model=AGENT_LLM_NAME,             # Or e.g., 'gemini-2.5-flash'
    openai_client=async_openai_client
  
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

  query = "get me all taylor swift events"

  asyncio.run(_main(query))

