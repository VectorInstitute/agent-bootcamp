"""Amadeus Hotel Search Demo using Gradio."""

import gradio as gr
from dotenv import load_dotenv
import os
import asyncio
from openai import AsyncOpenAI
import json

from hotel_api import AsyncAmadeusClient
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    function_tool,
)

DESCRIPTION = """\
Enter a hotel query below. The agent will use Amadeus to find relevant hotels. \
Try queries like: "find all hotels in Seoul which city code is ICN"
"""

load_dotenv()

async def agent_hotel_search(query: str) -> str:
    """Run the Hotel agent and return pretty-printed results."""    
    async_hotel_client = AsyncAmadeusClient(api_key=os.environ.get("AMADEUS_API_KEY"),
                                              api_secret=os.environ.get("AMADEUS_API_SECRET"))
    hotel_search_tool = function_tool(async_hotel_client.search_hotel)
    async_openai_client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    model = OpenAIChatCompletionsModel(
        model=os.environ.get("AGENT_LLM_NAME"),
        openai_client=async_openai_client,
    )
    hotel_agent = Agent(
        name="Hotel Agent",
        instructions="You are an assistant that search for hotels using city code. Find city code based on the city name provided and search for hotels in the city code",
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
    # Collect all new items and the final output
    results = []
    for item in response.new_items:
        # Always convert to string if not a dict or list
        if isinstance(item.raw_item, (dict, list)):
            results.append(item.raw_item)
        else:
            results.append(str(item.raw_item))
    results.append({"final_output": str(response.final_output)})
    return json.dumps(results, indent=2)


def sync_agent_hotel_search(query: str) -> str:
    """Sync wrapper for Gradio."""
    return asyncio.run(agent_hotel_search(query))

json_codeblock = gr.Code(language="json", wrap_lines=True)

demo = gr.Interface(
    fn=sync_agent_hotel_search,
    inputs=["text"],
    outputs=[json_codeblock],
    title="Amadeus Hotel Search Demo",
    description=DESCRIPTION,
    examples=[
        "find all hotels in Seoul which city code is ICN",
        "find all hotels in ICN"
    ],
)

demo.launch(server_name="0.0.0.0", share=True)