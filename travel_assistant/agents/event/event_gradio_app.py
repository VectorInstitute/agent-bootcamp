"""PredictHQ Event Search Demo using Gradio."""

import gradio as gr
from dotenv import load_dotenv
import os
import asyncio
from openai import AsyncOpenAI
import json  # <-- Move this to the top

from event_api import AsyncPredictHQClient
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    function_tool,
)

DESCRIPTION = """\
Enter a travel or event-related query below. The agent will use PredictHQ to find relevant events. \
Try queries like: "Are there any concerts in Toronto on August 9, 2025?" or "Show me Taylor Swift events in the US."
"""

load_dotenv()

async def agent_event_search(query: str) -> str:
    """Run the PredictHQ agent and return pretty-printed results."""
    async_predicthq_client = AsyncPredictHQClient(api_key=os.environ.get("PREDICTHQ_TOKEN"))
    event_search_tool = function_tool(async_predicthq_client.search_events)
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
        tools=[event_search_tool],
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

def sync_agent_event_search(query: str) -> str:
    """Sync wrapper for Gradio."""
    return asyncio.run(agent_event_search(query))

json_codeblock = gr.Code(language="json", wrap_lines=True)

demo = gr.Interface(
    fn=sync_agent_event_search,
    inputs=["text"],
    outputs=[json_codeblock],
    title="PredictHQ Event Search Demo",
    description=DESCRIPTION,
    examples=[
        "Are there any concerts in Toronto on August 9, 2025?",
        "Show me Taylor Swift events in the US.",
        "What major events are happening in New York City in 2025?",
        "Find me music festivals in California.",
    ],
)

demo.launch(server_name="0.0.0.0")