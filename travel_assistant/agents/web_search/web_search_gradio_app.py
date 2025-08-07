"""Web search demo using Gradio."""

import asyncio
import json
import os

import gradio as gr
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


DESCRIPTION = """\
Enter a search query. The agent will use DuckDuckGo first, then Tavily, and finally SerpAPI to find results.
"""

load_dotenv()


async def agent_web_search(query: str) -> str:
    """Run the web search agent and return pretty-printed results."""
    client = AsyncWebSearchClient()
    search_tool = function_tool(client.search)
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
        instructions="You search the web using DuckDuckGo, Tavily, and SerpAPI in that order.",
        tools=[search_tool],
        model=model,
    )
    response = await Runner.run(
        web_agent,
        input=query,
        run_config=RunConfig(model=model, tracing_disabled=True),
    )
    results = []
    for item in response.new_items:
        if isinstance(item.raw_item, (dict, list)):
            results.append(item.raw_item)
        else:
            results.append(str(item.raw_item))
    results.append({"final_output": str(response.final_output)})
    await async_openai_client.close()
    await client.close()
    return json.dumps(results, indent=2)


def sync_agent_web_search(query: str) -> str:
    """Sync wrapper for Gradio."""
    return asyncio.run(agent_web_search(query))


json_codeblock = gr.Code(language="json", wrap_lines=True)

demo = gr.Interface(
    fn=sync_agent_web_search,
    inputs=["text"],
    outputs=[json_codeblock],
    title="Web Search Demo",
    description=DESCRIPTION,
    examples=[
        "Latest AI research",
        "Best pizza places in New York",
        "Weather in Tokyo tomorrow",
    ],
)

demo.launch(server_name="0.0.0.0")
