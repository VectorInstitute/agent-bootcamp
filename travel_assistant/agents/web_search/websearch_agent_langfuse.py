"""Web search demo with Langfuse tracing."""

import os
import sys

import gradio as gr
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI
from web_search_api import AsyncWebSearchClient


# Make travel_assistant.utils importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils import (  # noqa: E402
    oai_agent_stream_to_gradio_messages,
    pretty_print,
    setup_langfuse_tracer,
)
from utils.langfuse.shared_client import langfuse_client  # noqa: E402


load_dotenv()

async_web_client = AsyncWebSearchClient()
web_search_tool = function_tool(async_web_client.search)
async_openai_client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)
model = OpenAIChatCompletionsModel(
    model=os.environ.get("AGENT_LLM_NAME", "gpt-4o-mini"),
    openai_client=async_openai_client,
)


async def _main(question: str, gr_messages: list[ChatMessage]):
    """Run the web search agent and stream results."""
    setup_langfuse_tracer()

    web_agent = Agent(
        name="Web Search Agent",
        instructions=(
            #"You search the web using DuckDuckGo first, then Tavily and SerpAPI if needed."
            "You search the hotel based on city, check in and check out dates, as well as price range using DuckDuckGo, Tavily, and SerpAPI in that order"
        ),
        tools=[web_search_tool],
        model=model,
    )

    with langfuse_client.start_as_current_span(name="Agents-SDK-Trace") as span:
        span.update(input=question)

        result_stream = Runner.run_streamed(web_agent, input=question)
        async for _item in result_stream.stream_events():
            gr_messages += oai_agent_stream_to_gradio_messages(_item)
            if gr_messages:
                yield gr_messages

        span.update(output=result_stream.final_output)

    pretty_print(gr_messages)
    yield gr_messages


demo = gr.ChatInterface(
    _main,
    title="Web Search Demo + Langfuse",
    type="messages",
    examples=[
        # "Latest AI research",
        # "Best pizza places in New York",
        # "Weather in Tokyo tomorrow",
        "I am planning a trip to Vancouver from Oct 1 until Oct 4. Find me a list of hotels below 200 CAD per night"
    ],
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
