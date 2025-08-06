"""PredictHQ Event Search Demo using Gradio."""
from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import os
from openai import AsyncOpenAI

from hotel_api import AsyncAmadeusClient
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    function_tool,
)

from gradio.components.chatbot import ChatMessage
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils import (
    oai_agent_stream_to_gradio_messages,
    pretty_print,
    setup_langfuse_tracer,
)
from utils.langfuse.shared_client import langfuse_client


async_amadeus_client = AsyncAmadeusClient(api_key=os.environ.get("AMADEUS_API_KEY"),
                                              api_secret=os.environ.get("AMADEUS_API_SECRET"))
hotel_search_tool = function_tool(async_amadeus_client.search_hotel)
async_openai_client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)
model = OpenAIChatCompletionsModel(
    model=os.environ.get("AGENT_LLM_NAME"),
    openai_client=async_openai_client,
)

async def _main(question: str, gr_messages: list[ChatMessage]):
    """Run the PredictHQ agent and return pretty-printed results."""

    setup_langfuse_tracer()

    hotel_agent = Agent(
        name="Amadeus Agent",
        instructions="""
            You are an assistant that search for hotels using city code. Find city code based on the city name provided and search for hotels in the city code.
            Example: find all hotels in ICN. ICN refers to Incheon, we need to find all hotels in Incheon. 
            
            """,
        tools=[hotel_search_tool],
        model=model,
    )

    with langfuse_client.start_as_current_span(name="Agents-SDK-Trace") as span:
      span.update(input=question)

      result_stream = Runner.run_streamed(hotel_agent, input=question)
      async for _item in result_stream.stream_events():
          gr_messages += oai_agent_stream_to_gradio_messages(_item)
          if len(gr_messages) > 0:
              yield gr_messages

      span.update(output=result_stream.final_output)

    pretty_print(gr_messages)
    yield gr_messages

demo = gr.ChatInterface(
    _main,
    title="Hotel Search Demo + LangFuse",
    type="messages",
    examples=[
        "Show me all hotels in ICN"
    ],
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")