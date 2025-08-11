"""Example code for planner-worker agent collaboration.

With reference to:

github.com/ComplexData-MILA/misinfo-datasets
/blob/3304e6e/misinfo_data_eval/tasks/web_search.py
"""

import asyncio
import contextlib
import logging
import signal
import sys

import agents
import gradio as gr
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI
from src.utils import (
    Configs,
    oai_agent_stream_to_gradio_messages
)

def search_historical_data(SQL):
    return SQL


load_dotenv(verbose=True)


logging.basicConfig(level=logging.INFO)


AGENT_LLM_NAMES = {
    "worker": "gemini-2.5-flash",  # less expensive,
    "planner": "gemini-2.5-pro",  # more expensive, better at reasoning and planning
}

configs = Configs.from_env_var()

async_openai_client = AsyncOpenAI()


async def _cleanup_clients() -> None:
    """Close async clients."""
    await async_openai_client.close()


def _handle_sigint(signum: int, frame: object) -> None:
    """Handle SIGINT signal to gracefully shutdown."""
    with contextlib.suppress(Exception):
        asyncio.get_event_loop().run_until_complete(_cleanup_clients())
    sys.exit(0)


# Worker Agent: handles long context efficiently
search_agent = agents.Agent(
    name="SearchAgent",
    instructions=(
        "You are a SQL search, You receive a single search query as input."
        "Generate an SQL query to gather data that can used to analyze the query"
        "Try to keep the data retireval consize without risking leaving out details"
        "Here is the schema of the data base which you will be writing quries againt"
        "The schema defines the column names and the column description"
        "schema start"
        "Year: Year the offer was promoted,"
        "Week: Week the offer was promoted"
        "Category_Group: What category the products fall under"
        "Category_Group_ID: Category group ID for the Category, unique for each category"
        "Product_Group_ID: Unique ID for each product"
        "Product_Group: Unique product names, unqiue for each product"
        "Sub_Product_ID: Sub product id unique for each subproduct"
        "Sub_Product: Sub product name unique for each sub product"
        "Gauging_unit: Unit of Gauging the number of unit sold in a pack ABC means each"
        "Locations: Store IDs for stores/location offering the promotions"
        "Locations_Name: Store names offering the promotions"
        "Container: promotion container unique to each promotion"
        "Advertisement: Amount of advertisement each promotion receives "
        "Shelf_Price: Product price on the Shelf in the stores"
        "Promo_Price: Product price for promotion purposes"
        "Total_Qty: Total quantity of "
        "Sales_Qty: "
        "Total_Revenue: "
        "Revenue: "
        "Margin	Weightage: "
        "Sales_Lift: "
        "Margin_Lift: "
        "Weightage_Lift: "
        "schema end"
        "Use the search_historical_data tool to run this SQL query you generated and return the output dataframe"
    ),
    tools=[
        agents.function_tool(search_historical_data),
    ],
    # a faster, smaller model for quick searches
    model=agents.OpenAIChatCompletionsModel(
        model="gemini-2.5-flash", openai_client=async_openai_client
    ),
)


REACT_INSTRUCTIONS = """\
Answer the question using the search tool. \
EACH TIME before invoking the function, you must explain your reasons for doing so. \
If the search tool did not return intended results, try again. \
For best performance, divide complex queries into simpler sub-queries. \
Do not make up information.
Stop after two tries.
"""
# Main Agent: more expensive and slower, but better at complex planning
main_agent = agents.Agent(
    name="MainAgent",
    instructions=REACT_INSTRUCTIONS,
    # Allow the planner agent to invoke the worker agent.
    # The long context provided to the worker agent is hidden from the main agent.
    tools=[
        search_agent.as_tool(
            tool_name="search",
            tool_description="Query the structured data and return enough data to help asnwer the user question.",
        )
    ],
    # a larger, more capable model for planning and reasoning over summaries
    model=agents.OpenAIChatCompletionsModel(
        model="gemini-2.5-pro", openai_client=async_openai_client
    ),
)


async def _main(question: str, gr_messages: list[ChatMessage]):

    result_stream = agents.Runner.run_streamed(main_agent, input=question)
    async for _item in result_stream.stream_events():
        gr_messages += oai_agent_stream_to_gradio_messages(_item)
        if len(gr_messages) > 0:
            yield gr_messages



demo = gr.ChatInterface(
    _main,
    title="2.2 Multi-Agent for Efficiency",
    type="messages",
    examples=[
        "Give me the best promotion for spicy sauce",
    ],
)

if __name__ == "__main__":
    async_openai_client = AsyncOpenAI()

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        demo.launch(share=True)
    finally:
        asyncio.run(_cleanup_clients())
