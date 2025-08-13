"""Example code for planner-worker agent collaboration.

With reference to:

github.com/ComplexData-MILA/misinfo-datasets
/blob/3304e6e/misinfo_data_eval/tasks/web_search.py
"""
import pandas as pd
from pandasql import sqldf
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
    oai_agent_stream_to_gradio_messages,
    setup_langfuse_tracer,  #add langfuse for traceback
)
import agent_prompt_lib

from src.utils.langfuse.shared_client import langfuse_client

import pandas as pd
from pandasql import sqldf

import pandas as pd
from pandasql import sqldf

def search_historical_data(sql: str) -> list[dict]:
    """
    Run a SQL query against the local LCL sales dataset.

    Args:
        sql (str): SQL query string. Reference the dataframe by name 'df'.

    Returns:
        list[dict]: Query results as a list of dictionaries (JSON-serializable).
    """
    # Load dataset (fixed location)
    df = pd.read_csv("/home/coder/agent-bootcamp_lcl1/lcl_sample_dataset.csv")

    # Ensure df is in the environment passed to sqldf
    context = {"historical_sales_data": df}

    # Execute SQL query
    result_df = sqldf(sql, context)

    # Convert result to JSON-friendly format
    return result_df.to_dict(orient="records")


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
    agent_prompt_lib.prompt_search_agent["v2"]    
    ),
    tools=[
        agents.function_tool(search_historical_data),
    ],
    # a faster, smaller model for quick searches
    model=agents.OpenAIChatCompletionsModel(
        model="gemini-2.5-flash", openai_client=async_openai_client
    ),
)

# "Rules:"
#         "1. The table name is 'df'."
#         "2. The year column is 'Year', not 'startYear'."
#         "3. Always use the LIKE operator with wildcards for text matches (e.g., "
#         "Product_Group LIKE '%spicy%')."
#         "4. When looking for the 'best' promotion, order by Sales_Lift DESC and LIMIT to 1."
#         "5. Keep queries concise but ensure they capture all relevant matches."
#         "Example query:"
#         "SELECT * FROM df "
#         "WHERE Year = 2024 AND (Product_Group LIKE '%spicy%' OR Category_Group LIKE '%spicy%') "
#         "ORDER BY Sales_Lift DESC LIMIT 1;"


REACT_INSTRUCTIONS = """\
Answer the question using the search tool. \
EACH TIME before invoking the function, you must explain your reasons for doing so. \
If the search tool did not return intended results, try again. \
For best performance, divide complex queries into simpler sub-queries. \
Do not make up information.
Stop after two tries.
Your search tool is an agent searching an sql database containing promotion data for products.
Use that to reason based on the information you get from the search agent/tool.
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
    setup_langfuse_tracer()
    
    with langfuse_client.start_as_current_span(name="Agents-SDK-Trace") as span:
        span.update(input=question)

        result_stream = agents.Runner.run_streamed(main_agent, input=question)
        async for _item in result_stream.stream_events():
            gr_messages += oai_agent_stream_to_gradio_messages(_item)
            if len(gr_messages) > 0:
                yield gr_messages

        span.update(output=result_stream.final_output)




demo = gr.ChatInterface(
    _main,
    title="Multi-Agent for Promo Planning at LCL",
    type="messages",
    examples=[
        "Give me the all promotions for spicy sauce",
    ],
)

if __name__ == "__main__":
    async_openai_client = AsyncOpenAI()

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        demo.launch(share=True)
    finally:
        asyncio.run(_cleanup_clients())
