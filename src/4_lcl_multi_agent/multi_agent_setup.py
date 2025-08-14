"""Example code for planner-worker agent collaboration.

With reference to:

github.com/ComplexData-MILA/misinfo-datasets
/blob/3304e6e/misinfo_data_eval/tasks/web_search.py
"""
import requests
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
import agent_example_lib

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



def submit_plan_to_api(plan: dict, url: str = "http://127.0.0.1:8000/update") -> None:
    """
    Sends a plan JSON to the FastAPI /update endpoint.

    Args:
        plan (dict): The plan data to send.
        url (str): The FastAPI endpoint URL (default is localhost).

    Returns:
        None
    """
    try:
        response = requests.post(url, json=plan)
        response.raise_for_status()
        print("✅ Plan submitted successfully.")
        print("Response:", response.text)
    except requests.exceptions.RequestException as e:
        print("❌ Failed to submit plan.")
        print("Error:", e)


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

Once you have the answer ready to share with the user, in addition to answer the user in natural 
language, use the api tool 'submit_plan_to_api' to create a reponse in the sample dictionary defined
below to the Front End through API call. In case you don't find the api tool, ignore sending the 
api call.


The table schema is defined in Pydantic class as follows:

class Plan(BaseModel):
    id: str  # main agent generate 10-digit random string 
    created_from: "agent" #default is "agent" no need to change
    offers: list[Offer]

class Offer(BaseModel):
    id: str
    year: int  # Ad year the offer starts on
    week: int  # Ad week the offer starts on
    category_group: str #
    container: str # Empty until created in JDA
    products: list[Product]

class Product(BaseModel):
    "product_group": str
    "product_group_id": str
    "shelf_price": float
    "promo_price": float
    "weightage": float
    "gauging_unit": str
    "sub_product": str
    "sub_product_id": str

The dictionary response is defined as follows:
{
    "plan_id": "random_str",
    "created_from": "agent",
    "offers": [
        {
            "id": "123",
            "year": 2025,
            "week": 12,
            "category_group": "soda",
            "container": "123-45-45",
            "products": [
                {
                    "product_group": "tasty soda",
                    "product_group_id": "1234",
                    "shelf_price": 23,
                    "promo_price": 23,
                    "weightage": 23,
                    "gauging_unit": "23",
                    "sub_product": "23",
                    "sub_product_id": 23
                },
                {
                    "product_group": "sweet soda",
                    "product_group_id": "1235",
                    "shelf_price": 24,
                    "promo_price": 24,
                    "weightage": 24,
                    "gauging_unit": "24",
                    "sub_product": "24",
                    "sub_product_id": 24
                }
            ]
        },
        {
            "id": "134",
            "year": 2024,
            "week": 14,
            "category_group": "chips",
            "container": "134-45-45",
            "products": [
                {
                    "product_group": "tasty chips",
                    "product_group_id": "1234",
                    "shelf_price": 23,
                    "promo_price": 23,
                    "weightage": 23,
                    "gauging_unit": "23",
                    "sub_product": "23",
                    "sub_product_id": 23
                },
                {
                    "product_group": "spicy chips",
                    "product_group_id": "1235",
                    "shelf_price": 24,
                    "promo_price": 24,
                    "weightage": 24,
                    "gauging_unit": "24",
                    "sub_product": "24",
                    "sub_product_id": 24
                }
            ]
        }
    ]
}
"""
# Examples of good responses can be found here: 
#     input: {agent_example_lib.main_agent["input"]}
#     output: {agent_example_lib.main_agent["output"]}

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
        ),
        agents.function_tool(submit_plan_to_api)
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
