"""Multi-agent Planner-Researcher Setup via OpenAI Agents SDK.

Note: this implementation does not unlock the full potential and flexibility
of LLM agents. Use this reference implementation only if your use case requires
the additional structures, and you are okay with the additional complexities.

Log traces to LangFuse for observability and evaluation.
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
from pydantic import BaseModel

from src.utils import (
    Configs,
    oai_agent_items_to_gradio_messages,
    pretty_print,
    setup_langfuse_tracer,
)
from src.utils.langfuse.shared_client import langfuse_client


load_dotenv(verbose=True)

logging.basicConfig(level=logging.INFO)


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
    print("______________________________________________________")
    print(df.head())
    print("______________________________________________________")
    print(sql)
    print("______________________________________________________")

    # Ensure df is in the environment passed to sqldf
    context = {"df": df}

    # Execute SQL query
    result_df = sqldf(sql, context)
    print(result_df)
    print("______________________________________________________")

    # Convert result to JSON-friendly format
    return result_df.to_dict(orient="records")



PLANNER_INSTRUCTIONS = """\
You are a search planner. \
Given a user's query, produce a list of search steps that can be used to retrieve
relevant information from a sql table, base to answer the user's question. \
These steps will be passed to an sql generation tool that will intern search an sql table as a tool. \
As you are not able to clarify from the user what they are looking for, \
For easch step you genenrate plan what information you might need to get to plan your next steps \
Do not make up information.
Stop after two tries.
Your search tool is an agent searching an sql database containing promotion data for products.
Use that to reason based on the information you get from the search agent/tool.
"""

RESEARCHER_INSTRUCTIONS = (
        "You are a SQL search tool, You receive a single query in natural language as an input." \
        "You are supposed to translate this natural language into an sql query based on the database,"
        "so that the result helps best answer the users question." \
        "If you think the results would not satisfy the user, please reason for the best query you can generate to give back better results."
        "If you do not have the exact product provide the next best product you can get."
        "Do not respond in natural language only respond back by providing the dataframe itself." 
        "However, you can reason in natural language."
        "Generate an SQL query to gather data that can used to analyze the query"
        "Try to keep the data retireval consize without risking leaving out details"
        "Here is the schema of the data base which you will be writing quries againt"
        "The schema defines the column names and the column description"
        "The schema is as follows 'column name': 'description', Use only the column names written before the : to generate the sql"
        "the table name is 'df'"
        "Only pick column names from below"
        "schema start" \
        "column: Description" \
        "Year: Year the offer was promoted," \
        "Week: Week the offer was promoted" \
        "Category_Group: What category the products fall under" \
        "Category_Group_ID: Category group ID for the Category, unique for each category" \
        "Product_Group_ID: Unique ID for each product" \
        "Product_Group: Unique product names, unqiue for each product" \
        "Sub_Product_ID: Sub product id unique for each subproduct" \
        "Sub_Product: Sub product name unique for each sub product" \
        "Gauging_unit: Unit of Gauging the number of unit sold in a pack ABC means each" \
        "Locations: Store IDs for stores/location offering the promotions" \
        "Locations_Name: Store names offering the promotions" \
        "Container: promotion container unique to each promotion" \
        "Advertisement: Amount of advertisement each promotion receives (this is a categorical column containng the following categories: not_promoted, most_promoted, least_promoted, medium_promoted)" \
        "Shelf_Price: Product price on the Shelf in the stores" \
        "Promo_Price: Product price for promotion purposes" \
        "Total_Qty: Total quantity of" \
        "Sales_Qty: No description" \
        "Total_Revenue: No description" \
        "Revenue: No description" \
        "Margin	Weightage: No description" \
        "Sales_Lift: No description" \
        "Margin_Lift: No description" \
        "Weightage_Lift: No description" \
        "schema end" \
        "Use the search_historical_data tool to run this SQL query you generated and return the output dataframe"
)

WRITER_INSTRUCTIONS = """\
You are an expert at synthesizing information and writing coherent reports. \
Given a user's query and a set of search summaries, synthesize these into a \
coherent report (at least a few paragraphs long) that answers the user's question. \
Do not make up any information outside of the search summaries.
"""


class SearchItem(BaseModel):
    """A single search item in the search plan."""

    # The search step to be used in the knowledge base search
    search_step: str

    # A description of the search step and its relevance to the query
    reasoning: str


class SearchPlan(BaseModel):
    """A search plan containing multiple search items."""

    search_steps: list[SearchItem]

    def __str__(self) -> str:
        """Return a string representation of the search plan."""
        return "\n".join(
            f"search_step: {step.search_step}\nReasoning: {step.reasoning}\n"
            for step in self.search_steps
        )


class ResearchReport(BaseModel):
    """Model for the final report generated by the writer agent."""

    # The summary of the research findings
    summary: str

    # full report text
    full_report: str


async def _create_search_plan(planner_agent: agents.Agent, query: str) -> SearchPlan:
    """Create a search plan using the planner agent."""
    with langfuse_client.start_as_current_span(
        name="create_search_plan", input=query
    ) as planner_span:
        response = await agents.Runner.run(planner_agent, input=query)
        search_plan = response.final_output_as(SearchPlan)
        planner_span.update(output=search_plan)

    return search_plan


async def _generate_final_report(
    writer_agent: agents.Agent, search_results: list[str], query: str
) -> agents.RunResult:
    """Generate the final report using the writer agent."""
    input_data = f"Original question: {query}\n"
    input_data += "Search summaries:\n" + "\n".join(
        f"{i + 1}. {result}" for i, result in enumerate(search_results)
    )

    with langfuse_client.start_as_current_span(
        name="generate_final_report", input=input_data
    ) as writer_span:
        response = await agents.Runner.run(writer_agent, input=input_data)
        writer_span.update(output=response.final_output)

    return response


async def _cleanup_clients() -> None:
    """Close async clients."""
    await async_weaviate_client.close()
    await async_openai_client.close()


def _handle_sigint(signum: int, frame: object) -> None:
    """Handle SIGINT signal to gracefully shutdown."""
    with contextlib.suppress(Exception):
        asyncio.get_event_loop().run_until_complete(_cleanup_clients())
    sys.exit(0)


async def _main(question: str, gr_messages: list[ChatMessage]):
    planner_agent = agents.Agent(
        name="Planner Agent",
        instructions=PLANNER_INSTRUCTIONS,
        model=agents.OpenAIChatCompletionsModel(
            model="gemini-2.5-flash", openai_client=async_openai_client
        ),
        output_type=SearchPlan,
    )
    research_agent = agents.Agent(
        name="Research Agent",
        instructions=RESEARCHER_INSTRUCTIONS,
        tools=[agents.function_tool(search_historical_data)],
        model=agents.OpenAIChatCompletionsModel(
            model="gemini-2.5-flash",
            openai_client=async_openai_client,
        ),
        model_settings=agents.ModelSettings(tool_choice="required"),
    )
    writer_agent = agents.Agent(
        name="Writer Agent",
        instructions=WRITER_INSTRUCTIONS,
        model=agents.OpenAIChatCompletionsModel(
            model="gemini-2.5-flash", openai_client=async_openai_client
        ),
        output_type=ResearchReport,
    )

    gr_messages.append(ChatMessage(role="user", content=question))
    yield gr_messages

    with langfuse_client.start_as_current_span(
        name="Multi-Agent-Trace", input=question
    ) as agents_span:
        # Create a search plan
        search_plan = await _create_search_plan(planner_agent, question)
        gr_messages.append(
            ChatMessage(role="assistant", content=f"Search Plan:\n{search_plan}")
        )
        pretty_print(gr_messages)
        yield gr_messages

        # Execute the search plan
        search_results = []
        for step in search_plan.search_steps:
            with langfuse_client.start_as_current_span(
                name="execute_search_step", input=step.search_step
            ) as search_span:
                response = await agents.Runner.run(
                    research_agent, input=step.search_step
                )
                search_result: str = response.final_output
                search_span.update(output=search_result)

            search_results.append(search_result)
            gr_messages += oai_agent_items_to_gradio_messages(response.new_items)
            yield gr_messages

        # Generate the final report
        writer_agent_response = await _generate_final_report(
            writer_agent, search_results, question
        )
        agents_span.update(output=writer_agent_response.final_output)

        report = writer_agent_response.final_output_as(ResearchReport)
        gr_messages.append(
            ChatMessage(
                role="assistant",
                content=f"Summary:\n{report.summary}\n\nFull Report:\n{report.full_report}",
            )
        )
        pretty_print(gr_messages)
        yield gr_messages


if __name__ == "__main__":
    configs = Configs.from_env_var()

    async_openai_client = AsyncOpenAI()
    setup_langfuse_tracer()

    with gr.Blocks(title="OAI Agent SDK - Multi-agent") as app:
        chatbot = gr.Chatbot(type="messages", label="Agent", height=600)
        chat_message = gr.Textbox(lines=1, label="Ask a question")
        chat_message.submit(_main, [chat_message, chatbot], [chatbot])

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        app.launch(server_name="0.0.0.0")
    finally:
        asyncio.run(_cleanup_clients())
