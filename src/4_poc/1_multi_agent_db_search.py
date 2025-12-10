import asyncio
import contextlib
import signal
import sys
from pathlib import Path
import logging

from agents import Agent
from agents import function_tool
from agents import OpenAIChatCompletionsModel
from agents import RunConfig
from agents import Runner

# import gradio as gr
from dotenv import load_dotenv
# from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI

from src.utils import (
    CodeInterpreter,
    # AsyncWeaviateKnowledgeBase,
    Configs,
    # get_weaviate_async_client,
    # oai_agent_stream_to_gradio_messages,
    set_up_logging,
    pretty_print,
    # setup_langfuse_tracer,
)
# from src.utils.langfuse.shared_client import langfuse_client
# from src.utils.tools.gemini_grounding import (
#     GeminiGroundingWithGoogleSearch,
#     ModelSettings,
# )


load_dotenv(verbose=True)

AGENT_LLM_NAMES = {
    "worker": "gemini-2.5-flash",  # less expensive,
    "planner": "gemini-2.5-pro",  # more expensive, better at reasoning and planning
}


# uv run --env-file .env ./src/4_poc/1_multi_agent_db_search.py


configs = Configs.from_env_var()
async_openai_client = AsyncOpenAI()
code_interpreter = CodeInterpreter(template_name="lobsuu8phhb16r4r04yr", local_files=[])
no_tracing_config = RunConfig(tracing_disabled=True)


# agents
general_query_agent = Agent(
    name="GeneralQuery",
    instructions="""
        The `code_interpreter` tool executes Python commands.
        Please note that data is not persisted. Each time you invoke this tool,
        you will need to run import and define all variables from scratch.

        You will be asked a general question about the database. Write a Python
        command to answer the question, then pass that command to the `code_interpreter`
        tool. Use sqlite3 to query the database.

        The database is a sqlite3 file located at /data/data.db.

        Here are brief descriptions of all the tables in the database.
        'transactions' - history of credit card transactions.
        'cards' - details for each credit card.
        'users' - details for each user.
        'mcc_codes' - maps merchant category codes to their plain text descriptions.

        Do not invent transaction data.
        Do not invent card data.
        Do not invent user data.
    """,
    tools=[
        function_tool(
            code_interpreter.run_code,
            name_override="code_interpreter",
        )
    ],
    # a faster, smaller model for quick searches
    model=OpenAIChatCompletionsModel(
        model=AGENT_LLM_NAMES["worker"], openai_client=async_openai_client
    ),
)

client_expert_agent = Agent(
    name="ClientExpert",
    instructions="""
        The `code_interpreter` tool executes Python commands.
        Please note that data is not persisted. Each time you invoke this tool,
        you will need to run import and define all variables from scratch.

        You will be asked to return information about a client. Write a Python
        command to answer the question, then pass that command to the `code_interpreter`
        tool. Use sqlite3 to query the database.

        The database is a sqlite3 file located at /data/data.db.  Query the 'users'
        table for client information.

        If you are only given a client ID, return all available information for that
        client.

        This is the schema of the 'users' table. It can help you write your query.
        (
            "client_id" INTEGER,
            "current_age" INTEGER,
            "retirement_age" INTEGER,
            "birth_year" INTEGER,
            "birth_month" INTEGER,
            "gender" TEXT,
            "address" TEXT,
            "latitude" REAL,
            "longitude" REAL,
            "per_capita_income" TEXT,
            "yearly_income" TEXT,
            "total_debt" TEXT,
            "credit_score" INTEGER,
            "num_credit_cards" INTEGER
        )
    """,
    tools=[
        function_tool(
            code_interpreter.run_code,
            name_override="code_interpreter",
        )
    ],
    # a faster, smaller model for quick searches
    model=OpenAIChatCompletionsModel(
        model=AGENT_LLM_NAMES["worker"], openai_client=async_openai_client
    ),
)

transaction_expert_agent = Agent(
    name="TransactionExpert",
    instructions="""
        The `code_interpreter` tool executes Python commands.
        Please note that data is not persisted. Each time you invoke this tool,
        you will need to run import and define all variables from scratch.

        You will be asked to return information about credit card transations. 
        Write a Python command to answer the question, then pass that command to the
        `code_interpreter` tool. Use sqlite3 to query the database.

        The database is a sqlite3 file located at /data/data.db.  Query the
        'transactions' table for a history of all transactions.

        You will be given one or more card IDs, client IDs or transaction IDs.
        Please return all transactions with matching IDs.

        This is the schema of the 'transactions' table. It can help you write your query.
        (
            "transaction_id" INTEGER,
            "date" TEXT,
            "client_id" INTEGER,
            "card_id" INTEGER,
            "amount" TEXT,
            "use_chip" TEXT,
            "merchant_id" INTEGER,
            "merchant_city" TEXT,
            "merchant_state" TEXT,
            "zip" REAL,
            "mcc" INTEGER,
            "errors" TEXT
        )
    """,
    tools=[
        function_tool(
            code_interpreter.run_code,
            name_override="code_interpreter",
        )
    ],
    # a faster, smaller model for quick searches
    model=OpenAIChatCompletionsModel(
        model=AGENT_LLM_NAMES["worker"], openai_client=async_openai_client
    ),
)

card_expert_agent = Agent(
    name="CardExpert",
    instructions="""
        The `code_interpreter` tool executes Python commands.
        Please note that data is not persisted. Each time you invoke this tool,
        you will need to run import and define all variables from scratch.

        You will be asked to return information about credit cards. 
        Write a Python command to answer the question, then pass that command to the
        `code_interpreter` tool. Use sqlite3 to query the database.

        The database is a sqlite3 file located at /data/data.db. Query the 'cards'
        table for credit card information, like the card IDs of cards owned by a
        given client.

        You will be given one or more card IDs or client IDs. Please return all details
        of the cards with matching IDs.

        This is the schema of the 'cards' table. It can help you write your query.
        (
            "card_id" INTEGER,
            "client_id" INTEGER,
            "card_brand" TEXT,
            "card_type" TEXT,
            "card_number" INTEGER,
            "expires" TEXT,
            "cvv" INTEGER,
            "has_chip" TEXT,
            "num_cards_issued" INTEGER,
            "credit_limit" TEXT,
            "acct_open_date" TEXT,
            "year_pin_last_changed" INTEGER,
            "card_on_dark_web" TEXT
        )
    """,
    tools=[
        function_tool(
            code_interpreter.run_code,
            name_override="code_interpreter",
        )
    ],
    # a faster, smaller model for quick searches
    model=OpenAIChatCompletionsModel(
        model=AGENT_LLM_NAMES["worker"], openai_client=async_openai_client
    ),
)

main_agent = Agent(
    name="MainAgent",
    instructions="""
        You are a deep research agent and your goal is to conduct in-depth, multi-turn
        research by breaking down complex queries, using the provided tools, and
        synthesizing the information into a comprehensive report.

        You have access to the following tools:
        1. 'client_expert' - use this tool to surface demographic details about a 
            given client. Pass this tool a concise, natural language query containing
            all client ID information.
        2. 'transaction_expert' - use this tool to surface details about credit card
            transactions. Pass this tool a concise, natural language query containing
            one or more card ID, client ID, or transaction ID.
        3. 'card_expert' - use this tool to surface details about credit cards,
            including ownership of credit cards. Pass this tool a concise, natural
            language query containing one or more card ID or client ID.
        4. 'general_query' - use this tool for general queries of the bank's database.
            Pass this tool a concise, natural language query.

        These tools will not return raw search results or the sources themselves.
        Instead, they will return a concise summary of the key findings.

        For best performance, divide complex queries into simpler sub-queries
        Before calling a tool, always explain your reasoning for doing so.

        **Routing Guidelines:**
        - When answering a question, you should first try to use the 'general_query'
          tool, unless the question requires information about a specific client.
        - If a tool returns insufficient information for a given query, try
          reformulating or using another tool. You can call a tool multiple
          times to get the information you need to answer the user's question.

        **Guidelines for synthesis**
        - After collecting results, write the final answer from your own synthesis.
        - Do not invent transaction data.
        - Do not invent card data.
        - Do not invent user data.
        - If all tools fail, say so and suggest 2-3 refined queries.

        Do not make up information.
    """,
    # Allow the planner agent to invoke the worker agent.
    # The long context provided to the worker agent is hidden from the main agent.
    tools=[
        client_expert_agent.as_tool(
            tool_name="client_expert",
            tool_description=(
                "Given a client ID, search the bank's database for information "
                "about that client."
            ),
        ),
        transaction_expert_agent.as_tool(
            tool_name="transaction_expert",
            tool_description=(
                "Given a set of card IDs, client IDs, or transaction IDs, search "
                "the bank's database for details of those transactions with "
                "matching IDs."
            ),
        ),
        card_expert_agent.as_tool(
            tool_name="card_expert",
            tool_description=(
                "Given a set of card IDs, or client IDs, search the bank's "
                "database for details of those cards with matching IDs. Use this "
                "tool to find card IDs of cards owned by a given client."
            ),
        ),
        general_query_agent.as_tool(
            tool_name="general_query",
            tool_description=(
                "Query the bank's database for general information and return a "
                "concise summary of the key findings."
            ),
        ),
    ],
    # a larger, more capable model for planning and reasoning over summaries
    model=OpenAIChatCompletionsModel(
        model=AGENT_LLM_NAMES["planner"], openai_client=async_openai_client
    ),
)


async def main(query: str) -> None:
    response = await Runner.run(
        main_agent,
        input=query,
        run_config=no_tracing_config,
    )

    for item in response.new_items:
        pretty_print(item.raw_item)
        print()

    # pretty_print(response.final_output)

    await async_openai_client.close()


if __name__ == "__main__":
    # query = ("What tables exist in the bank's database?")
    query = ("Tell me everything you know about the client with client ID 379")

    logging.basicConfig(level=logging.INFO)
    asyncio.run(main(query))
