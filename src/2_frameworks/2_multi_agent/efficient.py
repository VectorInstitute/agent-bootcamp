"""Example code for planner-worker agent collaboration.

With reference to:

github.com/ComplexData-MILA/misinfo-datasets
/blob/3304e6e/misinfo_data_eval/tasks/web_search.py
"""

import asyncio
from typing import Any, AsyncGenerator

import agents
import gradio as gr
from aieng.agents import (
    get_or_create_agent_session,
    oai_agent_stream_to_gradio_messages,
    set_up_logging,
)
from aieng.agents.client_manager import AsyncClientManager
from aieng.agents.gradio import COMMON_GRADIO_CONFIG
from aieng.agents.langfuse import langfuse_client, setup_langfuse_tracer
from aieng.agents.prompts import REACT_INSTRUCTIONS, SEARCH_AGENT_INSTRUCTIONS
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from langfuse import propagate_attributes


async def _main(
    query: str, history: list[ChatMessage], session_state: dict[str, Any]
) -> AsyncGenerator[list[ChatMessage], Any]:
    # Initialize list of chat messages for a single turn
    turn_messages: list[ChatMessage] = []

    # Construct an in-memory SQLite session for the agent to maintain
    # conversation history across multiple turns of a chat
    # This makes it possible to ask follow-up questions that refer to
    # previous turns in the conversation
    session = get_or_create_agent_session(history, session_state)

    # Use the main agent as the entry point- not the worker agent.
    with (
        langfuse_client.start_as_current_observation(
            name="Orchestrator-Worker", as_type="agent", input=query
        ) as obs,
        propagate_attributes(
            session_id=session.session_id  # Propagate session_id to all child observations
        ),
    ):
        # Run the agent in streaming mode to get and display intermediate outputs
        result_stream = agents.Runner.run_streamed(
            main_agent,
            input=query,
            session=session,
            max_turns=30,  # Increase max turns to support more complex queries
        )

        async for _item in result_stream.stream_events():
            turn_messages += oai_agent_stream_to_gradio_messages(_item)
            if len(turn_messages) > 0:
                yield turn_messages

        obs.update(output=result_stream.final_output)


if __name__ == "__main__":
    load_dotenv(verbose=True)

    # Set logging level and suppress some noisy logs from dependencies
    set_up_logging()

    # Set up LangFuse for tracing
    setup_langfuse_tracer()

    # Initialize client manager
    # This class initializes the OpenAI and Weaviate async clients, as well as the
    # Weaviate knowledge base tool. The initialization is done once when the clients
    # are first accessed, and the clients are reused for subsequent calls.
    client_manager = AsyncClientManager()

    # Use smaller, faster model for focused search tasks
    worker_model = client_manager.configs.default_worker_model
    # Use larger, more capable model for complex planning and reasoning
    planner_model = client_manager.configs.default_planner_model

    # Worker Agent: handles long context efficiently
    search_agent = agents.Agent(
        name="SearchAgent",
        instructions=SEARCH_AGENT_INSTRUCTIONS,
        tools=[
            agents.function_tool(client_manager.knowledgebase.search_knowledgebase),
        ],
        # a faster, smaller model for quick searches
        model=agents.OpenAIChatCompletionsModel(
            model=worker_model, openai_client=client_manager.openai_client
        ),
    )

    # Main Agent: more expensive and slower, but better at complex planning
    main_agent = agents.Agent(
        name="MainAgent",
        instructions=REACT_INSTRUCTIONS,
        # Allow the planner agent to invoke the worker agent.
        # The long context provided to the worker agent is hidden from the main agent.
        tools=[
            search_agent.as_tool(
                tool_name="search_knowledgebase",
                tool_description="Perform a search on a Wikipedia knowledge base for a query and return a concise summary.",
            )
        ],
        # a larger, more capable model for planning and reasoning over summaries
        model=agents.OpenAIChatCompletionsModel(
            model=planner_model, openai_client=client_manager.openai_client
        ),
        # NOTE: enabling parallel tool calls here can sometimes lead to issues with
        # with invalid arguments being passed to the search agent.
        model_settings=agents.ModelSettings(parallel_tool_calls=False),
    )

    demo = gr.ChatInterface(
        _main,
        **COMMON_GRADIO_CONFIG,
        examples=[
            [
                "Write a structured report on the history of AI, covering: "
                "1) the start in the 50s, 2) the first AI winter, 3) the second AI winter, "
                "4) the modern AI boom, 5) the evolution of AI hardware, and "
                "6) the societal impacts of modern AI"
            ],
            [
                "Compare the box office performance of 'Oppenheimer' with the third Avatar movie"
            ],
        ],
        title="2.2.2: Multi-Agent Orchestrator-worker for Retrieval-Augmented Generation",
    )

    try:
        demo.launch(share=True)
    finally:
        asyncio.run(client_manager.close())
