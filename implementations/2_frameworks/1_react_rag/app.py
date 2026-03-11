"""Reason-and-Act Knowledge Retrieval Agent via the OpenAI Agent SDK."""

from typing import Any, AsyncGenerator

import agents
import gradio as gr
from aieng.agents import (
    get_or_create_agent_session,
    oai_agent_stream_to_gradio_messages,
    register_async_cleanup,
    set_up_logging,
)
from aieng.agents.client_manager import AsyncClientManager
from aieng.agents.gradio import get_common_gradio_config
from aieng.agents.prompts import REACT_INSTRUCTIONS
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage


load_dotenv(verbose=True)

# Set logging level and suppress some noisy logs from dependencies
set_up_logging()

# Disable tracing to OpenAI platform since we are using Gemini models instead
# of OpenAI models
agents.set_tracing_disabled(disabled=True)

if gr.NO_RELOAD:
    # Initialize client manager
    # This class initializes the OpenAI and Weaviate async clients, as well as the
    # Weaviate knowledge base tool. The initialization is done once when the clients
    # are first accessed, and the clients are reused for subsequent calls.
    client_manager = AsyncClientManager()

    # Register async cleanup to ensure clients are properly closed on program exit
    register_async_cleanup(client_manager)


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

    # Define an agent using the OpenAI Agent SDK
    main_agent = agents.Agent(
        name="Wikipedia Agent",  # Agent name for logging and debugging purposes
        instructions=REACT_INSTRUCTIONS,  # System instructions for the agent
        # Tools available to the agent
        # We wrap the `search_knowledgebase` method with `function_tool`, which
        # will construct the tool definition JSON schema by extracting the necessary
        # information from the method signature and docstring.
        tools=[agents.function_tool(client_manager.knowledgebase.search_knowledgebase)],
        model=agents.OpenAIChatCompletionsModel(
            model=client_manager.configs.default_worker_model,
            openai_client=client_manager.openai_client,
        ),
    )

    # Run the agent in streaming mode to get and display intermediate outputs
    result_stream = agents.Runner.run_streamed(main_agent, input=query, session=session)

    async for _item in result_stream.stream_events():
        # Parse the stream events, convert to Gradio chat messages and append to
        # the chat history
        turn_messages += oai_agent_stream_to_gradio_messages(_item)
        if len(turn_messages) > 0:
            yield turn_messages


demo = gr.ChatInterface(
    _main,
    **get_common_gradio_config(),
    examples=[
        [
            "At which university did the SVP Software Engineering"
            " at Apple (as of June 2025) earn their engineering degree?"
        ],
    ],
    title="2.1: ReAct for Retrieval-Augmented Generation with OpenAI Agent SDK",
)

if __name__ == "__main__":
    demo.launch(share=True)
