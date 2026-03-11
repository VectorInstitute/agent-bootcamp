"""Git Agent via the git MCP server with OpenAI Agent SDK.

Log traces to LangFuse for observability and evaluation.
"""

import asyncio
import subprocess
from typing import Any, AsyncGenerator

import agents
import gradio as gr
from agents.mcp import MCPServerStdio, create_static_tool_filter
from aieng.agents import (
    get_or_create_agent_session,
    oai_agent_stream_to_gradio_messages,
    pretty_print,
    set_up_logging,
)
from aieng.agents.client_manager import AsyncClientManager
from aieng.agents.gradio import COMMON_GRADIO_CONFIG
from aieng.agents.langfuse import langfuse_client, setup_langfuse_tracer
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from langfuse import propagate_attributes


async def _main(
    query: str, history: list[ChatMessage], session_state: dict[str, Any]
) -> AsyncGenerator[list[ChatMessage], Any]:
    """Initialize MCP Git server and run the agent."""
    # Initialize list of chat messages for a single turn
    turn_messages: list[ChatMessage] = []

    # Construct an in-memory SQLite session for the agent to maintain
    # conversation history across multiple turns of a chat
    # This makes it possible to ask follow-up questions that refer to
    # previous turns in the conversation
    session = get_or_create_agent_session(history, session_state)

    # Get the absolute path to the current git repository, regardless of where
    # the script is run from
    repo_path = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()

    with (
        langfuse_client.start_as_current_observation(
            name="Git-Agent", as_type="agent", input=query
        ) as obs,
        propagate_attributes(
            session_id=session.session_id  # Propagate session_id to all child observations
        ),
    ):
        async with MCPServerStdio(
            name="Git server",
            params={
                "command": "uvx",
                "args": ["mcp-server-git"],
            },
            tool_filter=create_static_tool_filter(
                allowed_tool_names=["git_status", "git_log"]
            ),
        ) as mcp_server:
            agent = agents.Agent(
                name="Git Assistant",
                instructions=f"Answer questions about the git repository at {repo_path}, use that for repo_path",
                mcp_servers=[mcp_server],
                model=agents.OpenAIChatCompletionsModel(
                    model=client_manager.configs.default_planner_model,
                    openai_client=client_manager.openai_client,
                ),
            )

            result_stream = agents.Runner.run_streamed(
                agent, input=query, session=session
            )
            async for _item in result_stream.stream_events():
                turn_messages += oai_agent_stream_to_gradio_messages(_item)
                if len(turn_messages) > 0:
                    yield turn_messages

        obs.update(output=result_stream.final_output)

    pretty_print(turn_messages)
    yield turn_messages

    # Clear the turn messages after yielding to prepare for the next turn
    turn_messages.clear()


if __name__ == "__main__":
    load_dotenv(verbose=True)

    set_up_logging()
    setup_langfuse_tracer()

    # Initialize client manager
    # This class initializes the OpenAI and Weaviate async clients, as well as the
    # Weaviate knowledge base tool. The initialization is done once when the clients
    # are first accessed, and the clients are reused for subsequent calls.
    client_manager = AsyncClientManager()

    demo = gr.ChatInterface(
        _main,
        **COMMON_GRADIO_CONFIG,
        examples=[
            ["Summarize the last change in the repository."],
            ["How many branches currently exist on the remote?"],
        ],
        title="2.4 OAI Agent SDK + Git MCP Server",
    )

    try:
        demo.launch(share=True)
    finally:
        asyncio.run(client_manager.close())
