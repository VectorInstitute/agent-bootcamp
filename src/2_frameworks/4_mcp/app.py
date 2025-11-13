"""Git Agent via the git MCP server with OpenAI Agent SDK.

Log traces to LangFuse for observability and evaluation.
"""

import asyncio
import contextlib
import os
import signal
import sys

import agents
import gradio as gr
from agents.mcp import MCPServerStdio, create_static_tool_filter
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI

from src.utils import (
    Configs,
    oai_agent_stream_to_gradio_messages,
    pretty_print,
    set_up_logging,
    setup_langfuse_tracer,
)
from src.utils.langfuse.shared_client import langfuse_client


load_dotenv(verbose=True)

set_up_logging()

AGENT_LLM_NAME = "gemini-2.5-flash"

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


async def _main(question: str, gr_messages: list[ChatMessage]):
    """Initialize MCP Git server and run the agent."""
    setup_langfuse_tracer()

    repo_path = os.path.abspath("/home/coder/agent-bootcamp")

    with langfuse_client.start_as_current_span(name="Agents-SDK-Trace") as span:
        span.update(input=question)

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
                    model=AGENT_LLM_NAME, openai_client=async_openai_client
                ),
            )

            result_stream = agents.Runner.run_streamed(agent, input=question)
            async for _item in result_stream.stream_events():
                gr_messages += oai_agent_stream_to_gradio_messages(_item)
                if len(gr_messages) > 0:
                    yield gr_messages

        span.update(output=result_stream.final_output)

    pretty_print(gr_messages)
    yield gr_messages


demo = gr.ChatInterface(
    _main,
    title="2.1 OAI Agent SDK MCP",
    type="messages",
    examples=[
        "Summarize the last change in the repository.",
        "How many branches currently exist on the remote?",
    ],
)


if __name__ == "__main__":
    configs = Configs.from_env_var()

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        demo.launch(share=True)
    finally:
        asyncio.run(_cleanup_clients())
