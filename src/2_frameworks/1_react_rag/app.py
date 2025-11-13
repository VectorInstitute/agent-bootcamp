"""Reason-and-Act Knowledge Retrieval Agent via the OpenAI Agent SDK."""

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

from src.prompts import REACT_INSTRUCTIONS
from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    get_weaviate_async_client,
    oai_agent_stream_to_gradio_messages,
)


# Load env vars
load_dotenv(verbose=True)


logging.basicConfig(level=logging.INFO)


AGENT_LLM_NAME = "gemini-2.5-flash"

# Config using env vars
configs = Configs.from_env_var()
# Instantiate the Weaviate client (Knowledge bases are stored)
async_weaviate_client = get_weaviate_async_client(
    http_host=configs.weaviate_http_host,
    http_port=configs.weaviate_http_port,
    http_secure=configs.weaviate_http_secure,
    grpc_host=configs.weaviate_grpc_host,
    grpc_port=configs.weaviate_grpc_port,
    grpc_secure=configs.weaviate_grpc_secure,
    api_key=configs.weaviate_api_key,
)

# Instantiate OpenAi client
async_openai_client = AsyncOpenAI()

# Switch out collection name when ready
async_knowledgebase = AsyncWeaviateKnowledgeBase(
    async_weaviate_client,
    collection_name="enwiki_20250520",
)


# Close clients appropraitely
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
    # Build agent with tool access to knowledge base
    main_agent = agents.Agent(
        name="Wikipedia Agent",  # Documentation
        instructions=REACT_INSTRUCTIONS,  # system prompt
        tools=[
            agents.function_tool(async_knowledgebase.search_knowledgebase)
        ],  # tools passed to agent
        model=agents.OpenAIChatCompletionsModel(  # Which client to use
            model=AGENT_LLM_NAME, openai_client=async_openai_client
        ),
    )

    # Running agent as a stream so that can pass it gradio
    result_stream = agents.Runner.run_streamed(main_agent, input=question)

    # For every event in the stream
    async for _item in result_stream.stream_events():
        gr_messages += oai_agent_stream_to_gradio_messages(
            _item
        )  # Custom function to recognize class of streamed event and formulate it appropirately
        if len(gr_messages) > 0:
            yield gr_messages


demo = gr.ChatInterface(
    _main,
    title="2.1 OAI Agent SDK ReAct",  # See title on page
    type="messages",
    examples=[  # Pre-filled suggestions in the interface
        "At which university did the SVP Software Engineering"
        " at Apple (as of June 2025) earn their engineering degree?",
    ],
)


if __name__ == "__main__":
    configs = Configs.from_env_var()
    async_weaviate_client = get_weaviate_async_client(
        http_host=configs.weaviate_http_host,
        http_port=configs.weaviate_http_port,
        http_secure=configs.weaviate_http_secure,
        grpc_host=configs.weaviate_grpc_host,
        grpc_port=configs.weaviate_grpc_port,
        grpc_secure=configs.weaviate_grpc_secure,
        api_key=configs.weaviate_api_key,
    )
    async_knowledgebase = AsyncWeaviateKnowledgeBase(
        async_weaviate_client,
        collection_name="enwiki_20250520",
    )

    async_openai_client = AsyncOpenAI()

    # OpenAI has a built-in tracing mechanism we have to turn off
    agents.set_tracing_disabled(disabled=True)

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        demo.launch(share=True)  # Launch gradio app
    finally:
        asyncio.run(_cleanup_clients())
