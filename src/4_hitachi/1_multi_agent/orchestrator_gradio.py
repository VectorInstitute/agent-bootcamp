"""Example code for orchestrator-worker agent collaboration.

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

from src.prompts import REACT_INSTRUCTIONS, KB_SEARCH_INSTRUCTIONS
from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    get_weaviate_async_client,
    oai_agent_stream_to_gradio_messages,
    setup_langfuse_tracer,
)
from src.utils.langfuse.shared_client import langfuse_client


load_dotenv(verbose=True)


logging.basicConfig(level=logging.INFO)

DATASET_NAME = "hitachi-multi-agent-orchestrator"
AGENT_LLM_NAMES = {
    "worker": "gemini-2.5-flash",  # less expensive,
    "planner": "gemini-2.5-pro",  # more expensive, better at reasoning and planning
}

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
async_openai_client = AsyncOpenAI()
async_knowledgebase = AsyncWeaviateKnowledgeBase(
    async_weaviate_client,
    collection_name="enwiki_20250520",
)


async def _cleanup_clients() -> None:
    """Close async clients."""
    await async_weaviate_client.close()
    await async_openai_client.close()


def _handle_sigint(signum: int, frame: object) -> None:
    """Handle SIGINT signal to gracefully shutdown."""
    with contextlib.suppress(Exception):
        asyncio.get_event_loop().run_until_complete(_cleanup_clients())
    sys.exit(0)


# Knowledgebase Search Agent: a simple agent that searches the knowledge base
knowledgebase_agent = agents.Agent(
    name="KnowledgeBaseSearchAgent",
    instructions=KB_SEARCH_INSTRUCTIONS,
    tools=[
        agents.function_tool(async_knowledgebase.search_knowledgebase),
    ],
    # a faster, smaller model for quick searches
    model=agents.OpenAIChatCompletionsModel(
        model="gemini-2.5-flash", openai_client=async_openai_client
    ),
)

# Main Agent: more expensive and slower, but better at complex planning
orchestrator_agent = agents.Agent(
    name="OrchestratorAgent",
    instructions=REACT_INSTRUCTIONS,

    # Allow the planner agent to invoke the worker agent.
    # The long context provided to the worker agent is hidden from the main agent.
    tools=[
        knowledgebase_agent.as_tool(
            tool_name="KnowledgeBaseSearchAgent",
            tool_description="Perform a search in the knowledge base and return a concise answer.",
        )
    ],
    # a larger, more capable model for planning and reasoning over summaries
    model=agents.OpenAIChatCompletionsModel(
        model="gemini-2.5-pro", openai_client=async_openai_client
    ),
)


async def _main(question: str, gr_messages: list[ChatMessage]):
    setup_langfuse_tracer()

    # Use the main agent as the entry point- not the worker agent.
    with langfuse_client.start_as_current_span(name="Agents-SDK-Trace") as span:
        span.update(input=question)

        result_stream = agents.Runner.run_streamed(orchestrator_agent, input=question)
        async for _item in result_stream.stream_events():
            gr_messages += oai_agent_stream_to_gradio_messages(_item)
            if len(gr_messages) > 0:
                yield gr_messages

        span.update(output=result_stream.final_output)


demo = gr.ChatInterface(
    _main,
    title="Hitachi Multi-Agent Knowledge Retrieval System",
    type="messages",
    examples=[
        "What city are George Washington University Hospital"
        " and MedStar Washington Hospital Center located in?"
    ],
)


if __name__ == "__main__":
    async_openai_client = AsyncOpenAI()

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        demo.launch(server_name="0.0.0.0")
    finally:
        asyncio.run(_cleanup_clients())
