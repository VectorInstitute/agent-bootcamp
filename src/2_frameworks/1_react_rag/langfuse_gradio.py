"""Reason-and-Act Knowledge Retrieval Agent via the OpenAI Agent SDK.

Log traces to LangFuse for observability and evaluation.
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

from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    get_weaviate_async_client,
    oai_agent_items_to_gradio_messages,
    pretty_print,
    setup_langfuse_tracer,
)
from src.utils.langfuse.shared_client import langfuse as langfuse_client


load_dotenv(verbose=True)

logging.basicConfig(level=logging.INFO)

SYSTEM_MESSAGE = """\
Answer the question using the search tool. \
EACH TIME before invoking the function, you must explain your reasons for doing so. \
Be sure to mention the sources in your response. \
If the search did not return intended results, try again. \
Do not make up information."""


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
    setup_langfuse_tracer()

    main_agent = agents.Agent(
        name="Wikipedia Agent",
        instructions=SYSTEM_MESSAGE,
        tools=[agents.function_tool(async_knowledgebase.search_knowledgebase)],
        model=agents.OpenAIChatCompletionsModel(
            model="gemini-2.5-flash-lite-preview-06-17",
            openai_client=async_openai_client,
        ),
    )
    gr_messages.append(ChatMessage(role="user", content=question))
    yield gr_messages

    with langfuse_client.start_as_current_span(
        name="Agents-SDK-Trace", input=question
    ) as span:
        responses = await agents.Runner.run(main_agent, input=question)
        span.update_trace(output=responses.final_output)

    gr_messages += oai_agent_items_to_gradio_messages(responses.new_items)
    pretty_print(gr_messages)
    yield gr_messages


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
    agents.set_tracing_disabled(disabled=True)

    with gr.Blocks(title="OAI Agent SDK ReAct") as app:
        chatbot = gr.Chatbot(type="messages", label="Agent")
        chat_message = gr.Textbox(lines=1, label="Ask a question")
        chat_message.submit(_main, [chat_message, chatbot], [chatbot])

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        app.launch(server_name="0.0.0.0")
    finally:
        asyncio.run(_cleanup_clients())
