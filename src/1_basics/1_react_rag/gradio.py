"""Reason-and-Act Knowledge Retrieval Agent, no framework.

With reference to huggingface.co/spaces/gradio/langchain-agent
"""

import asyncio
import contextlib
import json
import signal
import sys

import gradio as gr
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionToolParam

from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    get_weaviate_async_client,
)


load_dotenv(verbose=True)

MAX_TURNS = 5
AGENT_LLM_NAME = "gemini-2.5-flash"

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

tools: list["ChatCompletionToolParam"] = [
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Get references on the specified topic from the English Wikipedia.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": ("Keyword for the search e.g. Apple TV"),
                    }
                },
                "required": ["keyword"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

system_message: "ChatCompletionSystemMessageParam" = {
    "role": "system",
    "content": (
        "Answer the question using the search tool. "
        "You must explain your reasons for invoking the tool. "
        "Be sure to mention the sources. "
        "If the search did not return intended results, try again. "
        "Do not make up information. You must use the search tool "
        "for all facts that might change over time."
    ),
}


async def _cleanup_clients() -> None:
    """Close async clients."""
    await async_weaviate_client.close()
    await async_openai_client.close()


def _handle_sigint(signum: int, frame: object) -> None:
    """Handle SIGINT signal to gracefully shutdown."""
    with contextlib.suppress(Exception):
        asyncio.get_event_loop().run_until_complete(_cleanup_clients())
    sys.exit(0)


async def react_rag(query: str, history: list[ChatMessage]):
    """Handles ReAct RAG chat for knowledgebase-augmented agents."""
    oai_messages = [system_message, {"role": "user", "content": query}]

    for _ in range(MAX_TURNS):
        completion = await async_openai_client.chat.completions.create(
            model=AGENT_LLM_NAME,
            messages=oai_messages,
            tools=tools,
        )

        # Print assistant output
        message = completion.choices[0].message
        oai_messages.append(message)

        # Execute tool calls and send results back to LLM if requested.
        # Otherwise, stop, as the conversation would have been finished.
        tool_calls = message.tool_calls
        if tool_calls is None:
            history.append(
                ChatMessage(
                    content=message.content or "",
                    role="assistant",
                )
            )
            yield history
            break

        for tool_call in tool_calls:
            arguments = json.loads(tool_call.function.arguments)
            results = await async_knowledgebase.search_knowledgebase(
                arguments["keyword"]
            )
            results_serialized = json.dumps(
                [_result.model_dump() for _result in results]
            )

            oai_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": results_serialized,
                }
            )
            history.append(
                ChatMessage(
                    role="assistant",
                    content=results_serialized,
                    metadata={
                        "title": f"Used tool {tool_call.function.name}",
                        "log": f"Arguments: {arguments}",
                    },
                )
            )
            yield history


demo = gr.ChatInterface(
    react_rag,
    title="1.1 ReAct Agent for Retrieval-Augmented Generation",
    type="messages",
    examples=[
        "At which university did the SVP Software Engineering"
        " at Apple (as of June 2025) earn their engineering degree?",
    ],
)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        demo.launch(server_name="0.0.0.0")
    finally:
        asyncio.run(_cleanup_clients())
