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
    gradio_messages_to_oai_chat,
)


load_dotenv(verbose=True)

MAX_TURNS = 5

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
        "Do not make up information."
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


async def _main(question: str, gr_message_history: list[ChatMessage]):
    gr_message_history = [
        ChatMessage(**msg) if isinstance(msg, dict) else msg
        for msg in gr_message_history
    ]

    gr_message_history.append(ChatMessage(role="user", content=question))
    yield gr_message_history

    oai_messages = [
        system_message,
        *gradio_messages_to_oai_chat(gr_message_history),
    ]

    for _ in range(MAX_TURNS):
        completion = await async_openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=oai_messages,
            tools=tools,
        )

        # Add assistant output
        message = completion.choices[0].message
        oai_messages.append(message)  # type: ignore[arg-type]
        gr_message_history.append(
            ChatMessage(
                content=message.content or "",
                role="assistant",
            )
        )
        yield gr_message_history

        # Execute function calls if requested.
        tool_calls = message.tool_calls
        if tool_calls is not None:
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
                gr_message_history.append(
                    ChatMessage(
                        role="assistant",
                        content=results_serialized,
                        metadata={
                            "title": f"Used tool {tool_call.function.name}",
                            "log": f"Arguments: {arguments}",
                        },
                    )
                )
                yield gr_message_history

        else:
            break


if __name__ == "__main__":
    with gr.Blocks() as app:
        chatbot = gr.Chatbot(type="messages", label="Agent")
        chat_message = gr.Textbox(lines=1, label="Ask a question")
        chat_message.submit(_main, [chat_message, chatbot], [chatbot])

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        app.launch(server_name="0.0.0.0")
    finally:
        asyncio.run(_cleanup_clients())
