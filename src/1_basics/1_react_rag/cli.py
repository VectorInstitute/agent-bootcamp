"""Reason-and-Act Knowledge Retrieval Agent, no framework."""

import asyncio
import json
import sys
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.prompts import REACT_INSTRUCTIONS
from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    get_weaviate_async_client,
    pretty_print,
)


if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionToolParam

load_dotenv(verbose=True)

MAX_TURNS = 5

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


async def _main():
    configs = Configs()
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
        collection_name=configs.weaviate_collection_name,
    )

    messages: list = [
        {
            "role": "system",
            "content": REACT_INSTRUCTIONS,
        },
        {
            "role": "user",
            "content": "At which university did the SVP Software Engineering"
            " at Apple (as of June 2025) earn their engineering degree?",
        },
    ]

    try:
        while True:
            for _ in range(MAX_TURNS):
                completion = await async_openai_client.chat.completions.create(
                    model=configs.default_planner_model,
                    messages=messages,
                    tools=tools,
                )

                # Add message to conversation history
                message = completion.choices[0].message
                messages.append(message)

                tool_calls = message.tool_calls

                # Execute function calls if requested.
                if tool_calls is not None:
                    for tool_call in tool_calls:
                        pretty_print(tool_call)
                        arguments = json.loads(tool_call.function.arguments)
                        results = await async_knowledgebase.search_knowledgebase(
                            arguments["keyword"]
                        )

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(
                                    [_result.model_dump() for _result in results]
                                ),
                            }
                        )

                # Otherwise, print final response and stop.
                else:
                    pretty_print(message.content)
                    break

                pretty_print(messages)

            # Get new user input
            timeout_secs = 60
            try:
                user_input = await asyncio.wait_for(
                    asyncio.to_thread(input, "Ask a question: "),
                    timeout=timeout_secs,
                )
            except asyncio.TimeoutError:
                print(f"\nNo response received within {timeout_secs} seconds. Exiting.")
                break

            # Break if user_input is empty or a quit command
            if not user_input.strip() or user_input.lower() in {"quit", "exit"}:
                print("Exiting.")
                break

            messages.append({"role": "user", "content": user_input})
    finally:
        await async_weaviate_client.close()
        await async_openai_client.close()
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(_main())
