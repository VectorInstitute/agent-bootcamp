"""Reason-and-Act Knowledge Retrieval Agent, no framework."""

import asyncio
import json
from typing import TYPE_CHECKING

from elasticsearch import AsyncElasticsearch
from openai import AsyncOpenAI

from src.utils import AsyncESKnowledgeBase, Configs, pretty_print


if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

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
    configs = Configs.from_env_var()
    async_es_client = AsyncElasticsearch(configs.es_host, api_key=configs.es_api_key)
    async_openai_client = AsyncOpenAI()
    async_knowledgebase = AsyncESKnowledgeBase(
        async_es_client,
        es_collection_name="enwiki-20250501",
    )

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                "Answer the question using the search tool. "
                "Be sure to mention the sources. "
                "If the search did not return intended results, try again. "
                "Do not make up information."
            ),
        },
        {
            "role": "user",
            "content": "When was the 4K (first generation) Apple TV released?",
        },
    ]
    for _ in range(MAX_TURNS):
        completion = await async_openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            tools=tools,
        )

        # Add message to conversation history
        message = completion.choices[0].message
        messages.append(message.model_dump())  # type: ignore[arg-type]

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
            pretty_print(message)
            break

        pretty_print(messages)
        input()

    await async_es_client.close()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(_main())
