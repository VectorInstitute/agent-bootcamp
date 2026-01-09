"""Reason-and-Act Knowledge Retrieval Agent, no framework."""

import asyncio
import json
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from src.prompts import REACT_INSTRUCTIONS
from src.utils import (
    AsyncClientManager,
    pretty_print,
)


if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionToolParam


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


async def _main() -> None:
    # Initialize client manager
    # This class initializes the OpenAI and Weaviate async clients, as well as the
    # Weaviate knowledge base tool. The initialization is done once when the clients
    # are first accessed, and the clients are reused for subsequent calls.
    client_manager = AsyncClientManager()

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

    # Show initial system prompt and user query
    print("System prompt: \n", REACT_INSTRUCTIONS)
    print("User query: ")
    pretty_print(messages[-1]["content"])

    try:
        while True:
            # Flag to track if final response is given
            agent_responded = False

            for _ in range(MAX_TURNS):
                completion = await client_manager.openai_client.chat.completions.create(
                    model=client_manager.configs.default_worker_model,
                    messages=messages,
                    tools=tools,
                )

                # Add message to conversation history
                message = completion.choices[0].message
                messages.append(message)

                tool_calls = message.tool_calls

                # Execute function calls if requested.
                if tool_calls is not None:
                    # Show thought that led to tool call
                    print("\nAgent Thought: ")
                    pretty_print(message.content)

                    for tool_call in tool_calls:
                        print("\nAgent Action: ")
                        pretty_print(tool_call)
                        arguments = json.loads(tool_call.function.arguments)
                        results = (
                            await client_manager.knowledgebase.search_knowledgebase(
                                arguments["keyword"]
                            )
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
                        print("\nAgent Observation: ")
                        pretty_print(results)

                # Otherwise, print final response and stop.
                elif message.content is not None:
                    print("\nAgent final response: \n", message.content)
                    agent_responded = True
                    break

            if not agent_responded:
                # Add message letting the agent know max turns reached
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "You have reached the maximum number of allowed reasoning "
                            "steps. Provide a final answer based on the information available."
                        ),
                    }
                )

                # Make one final LLM call to get a response given the history
                completion = await client_manager.openai_client.chat.completions.create(
                    model=client_manager.configs.default_worker_model,
                    messages=messages,
                )
                message = completion.choices[0].message
                print(
                    "\nAgent final response (after max turns): \n",
                    message.content,
                )

                # Remove the last system message for next iteration
                messages.pop()

                # Append the final message to history
                messages.append(message)

            # Get new user input
            timeout_secs = 300
            try:
                user_input = await asyncio.wait_for(
                    asyncio.to_thread(input, "Enter your prompt: "),
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
        await client_manager.close()


if __name__ == "__main__":
    load_dotenv(verbose=True)

    asyncio.run(_main())
