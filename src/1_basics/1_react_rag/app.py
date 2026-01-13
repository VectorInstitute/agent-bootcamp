"""Reason-and-Act Knowledge Retrieval Agent, no framework.

With reference to huggingface.co/spaces/gradio/langchain-agent
"""

import asyncio
import json
from typing import TYPE_CHECKING, Any, AsyncGenerator

import gradio as gr
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage

from src.prompts import REACT_INSTRUCTIONS
from src.utils.client_manager import AsyncClientManager


if TYPE_CHECKING:
    from openai.types.chat import (
        ChatCompletionSystemMessageParam,
        ChatCompletionToolParam,
    )

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

system_message: "ChatCompletionSystemMessageParam" = {
    "role": "system",
    "content": REACT_INSTRUCTIONS,
}


async def react_rag(
    query: str, history: list[ChatMessage]
) -> AsyncGenerator[list[ChatMessage], Any]:
    """Handle ReAct RAG chat for knowledgebase-augmented agents."""
    # Flag to track if the agent has provided a final response
    # If the agent exhausts all reasoning steps without a final answer,
    # we make one last call to get a final response based on the information available.
    agent_responded = False

    # Construct chat completion messages to pass to LLM
    oai_messages = [system_message, {"role": "user", "content": query}]

    for _ in range(MAX_TURNS):
        # Call OpenAI chat completions with tools enabled
        completion = await client_manager.openai_client.chat.completions.create(
            model=client_manager.configs.default_worker_model,
            messages=oai_messages,
            tools=tools,  # This makes the tool defined above available to the LLM
        )

        # Print assistant output
        message = completion.choices[0].message
        oai_messages.append(message)

        # Execute tool calls and send results back to LLM if requested.
        # Otherwise, stop, as the conversation would have been finished.
        tool_calls = message.tool_calls

        if tool_calls is None:  # No tool calls, assume final response
            history.append(ChatMessage(content=message.content or "", role="assistant"))
            agent_responded = True
            yield history
            break

        history.append(
            ChatMessage(
                role="assistant",
                content=message.content or "",
                metadata={"title": "🧠 Thought"},
            )
        )
        yield history

        for tool_call in tool_calls:
            arguments = json.loads(tool_call.function.arguments)
            results = await client_manager.knowledgebase.search_knowledgebase(
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
                    content=f"```\n{results_serialized}\n```",
                    metadata={
                        "title": f"🛠️ Used tool `{tool_call.function.name}`",
                        "log": f"Arguments: {arguments}",
                        "status": "done",  # This makes it collapsed by default
                    },
                )
            )
            yield history

    if not agent_responded:
        # Make one final LLM call to get a response given the history
        oai_messages.append(
            {
                "role": "system",
                "content": (
                    "You have reached the maximum number of allowed reasoning "
                    "steps. Provide a final answer based on the information available."
                ),
            }
        )
        completion = await client_manager.openai_client.chat.completions.create(
            model=client_manager.configs.default_planner_model,
            messages=oai_messages,
        )
        message = completion.choices[0].message
        history.append(ChatMessage(content=message.content or "", role="assistant"))
        oai_messages.pop()  # Remove the last system message for next iteration
        oai_messages.append(message)  # Append the final message to history
        yield history


if __name__ == "__main__":
    load_dotenv(verbose=True)

    # Initialize client manager
    # This class initializes the OpenAI and Weaviate async clients, as well as the
    # Weaviate knowledge base tool. The initialization is done once when the clients
    # are first accessed, and the clients are reused for subsequent calls.
    client_manager = AsyncClientManager()

    demo = gr.ChatInterface(
        react_rag,
        chatbot=gr.Chatbot(height=600),
        textbox=gr.Textbox(lines=1, placeholder="Enter your prompt"),
        examples=[
            [
                "At which university did the SVP Software Engineering"
                " at Apple (as of June 2025) earn their engineering degree?"
            ],
            [
                "Où le vice-président senior actuel d'Apple en charge de l'ingénierie "
                "logicielle a-t-il obtenu son diplôme d'ingénieur?"
            ],
        ],
        title="1.1: ReAct Agent for Retrieval-Augmented Generation",
    )

    try:
        demo.launch(share=True)
    finally:
        asyncio.run(client_manager.close())
