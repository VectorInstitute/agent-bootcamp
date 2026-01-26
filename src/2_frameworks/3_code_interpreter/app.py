"""Code Interpreter example.

Logs traces to LangFuse for observability and evaluation.

You will need your E2B API Key.
"""

import asyncio
from pathlib import Path
from typing import Any, AsyncGenerator

import agents
import gradio as gr
from aieng.agents import (
    get_or_create_agent_session,
    oai_agent_stream_to_gradio_messages,
    pretty_print,
    set_up_logging,
)
from aieng.agents.client_manager import AsyncClientManager
from aieng.agents.gradio import COMMON_GRADIO_CONFIG
from aieng.agents.langfuse import langfuse_client, setup_langfuse_tracer
from aieng.agents.prompts import CODE_INTERPRETER_INSTRUCTIONS
from aieng.agents.tools import CodeInterpreter
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from langfuse import propagate_attributes


async def _main(
    query: str, history: list[ChatMessage], session_state: dict[str, Any]
) -> AsyncGenerator[list[ChatMessage], Any]:
    # Initialize list of chat messages for a single turn
    turn_messages: list[ChatMessage] = []

    # Construct an in-memory SQLite session for the agent to maintain
    # conversation history across multiple turns of a chat
    # This makes it possible to ask follow-up questions that refer to
    # previous turns in the conversation
    session = get_or_create_agent_session(history, session_state)

    with (
        langfuse_client.start_as_current_observation(
            name="Code-Interpreter-Agent", as_type="agent", input=query
        ) as obs,
        propagate_attributes(
            session_id=session.session_id  # Propagate session_id to all child observations
        ),
    ):
        # Run the agent in streaming mode to get and display intermediate outputs
        result_stream = agents.Runner.run_streamed(
            main_agent, input=query, session=session
        )

        async for _item in result_stream.stream_events():
            turn_messages += oai_agent_stream_to_gradio_messages(_item)
            if len(turn_messages) > 0:
                yield turn_messages

        obs.update(output=result_stream.final_output)

    pretty_print(turn_messages)
    yield turn_messages

    # Clear the turn messages after yielding to prepare for the next turn
    turn_messages.clear()


if __name__ == "__main__":
    load_dotenv(verbose=True)

    set_up_logging()
    setup_langfuse_tracer()

    # Initialize client manager
    # This class initializes the OpenAI and Weaviate async clients, as well as the
    # Weaviate knowledge base tool. The initialization is done once when the clients
    # are first accessed, and the clients are reused for subsequent calls.
    client_manager = AsyncClientManager()

    # Initialize code interpreter with local files that will be available to the agent
    code_interpreter = CodeInterpreter(
        local_files=[
            Path("sandbox_content/"),
            Path("aieng-agents-utils/tests/example_files/example_a.csv"),
        ]
    )

    main_agent = agents.Agent(
        name="Data Analysis Agent",
        instructions=CODE_INTERPRETER_INSTRUCTIONS,
        tools=[
            agents.function_tool(
                code_interpreter.run_code, name_override="code_interpreter"
            )
        ],
        model=agents.OpenAIChatCompletionsModel(
            model=client_manager.configs.default_planner_model,
            openai_client=client_manager.openai_client,
        ),
    )

    demo = gr.ChatInterface(
        _main,
        **COMMON_GRADIO_CONFIG,
        examples=[
            ["What is the sum of the column `x` in this example_a.csv?"],
            ["What is the sum of the column `y` in this example_a.csv?"],
            ["Create a linear best-fit line for the data in example_a.csv."],
        ],
        title="2.3. OAI Agent SDK ReAct + Code Interpreter Tool",
    )

    try:
        demo.launch(share=True)
    finally:
        asyncio.run(client_manager.close())
