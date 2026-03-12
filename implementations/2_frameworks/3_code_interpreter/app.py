"""Code Interpreter example.

Logs traces to LangFuse for observability and evaluation.

You will need your E2B API Key.
"""

from pathlib import Path
from typing import Any, AsyncGenerator

import agents
import gradio as gr
from aieng.agents import (
    get_or_create_agent_session,
    oai_agent_stream_to_gradio_messages,
    pretty_print,
    register_async_cleanup,
    set_up_logging,
)
from aieng.agents.client_manager import AsyncClientManager
from aieng.agents.gradio import get_common_gradio_config
from aieng.agents.langfuse import langfuse_client, setup_langfuse_tracer
from aieng.agents.prompts import CODE_INTERPRETER_INSTRUCTIONS
from aieng.agents.tools import CodeInterpreter
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from langfuse import propagate_attributes


load_dotenv(verbose=True)

# Set logging level and suppress some noisy logs from dependencies
set_up_logging()

if gr.NO_RELOAD:
    # Set up LangFuse for tracing
    setup_langfuse_tracer()

    # Initialize client manager
    # This class initializes the OpenAI and Weaviate async clients, as well as the
    # Weaviate knowledge base tool. The initialization is done once when the clients
    # are first accessed, and the clients are reused for subsequent calls.
    client_manager = AsyncClientManager()

    # Register async cleanup to ensure clients are properly closed on program exit
    register_async_cleanup(client_manager)


def _get_main_agent() -> agents.Agent:
    # Initialize code interpreter with local files that will be available to the agent
    code_interpreter = CodeInterpreter(
        local_files=[
            Path("sandbox_content/"),
            Path("aieng-agents/tests/example_files/example_a.csv"),
        ]
    )

    return agents.Agent(
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

    main_agent = _get_main_agent()

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


demo = gr.ChatInterface(
    _main,
    **get_common_gradio_config(),
    examples=[
        ["What is the sum of the column `x` in this example_a.csv?"],
        ["What is the sum of the column `y` in this example_a.csv?"],
        ["Create a linear best-fit line for the data in example_a.csv."],
    ],
    title="2.3. OAI Agent SDK ReAct + Code Interpreter Tool",
)

if __name__ == "__main__":
    demo.launch(share=True)
