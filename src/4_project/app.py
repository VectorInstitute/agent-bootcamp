from dotenv import load_dotenv
from logging import basicConfig, INFO
from src.utils.client_manager import AsyncClientManager
from agents import (
    set_tracing_disabled,
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    InputGuardrailTripwireTriggered,
    function_tool,
)
import gradio as gr
from gradio.components.chatbot import ChatMessage
from src.utils.gradio import COMMON_GRADIO_CONFIG
from src.utils import oai_agent_stream_to_gradio_messages
from src.utils.agent_session import get_or_create_session
import asyncio
from sub_agents.triage_agent import (
    off_topic_guardrail,
    dynamic_triage_agent_instructions,
)
from typing import Any, AsyncGenerator
from models import UserAccountContext
from src.utils.tools.gemini_grounding import (
    GeminiGroundingWithGoogleSearch,
    ModelSettings,
)
from sub_agents.code_interpreter_agent import code_interpreter_agent

# Context is available locally to your code
# LLM only sees the conversation history
# Need to pass in the context to the prompt
user_account_ctx = UserAccountContext(
    customer_id="C5841053",
    name="Bonbon",
    nra=65,
    status="active",
)

# Make a tool that can work with user data without exposing the user data to LLM
# e.g. change email -> POST request is handled under the function
# @function_tool
# def get_user_nra(wrapper: RunContextWrapper[UserAccountContext]):
#     return f"The user {wrapper.context.customer_id} has a NRA {wrapper.context.nra}"


async def _main(
    query: str, history: list[ChatMessage], session_state: dict[str, Any]
) -> AsyncGenerator[list[ChatMessage], Any]:
    turn_messages: list[ChatMessage] = []

    session = get_or_create_session(history, session_state)

    worker_model = client_manager.configs.default_worker_model

    gemini_grounding_tool = GeminiGroundingWithGoogleSearch(
        model_settings=ModelSettings(model=worker_model)
    )

    try:
        main_agent = Agent(
            name="Pension Support Agent",
            instructions=dynamic_triage_agent_instructions,
            model=OpenAIChatCompletionsModel(
                model=worker_model, openai_client=client_manager.openai_client
            ),
            # model_settings=ModelSettings(parallel_tool_calls=False),
            # TODO: This is not compatible with gemini. Need to use openAI
            # INFO:httpx:HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/openai/responses "HTTP/1.1 404 Not Found"
            # input_guardrails=[off_topic_guardrail],
            tools=[
                function_tool(
                    gemini_grounding_tool.get_web_search_grounded_response,
                    name_override="search_web",
                ),
                code_interpreter_agent.as_tool(
                    tool_name="code_interpreter",
                    tool_description=(
                        "Use this tool when you need to create code from a local csv file"
                        "in order to calculate projected pension amounts or any other pension data"
                        "Make sure only to provide the information regarding the current member"
                    ),
                ),
            ],
        )
        print("user_account_ctx", user_account_ctx)
        result_stream = Runner.run_streamed(
            main_agent, input=query, session=session, context=user_account_ctx
        )

        async for _item in result_stream.stream_events():
            # Parse the stream events, convert to Gradio chat messages and append to
            # the chat history``
            turn_messages += oai_agent_stream_to_gradio_messages(_item)
            if len(turn_messages) > 0:
                yield turn_messages

    except InputGuardrailTripwireTriggered as e:
        print("InputGuardrailException", e)
        turn_messages = [
            ChatMessage(
                role="assistant",
                content="I cannot help you with that.",
                metadata={
                    "title": "*Guardrail*",
                    "status": "done",  # This makes it collapsed by default
                },
            )
        ]


if __name__ == "__main__":
    load_dotenv(verbose=True)
    basicConfig(level=INFO)

    # openAI and Weaviate async clients
    client_manager = AsyncClientManager()

    # Disable openAI platform tracing
    set_tracing_disabled(disabled=True)

    demo = gr.ChatInterface(
        _main,
        **COMMON_GRADIO_CONFIG,
        examples=[
            ["What is the expected pension amount when I retire at the age of 60?"],
        ],
        title="Pension Bot",
    )

    try:
        demo.launch(share=True)
    finally:
        asyncio.run(client_manager.close())
