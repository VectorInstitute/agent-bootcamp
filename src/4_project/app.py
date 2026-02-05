from dotenv import load_dotenv
from logging import basicConfig, INFO
from src.utils.client_manager import AsyncClientManager
from agents import set_tracing_disabled, Agent, OpenAIChatCompletionsModel, Runner
import gradio as gr
from gradio.components.chatbot import ChatMessage
from src.utils.gradio import COMMON_GRADIO_CONFIG
from src.utils.agent_session import get_or_create_session
import asyncio

async def _main(query: str, history: list[ChatMessage], session_state: dict[str, Any]):
    turn_messages: list[ChatMessage] = []

    session = get_or_create_session(history, session_state)

    main_agent = Agent(
        name="Pension System Agent",
        instruction="You're amazing",
        model= OpenAIChatCompletionsModel(
            model=client_manager.configs.default_worker_model,
            openai_client=client_manager.openai_client,
        ),
    )

    result = Runner.run_streamed(
        main_agent,
        input=query,
        session=session
    )

    


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
        examples = [
            [
                "What is maximum CPP?"
            ]
        ],
        title="Pension Bot"

    )

    try:
        demo.launch(share=True)
    finally:
        asyncio.run(client_manager.close())