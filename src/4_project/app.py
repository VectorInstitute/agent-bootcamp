from dotenv import load_dotenv
from logging import basicConfig, INFO
from src.utils.client_manager import AsyncClientManager
from agents import set_tracing_disabled
import gradio as gr
from src.utils.gradio import COMMON_GRADIO_CONFIG
import asyncio

async def _main():
    pass


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