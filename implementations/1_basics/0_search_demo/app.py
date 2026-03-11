"""Knowledge Base Search Demo using Gradio."""

import asyncio

import gradio as gr
from aieng.agents import AsyncClientManager, pretty_print
from dotenv import load_dotenv


DESCRIPTION = """\
In the example below, your goal is to find out where \
Apple's SVP Software Engineering got his degree in engineering - \
without knowing the full name of that person ahead of time. \
\
Did you see why traditional RAG systems like this one \
can't reliably handle this type of "multi-hop" queries? \
Can you come up with more examples? Make note of your \
findings and share them with your teammates! \
\
The output format you see is also what the Agent LLM \
would receive when interacting with the knowledge base search \
tool in subsequent sections of this bootcamp - both when using \
the Wikipedia database we provided and when using your own \
public dataset.
"""


async def search_and_pretty_format(keyword: str) -> str:
    """Search knowledgebase and pretty-format output."""
    output = await client_manager.knowledgebase.search_knowledgebase(keyword)
    return pretty_print(output)


if __name__ == "__main__":
    load_dotenv(verbose=True)

    # Initialize client manager
    # This class initializes the OpenAI and Weaviate async clients, as well as the
    # Weaviate knowledge base tool. The initialization is done once when the clients
    # are first accessed, and the clients are reused for subsequent calls.
    client_manager = AsyncClientManager()

    # Gradio UI
    # The UI consists of a text input for the search keyword
    # and a code block to display the JSON-formatted search results.
    demo = gr.Interface(
        fn=search_and_pretty_format,
        inputs=["text"],
        outputs=[gr.Code(language="json", wrap_lines=True)],
        title="1.0: Knowledge Base Search Demo",
        description=DESCRIPTION,
        examples=[
            "Apple SVP Software Engineering academic background",
            "Apple SVP Software Engineering",
            "Craig Federighi",
            "Craig Federighi academic background",
        ],
    )

    try:
        demo.launch(share=True)
    finally:
        # Ensure clients are closed on exit
        asyncio.run(client_manager.close())
