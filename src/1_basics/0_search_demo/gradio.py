import gradio as gr
from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    get_weaviate_async_client,
    pretty_print,
)


DESCRIPTION = """\
In the example below, your goal is to find out where \
Apple's SVP Software Engineering got his degree in engineering- \
without knowing the full name of that person ahead of time. \
\
Did you see why traditional RAG systems like this one \
can't reliably handle this type of "multi-hop" queries? \
Can you come up with more examples? Make note of your \
findings and share them with your teammates! \
\
The output format you see is also what the Agent LLM \
would receive when interacting with the knowledge base search \
tool in subsequent sections of this bootcamp- both when using \
the Wikipedia database we provided and when using your own \
public dataset. 
"""


load_dotenv(verbose=True)

configs = Configs.from_env_var()
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
    collection_name="enwiki_20250520",
)


async def search_and_pretty_format(keyword: str) -> str:
    """Search knowledgebase and pretty-format output."""
    output = await async_knowledgebase.search_knowledgebase(keyword)
    return pretty_print(output)


json_codeblock = gr.Code(language="json", wrap_lines=True)

demo = gr.Interface(
    fn=search_and_pretty_format,
    inputs=["text"],
    outputs=[json_codeblock],
    title="1.0: Knowledge Base Search Demo",
    description=DESCRIPTION,
    examples=[
        "Apple SVP Software Engineering",
        "Craig Federighi",
        "Apple SVP Software Engineering academic background",
        "Craig Federighi academic background",
    ],
)

demo.launch(server_name="0.0.0.0")
