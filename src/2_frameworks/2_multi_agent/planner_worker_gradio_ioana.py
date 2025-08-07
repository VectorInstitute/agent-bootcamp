"""Example code for planner-worker agent collaboration.

With reference to:

github.com/ComplexData-MILA/misinfo-datasets
/blob/3304e6e/misinfo_data_eval/tasks/web_search.py
"""

import asyncio
import contextlib
import logging
import signal
import sys

import agents
import gradio as gr
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI

from prompts_i import REACT_INSTRUCTIONS
from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    get_weaviate_async_client,
    oai_agent_stream_to_gradio_messages,
    setup_langfuse_tracer,
)
from src.utils.langfuse.shared_client import langfuse_client


load_dotenv(verbose=True)


logging.basicConfig(level=logging.INFO)


AGENT_LLM_NAMES = {
    "worker": "gemini-2.5-flash",  # less expensive,
    "planner": "gemini-2.5-pro",  # more expensive, better at reasoning and planning
}

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


async def _cleanup_clients() -> None:
    """Close async clients."""
    await async_weaviate_client.close()
    await async_openai_client.close()


def _handle_sigint(signum: int, frame: object) -> None:
    """Handle SIGINT signal to gracefully shutdown."""
    with contextlib.suppress(Exception):
        asyncio.get_event_loop().run_until_complete(_cleanup_clients())
    sys.exit(0)

# Worker Agent QA: handles long context efficiently
import os
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from agents import function_tool
# from pydantic import BaseModel



@function_tool
async def faq_match_tool(user_query:str)->list:
    '''Return a list of FAQ sorted from the most simialr to least similar to the user query by cosine similarity.
    '''
    faq_list = ["Where are Toyota cars manufactured? Toyota cars are produced in Germany and Japan", 
                "What is the engine power of Toyota RAV4? 180HP","Where is Japan?",
                "What is horse power in cars?", "What is the capital of Germany? it's Berlin", 
                "Is Toyota a German brand? No, it's a Japanese automobile brand."]

    _embed_client = openai.OpenAI(
        api_key=os.getenv("EMBEDDING_API_KEY"),
        base_url=os.getenv("EMBEDDING_BASE_URL"),
        max_retries=5)
    
    #embed user query
    user_query_embedding = _embed_client.embeddings.create(input=user_query, model=os.getenv('EMBEDDING_MODEL_NAME'))
    user_query_embedding = np.array(user_query_embedding.data[0].embedding)
    user_query_embedding = user_query_embedding.reshape(1, -1)

    cosi_list = []
    faq_embedding_list = _embed_client.embeddings.create(input=faq_list, model=os.getenv('EMBEDDING_MODEL_NAME'))
    for i, faq_embedding in enumerate(faq_embedding_list.data):
            faq_embedding = np.array(faq_embedding.embedding)
            faq_embedding = faq_embedding.reshape(1,-1)
            similarity_score = cosine_similarity(user_query_embedding, faq_embedding)[0][0]
            cosi_list.append({"faq":faq_list[i], "sim":similarity_score})

    sorted_faqs = sorted(cosi_list, key=lambda d: d["sim"], reverse=True)
    sorted_faqs_list = [i["faq"] for i in sorted_faqs]
    return "\n".join(f" {i}\n"for i in sorted_faqs_list)
    # return sorted_faqs_list

# class FAQ(BaseModel):
#     user_query: str
#     similar_faqs: list[str]
#     def __str__(self) -> str:
#         """Return a string representation of the faq list"""
#         return "\n".join(
#             f"faq {step}\n"
#             for step in self.similar_faqs
#         )

faq_agent = agents.Agent(
    name="QAMatchAgent",
    instructions=(
        "You are an agent specializing in matching user queries to FAQ. You receive a single user query as input. "
        "Use the faq_match_tool tool to return a sorted list of FAQ based on how similar they are with the user query."
        "Return maximum 3 FAQs that best match the user query. If you can't find any match, return the raw user query."
        "ALWAYS structure the output as a list that contains [user request, faqs if any]."
        
    ),
    tools=[faq_match_tool],
    # a faster, smaller model for quick searches
    model=agents.OpenAIChatCompletionsModel(
        model="gemini-2.5-flash", openai_client=async_openai_client
    ),
    model_settings=agents.ModelSettings(tool_choice="required"),
    # output_type=FAQ
)

# Worker Agent: handles long context efficiently
search_agent = agents.Agent(
    name="SearchAgent",
    instructions=(
        "You are a search agent. You receive a search query and a list of FAQ as input. "
        "Use the WebSearchTool to perform a web search on the search query, then produce a concise "
        "'search summary' of the key findings. Corroborate the findings with the FAQ into a final answer. Do NOT return raw search results."
    ),
    tools=[
        agents.function_tool(async_knowledgebase.search_knowledgebase),
    ],
    # a faster, smaller model for quick searches
    model=agents.OpenAIChatCompletionsModel(
        model="gemini-2.5-flash", openai_client=async_openai_client
    ),
)

# Main Agent: more expensive and slower, but better at complex planning
main_agent = agents.Agent(
    name="MainAgent",
    instructions=REACT_INSTRUCTIONS,
    # Allow the planner agent to invoke the worker agents.
    # The long context provided to the worker agent is hidden from the main agent.
    tools=[
        faq_agent.as_tool(
            tool_name="faq_match",
            tool_description = "Identify the matching FAQs in the database."
       ),
        search_agent.as_tool(
            tool_name="search",
            tool_description="Perform a web search for a query and return a concise summary.",
        )
    ],
    # a larger, more capable model for planning and reasoning over summaries
    model=agents.OpenAIChatCompletionsModel(
        model="gemini-2.5-pro", openai_client=async_openai_client
    ),
)


async def _main(question: str, gr_messages: list[ChatMessage]):
    setup_langfuse_tracer()

    # Use the main agent as the entry point- not the worker agent.
    with langfuse_client.start_as_current_span(name="Agents-SDK-Trace") as span:
        span.update(input=question)

        result_stream = agents.Runner.run_streamed(main_agent, input=question)
        async for _item in result_stream.stream_events():
            gr_messages += oai_agent_stream_to_gradio_messages(_item)
            if len(gr_messages) > 0:
                yield gr_messages

        span.update(output=result_stream.final_output)


demo = gr.ChatInterface(
    _main,
    title="2.2 Multi-Agent for Efficiency",
    type="messages",
    examples=[
        "what is toyota? ",
        "How does the annual growth in the 50th-percentile income "
        "in the US compare with that in Canada?",
    ],
)

if __name__ == "__main__":
    async_openai_client = AsyncOpenAI()

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        demo.launch(server_name="0.0.0.0")
    finally:
        asyncio.run(_cleanup_clients())
