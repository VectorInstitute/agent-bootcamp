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
import os
import json
import openai
import numpy as np

import agents
from agents import function_tool
import gradio as gr
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel

from src.prompts import REACT_INSTRUCTIONS, KB_SEARCH_INSTRUCTIONS, QA_SEARCH_INSTRUCTIONS, EVALUATOR_INSTRUCTIONS, EVALUATOR_TEMPLATE
from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    get_weaviate_async_client,
    oai_agent_stream_to_gradio_messages,
    setup_langfuse_tracer,
)
from src.utils.langfuse.shared_client import langfuse_client
from rich.progress import track

load_dotenv(verbose=True)


logging.basicConfig(level=logging.INFO)



class QASearchSingleResponse(BaseModel):
    """Typed response from the QA search agent."""

    user_query: str | None
    question: str | None
    answer: str | None
    context: str | None

    def __str__(self) -> str:
        """String representation of the response."""
        return f"QASearchSingleResponse(user_query={self.user_query}, question={self.question}, answer={self.answer}, context={self.context})"


class KBSearchResponse(BaseModel):
    """Query to the knowledge base search agent."""

    answer: str | None
    context: str | None

class EvaluatorQuery(BaseModel):
    """Query to the evaluator agent."""

    question: str
    ground_truth: str
    proposed_response: str

    def get_query(self) -> str:
        """Obtain query string to the evaluator agent."""
        return EVALUATOR_TEMPLATE.format(**self.model_dump())


class EvaluatorResponse(BaseModel):
    """Typed response from the evaluator."""

    explanation: str
    is_answer_correct: bool


class EvaluatorAgent(agents.Agent[EvaluatorResponse]):
    async def run_tool(self, tool_input, *args, **kwargs):
        # Convert dict → EvaluatorQuery → formatted prompt
        if not isinstance(tool_input, EvaluatorQuery):
            tool_input = EvaluatorQuery(**tool_input)
        return await super().run_tool(tool_input.get_query(), *args, **kwargs)


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

@function_tool
async def qa_search_tool(user_query:str) -> list:
    '''Return a list of questions sorted from the most similar to least similar to the user query by cosine similarity.
    '''
    qa_dataset = { 
        1 : {
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris.",
            "context": "Paris is the capital city of France, known for its art, fashion, and culture."
        }
    }

    # _embed_client = openai.OpenAI(
    #     api_key=os.getenv("EMBEDDING_API_KEY"),
    #     base_url=os.getenv("EMBEDDING_BASE_URL"),
    #     max_retries=5)
    
    # #embed user query
    # user_query_embedding = _embed_client.embeddings.create(input=user_query, model=os.getenv('EMBEDDING_MODEL_NAME'))
    # user_query_embedding = np.array(user_query_embedding.data[0].embedding)
    # user_query_embedding = user_query_embedding.reshape(1, -1)

    # cosi_list = []
    # qa_embedding_list = _embed_client.embeddings.create(input=qa_dataset, model=os.getenv('EMBEDDING_MODEL_NAME'))
    # for i, qa_embedding in enumerate(qa_embedding_list.data):
    #         qa_embedding = np.array(qa_embedding.embedding)
    #         qa_embedding = qa_embedding.reshape(1,-1)
    #         similarity_score = cosine_similarity(user_query_embedding, qa_embedding)[0][0]
    #         cosi_list.append({"faq":faq_list[i], "sim":similarity_score})

    # sorted_qa = sorted(cosi_list, key=lambda d: d["sim"], reverse=True)
    # sorted_faqs_list = [i["faq"] for i in sorted_qa]
    # 
    # return "\n".join(f" {i}\n"for i in sorted_faqs_list)

    return json.dumps(qa_dataset)

qa_search_agent = agents.Agent(
    name="QASearchAgent",
    instructions=QA_SEARCH_INSTRUCTIONS,
    tools=[qa_search_tool],
    # a faster, smaller model for quick searches
    model=agents.OpenAIChatCompletionsModel(
        model="gemini-2.5-flash", openai_client=async_openai_client
    )
)

# Worker Agent: handles long context efficiently
kb_search_agent = agents.Agent(
    name="KBSearchAgent",
    instructions=KB_SEARCH_INSTRUCTIONS,
    tools=[
        agents.function_tool(async_knowledgebase.search_knowledgebase),
    ],
    # a faster, smaller model for quick searches
    model=agents.OpenAIChatCompletionsModel(
        model="gemini-2.5-flash", openai_client=async_openai_client
    )
)

evaluator_agent = agents.Agent(
    name="EvaluatorAgent",
    instructions=EVALUATOR_INSTRUCTIONS,
    output_type=EvaluatorResponse,
    model=agents.OpenAIChatCompletionsModel(
        model="gemini-2.5-flash", openai_client=async_openai_client
    )
)

# Main Agent: more expensive and slower, but better at complex planning
main_agent = agents.Agent(
    name="MainAgent",
    instructions=REACT_INSTRUCTIONS,
    # Allow the planner agent to invoke the worker agents.
    # The long context provided to the worker agent is hidden from the main agent.
    tools=[
        qa_search_agent.as_tool(
            tool_name="qa_search_Agent",
            tool_description = "Perform a search of the QA database and retrieve question/answer/context tuples related to input query."
       ),
        kb_search_agent.as_tool(
            tool_name="kb_search_agent",
            tool_description="Perform a search of a knowledge base and synthesize the search results to answer input question.",
        ),
        evaluator_agent.as_tool(
            tool_name="evaluator_agent",
            tool_description="Evaluate the output of the knowledge base search agent.",
        )
    ],
    # a larger, more capable model for planning and reasoning over summaries
    model=agents.OpenAIChatCompletionsModel(
        model="gemini-2.5-pro", openai_client=async_openai_client
    ),
)

import json
async def _main(question: str, gr_messages: list[ChatMessage]):
    setup_langfuse_tracer()

    # Use the main agent as the entry point- not the worker agent.
    with langfuse_client.start_as_current_span(name="Calen-Multi-Agent") as span:
        span.update(input=question)

        result_stream = agents.Runner.run_streamed(main_agent, input=question)
        score_is_answer_correct = []
        score_explanation = []
        async for _item in result_stream.stream_events():
            # print(f"Item: {_item}")
            gr_messages += oai_agent_stream_to_gradio_messages(_item)
            if len(gr_messages) > 0:
                yield gr_messages

            try:
                # Assume `event` is your RunItemStreamEvent
                if _item.name == "tool_output" and _item.item.type == "tool_call_output_item":
                        tool_output = json.loads(_item.item.output)

                        explanation = tool_output.get("explanation")
                        is_correct = tool_output.get("is_answer_correct")

                        score_is_answer_correct.append(is_correct)
                        score_explanation.append(explanation)

                        print("✅ is_answer_correct:", is_correct)
                        print("🧠 explanation:", explanation)
            except: 
                continue
        
        print(result_stream.final_output)
        span.update(output=result_stream.final_output)

        langfuse_client.create_score(
                name="is_answer_correct",
                value=score_is_answer_correct[0],
                comment=score_explanation[0],
                trace_id=langfuse_client.get_current_trace_id()
            )


demo = gr.ChatInterface(
    _main,
    title="Hitachi Multi-Agent Knowledge Retrieval System",
    type="messages",
    examples=[
        "Where should I go in France? ",
        "Where is the government of France located? "
    ],
)

if __name__ == "__main__":
    async_openai_client = AsyncOpenAI()

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        demo.launch(server_name="0.0.0.0")
    finally:
        asyncio.run(_cleanup_clients())
