"""Evaluate agent on dataset using LLM-as-a-Judge."""

import argparse
import asyncio

import agents
import pydantic
from dotenv import load_dotenv
from langfuse._client.datasets import DatasetItemClient
from openai import AsyncOpenAI
from rich.progress import track

from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    gather_with_progress,
    get_weaviate_async_client,
    set_up_logging,
    setup_langfuse_tracer,
)
from src.utils.langfuse.shared_client import flush_langfuse, langfuse_client


load_dotenv(verbose=True)
set_up_logging()


from src.prompts import REACT_INSTRUCTIONS, EV_INSTRUCTIONS_HALLUCINATIONS, EV_TEMPLATE_HALLUCINATIONS
# Worker Agent QA: handles long context efficiently
import os
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from agents import function_tool
# from pydantic import BaseModel

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

EVALUATOR_INSTRUCTIONS = """\
Evaluate whether the "Proposed Answer" to the given "Question" matches the "Ground Truth"."""

EVALUATOR_TEMPLATE = """\
# Question

{question}

# Ground Truth

{ground_truth}

# Proposed Answer

{proposed_response}

"""




class LangFuseTracedResponse(pydantic.BaseModel):
    """Agent Response and LangFuse Trace info."""

    answer: str | None
    trace_id: str | None


class EvaluatorQuery(pydantic.BaseModel):
    """Query to the evaluator agent."""

    question: str
    generation: str

    def get_query(self) -> str:
        """Obtain query string to the evaluator agent."""
        return EV_TEMPLATE_HALLUCINATIONS.format(**self.model_dump())
        # return EVALUATOR_TEMPLATE.format(**self.model_dump())


class EvaluatorResponse(pydantic.BaseModel):
    """Typed response from the evaluator."""

    explanation: str
    hallucination: float


async def run_agent_with_trace(
    agent: agents.Agent, query: str
) -> "LangFuseTracedResponse":
    """Run OpenAI Agent on query, returning response and trace_id.

    Returns None if agent exceeds max_turn limit.
    """
    try:
        result = await agents.Runner.run(agent, query)
        if "|" in result.final_output:
            answer = result.final_output.split("|")[-1].strip()
        else:
            answer = result.final_output

    except agents.MaxTurnsExceeded:
        answer = None

    return LangFuseTracedResponse(
        answer=answer, trace_id=langfuse_client.get_current_trace_id()
    )


async def run_evaluator_agent(evaluator_query: EvaluatorQuery) -> EvaluatorResponse:
    """Evaluate using evaluator agent."""
    evaluator_agent = agents.Agent(
        name="Evaluator Agent",
        instructions=EV_INSTRUCTIONS_HALLUCINATIONS,
        output_type=EvaluatorResponse,
        model=agents.OpenAIChatCompletionsModel(
            model="gemini-2.5-flash", openai_client=async_openai_client
        ),
    )

    result = await agents.Runner.run(evaluator_agent, input=evaluator_query.get_query())
    return result.final_output_as(EvaluatorResponse)


async def run_and_evaluate(
    run_name: str, main_agent: agents.Agent, lf_dataset_item: "DatasetItemClient"
) -> "tuple[LangFuseTracedResponse, EvaluatorResponse | None]":
    """Run main agent and evaluator agent on one dataset instance.

    Returns None if main agent returned a None answer.
    """
    # expected_output = lf_dataset_item.expected_output
    # assert expected_output is not None

    with lf_dataset_item.run(run_name=run_name) as root_span:
        root_span.update(input=lf_dataset_item.input["text"])
        traced_response = await run_agent_with_trace(
            main_agent, query=lf_dataset_item.input["text"]
        )
        root_span.update(output=traced_response.answer)

    answer = traced_response.answer
    if answer is None:
        return traced_response, None

    evaluator_response = await run_evaluator_agent(
        EvaluatorQuery(
            question=lf_dataset_item.input["text"],
            # ground_truth=expected_output["text"],
            generation=answer,
        )
    )

    return traced_response, evaluator_response


parser = argparse.ArgumentParser()
parser.add_argument("--langfuse_dataset_name", required=True)
parser.add_argument("--run_name", required=True)
parser.add_argument("--limit", type=int)


if __name__ == "__main__":
    args = parser.parse_args()

    lf_dataset_items = langfuse_client.get_dataset(args.langfuse_dataset_name).items
    if args.limit is not None:
        lf_dataset_items = lf_dataset_items[: args.limit]

    configs = Configs.from_env_var()
    # async_weaviate_client = get_weaviate_async_client(
    #     http_host=configs.weaviate_http_host,
    #     http_port=configs.weaviate_http_port,
    #     http_secure=configs.weaviate_http_secure,
    #     grpc_host=configs.weaviate_grpc_host,
    #     grpc_port=configs.weaviate_grpc_port,
    #     grpc_secure=configs.weaviate_grpc_secure,
    #     api_key=configs.weaviate_api_key,
    # )
    # async_openai_client = AsyncOpenAI()
    # async_knowledgebase = AsyncWeaviateKnowledgeBase(
    #     async_weaviate_client,
    #     collection_name="enwiki_20250520",
    # )

    tracer = setup_langfuse_tracer()

    # main_agent = agents.Agent(
    #     name="Wikipedia Agent",
    #     instructions=REACT_INSTRUCTIONS,
    #     tools=[agents.function_tool(async_knowledgebase.search_knowledgebase)],
    #     model=agents.OpenAIChatCompletionsModel(
    #         model="gemini-2.5-flash", openai_client=async_openai_client
    #     ),


    ##############################################################################################
    ###### Our Pipeline
    ##############################################################################################
    

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

    coros = [
        run_and_evaluate(
            run_name=args.run_name,
            main_agent=main_agent,
            lf_dataset_item=_item,
        )
        for _item in lf_dataset_items
    ]
    results = asyncio.run(
        gather_with_progress(coros, description="Running agent and evaluating")
    )

    for _traced_response, _eval_output in track(
        results, total=len(results), description="Uploading scores"
    ):
        # Link the trace to the dataset item for analysis
        if _eval_output is not None:
            langfuse_client.create_score(
                name="hallucination_score",
                value=_eval_output.hallucination,
                comment=_eval_output.explanation,
                trace_id=_traced_response.trace_id,
            )

    flush_langfuse()

    asyncio.run(async_weaviate_client.close())
