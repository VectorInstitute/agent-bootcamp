"""Evaluate agent on dataset using LLM-as-a-Judge."""

import argparse
import asyncio

import agents
import pydantic
from dotenv import load_dotenv
from langfuse._client.datasets import DatasetItemClient
from openai import AsyncOpenAI
from rich.progress import Progress, SpinnerColumn, TextColumn, track

from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    gather_with_progress,
    get_weaviate_async_client,
    set_up_logging,
    setup_langfuse_tracer,
)
from src.utils.langfuse.shared_client import langfuse_client


load_dotenv(verbose=True)
set_up_logging()


SYSTEM_MESSAGE = """\
Answer the question using the search tool. \
EACH TIME before invoking the function, you must explain your reasons for doing so. \
Be sure to mention the sources in your response. \
If the search did not return intended results, try again. \
Do not make up information.

Finally, write "|" and include a one-sentence summary of your answer.
"""

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
    ground_truth: str
    proposed_response: str

    def get_query(self) -> str:
        """Obtain query string to the evaluator agent."""
        return EVALUATOR_TEMPLATE.format(**self.model_dump())


class EvaluatorResponse(pydantic.BaseModel):
    """Typed response from the evaluator."""

    explanation: str
    is_answer_correct: bool


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
        instructions=EVALUATOR_INSTRUCTIONS,
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
    expected_output = lf_dataset_item.expected_output
    assert expected_output is not None

    with lf_dataset_item.run(run_name=run_name):
        traced_response = await run_agent_with_trace(
            main_agent, query=lf_dataset_item.input["text"]
        )

    answer = traced_response.answer
    if answer is None:
        return traced_response, None

    evaluator_response = await run_evaluator_agent(
        EvaluatorQuery(
            question=lf_dataset_item.input["text"],
            ground_truth=expected_output["text"],
            proposed_response=answer,
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

    tracer = setup_langfuse_tracer()

    main_agent = agents.Agent(
        name="Wikipedia Agent",
        instructions=SYSTEM_MESSAGE,
        tools=[agents.function_tool(async_knowledgebase.search_knowledgebase)],
        model=agents.OpenAIChatCompletionsModel(
            model="gemini-2.5-flash", openai_client=async_openai_client
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
                name="is_answer_correct",
                value=_eval_output.is_answer_correct,
                comment=_eval_output.explanation,
                trace_id=_traced_response.trace_id,
            )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Finalizing Langfuse annotations...", total=None)
        langfuse_client.flush()

    asyncio.run(async_weaviate_client.close())
