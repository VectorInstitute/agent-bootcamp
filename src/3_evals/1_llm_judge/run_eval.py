"""Evaluate agent on dataset using LLM-as-a-Judge."""

import argparse
import asyncio

import agents
import pydantic
from elasticsearch import AsyncElasticsearch
from langfuse.client import DatasetItemClient, StatefulTraceClient
from openai import AsyncOpenAI
from opentelemetry import trace as otlp_trace
from rich.progress import track

from src.utils import (
    AsyncESKnowledgeBase,
    Configs,
    gather_with_progress,
    get_langfuse_tracer,
)
from src.utils.langfuse.shared_client import langfuse as langfuse_client


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
) -> "tuple[StatefulTraceClient, str]":
    """Run OpenAI Agent on query, returning response and trace."""
    with tracer.start_as_current_span("OpenAI-Agent-Trace") as span:
        span.set_attribute("langfuse.tag", "dataset-run")

        result = await agents.Runner.run(agent, query)

        # Get the Langfuse trace_id to link the dataset run item to the agent trace
        current_span = otlp_trace.get_current_span()
        span_context = current_span.get_span_context()
        trace_id = span_context.trace_id
        formatted_trace_id = otlp_trace.format_trace_id(trace_id)

        langfuse_trace = langfuse_client.trace(
            id=formatted_trace_id, input=query, output=result.final_output
        )

    if "|" in result.final_output:
        short_answer = result.final_output.split("|")[-1]
    else:
        short_answer = result.final_output

    return langfuse_trace, short_answer


async def run_evaluator_agent(evaluator_query: EvaluatorQuery) -> EvaluatorResponse:
    """Evaluate using evaluator agent."""
    evaluator_agent = agents.Agent(
        "Evaluator Agent",
        instructions=EVALUATOR_INSTRUCTIONS,
        output_type=EvaluatorResponse,
    )

    result = await agents.Runner.run(evaluator_agent, input=evaluator_query.get_query())
    return result.final_output_as(EvaluatorResponse)


async def run_and_evaluate(
    main_agent: agents.Agent, lf_dataset_item: "DatasetItemClient"
) -> "tuple[StatefulTraceClient, EvaluatorResponse]":
    """Run main agent and evaluator agent on one dataset instance."""
    expected_output = lf_dataset_item.expected_output
    assert expected_output is not None

    langfuse_trace, short_answer = await run_agent_with_trace(
        main_agent, query=lf_dataset_item.input["text"]
    )
    evaluator_response = await run_evaluator_agent(
        EvaluatorQuery(
            question=lf_dataset_item.input["text"],
            ground_truth=expected_output["text"],
            proposed_response=short_answer,
        )
    )

    return langfuse_trace, evaluator_response


parser = argparse.ArgumentParser()
parser.add_argument("--langfuse_dataset_name", required=True)
parser.add_argument("--run_name", required=True)


if __name__ == "__main__":
    args = parser.parse_args()

    lf_dataset = langfuse_client.get_dataset(args.langfuse_dataset_name)
    configs = Configs.from_env_var()
    async_es_client = AsyncElasticsearch(configs.es_host, api_key=configs.es_api_key)
    async_openai_client = AsyncOpenAI()
    async_knowledgebase = AsyncESKnowledgeBase(
        async_es_client,
        es_collection_name="enwiki-20250501",
    )
    tracer = get_langfuse_tracer()

    main_agent = agents.Agent(
        name="Wikipedia Agent",
        instructions=SYSTEM_MESSAGE,
        tools=[agents.function_tool(async_knowledgebase.search_knowledgebase)],
        model=agents.OpenAIChatCompletionsModel(
            model="gpt-4o-mini", openai_client=async_openai_client
        ),
    )
    coros = [run_and_evaluate(main_agent, _item) for _item in lf_dataset.items]
    results = asyncio.run(
        gather_with_progress(coros, description="Running agent and evaluating")
    )

    for _dataset_item, (_trace, _eval_output) in track(
        zip(lf_dataset.items, results),
        total=len(results),
        description="Uploading scores",
    ):
        # Link the trace to the dataset item for analysis
        _dataset_item.link(_trace, run_name=args.run_name)
        _trace.score(
            name="is_answer_correct",
            value=_eval_output.is_answer_correct,
            comment=_eval_output.explanation,
        )

    asyncio.run(async_es_client.close())
