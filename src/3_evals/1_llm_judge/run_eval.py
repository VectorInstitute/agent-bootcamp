"""Evaluate agent on dataset using LLM-as-a-Judge."""

import argparse
import asyncio
import logging

import agents
import pydantic
from dotenv import load_dotenv
from langfuse._client.datasets import DatasetItemClient
from openai import AsyncOpenAI
from opentelemetry import trace as otlp_trace
from rich.progress import Progress, SpinnerColumn, TextColumn, track
from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    get_weaviate_async_client,
    setup_langfuse_tracer,
)
from src.utils.langfuse.shared_client import langfuse as langfuse_client

load_dotenv(verbose=True)
logger = logging.getLogger(__name__)

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


async def run_agent_with_trace(agent: agents.Agent, query: str) -> str | None:
    """Run OpenAI Agent on query, returning response and trace.

    Returns None if agent exceeds max_turn limit.
    """
    with langfuse_client.start_as_current_span(
        name="OpenAI-Agent-Trace", input=query
    ) as span:
        span.update_trace(tags=["dataset-run"])

        try:
            result = await agents.Runner.run(agent, query)
            if "|" in result.final_output:
                answer = result.final_output.split("|")[-1].strip()
            else:
                answer = result.final_output

        except agents.MaxTurnsExceeded:
            answer = None

        # Get the Langfuse trace_id to link the dataset run item to the agent trace
        current_span = otlp_trace.get_current_span()
        span_context = current_span.get_span_context()
        trace_id = span_context.trace_id
        formatted_trace_id = otlp_trace.format_trace_id(trace_id)

        span.update_trace(user_id=formatted_trace_id, output=answer)

    return answer


async def run_evaluator_agent(
    evaluator_agent: agents.Agent, evaluator_query: EvaluatorQuery
) -> EvaluatorResponse:
    """Evaluate using evaluator agent."""
    result = await agents.Runner.run(evaluator_agent, input=evaluator_query.get_query())
    return result.final_output_as(EvaluatorResponse)


async def run_and_evaluate(
    main_agent: agents.Agent,
    evaluator_agent: agents.Agent,
    lf_dataset_item: "DatasetItemClient",
) -> EvaluatorResponse | None:
    """Run main agent and evaluator agent on one dataset instance.

    Returns None if main agent returned a None answer.
    """
    expected_output = lf_dataset_item.expected_output
    assert expected_output is not None

    answer = await run_agent_with_trace(main_agent, query=lf_dataset_item.input["text"])

    if answer is None:
        return None

    return await run_evaluator_agent(
        evaluator_agent,
        EvaluatorQuery(
            question=lf_dataset_item.input["text"],
            ground_truth=expected_output["text"],
            proposed_response=answer,
        ),
    )


async def _main() -> None:
    args = parser.parse_args()

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
    async_knowledgebase = AsyncWeaviateKnowledgeBase(
        async_weaviate_client, collection_name="enwiki_20250520"
    )

    async_openai_client = AsyncOpenAI()
    agents.set_tracing_disabled(disabled=True)

    try:
        # Setup langfuse
        setup_langfuse_tracer()

        lf_dataset_items = langfuse_client.get_dataset(args.langfuse_dataset_name).items
        if args.limit is not None:
            lf_dataset_items = lf_dataset_items[: args.limit]

        main_agent = agents.Agent(
            name="Wikipedia Agent",
            instructions=SYSTEM_MESSAGE,
            tools=[agents.function_tool(async_knowledgebase.search_knowledgebase)],
            model=agents.OpenAIChatCompletionsModel(
                model="gemini-2.5-flash", openai_client=async_openai_client
            ),
        )
        evaluator_agent = agents.Agent(
            name="Evaluator Agent",
            instructions=EVALUATOR_INSTRUCTIONS,
            output_type=EvaluatorResponse,
            model=agents.OpenAIChatCompletionsModel(
                model="gemini-2.5-flash", openai_client=async_openai_client
            ),
        )

        for _item in track(
            lf_dataset_items,
            total=len(lf_dataset_items),
            description="Running agent and evaluating",
        ):
            _eval_output = await run_and_evaluate(main_agent, evaluator_agent, _item)

            with _item.run(run_name=args.run_name) as span:
                if _eval_output is not None:
                    span.score(
                        name="is_answer_correct",
                        value=_eval_output.is_answer_correct,
                        comment=_eval_output.explanation,
                    )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            _ = progress.add_task("Finalizing Langfuse annotations...", total=None)
            langfuse_client.flush()
    finally:
        await async_weaviate_client.close()
        await async_openai_client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--langfuse_dataset_name", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--limit", type=int)

    asyncio.run(_main())
