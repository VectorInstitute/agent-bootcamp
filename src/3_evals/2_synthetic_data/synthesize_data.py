"""Synthesize additional test data using Agent to match style but maintain diversity.

Overview:
- Obtain a list of topics.
- Run a knowledgebase agent to generate question-answer pairs regarding the list
  of topics. Use few-shot in-context learning to match style of input.
- Upload dataset to Langfuse
"""

import argparse
import asyncio
import random

import agents
import pydantic
from dotenv import load_dotenv
from openai import AsyncOpenAI
from rich.progress import track

from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    get_weaviate_async_client,
    pretty_print,
    setup_langfuse_tracer,
)
from src.utils.data import get_dataset, get_dataset_url_hash
from src.utils.langfuse.shared_client import langfuse
from src.utils.tools.news_events import NewsEvent, get_news_events


load_dotenv(verbose=True)

SYSTEM_MESSAGE = """\
Example questions: \
{example_questions}

Given the example questions, produce 5 additional questions of the same format \
regarding the topic that the user specified with the help of the knowledge base search tool. \
Be sure to include URL citations (citation_urls) in your response. \

You MUST produce a JSON output of this format:
{json_schema}
"""


class _SyntheticTestCase(pydantic.BaseModel):
    """Represents one synthetic test case."""

    question: str
    expected_answer: str
    citation_urls: list[str]


class _SyntheticTestCases(pydantic.RootModel):
    """List of synthetic test cases."""

    root: list[_SyntheticTestCase]


async def generate_synthetic_test_cases(
    test_case_generator_agent: agents.Agent,
    structured_output_agent: agents.Agent,
    news_event: "NewsEvent",
) -> list[_SyntheticTestCase]:
    """Generate synthetic test cases using agent pipeline.

    Params
    ------
        test_case_generator_agent: main agent for generating test case.
        structured_output_agent: agent for structuring the output.
        news_event: news event to write about.

    Returns
    -------
        _SyntheticTestCases
    """
    with langfuse.start_as_current_span(name="generate_synthetic_test_cases") as span:
        test_case_query = (
            "Generate test question-answer pairs based on this news event: \n"
            + news_event.model_dump_json(indent=2)
        )
        raw_response = await agents.Runner.run(
            test_case_generator_agent, input=test_case_query
        )
        print(raw_response.final_output)

        structured_response = await agents.Runner.run(
            structured_output_agent,
            input=raw_response.final_output,
        )
        synthetic_test_cases = structured_response.final_output_as(
            list[_SyntheticTestCase]
        )

        span.update_trace(
            input=test_case_query,
            output=synthetic_test_cases,
            metadata={"generated_test_case": raw_response.final_output},
        )

    return synthetic_test_cases


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

    setup_langfuse_tracer()

    generator = random.Random(0)
    dataset_name_hash = get_dataset_url_hash(args.langfuse_dataset_name)

    async_openai_client = AsyncOpenAI()
    agents.set_tracing_disabled(disabled=True)

    try:
        # Create langfuse dataset and upload.
        langfuse.create_dataset(
            name=args.langfuse_dataset_name,
            description=f"[{dataset_name_hash}] Synthetic data based on {args.source_dataset}",
            metadata={
                "name_hash": dataset_name_hash,
                "reference_source": args.source_dataset,
                "type": "synthetic_benchmark",
            },
        )

        df = get_dataset(args.source_dataset, limit=args.limit)
        rows_news_only = [
            row.to_dict() for _, row in df[df["area"] == "Knowledge"].iterrows()
        ]
        rows_filtered = [
            {k: v for k, v in row.items() if k not in ("id", "category", "area")}
            for row in rows_news_only
        ]

        example_questions = generator.choices(rows_filtered, k=5)
        example_questions_str = pretty_print(example_questions)
        test_case_generator_agent = agents.Agent(
            name="Test Case Generator Agent",
            instructions=SYSTEM_MESSAGE.format(
                example_questions=example_questions_str,
                json_schema=_SyntheticTestCases.model_json_schema(),
            ),
            # Hint: replace this tool with your own knowledge base search tool.
            tools=[agents.function_tool(async_knowledgebase.search_knowledgebase)],
            model=agents.OpenAIChatCompletionsModel(
                model="gemini-2.5-flash", openai_client=async_openai_client
            ),
        )
        structured_output_agent = agents.Agent(
            name="Structured Output Agent",
            instructions="Extract the structured output from the given text.",
            output_type=list[_SyntheticTestCase],
            model=agents.OpenAIChatCompletionsModel(
                model="gemini-2.5-flash", openai_client=async_openai_client
            ),
        )

        news_events_by_category = await get_news_events()
        all_news_events = [
            item for items in news_events_by_category.root.values() for item in items
        ]
        if len(all_news_events) == 0:
            raise ValueError("Cannot retrieve list of news headlines.")

        # Randomly sample (might include repetitons) up to news event.
        news_events = generator.sample(all_news_events, k=args.limit)

        # Run generation async
        # _coros = [
        #     generate_synthetic_test_cases(
        #         test_case_generator_agent=test_case_generator_agent,
        #         news_event=_event,
        #     )
        #     for _event in news_events
        # ]
        # results = asyncio.run(
        #     gather_with_progress(_coros, description="Generating synthetic test cases...")
        # )

        # all_examples = [_test_case for _test_cases in results for _test_case in _test_cases]

        # # Upload to Langfuse
        # for idx, _test_case in enumerate(
        #     track(all_examples, description="Uploading to Langfuse")
        # ):
        #     langfuse.create_dataset_item(
        #         dataset_name=args.langfuse_dataset_name,
        #         input={"text": _test_case.question},
        #         expected_output={"text": _test_case.expected_answer},
        #         metadata=_test_case.model_dump(),
        #         # unique id to enable upsert without creating duplicates
        #         id=f"{dataset_name_hash}-{idx:05}",
        #     )

        results: list[list[_SyntheticTestCase]] = []
        for _event in track(
            news_events,
            total=len(news_events),
            description="Generating synthetic test cases...",
        ):
            # Generate synthetic test cases
            result = await generate_synthetic_test_cases(
                test_case_generator_agent=test_case_generator_agent,
                structured_output_agent=structured_output_agent,
                news_event=_event,
            )
            results.append(result)

        all_examples = [
            _test_case for _test_cases in results for _test_case in _test_cases
        ]

        # Upload to Langfuse
        for idx, _test_case in enumerate(
            track(all_examples, description="Uploading to Langfuse")
        ):
            langfuse.create_dataset_item(
                dataset_name=args.langfuse_dataset_name,
                input={"text": _test_case.question},
                expected_output={
                    "text": _test_case.expected_answer,
                    "citation_urls": _test_case.citation_urls,
                },
                metadata=_test_case.model_dump(),
                # unique id to enable upsert without creating duplicates
                id=f"{dataset_name_hash}-{idx:05}",
            )
    finally:
        await async_weaviate_client.close()
        await async_openai_client.close()

        # Flush Langfuse traces
        langfuse.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dataset", required=True)
    parser.add_argument("--langfuse_dataset_name", required=True)
    parser.add_argument("--limit", type=int, default=18)

    asyncio.run(_main())
