"""Synthesize additional test data using Agent to match style but maintain diversity.

Overview:
- Define a list of example questions (few-shot prompting.)
- Prompt agent to explore the data files using the code-interpreter and
    generate more questions of that format.
- Extract structured data from agent output.
- Upload dataset to Langfuse.

Example:
```
uv run --env-file .env src/3_evals/2_synthetic_data/synthesize_data_e2b.py \
--langfuse_dataset_name e2b-synthetic-20251113-1a \
--limit 36 \
--max_concurrency 20
```
(Same concurrency as E2B limit.)
"""

import argparse
import asyncio
import random
from pathlib import Path

import agents
import pydantic
from dotenv import load_dotenv
from rich.progress import track

from src.utils import (
    CodeInterpreter,
    gather_with_progress,
    pretty_print,
    rate_limited,
    set_up_logging,
    setup_langfuse_tracer,
)
from src.utils.client_manager import AsyncClientManager
from src.utils.data import get_dataset_url_hash
from src.utils.langfuse.shared_client import langfuse_client


SYSTEM_MESSAGE = """\
Example questions: \
{example_questions}

Explore the SQL databases under /data using the `run_code` tool.
Produce 5 additional questions of the same format as the example questions.
For each test case, you should produce a JSON output of this format:
{json_schema}

Be specific with your question and name the SQL file and table involved,
so each "question" is self-contained and can be answered on its own.
"""


class _Citation(pydantic.BaseModel):
    """Represents one cited source."""

    title: str
    section: str | None = None


class _SyntheticTestCase(pydantic.BaseModel):
    """Represents one synthetic test case."""

    question: str
    expected_answer: str
    citations: list[_Citation]


async def generate_synthetic_test_cases(
    test_case_generator_agent: agents.Agent,
) -> list[_SyntheticTestCase] | None:
    """Generate synthetic test cases using agent pipeline.

    Params
    ------
        test_case_generator_agent: main agent for generating test case.

    Returns
    -------
        list[_SyntheticTestCase] or None if error is encountered.
    """
    try:
        structured_output_agent = agents.Agent(
            name="Structured Output Agent",
            instructions="Extract the structured output from the given text.",
            output_type=list[_SyntheticTestCase],
            model=agents.OpenAIChatCompletionsModel(
                model=client_manager.configs.default_planner_model,
                openai_client=client_manager.openai_client,
            ),
        )

        with langfuse_client.start_as_current_observation(
            name="generate_synthetic_test_cases", as_type="agent"
        ):
            raw_response = await agents.Runner.run(
                test_case_generator_agent,
                input="Generate test question-answer pairs based on files under /data",
            )
            structured_response = await agents.Runner.run(
                structured_output_agent,
                input=raw_response.final_output,
            )

        return structured_response.final_output_as(list[_SyntheticTestCase])
    except Exception as e:
        print(f"Encountered exception; skipping: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--langfuse_dataset_name", required=True)
    parser.add_argument("--limit", type=int, default=18)
    parser.add_argument("--max_concurrency", type=int, default=3)
    args = parser.parse_args()

    load_dotenv(verbose=True)

    set_up_logging()

    client_manager = AsyncClientManager()
    setup_langfuse_tracer()

    code_interpreter = CodeInterpreter(
        template_name=client_manager.configs.default_code_interpreter_template,
        local_files=[
            Path("sandbox_content/"),
            Path("tests/tool_tests/example_files/example_a.csv"),
        ],
    )

    example_questions = [
        _SyntheticTestCase(
            question="How many airports are listed in Airlines.sqlite?",
            expected_answer="104",
            citations=[
                _Citation(
                    title="Airlines.sqlite",
                    section="SELECT COUNT(DISTINCT airport_code) FROM airports_data;",
                )
            ],
        ),
        _SyntheticTestCase(
            question="What unique aircraft codes are listed in Airlines.sqlite?",
            expected_answer="773,763,SU9,320,321,319,733,CN1,CR2",
            citations=[
                _Citation(
                    title="Airlines.sqlite",
                    section="SELECT DISTINCT aircraft_code FROM aircrafts_data;",
                )
            ],
        ),
    ]
    example_questions_str = pretty_print(example_questions)

    test_case_generator_agent = agents.Agent(
        name="Test Case Generator Agent",
        instructions=SYSTEM_MESSAGE.format(
            example_questions=example_questions_str,
            json_schema=_SyntheticTestCase.model_json_schema(),
        ),
        tools=[agents.function_tool(code_interpreter.run_code)],
        model=agents.OpenAIChatCompletionsModel(
            model=client_manager.configs.default_planner_model,
            openai_client=client_manager.openai_client,
        ),
    )

    generator = random.Random(0)
    dataset_name_hash = get_dataset_url_hash(args.langfuse_dataset_name)

    # Create langfuse dataset and upload.
    langfuse_client.create_dataset(
        name=args.langfuse_dataset_name,
        description=f"[{dataset_name_hash}] Synthetic data",
        metadata={
            "name_hash": dataset_name_hash,
            "type": "synthetic_benchmark",
        },
    )

    # Run generation async
    semaphore = asyncio.Semaphore(args.max_concurrency)
    _coros = [
        rate_limited(
            lambda: generate_synthetic_test_cases(
                test_case_generator_agent=test_case_generator_agent,
            ),
            semaphore=semaphore,
        )
        for _ in range(args.limit)
    ]
    results = asyncio.run(
        gather_with_progress(_coros, description="Generating synthetic test cases...")
    )

    all_examples = [
        _test_case
        for _test_cases in results
        if _test_cases is not None
        for _test_case in _test_cases
    ]

    # Upload to Langfuse
    for idx, _test_case in enumerate(
        track(all_examples, description="Uploading to Langfuse")
    ):
        langfuse_client.create_dataset_item(
            dataset_name=args.langfuse_dataset_name,
            input={"text": _test_case.question},
            expected_output={"text": _test_case.expected_answer},
            metadata=_test_case.model_dump(),
            # unique id to enable upsert without creating duplicates
            id=f"{dataset_name_hash}-{idx:05}",
        )
