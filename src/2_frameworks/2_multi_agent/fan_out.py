"""Example showing fan-out for identifying conflicting info between document chunks.

N Documents
-> O(N^2) first-pass agents
-> O(N) summaries

TODO:
- replace conflict-detection with your own "second-order" task
- revise error handling- at the moment, if there is an exception,
    the pair would be skipped. You might want to set up e.g., retry.

uv run --env-file .env src/2_frameworks/2_multi_agent/fan_out.py \
--source_dataset laliyepeng/test-cra-dataset \
--num_rows 10 \
--output_report report.md
"""

import argparse
import asyncio
import json
import re
from collections import defaultdict
from typing import Any, Sequence

import agents
import datasets
import openai
import pydantic

from src.utils import set_up_logging, setup_langfuse_tracer
from src.utils.async_utils import gather_with_progress, rate_limited
from src.utils.client_manager import AsyncClientManager
from src.utils.langfuse.shared_client import langfuse_client


MAX_CONCURRENCY = {"worker": 50, "reviewer": 50}
MAX_GENERATED_TOKENS = {"worker": 16384, "reviewer": 32768}


Document = dict[str, Any]


def _condense_whitespaces(text: str) -> str:
    """Compress consecutive whitespaces into one.

    If there is a new line among the whitespaces, replace with a new line.

    Otherwise, replace with a space.
    """
    return re.sub(r"\s+", lambda match: "\n" if "\n" in match.group(0) else " ", text)


def _render_dict_as_markdown(
    data: dict[str, Any], as_codeblock: Sequence[str] = ("text",)
) -> str:
    """Render key-value pairs in a dictionary as a markdown list.

    Fields marked as `as_codeblock`, if present, are rendered as a codeblock instead.
    """
    return "\n".join(
        f"- **{k}**: \n```\n{v}\n```" if k in as_codeblock else f"- **{k}**: {v}"
        for k, v in data.items()
    )


class _BaseModel(pydantic.BaseModel):
    """Base Model but use JSON for __repr__."""

    def __repr__(self) -> str:
        return _condense_whitespaces(self.model_dump_json())


class DocumentPair(_BaseModel):
    """One pair of documents to compare."""

    document_ids: list[int]
    documents: list[Document]

    def get_prompt(self) -> str:
        """Get LLM query text."""
        return _condense_whitespaces(json.dumps(self.documents))


class Conflict(_BaseModel):
    """Description of a conflict."""

    source_snippets: list[str]
    explanation: str

    def as_markdown(self) -> str:
        """Markdown representation of self."""
        source_snippets = "\n---\n".join(self.source_snippets)

        return f"""\

{source_snippets}

---

{self.explanation}

"""


class ConflictSummary(_BaseModel):
    """Summary of conflicts found, if any."""

    analysis: str
    conflicts: list[Conflict] = []


class DocumentPairFlagged(DocumentPair):
    """A document pair for which there is a conflict."""

    summary: ConflictSummary


class ConflictingDocument(_BaseModel):
    """Conflicts with a document."""

    summary: ConflictSummary
    document: Document
    document_id: int = pydantic.Field(exclude=True)

    def get_report(self) -> str:
        """Render conflict as markdown."""
        return f"""\
### Conflict with document {self.document_id}

{_render_dict_as_markdown(self.document)}

---

{self.summary.analysis}

---

{"\n---\n".join(_conflict.as_markdown() for _conflict in self.summary.conflicts)}
"""


class ConflictedDocument(_BaseModel):
    """Conflicts related to one document."""

    document: Document
    document_ids: list[int] = pydantic.Field(exclude=True)
    conflicting_documents: list[ConflictingDocument]


class ConflictReview(_BaseModel):
    """Summary and review of suggested conflicts."""

    analysis: str
    is_conflict_valid: bool


class ReviewedDocument(_BaseModel):
    """Conflict review on a document, plus that document itself."""

    conflicted_document: ConflictedDocument
    review: ConflictReview

    def get_report(self) -> str:
        """Markdown report on this document."""
        conflict_reports = "\n---\n".join(
            _conflicting.get_report()
            for _conflicting in self.conflicted_document.conflicting_documents
        )

        return f"""\
## Document {self.conflicted_document.document_ids[0]}: \
{"likely conflict" if self.review.is_conflict_valid else "possible conflict"}

{self.review.analysis}

### Original Document

{_condense_whitespaces(_render_dict_as_markdown(self.conflicted_document.document))}

### Individual Reports

{conflict_reports}

"""


def build_document_pairs(documents: list[Document]) -> list[DocumentPair]:
    """Build ((N - 1) * (N - 2) / 2) document pairs from N documents."""
    output: list[DocumentPair] = []
    # Document 1 pairs with 2, ..., N; Document 2 pairs with 3, ..., N, etc.
    for index, document_a in enumerate(documents):
        for offset, document_b in enumerate(documents[index + 1 :]):
            output.append(
                DocumentPair(
                    document_ids=[index, index + (offset + 1)],
                    documents=[document_a, document_b],
                )
            )

    return output


def group_conflicts(
    flagged_pairs: list[DocumentPairFlagged],
) -> list[ConflictedDocument]:
    """Group conflict pairs by the first document.

    Example:
        Input: [(a, b), (a, c), (b, c)].
        Output: [(a, b, c), (b, c)]
    """
    document_by_id: dict[int, Document] = {
        _id: _document
        for _pair in flagged_pairs
        for _id, _document in zip(_pair.document_ids, _pair.documents)
    }

    # Map document id to a list of second documents in each pair.
    conflicts_by_id: dict[int, list[ConflictingDocument]] = defaultdict(list)
    for _pair in flagged_pairs:
        _conflict = ConflictingDocument(
            summary=_pair.summary,
            document=_pair.documents[-1],
            document_id=_pair.document_ids[-1],
        )
        conflicts_by_id[_pair.document_ids[0]].append(_conflict)

    # Replace document_id values in conflicts_by_id with actual documents.
    return [
        ConflictedDocument(
            document=document_by_id[_id],
            document_ids=[_id, *[_conflict.document_id for _conflict in _conflicts]],
            conflicting_documents=_conflicts,
        )
        for _id, _conflicts in conflicts_by_id.items()
    ]


async def process_document_pair(document_pair: DocumentPair) -> ConflictSummary | None:
    """Process one document pair.

    Returns None if exception is encountered.
    """
    with langfuse_client.start_as_current_observation(
        name="Conflict - suggest", as_type="agent"
    ) as obs:
        try:
            result = await agents.Runner.run(
                worker_agent, input=document_pair.get_prompt()
            )
            output = result.final_output_as(ConflictSummary)
        except agents.AgentsException as e:
            print(e)
            return None

        obs.update(input=document_pair, output=output)

    return output


async def process_fan_out(
    document_pairs: list[DocumentPair], max_concurrency: int = MAX_CONCURRENCY["worker"]
) -> list[DocumentPairFlagged]:
    """Process fan out on the given document pairs.

    O(N^2) pairs for N documents.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    coros = [
        rate_limited(lambda _pair=_pair: process_document_pair(_pair), semaphore)
        for _pair in document_pairs
    ]
    results = await gather_with_progress(coros, "Fan out...")
    return [
        DocumentPairFlagged(**pair.model_dump(), summary=summary)
        for pair, summary in zip(document_pairs, results)
        if summary and (len(summary.conflicts) > 0)
    ]


async def process_one_review(
    conflicted_document: ConflictedDocument,
) -> ConflictReview | None:
    """Generate review for conflicts proposed around one document.

    Return None upon error.
    """
    with langfuse_client.start_as_current_observation(
        name="Review proposal", as_type="agent"
    ) as obs:
        try:
            result = await agents.Runner.run(
                conflict_review_agent, input=conflicted_document.model_dump_json()
            )
            output = result.final_output_as(ConflictReview)
        except agents.AgentsException as e:
            print(e)
            return None

        obs.update(input=conflicted_document, output=output)

    return output


async def process_conflict_reviews(
    conflicted_documents: list[ConflictedDocument],
    max_concurrency: int = MAX_CONCURRENCY["reviewer"],
    valid_only: bool = False,
) -> list[ReviewedDocument]:
    """Generate review of identified conflicts for each document.

    O(N) for N documents.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    coros = [
        rate_limited(
            lambda _conflicted=_conflicted: process_one_review(_conflicted),
            semaphore=semaphore,
        )
        for _conflicted in conflicted_documents
    ]
    reviews = await gather_with_progress(coros, description="Reviewing conflicts...")
    return [
        ReviewedDocument(conflicted_document=_conflicted, review=_review)
        for _conflicted, _review in zip(conflicted_documents, reviews)
        if _review and (_review.is_conflict_valid or not valid_only)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dataset", required=True)
    parser.add_argument(
        "--num_rows", default=-1, type=int, help="Set to -1 to select all rows."
    )
    parser.add_argument("--output_report", default="report.md")
    args = parser.parse_args()

    set_up_logging()
    setup_langfuse_tracer()

    client_manager = AsyncClientManager()

    worker_agent = agents.Agent(
        "Conflict-detection Agent",
        instructions=(
            "Identify conflicting information (if any) in and between these documents. "
            "Be sure to show your reasoning, even if there is no conflict. "
            "If no conflict is found between the two documents, use an empty list."
        ),
        output_type=ConflictSummary,
        model=agents.OpenAIChatCompletionsModel(
            model=client_manager.configs.default_worker_model,
            openai_client=client_manager.openai_client,
        ),
        model_settings=agents.ModelSettings(
            reasoning=openai.types.Reasoning(
                effort="high", generate_summary="detailed"
            ),
            max_tokens=MAX_GENERATED_TOKENS["worker"],
        ),
    )

    conflict_review_agent = agents.Agent(
        "Conflict-review agent",
        instructions=(
            "Given the documents suggested to be in conflict with information "
            "in the current document, analyze whether the suggestions are valid."
        ),
        output_type=ConflictReview,
        model=agents.OpenAIChatCompletionsModel(
            model=client_manager.configs.default_planner_model,
            openai_client=client_manager.openai_client,
        ),
        model_settings=agents.ModelSettings(
            reasoning=openai.types.Reasoning(
                effort="high", generate_summary="detailed"
            ),
            max_tokens=MAX_GENERATED_TOKENS["reviewer"],
        ),
    )

    dataset_dict = datasets.load_dataset(args.source_dataset)
    assert isinstance(dataset_dict, datasets.DatasetDict)
    documents = list(dataset_dict["train"])[: args.num_rows]

    with langfuse_client.start_as_current_observation(
        name="Fan-Out", as_type="chain", input=args.source_dataset
    ) as span:
        # Run O(N^2) agents on N documents to identify pairwise e.g., conflicts.
        document_pairs = build_document_pairs(documents)  # type: ignore[arg-type]
        print(f"Built {len(document_pairs)} pair(s) from {len(documents)} document(s).")

        with langfuse_client.start_as_current_observation(
            name="Conflicts - Pairwise", as_type="chain"
        ) as obs:
            flagged_pairs = asyncio.get_event_loop().run_until_complete(
                process_fan_out(document_pairs)
            )
            obs.update(
                input=args.source_dataset,
                output=f"{len(flagged_pairs)} pairs identified.",
            )

        # Collect conflicts related to each document.
        # from O(N^2) pairs to O(N) summarized per-document conflicts.
        conflicted_documents = group_conflicts(flagged_pairs)

        # Review these O(N) per-document conflicts.
        with langfuse_client.start_as_current_observation(
            name="Conflicts - Review", as_type="chain"
        ) as obs:
            conflict_reviews: list[ReviewedDocument] = (
                asyncio.get_event_loop().run_until_complete(
                    process_conflict_reviews(conflicted_documents)
                )
            )
            obs.update(input=conflicted_documents, output=conflict_reviews)

        # Generate markdown output
        with open(args.output_report, "w") as output_file:
            reports = [_review.get_report() for _review in conflict_reviews]
            output_file.write("\n".join(reports))
            print(f"Writing report to {args.output_report}.")

        span.update(output="Wrote report to " + args.output_report)
