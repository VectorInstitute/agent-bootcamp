"""Add diversity metrics to the given LangFuse dataset.

Usage:

uv run \
--env-file .env \
-m src.3_evals.2_synthetic_data.annotate_diversity \
--langfuse_dataset_name ${DATASET_NAME} \
--run_name cosine_similarity_bge_m3_20250716 \
--limit 18
"""

import argparse
import asyncio
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pydantic
from openai import AsyncOpenAI
from rich.progress import track

from src.utils import Configs, create_batches, gather_with_progress
from src.utils.langfuse.shared_client import flush_langfuse, langfuse_client


if TYPE_CHECKING:
    from langfuse._client.datasets import DatasetItemClient


parser = argparse.ArgumentParser()
parser.add_argument("--langfuse_dataset_name", required=True)
parser.add_argument("--run_name", default="cosine_similarity")
parser.add_argument("--limit", type=int)
parser.add_argument("--embed_batch_size", type=int, default=18)


class EmbeddingResult(pydantic.BaseModel):
    """Tracks trace_id and embedding vector for an instance."""

    langfuse_trace_id: str
    embedding: list[float]


async def batch_embed(
    dataset_items: list["DatasetItemClient"],
    oai_client: AsyncOpenAI,
    model_name: str,
    run_name: str | None = None,
) -> list[EmbeddingResult]:
    """Run embedding on the given LangFuse dataset items."""
    output: list[EmbeddingResult] = []
    run_name = run_name or f"embed_{model_name}"

    texts = [_item.input["text"] for _item in dataset_items]
    embedding_response = await oai_client.embeddings.create(
        input=texts, model=model_name
    )
    embeddings = [_data.embedding for _data in embedding_response.data]
    assert len(dataset_items) == len(embeddings), (len(dataset_items), len(embeddings))

    for _lf_dataset_item, _embedding in zip(dataset_items, embeddings):
        with _lf_dataset_item.run(run_name=run_name) as span:
            _result = EmbeddingResult(
                langfuse_trace_id=span.trace_id,
                embedding=_embedding,
            )
            output.append(_result)

    return output


def _avg_cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    """
    Memory-efficient calculation of instance-wise cosine similarity with population.

    Copied from:

    github.com/ComplexData-MILA/AIF-Gen/\
    blob/10f500c68517b34aa4c07381f2cf67ea6043d73c/\
    aif_gen/validation/embedding_diversity.py#L155

    (MIT)
    """
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1e-10  # avoid division by zero

    normalized_matrix = matrix / row_norms  # shape: (n, m)
    sum_normalized = normalized_matrix.sum(axis=0)  # shape: (m,)
    return normalized_matrix.dot(sum_normalized) / matrix.shape[0]


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.embed_batch_size > 0, "args.embed_batch_size must be at least 1."

    lf_dataset_items = langfuse_client.get_dataset(args.langfuse_dataset_name).items
    limit = (
        min(len(lf_dataset_items), args.limit) if args.limit else len(lf_dataset_items)
    )
    print(f"Items to process: {limit}")

    configs = Configs()
    async_embed_client = AsyncOpenAI(
        api_key=configs.embedding_api_key,
        base_url=configs.embedding_base_url,
        max_retries=5,
    )

    # Construct embed batches.
    batches: list[list["DatasetItemClient"]] = create_batches(
        lf_dataset_items,
        batch_size=args.embed_batch_size,
        limit=args.limit,
        keep_trailing=True,
    )

    # Async embed, traced.
    embed_coros = [
        batch_embed(
            _batch,
            oai_client=async_embed_client,
            model_name=configs.embedding_model_name,
            run_name=args.run_name,
        )
        for _batch in batches
    ]
    batched_embed_results = asyncio.run(
        gather_with_progress(embed_coros, description="Embedding")
    )
    embed_results = [
        _result for _results in batched_embed_results for _result in _results
    ]  # Unpacked

    # Compute per-row cosine similarity offsets
    embeddings = [_result.embedding for _result in embed_results]
    embeddings_np = np.asarray(embeddings)  # (N, L)
    cosine_similarities = _avg_cosine_similarity(embeddings_np)  # (N,)
    mean_similarity = np.mean(cosine_similarities).item()
    assert cosine_similarities.shape == (len(embeddings),), cosine_similarities.shape
    print(
        f"Cosine similarity of {args.langfuse_dataset_name}\n",
        pd.Series(cosine_similarities).describe(),
    )

    # Annotate dataset rows in LangFuse
    for _similarity, _trace_id in zip(
        track(cosine_similarities.flatten(), description="Uploading scores..."),
        [_result.langfuse_trace_id for _result in embed_results],
    ):
        langfuse_client.create_score(
            name="similarity_to_mean",
            value=_similarity,
            comment=f"Mean similarity: {mean_similarity:.3f}",
            trace_id=_trace_id,
        )

    flush_langfuse()
