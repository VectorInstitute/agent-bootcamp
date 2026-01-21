"""Create a knowledge base in Weaviate from a HuggingFace dataset."""

import asyncio
import logging
import os
from typing import Any

import click
import datasets
import weaviate
import weaviate.classes.config as wc
from datasets import Features, Sequence, Value, load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAIError
from weaviate.classes.config import DataType as WDataType
from weaviate.classes.init import Auth


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
load_dotenv(".env.participants")
logger = logging.getLogger(__name__)

HF_TO_WEAVIATE_TYPE_MAP = {
    "string": WDataType.TEXT,
    "list[string]": WDataType.TEXT_ARRAY,
    "int64": WDataType.INT,
    "int32": WDataType.INT,
    "list[int64]": WDataType.INT_ARRAY,
    "list[int32]": WDataType.INT_ARRAY,
    "bool": WDataType.BOOL,
    "list[bool]": WDataType.BOOL_ARRAY,
    "float64": WDataType.NUMBER,
    "float32": WDataType.NUMBER,
    "list[float64]": WDataType.NUMBER_ARRAY,
    "list[float32]": WDataType.NUMBER_ARRAY,
    "timestamp[s]": WDataType.DATE,
    "list[timestamp[s]]": WDataType.DATE_ARRAY,
    "binary": WDataType.BLOB,
    "dict": WDataType.OBJECT,
}


def _hf_type_to_str(feature: Any) -> str:
    """Convert HuggingFace feature type to a string representation."""
    if isinstance(feature, Value):
        return feature.dtype  # e.g., 'string', 'int64', 'float64', 'bool', 'binary'
    if isinstance(feature, Sequence):
        # Recursively get the type of the inner feature
        return f"list[{_hf_type_to_str(feature.feature)}]"
    if isinstance(feature, list):
        return f"list[{_hf_type_to_str(feature[0])}]"
    if isinstance(feature, dict):
        return "dict"
    return str(type(feature))


def _create_properties(features: Features) -> list[wc.Property]:
    """Create Weaviate properties from HuggingFace dataset features."""
    properties = []
    for name, feature in features.items():
        print(name, feature)
        hf_type = _hf_type_to_str(feature)
        weaviate_type = HF_TO_WEAVIATE_TYPE_MAP.get(hf_type)
        if weaviate_type is None:
            raise ValueError(f"Unsupported feature type: {feature}")

        properties.append(
            wc.Property(
                name=name.replace("-", "_").replace(" ", "_")
                if name != "id"
                else "id_",
                data_type=weaviate_type,
                index_filterable=name != "searchable_embedding_text",
                index_searchable=name != "searchable_embedding_text"
                and weaviate_type in [WDataType.TEXT, WDataType.TEXT_ARRAY],
                skip_vectorization=name != "searchable_embedding_text",
            )
        )
    return properties


def _create_or_get_weaviate_collection(
    client: weaviate.WeaviateClient, collection_name: str, features: Features
) -> weaviate.collections.Collection:
    """Get a Weaviate collection by name."""
    if client.collections.exists(collection_name):
        collection = client.collections.get(collection_name)
    else:
        properties = _create_properties(features)
        collection = client.collections.create(
            collection_name,
            properties=properties,
            vectorizer_config=wc.Configure.Vectorizer.none(),
            vector_index_config=wc.Configure.VectorIndex.dynamic(
                distance_metric=wc.VectorDistances.COSINE,
                threshold=10_000,
                flat=wc.Configure.VectorIndex.flat(
                    quantizer=wc.Configure.VectorIndex.Quantizer.bq(cache=True)
                ),
                hnsw=wc.Configure.VectorIndex.hnsw(
                    quantizer=wc.Configure.VectorIndex.Quantizer.pq(
                        segments=128, training_limit=50_000
                    ),
                ),
            ),
        )
        logger.info(
            f"Created Weaviate collection '{collection_name}' with properties: "
            f"{properties}"
        )
    return collection


async def _get_embeddings(
    texts: list[str], embedding_client: AsyncOpenAI, model_name: str
) -> list[list[float]]:
    try:
        embeddings = await embedding_client.embeddings.create(
            input=texts, model=model_name
        )
    except OpenAIError as e:
        if hasattr(e, "status_code") and e.status_code == 400 and "context" in str(e):
            embeddings = await embedding_client.embeddings.create(
                input=[text[:10000] for text in texts], model=model_name
            )
        else:
            raise
    return [embedding.embedding for embedding in embeddings.data]


async def _producer(
    dataset: datasets.Dataset,
    batch_size: int,
    embedding_client: AsyncOpenAI,
    model_name: str,
    obj_queue: asyncio.Queue,
) -> None:
    """Create batches of objects from the dataset with the vector included."""
    for batch in dataset.iter(batch_size=batch_size):
        objects: dict[str, list[Any]] = {}

        # Filter out None or empty strings from the batch
        # Get index of empty or None text entries
        null_indices = [
            i
            for i, text in enumerate(batch["searchable_embedding_text"])
            if text is None or text == ""
        ]
        # Remove empty or None text entries from the batch
        if null_indices:
            for key in batch:
                objects[key.replace("-", "_").replace(" ", "_")] = [
                    v for i, v in enumerate(batch[key]) if i not in null_indices
                ]
        else:
            objects = batch

        # Get embeddings for the batch
        embeddings = await _get_embeddings(
            objects["searchable_embedding_text"], embedding_client, model_name
        )

        # Rename "id" to "id_" to avoid conflict with Weaviate's reserved field
        if "id" in objects:
            objects["id_"] = objects.pop("id")

        objects["vector"] = embeddings

        await obj_queue.put(objects)


async def _consumer(
    collection: weaviate.collections.Collection, obj_queue: asyncio.Queue
) -> None:
    """Consume objects from the queue and add them to Weaviate."""
    while True:
        objects: dict[str, list[Any]] = await obj_queue.get()
        if objects is None:
            break  # Exit signal

        with collection.batch.fixed_size(
            batch_size=len(objects), concurrent_requests=1
        ) as batch:
            vectors = objects.pop("vector")
            for i in range(len(objects["searchable_embedding_text"])):
                obj = {
                    k.replace("-", "_").replace(" ", "_"): v[i]
                    for k, v in objects.items()
                }
                batch.add_object(obj, vector=vectors[i])

        # Flush the batch to Weaviate
        batch.flush()
        obj_queue.task_done()


@click.command()
@click.option("--dataset-name", required=True, help="HuggingFace dataset name")
@click.option("--collection-name", required=True, help="Name of weaviate collection")
@click.option(
    "--model-name", envvar="EMBEDDING_MODEL_NAME", help="Embedding model name"
)
@click.option("--batch-size", default=5, help="Batch size for processing")
@click.option("--max-concurrent", default=50, help="Max concurrent consumers")
@click.option(
    "--retry-attempts", default=5, help="Number of retry attempts for requests"
)
@click.option("--timeout-seconds", default=30, help="Timeout for requests in seconds")
@click.option("--cache-dir", help="Dataset cache directory")
def _cli(
    dataset_name,
    collection_name,
    model_name,
    batch_size,
    max_concurrent,
    retry_attempts,
    timeout_seconds,
    cache_dir,
):
    asyncio.run(
        main(
            dataset_name,
            collection_name,
            model_name,
            batch_size,
            max_concurrent,
            retry_attempts,
            timeout_seconds,
            cache_dir,
        )
    )


async def main(
    dataset_name: str,
    collection_name: str,
    model_name: str,
    batch_size: int,
    max_concurrent: int,
    retry_attempts: int,
    timeout_seconds: int,
    cache_dir: str | None = None,
) -> None:
    """Generate embeddings for a streaming HuggingFace dataset."""
    # Load dataset
    dataset = load_dataset(dataset_name, split="train", cache_dir=cache_dir)
    # Remove columns with nested/dict features
    dataset = dataset.remove_columns(
        [
            col
            for col in dataset.column_names
            if isinstance(dataset.features[col], dict) or "Unnamed" in col
        ]
    )
    assert "searchable_embedding_text" in dataset.column_names, (
        "Dataset must contain a 'text' column"
    )

    # Create Weaviate Client
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.environ["WEAVIATE_HTTP_HOST"],
        auth_credentials=Auth.api_key(os.environ["WEAVIATE_API_KEY"]),
    )
    logger.info(f"Weaviate client ready: {weaviate_client.is_ready()}")

    # Get or create Weaviate collection
    collection = _create_or_get_weaviate_collection(
        weaviate_client, collection_name, dataset.features
    )

    # Initialize OpenAI client for embedding service
    embedding_client = AsyncOpenAI(
        api_key=os.environ["EMBEDDING_API_KEY"],
        base_url=os.environ["EMBEDDING_BASE_URL"],
        timeout=timeout_seconds,
        max_retries=retry_attempts,
    )

    # Orchestrate producer and consumers
    obj_queue = asyncio.Queue(maxsize=max_concurrent * 5)
    producer = asyncio.create_task(
        _producer(dataset, batch_size, embedding_client, model_name, obj_queue)
    )
    # Create consumers to process the queue
    consumers = [
        asyncio.create_task(_consumer(collection, obj_queue))
        for _ in range(max_concurrent)
    ]

    try:
        # Wait for the producer to finish
        await producer

        # Signal consumers to stop
        for _ in range(max_concurrent):
            await obj_queue.put(None)

        # Wait for all consumers to finish
        await asyncio.gather(*consumers)
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}", exc_info=True)
    finally:
        weaviate_client.close()
        await embedding_client.close()
        logger.info("Finished processing dataset and closing clients.")


if __name__ == "__main__":
    _cli()
