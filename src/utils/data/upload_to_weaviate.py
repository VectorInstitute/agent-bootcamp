"""Upload Wikipedia embeddings to Weaviate vector database."""

import hashlib
import os
import uuid
from typing import Any

import h5py
import weaviate
import weaviate.classes.config as wc
from datasets import load_dataset
from dotenv import load_dotenv
from h5py._hl.dataset import Dataset
from h5py._hl.datatype import Datatype
from h5py._hl.files import File
from h5py._hl.group import Group
from tqdm import tqdm
from weaviate.classes.init import Auth


COLLECTION_NAME = "enwiki_20250520"


def load_embeddings_data(
    embeddings_path: str,
) -> tuple[Group | Dataset | Datatype, dict[Any, int], File]:
    """Load embeddings with memory mapping for faster access."""
    h5f = h5py.File(embeddings_path, "r")

    # Memory map the datasets for faster access
    embeddings_ds = h5f["embeddings"]
    uuids_ds = h5f["uuids"]

    # Pre-load UUIDs into memory for faster lookup
    uuids_array = uuids_ds[:]  # Load all UUIDs into memory
    if isinstance(uuids_array[0], bytes):
        uuids_array = [u.decode("utf-8") for u in uuids_array]

    # Create index mapping
    uuid_to_index = {uuid_str: i for i, uuid_str in enumerate(uuids_array)}

    return embeddings_ds, uuid_to_index, h5f


def generate_uuid(record: dict[str, Any]) -> str:
    """Generate a deterministic UUID from record content."""
    text: str = record.get("text", "")

    # Use SHA256 for a stable hash
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # Create the stable identifier
    identifier = f"{record.get('title', '')}-{record.get('section', '')}-{text_hash}"

    return str(uuid.uuid5(uuid.NAMESPACE_DNS, identifier))


def process_in_batches(uuid_to_index: dict[str, int], batch_size: int = 5_000):
    """Process records in batches for better efficiency."""
    batch_records = []

    for record in ds:
        record_uuid = generate_uuid(record)
        if record_uuid in uuid_to_index:
            batch_records.append((record, record_uuid))

        if len(batch_records) >= batch_size:
            yield batch_records
            batch_records = []

    if batch_records:  # Don't forget the last batch
        yield batch_records


if __name__ == "__main__":
    load_dotenv(".env")

    weaviate_url = os.environ["WEAVIATE_HTTP_HOST"]
    weaviate_api_key = os.environ["WEAVIATE_API_KEY"]

    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
    )

    if weaviate_client.collections.exists(COLLECTION_NAME):
        enwiki = weaviate_client.collections.get(COLLECTION_NAME)
    else:
        enwiki = weaviate_client.collections.create(
            name=COLLECTION_NAME,
            description="A subset of English Wikipedia dump from May 2025",
            properties=[
                wc.Property(
                    name="title",
                    data_type=wc.DataType.TEXT,
                    description="Title of the Wikipedia article",
                ),
                wc.Property(
                    name="section",
                    data_type=wc.DataType.TEXT,
                    description="Section title within the Wikipedia article",
                ),
                wc.Property(
                    name="text",
                    data_type=wc.DataType.TEXT,
                    description="Content of the Wikipedia article section",
                ),
            ],
            vectorizer_config=wc.Configure.Vectorizer.none(),
            vector_index_config=wc.Configure.VectorIndex.hnsw(
                quantizer=wc.Configure.VectorIndex.Quantizer.pq(
                    segments=128, training_limit=100_000
                )
            ),
            sharding_config=wc.Configure.sharding(
                virtual_per_physical=128, desired_count=3
            ),
            replication_config=wc.Configure.replication(
                factor=3,
                async_enabled=True,
                deletion_strategy=wc.ReplicationDeletionStrategy.TIME_BASED_RESOLUTION,
            ),
        )

    # Load embeddings and UUIDs
    print("Loading embeddings and UUIDs from HDF5 file...")
    embeddings_ds, uuid_to_index, h5f = load_embeddings_data(
        "/projects/llm/agent-bootcamp/embeddings/enwiki_20250520.h5"
    )

    ds = load_dataset(
        "ComplexDataLab/cdl-data-chai-enwiki-20250520",
        split="train",
        streaming=True,
        cache_dir=None,
    ).shuffle(seed=0, buffer_size=100_000)

    count = 0
    pbar = tqdm(total=len(uuid_to_index), desc="Adding records to Weaviate")
    try:
        pbar.set_postfix({"error_count": 0})
        pbar.update(0)

        with enwiki.batch.fixed_size(batch_size=200) as batch:
            for batch_num, batch_records in enumerate(
                process_in_batches(uuid_to_index, batch_size=10_000)
            ):
                if batch_num < 408:
                    continue

                if batch_num >= 500:
                    break

                for record, record_uuid in batch_records:
                    index = uuid_to_index[record_uuid]
                    vector = embeddings_ds[index]

                    batch.add_object(
                        {
                            "title": record["title"],
                            "section": record["section"],
                            "text": record["text"],
                        },
                        uuid=record_uuid,
                        vector=vector,
                    )

                    count += 1
                    if count == len(uuid_to_index):
                        print("All records have been added successfully.")
                        break

                pbar.set_postfix({"error_count": batch.number_errors})
                pbar.update(len(batch_records))

                if len(enwiki.batch.failed_objects) > 1000:
                    print("Batch import stopped due to excessive errors.")
                    break
    finally:
        pbar.close()
        h5f.close()
        weaviate_client.close()

    if len(enwiki.batch.failed_objects) > 0:
        print(f"Failed to import {len(enwiki.batch.failed_objects)} objects")
