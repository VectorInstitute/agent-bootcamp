"""Generate and store embeddings for a Wikipedia dataset."""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Optional

import aiohttp
import click
import h5py
import numpy as np
import openai
from aiolimiter import AsyncLimiter
from datasets import load_dataset
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from tqdm import tqdm
from transformers import AutoTokenizer


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class _EmbeddingConfig:
    """Configuration for embedding generation."""

    base_url: str
    api_key: str
    model_name: str
    embedding_dim: int
    max_concurrent_requests: int = 64
    max_requests_per_minute: int = 8000
    batch_size: int = 10_000
    max_tokens: int = 8192
    min_text_length: int = 10
    shuffle_buffer_size: int = 100_000
    shuffle_seed: int = 0
    retry_attempts: int = 5
    timeout_seconds: int = 30


@dataclass
class ProcessingState:
    """State tracking for resumable processing."""

    processed_count: int = 0
    batch_num: int = 0
    total_failed: int = 0
    timestamp: float = 0.0


class DatasetProcessor(ABC):
    """Abstract base class for dataset processors."""

    @abstractmethod
    def extract_text_chunks(self, record: dict[str, Any]) -> list[str]:
        """Extract text from a dataset record."""
        pass

    @abstractmethod
    def generate_uuid(self, record: dict[str, Any]) -> str:
        """Generate a deterministic UUID for a record."""
        pass

    @abstractmethod
    def should_skip_record(self, record: dict[str, Any]) -> bool:
        """Determine if a record should be skipped."""
        pass


class _WikiDatasetProcessor(DatasetProcessor):
    """Processor for Wikipedia-style datasets."""

    def __init__(
        self,
        tokenizer: Optional[AutoTokenizer] = None,
        min_text_length: int = 100,
        max_tokens: Optional[int] = 8192,
    ) -> None:
        self.tokenizer = tokenizer
        self.min_text_length = min_text_length
        self.max_tokens = max_tokens

    def extract_text_chunks(self, record: dict[str, Any]) -> list[str]:
        """Extract and process text from Wikipedia record."""
        text = "\n".join(
            [record.get("title", ""), record.get("section", ""), record.get("text", "")]
        )

        if not self.tokenizer or not self.max_tokens:
            return [text]

        # Use fast tokenizer to get sliding windows of tokens
        enc = self.tokenizer(
            text,
            return_overflowing_tokens=True,
            truncation=True,
            max_length=self.max_tokens,
            stride=self.max_tokens // 10,  # 10% overlap
            return_attention_mask=False,
        )

        # Decode the token IDs back to text chunks
        return [
            self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            for chunk_ids in enc["input_ids"]
        ]

    @staticmethod
    def generate_uuid(record: dict[str, Any]) -> str:
        """Generate a deterministic UUID from record content."""
        text: str = record.get("text", "")

        # Use SHA256 for a stable hash
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        # Create the stable identifier
        identifier = (
            f"{record.get('title', '')}-{record.get('section', '')}-{text_hash}"
        )

        return str(uuid.uuid5(uuid.NAMESPACE_DNS, identifier))

    def should_skip_record(self, record: dict[str, Any]) -> bool:
        """Skip records with insufficient text content."""
        text: str = record.get("text", "")
        return len(text.strip()) < self.min_text_length


class _EmbeddingGenerator:
    """Main class for generating embeddings from streaming datasets."""

    def __init__(self, config: _EmbeddingConfig, processor: DatasetProcessor) -> None:
        self.config = config
        self.processor = processor
        self.rate_limiter = AsyncLimiter(config.max_requests_per_minute)

    def create_dataset_generator(
        self, dataset_name: str, cache_dir: Optional[str] = None, split: str = "train"
    ) -> Generator[tuple[list[dict], int], None, None]:
        """Create a generator that yields batches from the dataset."""
        try:
            ds = (
                load_dataset(
                    dataset_name,
                    split=split,
                    cache_dir=cache_dir,
                    streaming=True,
                )
                .shuffle(
                    seed=self.config.shuffle_seed,
                    buffer_size=self.config.shuffle_buffer_size,
                )
                .batch(self.config.batch_size)
            )

            for batch_num, batch in enumerate(ds):
                batch_size = len(next(iter(batch.values())))
                batch_records = [
                    {k: batch[k][i] for k in batch} for i in range(batch_size)
                ]

                # Filter out records that should be skipped
                filtered_records = []
                for record in batch_records:
                    if not self.processor.should_skip_record(record):
                        filtered_records.append(record)

                if filtered_records:
                    yield filtered_records, batch_num

        except Exception as e:
            logger.error(f"Error creating dataset generator: {e}")
            raise

    def _initialize_hdf5_file(self, file_path: str) -> None:
        """Initialize HDF5 file with extensible datasets."""
        mode = "r+" if os.path.exists(file_path) else "w"

        with h5py.File(file_path, mode, libver="latest") as h5f:
            if mode == "w":
                # Create extensible datasets
                h5f.create_dataset(
                    "embeddings",
                    shape=(0, self.config.embedding_dim),
                    maxshape=(None, self.config.embedding_dim),
                    dtype="f4",
                    chunks=(2048, self.config.embedding_dim),
                    compression="gzip",
                    compression_opts=6,
                )

                h5f.create_dataset(
                    "uuids",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                    chunks=(2048,),
                    compression="gzip",
                    compression_opts=6,
                )

                # Metadata dataset for tracking
                h5f.create_dataset(
                    "metadata", shape=(1,), dtype=h5py.string_dtype(encoding="utf-8")
                )
                h5f["metadata"][0] = json.dumps(
                    {
                        "model_name": self.config.model_name,
                        "embedding_dim": self.config.embedding_dim,
                        "created_at": time.time(),
                        "version": "1.0",
                    }
                )

            # Enable SWMR for concurrent readers
            h5f.swmr_mode = True

    def test_connection(self, num_tries: int = 10) -> bool:
        """Test connection to embedding server."""
        client = openai.OpenAI(
            api_key=self.config.api_key, base_url=self.config.base_url
        )

        for attempt in range(num_tries):
            try:
                output = client.embeddings.create(
                    model=self.config.model_name, input="Test connection"
                )
                if output is not None:
                    logger.info("Connection test successful")
                    return True
            except openai.APIConnectionError as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                time.sleep(10)

        return False

    @staticmethod
    def _is_retryable_exception(exception: Exception) -> bool:
        """Check if exception is retryable."""
        return (
            isinstance(exception, aiohttp.ClientResponseError)
            and 500 <= exception.status < 600
        )

    @retry(
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(5),
        retry=retry_if_exception(_is_retryable_exception),
    )
    async def _fetch_embedding(
        self,
        session: aiohttp.ClientSession,
        record_uuid: str,
        text: str,
        semaphore: asyncio.Semaphore,
    ) -> tuple[str, Optional[list[float]]]:
        """Fetch embedding with proper rate limiting and error handling."""
        async with semaphore, self.rate_limiter:
            payload = {"model": self.config.model_name, "input": text}
            headers = {"Authorization": f"Bearer {self.config.api_key}"}

            try:
                async with session.post(
                    f"{self.config.base_url}/embeddings",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return record_uuid, data["data"][0]["embedding"]
            except Exception as e:
                logger.error(f"Failed embedding for UUID {record_uuid}: {e}")
                return record_uuid, None

    async def _process_batch(
        self,
        session: aiohttp.ClientSession,
        batch_records: list[dict],
        semaphore: asyncio.Semaphore,
    ) -> tuple[list[tuple[str, np.ndarray]], list[str]]:
        """Fetch embeddings for a batch of records."""
        queue = asyncio.Queue(maxsize=self.config.batch_size * 2)
        partials: dict[str, list[np.ndarray]] = {}
        failures: set[str] = set()

        rec_bar = tqdm(total=len(batch_records), desc="Chunking records", leave=False)
        fetch_bar = tqdm(total=0, desc="Fetching embeddings", leave=False)

        async def _producer():
            """Extract & chunk texts in threads, push (uuid, chunk_text) onto queue."""
            for record in batch_records:
                uuid = self.processor.generate_uuid(record)

                # Offload the possibly expensive tokenization into a thread
                text_chunks = await asyncio.to_thread(
                    self.processor.extract_text_chunks, record
                )

                for chunk in text_chunks:
                    await queue.put((uuid, chunk))
                    fetch_bar.total += 1

                rec_bar.update(1)

            # After all records, signal the consumers to exit:
            for _ in range(self.config.max_concurrent_requests):
                await queue.put((None, None))  # sentinel

            rec_bar.close()

        async def _consumer():
            """Continuously pull from queue, fetch embeddings, and bucket results."""
            while True:
                uuid, chunk = await queue.get()
                if uuid is None:
                    queue.task_done()
                    break

                # Fetch embeddings as soon as a chunk is ready, even if producer
                # is still working
                _, emb = await self._fetch_embedding(session, uuid, chunk, semaphore)
                if emb is not None:
                    partials.setdefault(uuid, []).append(
                        np.array(emb, dtype=np.float32)
                    )
                else:
                    failures.add(uuid)

                queue.task_done()

                fetch_bar.update(1)

        # Start producer
        producer_task = asyncio.create_task(_producer())

        # Start a fixed pool of fetchers
        consumers = [
            asyncio.create_task(_consumer())
            for _ in range(self.config.max_concurrent_requests)
        ]

        # Wait until everything is done
        await producer_task
        await queue.join()  # wait until all items processed
        await asyncio.gather(*consumers)
        fetch_bar.close()

        # Average chunk embeddings
        all_uuids = {self.processor.generate_uuid(r) for r in batch_records}
        total_failures = all_uuids - partials.keys()
        results = [
            (u, np.mean(embs, axis=0, dtype=np.float32)) for u, embs in partials.items()
        ]
        return results, list(total_failures)

    def _write_embeddings_batch(
        self, h5_file_path: str, results: list[tuple[str, np.ndarray]]
    ) -> None:
        """Write embeddings and UUIDs to HDF5 file."""
        if not results:
            return

        with h5py.File(h5_file_path, "r+", libver="latest") as h5f:
            embeddings_ds = h5f["embeddings"]
            uuids_ds = h5f["uuids"]

            current_size = embeddings_ds.shape[0]
            new_size = current_size + len(results)

            # Resize datasets
            embeddings_ds.resize((new_size, self.config.embedding_dim))
            uuids_ds.resize((new_size,))

            # Bulk write embeddings and UUIDs
            embeddings_ds[current_size:new_size, :] = np.stack(
                [embedding for _, embedding in results], axis=0
            )
            uuids_ds[current_size:new_size] = np.array(
                [record_uuid.encode("utf-8") for record_uuid, _ in results],
                dtype=h5py.string_dtype(),
            )

    def _save_checkpoint(self, checkpoint_path: str, state: ProcessingState) -> None:
        """Save processing checkpoint."""
        checkpoint_data = {
            "processed_count": state.processed_count,
            "batch_num": state.batch_num,
            "total_failed": state.total_failed,
            "timestamp": time.time(),
        }

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

    def _load_checkpoint(self, checkpoint_path: str) -> ProcessingState:
        """Load processing checkpoint."""
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r") as f:
                data = json.load(f)
                return ProcessingState(
                    processed_count=data.get("processed_count", 0),
                    batch_num=data.get("batch_num", 0),
                    total_failed=data.get("total_failed", 0),
                    timestamp=data.get("timestamp", 0.0),
                )
        return ProcessingState()

    def _save_failed_records(
        self, failed_records_path: str, failed_uuids: list[str]
    ) -> None:
        """Save failed record UUIDs for retry."""
        failed_data = {"failed_uuids": failed_uuids, "timestamp": time.time()}

        # Append to existing failed records
        existing_failed = []
        if os.path.exists(failed_records_path):
            try:
                with open(failed_records_path, "r") as f:
                    existing_data = json.load(f)
                    existing_failed = existing_data.get("failed_uuids", [])
            except json.JSONDecodeError:
                logger.error("Failed to decode existing failed records JSON")

        all_failed = list(set(existing_failed + failed_uuids))
        failed_data["failed_uuids"] = all_failed

        os.makedirs(os.path.dirname(failed_records_path), exist_ok=True)
        with open(failed_records_path, "w") as f:
            json.dump(failed_data, f, indent=2)

        logger.info(
            f"Saved {len(failed_uuids)} new failed records, {len(all_failed)} total"
        )

    async def generate_embeddings(
        self,
        dataset_name: str,
        output_path: str,
        checkpoint_path: str,
        failed_records_path: str,
        cache_dir: Optional[str] = None,
    ) -> None:
        """Generate embeddings."""
        if not self.test_connection():
            logger.error("Failed to connect to embedding server")
            return

        # Initialize HDF5 file
        self._initialize_hdf5_file(output_path)

        # Load checkpoint
        state = self._load_checkpoint(checkpoint_path)
        logger.info(f"Starting from processed count: {state.processed_count}")

        # Initialize concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        async with aiohttp.ClientSession() as session:
            dataset_gen = self.create_dataset_generator(dataset_name, cache_dir)

            for batch_records, batch_num in dataset_gen:
                # Skip batches if resuming from checkpoint
                if batch_num < state.batch_num:
                    continue

                logger.info(
                    f"Processing batch {batch_num} with {len(batch_records)} records"
                )

                # Process batch
                results, failed_uuids = await self._process_batch(
                    session, batch_records, semaphore
                )

                # Write results
                if results:
                    self._write_embeddings_batch(output_path, results)

                # Handle failures
                if failed_uuids:
                    self._save_failed_records(failed_records_path, failed_uuids)
                    state.total_failed += len(failed_uuids)

                # Update state
                state.processed_count += len(batch_records)
                state.batch_num = batch_num + 1

                # Save checkpoint
                self._save_checkpoint(checkpoint_path, state)

                success_rate = (len(results) / len(batch_records)) * 100
                logger.info(
                    f"Batch {batch_num} complete: {len(results)}/{len(batch_records)} "
                    f"successful ({success_rate:.1f}%)"
                )

        logger.info(f"Processing complete! Total failed: {state.total_failed}")


@click.command()
@click.option("--dataset-name", required=True, help="HuggingFace dataset name")
@click.option("--output-path", required=True, help="Output HDF5 file path")
@click.option(
    "--base-url", envvar="EMBEDDING_BASE_URL", required=True, help="API base URL"
)
@click.option("--api-key", envvar="EMBEDDING_API_KEY", required=True, help="API key")
@click.option("--model-name", default="bge-m3", help="Embedding model name")
@click.option("--embedding-dim", default=1024, help="Embedding dimension")
@click.option(
    "--max-tokens",
    default=8192,
    help="Maximum tokens per text. If tokenizer_name is provided, this will be used to limit the text length.",
)
@click.option("--batch-size", default=3_000, help="Batch size for processing")
@click.option(
    "--min-text-length",
    default=200,
    help="Minimum text length (in characters) for processing",
)
@click.option("--max-concurrent", default=64, help="Max concurrent requests to API")
@click.option("--max-requests-per-minute", default=3_000, help="Rate limit")
@click.option(
    "--retry-attempts", default=5, help="Number of retry attempts for requests"
)
@click.option("--timeout-seconds", default=30, help="Timeout for requests in seconds")
@click.option("--tokenizer-name", help="Tokenizer name (optional)")
@click.option("--cache-dir", help="Dataset cache directory")
@click.option("--checkpoint-file", help="Checkpoint file path (optional)")
@click.option("--failed-records-file", help="Failed records file path (optional)")
def main(
    dataset_name: str,
    output_path: str,
    base_url: str,
    api_key: str,
    model_name: str,
    embedding_dim: int,
    max_tokens: int,
    batch_size: int,
    min_text_length: int,
    max_concurrent: int,
    max_requests_per_minute: int,
    retry_attempts: int,
    timeout_seconds: int,
    tokenizer_name: str,
    cache_dir: str,
    checkpoint_file: Optional[str],
    failed_records_file: Optional[str],
):
    """Generate embeddings for a streaming HuggingFace dataset."""
    # Load environment variables
    load_dotenv()

    # Set default paths if not provided
    output_dir = Path(output_path).parent
    if not checkpoint_file:
        checkpoint_file = output_dir / "checkpoint.json"
    if not failed_records_file:
        failed_records_file = output_dir / "failed_records.json"

    os.makedirs(output_dir, exist_ok=True)

    # Create configuration
    config = _EmbeddingConfig(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        embedding_dim=embedding_dim,
        max_concurrent_requests=max_concurrent,
        max_requests_per_minute=max_requests_per_minute,
        batch_size=batch_size,
        min_text_length=min_text_length,
        max_tokens=max_tokens,
        retry_attempts=retry_attempts,
        timeout_seconds=timeout_seconds,
    )

    tokenizer = None
    if tokenizer_name:
        # Load tokenizer if specified
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        except Exception as e:
            logging.error(f"Failed to load tokenizer {tokenizer_name}: {e}")
            raise

    processor = _WikiDatasetProcessor(tokenizer, min_text_length, max_tokens)

    # Create generator and run
    generator = _EmbeddingGenerator(config, processor)

    asyncio.run(
        generator.generate_embeddings(
            dataset_name=dataset_name,
            output_path=output_path,
            checkpoint_path=str(checkpoint_file),
            failed_records_path=str(failed_records_file),
            cache_dir=cache_dir,
        )
    )


if __name__ == "__main__":
    main()
