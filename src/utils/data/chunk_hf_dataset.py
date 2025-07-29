"""Script to chunk text data from a HuggingFace dataset."""

import os
from collections import defaultdict
from functools import partial

import click
from datasets import load_dataset
from transformers import AutoTokenizer


def chunk_texts(
    record: dict[str, list[str]],
    tokenizer: AutoTokenizer,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
) -> dict[str, list[str]]:
    """Chunk the text in a record using a sliding window approach.

    Args:
        record (dict): A dictionary containing the text to be chunked.
        tokenizer (AutoTokenizer): The tokenizer to use for encoding the text.
        chunk_size (int): The maximum size of each text chunk.
        chunk_overlap (int): The number of tokens to overlap between chunks.

    Returns
    -------
        dict: A dictionary with the same keys as the input record, but with the
        "text" key containing a list of text chunks.
    """
    texts = record.get("text", [])

    # Use fast tokenizer to get sliding windows of tokens
    encodings = tokenizer(
        texts,
        return_overflowing_tokens=True,
        truncation=True,
        max_length=chunk_size,
        stride=chunk_overlap,
        return_attention_mask=False,
    )

    chunked_records = defaultdict(list)

    # The `overflow_to_sample_mapping` is a list where each element is the index
    # of the original sample in the input batch.
    # For each new chunk, we can use this to find its original metadata.
    for i, sample_idx in enumerate(encodings["overflow_to_sample_mapping"]):
        chunked_records["text"].append(
            tokenizer.decode(encodings["input_ids"][i], skip_special_tokens=True)
        )

        # For all other columns, copy the metadata from the original sample.
        for key, values in record.items():
            if key == "text":
                continue
            chunked_records[key].append(values[sample_idx])

    return chunked_records


@click.command()
@click.option(
    "--hf_dataset_path_or_name", required=True, help="Name of the dataset to load"
)
@click.option(
    "--hf_tokenizer_name", default="BAAI/bge-m3", help="Name of the tokenizer to use"
)
@click.option(
    "--hf_dataset_split", default="train", help="Split of the dataset to load"
)
@click.option(
    "--hf_dataset_cache_dir", default=None, help="Cache directory for the dataset"
)
@click.option("--chunk_size", default=512, help="Size of each text chunk")
@click.option("--chunk_overlap", default=128, help="Overlap between text chunks")
@click.option("--batch_size", default=10, help="Batch size for processing the dataset")
@click.option(
    "--save_to_hub", is_flag=True, help="Save the processed dataset to HuggingFace Hub"
)
@click.option(
    "--hub_repo_id",
    default=None,
    help="HuggingFace Hub repository ID to save the dataset",
)
def main(
    hf_dataset_path_or_name: str,
    hf_tokenizer_name: str,
    hf_dataset_split: str = "train",
    hf_dataset_cache_dir: str | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    batch_size: int = 10,
    save_to_hub: bool = False,
    hub_repo_id: str | None = None,
) -> None:
    """Chunk text data from a HuggingFace dataset."""
    dataset = load_dataset(
        hf_dataset_path_or_name, split=hf_dataset_split, cache_dir=hf_dataset_cache_dir
    )

    # Create a new dataset with chunked texts, keeping all the original fields
    # one chunk per row
    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name, use_fast=True)

    chunked_dataset = dataset.map(
        partial(
            chunk_texts,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ),
        batched=True,
        batch_size=batch_size,
        num_proc=min(batch_size, os.cpu_count() or 1, len(dataset) // batch_size),
    )
    chunked_dataset.to_json("chunked_dataset.json", orient="records", lines=True)

    if save_to_hub:
        if not hub_repo_id:
            raise ValueError("hub_repo_id must be provided when save_to_hub is True")
        chunked_dataset.push_to_hub(hub_repo_id, private=False)


if __name__ == "__main__":
    main()
