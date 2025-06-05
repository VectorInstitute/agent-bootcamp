"""Script for uploading test data to Langfuse.

See:
langfuse.com/docs/integrations/openaiagentssdk/example-evaluating-openai-agents
"""

import argparse

from rich.progress import track

from src.utils.data import get_dataset, get_dataset_url_hash
from src.utils.env_vars import Configs
from src.utils.langfuse.shared_client import langfuse


parser = argparse.ArgumentParser()
parser.add_argument("--source_dataset")
parser.add_argument("--langfuse_dataset_name")
parser.add_argument("--limit", type=int)


if __name__ == "__main__":
    args = parser.parse_args()
    configs = Configs.from_env_var()
    dataset_url_hash = get_dataset_url_hash(args.source_dataset)

    # Create a dataset in Langfuse
    langfuse.create_dataset(
        name=args.langfuse_dataset_name,
        description=f"[{dataset_url_hash}] Data from {args.source_dataset}",
        metadata={
            "url_hash": dataset_url_hash,
            "source": args.source_dataset,
            "type": "benchmark",
        },
    )

    df = get_dataset(args.source_dataset, limit=args.limit)

    for idx, row in track(
        df.iterrows(),
        total=len(df),
        description="Uploading to Langfuse",
    ):
        langfuse.create_dataset_item(
            dataset_name=args.langfuse_dataset_name,
            input={"text": row["question"]},
            expected_output={"text": row["expected_answer"]},
            # unique id to enable upsert without creating duplicates
            id=f"{dataset_url_hash}-{idx:05}",
        )
