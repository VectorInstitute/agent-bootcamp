"""Script for uploading test data to Langfuse.

See:
langfuse.com/docs/integrations/openaiagentssdk/example-evaluating-openai-agents
"""

import argparse

from aieng.agents import Configs
from aieng.agents.data import get_dataset, get_dataset_url_hash
from aieng.agents.langfuse import langfuse_client, set_up_langfuse_otlp_env_vars
from dotenv import load_dotenv
from rich.progress import track


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dataset", required=True)
    parser.add_argument("--langfuse_dataset_name", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    load_dotenv(verbose=True)

    configs = Configs()

    set_up_langfuse_otlp_env_vars()

    dataset_url_hash = get_dataset_url_hash(args.source_dataset)

    # Create a dataset in Langfuse
    assert langfuse_client.auth_check()
    langfuse_client.create_dataset(
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
        langfuse_client.create_dataset_item(
            dataset_name=args.langfuse_dataset_name,
            input={"text": row["question"]},
            expected_output={"text": row["expected_answer"]},
            # unique id to enable upsert without creating duplicates
            id=f"{dataset_url_hash}-{idx:05}",
        )
