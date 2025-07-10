"""Pressure test for Weaviate."""

import argparse
import asyncio
import time
from typing import Callable, Coroutine, Sequence

import dotenv
import plotly.graph_objs as go

from src.utils import AsyncWeaviateKnowledgeBase, Configs, get_weaviate_async_client

from ..async_utils import gather_with_progress, rate_limited
from ..pretty_printing import pretty_print


async def run_weaviate_query(keyword: str):
    """Run Weaviate query."""
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
        async_weaviate_client,
        collection_name="enwiki_20250520",
    )

    try:
        return await async_knowledgebase.search_knowledgebase(keyword=keyword)
    finally:
        await async_weaviate_client.close()


async def time_async_workload(
    async_fn: Callable[[], Coroutine], time_limit: int = 30
) -> float | None:
    """
    Run async call and return duration.

    Args:
        async_fn: async function to invoke and time.
        time_limit: in seconds

    Returns
    -------
        The latency (in seconds) of the simulated task.
    """
    start = time.perf_counter()
    try:
        await asyncio.wait_for(async_fn(), timeout=time_limit)
        end = time.perf_counter()
        return end - start
    except Exception as e:
        print(e)


async def run_with_concurrency(
    async_fn: Callable[[str], Coroutine[None, None, float | None]],
    concurrency: int,
    iterations: int,
) -> Sequence[float | None]:
    """
    Run the simulated workload concurrently, limited by a semaphore.

    Args:
        concurrency: Maximum number of concurrent tasks.
        iterations: Total number of tasks to run.

    Returns
    -------
        List of task durations.
    """
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [
        rate_limited(
            lambda index=index: time_async_workload(
                lambda: async_fn(f"{concurrency}/{index}")
            ),
            semaphore=semaphore,
        )
        for index in range(max(concurrency, iterations))
    ]
    return await gather_with_progress(tasks, description=f"Concurrency {concurrency}")


async def gather_all_concurrency_levels(
    async_fn: Callable[[str], Coroutine],
    concurrency_levels: Sequence[int],
    iterations_per_level: int,
) -> dict[int, Sequence[float | None]]:
    """
    Execute tasks at various concurrency levels and gathers timing data.

    Args:
        concurrency_levels: List of concurrency limits to test.
        iterations_per_level: Number of tasks to run per level.
        async_fn: Function to invoke, accepting a unique identifier to prevent caching.

    Returns
    -------
        Dictionary mapping concurrency level to list of durations.
    """
    results_by_level: dict[int, Sequence[float | None]] = {}
    for level in concurrency_levels:
        durations = await run_with_concurrency(async_fn, level, iterations_per_level)
        results_by_level[level] = durations
    return results_by_level


def plot_box_whisker(
    results_by_level: dict[int, Sequence[float | None]], subtitle: str | None = None
) -> go.Figure:
    """
    Plot a box-whisker plot using Plotly for each concurrency level.

    Args:
        results_by_level: Dictionary of durations keyed by concurrency level.
    """
    fig = go.Figure()
    for level, durations in results_by_level.items():
        valid_durations = [x for x in durations if x is not None]
        valid_percentage = len(valid_durations) / (len(durations) + 1e-6) * 100
        fig.add_trace(
            go.Box(
                y=valid_durations,
                name=f"{level} ({valid_percentage:.3f}% valid)",
                boxmean=True,
            )
        )

    title = "Box-Whisker Plot of Task Durations by Concurrency Level"
    if subtitle is not None:
        title += f"<br><sup>{subtitle}</sup>"

    fig.update_layout(
        title=title,
        xaxis_title="Concurrency Level",
        yaxis_title="Duration (seconds)",
    )
    return fig


parser = argparse.ArgumentParser()
parser.add_argument("--output_file", default="/tmp/plot.png")
parser.add_argument("--iterations_per_level", default=1, type=int)
parser.add_argument("--max_power", default=9, type=int)

if __name__ == "__main__":
    import uvloop

    uvloop.install()

    dotenv.load_dotenv()
    args = parser.parse_args()

    concurrency_levels = [2**i for i in range(1, args.max_power + 1)]

    results = asyncio.run(
        gather_all_concurrency_levels(
            lambda prefix: run_weaviate_query(f"{prefix}: Toronto"),
            concurrency_levels,
            args.iterations_per_level,
        )
    )
    pretty_print(results)

    configs = Configs.from_env_var()
    subtitle = f"{args.output_file} ({configs.weaviate_grpc_host})"
    plot = plot_box_whisker(results, subtitle=subtitle)
    plot.write_image(args.output_file, scale=5)
