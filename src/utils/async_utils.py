"""Utils for async workflows."""

import asyncio
import types
from typing import Any, Coroutine, Sequence, TypeVar

from rich.progress import Progress


T = TypeVar("T")


async def indexed(index: int, coro: Coroutine[None, None, T]) -> tuple[int, T]:
    """Return (index, await coro)."""
    return index, (await coro)


async def gather_with_progress(
    coros: "list[types.CoroutineType[Any, Any, T]]",
    description: str = "Running tasks",
) -> Sequence[T | Exception]:
    """
    Run a list of coroutines concurrently, display a rich.Progress bar as each finishes.

    Returns the results in the same order as the input list.

    :param coros: List of coroutines to run.
    :return: List of results, ordered to match the input coroutines.
    """
    # Wrap each coroutine in a Task and remember its original index
    tasks = [
        asyncio.create_task(indexed(index=index, coro=coro))
        for index, coro in enumerate(coros)
    ]

    # Pre‐allocate a results list; we'll fill in each slot as its Task completes
    results: list[T | None] = [None] * len(tasks)

    # Create and start a Progress bar with a total equal to the number of tasks
    with Progress() as progress:
        progress_task = progress.add_task(description, total=len(tasks))

        # as_completed yields each Task as soon as it finishes
        for finished in asyncio.as_completed(tasks):
            index, result = await finished
            results[index] = result
            progress.update(progress_task, advance=1)

    # At this point, every slot in `results` is guaranteed to be non‐None
    # so we can safely cast it back to List[T]
    return results  # type: ignore
