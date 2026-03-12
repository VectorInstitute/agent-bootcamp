"""Utils for async workflows."""

import asyncio
import atexit
import types
from typing import Any, Awaitable, Callable, Coroutine, Protocol, Sequence, TypeVar

from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


T = TypeVar("T")


class AsyncCloseable(Protocol):
    """Protocol for objects with an async close method."""

    async def close(self) -> None:
        """Close the resource asynchronously."""
        ...


def register_async_cleanup(*resources: AsyncCloseable) -> None:
    """Register async resources for cleanup at exit.

    Safely handles cleanup whether or not an event loop is running,
    making it suitable for Gradio apps and other async frameworks.

    Parameters
    ----------
    *resources : AsyncCloseable
        One or more objects with an async `close()` method to clean up at exit.

    Examples
    --------
    >>> client_manager = AsyncClientManager()
    >>> register_async_cleanup(client_manager)
    >>> # Resources will be closed when the program exits
    """

    def cleanup() -> None:
        """Cleanup function that safely closes async resources."""
        try:
            # Try to get the current running event loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, safe to create a new one with asyncio.run()
            async def close_all() -> None:
                await asyncio.gather(
                    *(resource.close() for resource in resources),
                    return_exceptions=True,
                )

            asyncio.run(close_all())
        else:
            # There's a running loop, schedule the cleanup as a task
            # This will execute after the current event loop iteration completes
            async def close_all() -> None:
                await asyncio.gather(
                    *(resource.close() for resource in resources),
                    return_exceptions=True,
                )

            loop.create_task(close_all())

    atexit.register(cleanup)


async def indexed(index: int, coro: Coroutine[None, None, T]) -> tuple[int, T]:
    """Return (index, await coro)."""
    return index, (await coro)


async def rate_limited(
    _fn: Callable[[], Awaitable[T]], semaphore: asyncio.Semaphore
) -> T:
    """Run _fn with semaphore rate limit."""
    async with semaphore:
        return await _fn()


async def gather_with_progress(
    coros: "list[types.CoroutineType[Any, Any, T]]",
    description: str = "Running tasks",
) -> Sequence[T]:
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
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    ) as progress:
        progress_task = progress.add_task(description, total=len(tasks))

        # as_completed yields each Task as soon as it finishes
        for finished in asyncio.as_completed(tasks):
            index, result = await finished
            results[index] = result
            progress.update(progress_task, advance=1)

    # At this point, every slot in `results` is guaranteed to be non‐None
    # so we can safely cast it back to List[T]
    return results  # type: ignore


__all__ = ["gather_with_progress", "rate_limited", "register_async_cleanup"]
