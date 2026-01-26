"""Utils for creating batches of data for performance."""

from typing import TypeVar


V = TypeVar("V")


def create_batches(
    items: list[V],
    batch_size: int,
    limit: int | None = None,
    keep_trailing: bool = True,
) -> list[list[V]]:
    """Transform the list of items into batches.

    Params:
        limit: number of items to include in total
        keep_trailing: if False, the last few items that
            does not fit in a full batch will not be returned.

    Return:
        List of batches.
    """
    batches: list[list[V]] = [[]]
    for _index, _item in enumerate(items):
        if (limit is not None) and (_index >= limit):
            break

        batches[-1].append(_item)
        if len(batches[-1]) == batch_size:
            batches.append([])

    # Discard trailing batch if empty or required
    if (len(batches[-1]) == 0) or (
        (not keep_trailing) and (len(batches[-1]) < batch_size)
    ):
        batches.pop(-1)

    return batches
