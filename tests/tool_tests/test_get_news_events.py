"""Test the tool for getting news events."""

import pytest

from src.utils.tools import get_news_events


@pytest.mark.asyncio
async def test_get_news_events():
    """Test tool for retrieving news events from enwiki."""
    events_by_category = await get_news_events()
    all_events = [item for items in events_by_category.root.values() for item in items]
    assert len(all_events) > 0
    for _category, _events in events_by_category.root.items():
        print(f"Category: {_category}; {len(_events)} news events.")

    print(f"Example event: {all_events[0].model_dump_json(indent=2)}")
