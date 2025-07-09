#!/usr/bin/env python3
"""Fetch and parse Wikipedia Current Events into structured data using Pydantic."""

from __future__ import annotations

import argparse
import asyncio
import random
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, RootModel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn


class NewsEvent(BaseModel):
    """Represents a single current event item."""

    date: date
    category: str
    description: str


class CurrentEvents(RootModel):
    """Mapping of event category to a list of Event items."""

    root: dict[str, list[NewsEvent]]


async def _fetch_current_events_html() -> str:
    """
    Retrieve the HTML for the Wikipedia Current Events page for a given date.

    Returns
    -------
        Raw HTML string of the parsed page.
    """
    # pick a random month between January and May
    # (the knowledge base is not updated after May 30, 2025)
    # and a random day in that month
    random.seed(42)
    random_date = date(2025, 1, 1) + timedelta(
        days=random.randint(0, (date(2025, 5, 20) - date(2025, 1, 1)).days)
    )
    # convert to Year_Month_day format (example: 2025_May_6)
    date_str = random_date.strftime("%Y_%B_%d")

    api_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "page": f"Portal:Current_events/{date_str}",
        "prop": "text",
        "format": "json",
    }
    client = httpx.AsyncClient()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    ) as progress:
        progress.add_task("GET wikipedia/Portal:Current_events...")
        resp = await client.get(api_url, params=params)

    resp.raise_for_status()
    data = resp.json()
    return data["parse"]["text"]["*"]


def _parse_current_events(html: str) -> dict[str, list[NewsEvent]]:
    """
    Parse the HTML of the Wikipedia Current Events portal and extract a list of events.

    Args:
        html: The HTML content of the portal or date subpage.

    Returns
    -------
        A dict mapping category -> list of Events
    """
    soup = BeautifulSoup(html, "lxml")
    events_by_category: dict[str, list[NewsEvent]] = defaultdict(list)
    # Find each date block
    date_divs = soup.find_all("div", class_="current-events-main vevent")

    for date_div in date_divs:
        date_div: Any
        # Extract ISO date
        date_span = date_div.find("span", class_="bday")
        date_str = date_span.get_text(strip=True) if date_span else ""

        # Find the content section
        content_div = date_div.find("div", class_="current-events-content")
        if not content_div:
            continue

        # Iterate through each category heading and its events
        for p_tag in content_div.find_all("p"):
            b_tag = p_tag.find("b")
            if not b_tag:
                continue
            category = b_tag.get_text(strip=True)

            # The next sibling <ul> contains the list of events for this category
            ul = p_tag.find_next_sibling(lambda tag: tag.name == "ul")
            if not ul:
                continue

            # Iterate top-level list items as individual events
            for li in ul.find_all("li", recursive=False):
                # Join all text fragments for a clean description
                description = " ".join(li.stripped_strings)
                events_by_category[category].append(
                    NewsEvent(
                        date=date.fromisoformat(date_str),
                        category=category,
                        description=description,
                    )
                )

    return events_by_category


async def get_news_events() -> CurrentEvents:
    """Return a list of current news events from the English Wikipedia.

    Returns
    -------
        dict mapping category of news events to list of news headlines.
    """
    html = await _fetch_current_events_html()
    events_dict = _parse_current_events(html)

    return CurrentEvents.model_validate(events_dict)


async def main() -> None:
    """Fetch, parse, and output events as JSON."""
    parser = argparse.ArgumentParser(
        description="Fetch and parse Wikipedia Current Events into structured JSON."
    )
    parser.add_argument(
        "--output", "-o", help="Output JSON file path (default: stdout)"
    )
    args = parser.parse_args()

    news_events = await get_news_events()
    output = news_events.model_dump_json(indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
    else:
        print(output)


if __name__ == "__main__":
    asyncio.run(main())
