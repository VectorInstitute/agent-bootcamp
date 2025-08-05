"""Implements event retrieval tool for PredictHQ."""

import httpx
import os
from typing import Optional

class AsyncPredictHQClient:
    BASE_URL = "https://api.predicthq.com/v1/events/"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        )

    async def search_events(
        self,
        q: Optional[str] = None,
        category: Optional[str] = None,
        country: Optional[str] = None,
        start: Optional[str] = None,  # Format: YYYY-MM-DD
        end: Optional[str] = None,
        limit: int = 5
    ) -> list[dict]:
        """
        Search for events using the PredictHQ API.

        Args:
            q (Optional[str]): Free text search query for event titles or descriptions.
            category (Optional[str]): Event category (e.g., 'sports', 'concerts').
            country (Optional[str]): 2-letter country code (e.g., 'US', 'GB').
            start (Optional[str]): Start date (inclusive) in 'YYYY-MM-DD' format.
            end (Optional[str]): End date (inclusive) in 'YYYY-MM-DD' format.
            limit (int): Maximum number of results to return (default: 5).

        Returns:
            list[dict]: A list of event dictionaries matching the search criteria.

        Raises:
            httpx.HTTPStatusError: If the API request fails.

        Example:
            events = await client.search_events(q="music", country="US", start="2024-08-01", end="2024-08-31")
        """
        params = {
            "q": q,
            "category": category,
            "country": country,
            "start.gte": start,
            "start.lte": end,
            "limit": limit,
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        resp = await self.client.get(url=self.BASE_URL, params=params)
        resp.raise_for_status()

        return resp.json().get("results", [])
