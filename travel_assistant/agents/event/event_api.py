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
        # category: Optional[str] = None,
        # country: Optional[str] = None,
        # start: Optional[str] = None,  # Format: YYYY-MM-DD
        # end: Optional[str] = None,
        # limit: int = 5
    ) -> list[dict]:
        params = {
            "q": q,
            # "category": category,
            # "country": country,
            # "start.gte": start,
            # "start.lte": end,
            # "limit": limit,
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        resp = await self.client.get(url=self.BASE_URL, params=params)
        resp.raise_for_status()

        # response = requests.get(
        #     url="https://api.predicthq.com/v1/events/",
        #     headers={
        #       "Authorization": f"Bearer {token}",
        #       "Accept": "application/json"
        #     },
        #     params={
        #         "q": "taylor swift"
        #     }
        # )
        return resp.json().get("results", [])
