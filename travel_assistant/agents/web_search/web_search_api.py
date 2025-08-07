"""Web search client using DuckDuckGo, Tavily, and SerpAPI."""

from __future__ import annotations

import os
from typing import Optional

import httpx


class AsyncWebSearchClient:
    """Async client that searches the web with fallback providers."""

    def __init__(self, tavily_api_key: Optional[str] | None = None, serp_api_key: Optional[str] | None = None) -> None:
        self.client = httpx.AsyncClient()
        self.tavily_api_key = tavily_api_key or os.environ.get("TAVILY_API_KEY")
        self.serp_api_key = serp_api_key or os.environ.get("SERPAPI_API_KEY")

    async def _duckduckgo_search(self, query: str, max_results: int) -> list[dict]:
        """Search DuckDuckGo for the query."""
        params = {"q": query, "format": "json", "no_redirect": 1, "no_html": 1}
        resp = await self.client.get("https://api.duckduckgo.com/", params=params)
        resp.raise_for_status()
        data = resp.json()
        results: list[dict] = []
        for item in data.get("RelatedTopics", []):
            if isinstance(item, dict) and item.get("Text") and item.get("FirstURL"):
                results.append({"title": item["Text"], "url": item["FirstURL"]})
                if len(results) >= max_results:
                    break
        return results

    async def _tavily_search(self, query: str, max_results: int) -> list[dict]:
        """Search Tavily for the query."""
        if not self.tavily_api_key:
            return []
        payload = {"api_key": self.tavily_api_key, "query": query, "num_results": max_results}
        resp = await self.client.post("https://api.tavily.com/search", json=payload)
        resp.raise_for_status()
        data = resp.json()
        results: list[dict] = []
        for item in data.get("results", [])[:max_results]:
            results.append({"title": item.get("title"), "url": item.get("url"), "content": item.get("content")})
        return results

    async def _serpapi_search(self, query: str, max_results: int) -> list[dict]:
        """Search SerpAPI for the query."""
        if not self.serp_api_key:
            return []
        params = {"engine": "google", "q": query, "api_key": self.serp_api_key, "num": max_results}
        resp = await self.client.get("https://serpapi.com/search", params=params)
        resp.raise_for_status()
        data = resp.json()
        results: list[dict] = []
        for item in data.get("organic_results", [])[:max_results]:
            results.append({"title": item.get("title"), "url": item.get("link")})
        return results

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Search the web using providers in priority order."""
        for provider in [self._duckduckgo_search, self._tavily_search, self._serpapi_search]:
            results = await provider(query, max_results)
            if results:
                return results
        return []

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self.client.aclose()
