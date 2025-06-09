"""Implements knowledge retrieval tool for Weaviate."""

import logging
from typing import TYPE_CHECKING

import pydantic


if TYPE_CHECKING:
    from weaviate import WeaviateAsyncClient


class _Source(pydantic.BaseModel):
    """Type hints for the "_source" field in ES Search Results."""

    title: str
    section: str | None = None


class _Highlight(pydantic.BaseModel):
    """Type hints for the "highlight" field in ES Search Results."""

    text: list[str]


class _SearchResult(pydantic.BaseModel):
    """Type hints for knowledge base search result."""

    source: _Source = pydantic.Field(alias="_source")
    highlight: _Highlight


SearchResults = list[_SearchResult]


class AsyncWeaviateKnowledgeBase:
    """Configurable search tools for Weaviate knowledge base."""

    def __init__(
        self,
        async_client: "WeaviateAsyncClient",
        collection_name: str,
        num_results: int = 5,
        snippet_length: int = 1000,
    ):
        self.async_client = async_client
        self.collection_name = collection_name
        self.num_results = num_results
        self.snippet_length = snippet_length
        self.logger = logging.getLogger(__name__)

    async def search_knowledgebase(self, keyword: str) -> SearchResults:
        """Search knowledge base.

        Parameters
        ----------
        keyword : str
            The search keyword to query the knowledge base.

        Returns
        -------
        SearchResults
            A list of search results. Each result contains source and highlight.
            If no results are found, returns an empty list.

        Raises
        ------
        Exception
            If Weaviate is not ready to accept requests (HTTP 503).

        """
        async with self.async_client:
            if not await self.async_client.is_ready():
                raise Exception("Weaviate is not ready to accept requests (HTTP 503).")

            collection = self.async_client.collections.get(self.collection_name)
            response = await collection.query.hybrid(keyword, limit=self.num_results)

        self.logger.info(f"Query: {keyword}; Returned matches: {len(response.objects)}")

        hits = []
        for obj in response.objects:
            hit = {
                "_source": {
                    "title": obj.properties.get("title", ""),
                    "section": obj.properties.get("section", None),
                },
                "highlight": {
                    "text": [obj.properties.get("text", "")[: self.snippet_length]]
                },
            }
            hits.append(hit)

        return [_SearchResult.model_validate(_hit) for _hit in hits]
