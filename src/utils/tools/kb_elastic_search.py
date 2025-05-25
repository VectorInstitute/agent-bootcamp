"""Implements knowledge retrieval tool for ElasticSearch."""

import logging
from typing import TYPE_CHECKING

import pydantic


if TYPE_CHECKING:
    from elasticsearch import AsyncElasticsearch


class _Source(pydantic.BaseModel):
    """Type hints for the "_source" field in ES Search Results."""

    title: str
    section: str | None = None


class _Highlight(pydantic.BaseModel):
    """Type hints for the "highlight" field in ES Search Results."""

    text: list[str]


class _SearchResult(pydantic.BaseModel):
    """Type hints for ES Search Result."""

    source: _Source = pydantic.Field(alias="_source")
    highlight: _Highlight


SearchResults = list[_SearchResult]


class AsyncESKnowledgeBase:
    """Configurable search tools for ElasticSearch knowledge base."""

    def __init__(
        self,
        async_es_client: "AsyncElasticsearch",
        es_collection_name: str,
        num_results: int = 5,
        snippet_length: int = 1000,
    ):
        self.es = async_es_client
        self.collection_name = es_collection_name
        self.num_results = num_results
        self.snippet_length = snippet_length
        self.logger = logging.getLogger(__name__)

    async def search_knowledgebase(self, keyword: str) -> SearchResults:
        """Search knowledge base.

        Returns
        -------
            a list of results. Empty list of no match is found.
        """
        title_match = {
            "match": {
                "title.fuzzy": {
                    "query": keyword,
                    "fuzziness": "AUTO",
                    "operator": "and",
                }
            }
        }

        text_match = {
            "match": {
                "text": {
                    "query": keyword,
                    "fuzziness": "AUTO",
                    "operator": "and",
                }
            }
        }

        response = await self.es.search(
            index=self.collection_name,
            body={
                "query": {"bool": {"should": [title_match, text_match]}},
                "highlight": {
                    "fields": {
                        "text": {
                            "fragment_size": self.snippet_length,
                            "number_of_fragments": 1,
                        },
                    },
                },
                "size": 5,
                # Return only snippets, not full documents
                "_source": ["title", "section"],
            },
        )
        hits = response["hits"]["hits"]
        self.logger.info(f"Query: {keyword}; Returned matches: {len(hits)}")
        return [_SearchResult.model_validate(_hit) for _hit in hits]
