"""Implements knowledge retrieval tool for Weaviate."""

import asyncio
import logging
import os

import backoff
import openai
import pydantic
import weaviate
from weaviate import WeaviateAsyncClient
from weaviate.config import AdditionalConfig

from ..async_utils import rate_limited


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

    def __repr__(self) -> str:
        return self.model_dump_json(indent=2)


SearchResults = list[_SearchResult]


class AsyncWeaviateKnowledgeBase:
    """Configurable search tools for Weaviate knowledge base."""

    def __init__(
        self,
        async_client: WeaviateAsyncClient,
        collection_name: str,
        num_results: int = 5,
        snippet_length: int = 1000,
        max_concurrency: int = 3,
        embedding_model_name: str = "@cf/baai/bge-m3",
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
    ) -> None:
        self.async_client = async_client
        self.collection_name = collection_name
        self.num_results = num_results
        self.snippet_length = snippet_length
        self.logger = logging.getLogger(__name__)
        self.semaphore = asyncio.Semaphore(max_concurrency)

        self.embedding_model_name = embedding_model_name
        self.embedding_api_key = embedding_api_key
        self.embedding_base_url = embedding_base_url

        self._embed_client = openai.OpenAI(
            api_key=self.embedding_api_key or os.getenv("EMBEDDING_API_KEY"),
            base_url=self.embedding_base_url or os.getenv("EMBEDDING_BASE_URL"),
            max_retries=5,
        )

    @backoff.on_exception(backoff.expo, exception=asyncio.CancelledError)  # type: ignore
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
            vector = self._vectorize(keyword)
            response = await rate_limited(
                lambda: collection.query.hybrid(
                    keyword, vector=vector, limit=self.num_results
                ),
                semaphore=self.semaphore,
            )

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

    def _vectorize(self, text: str) -> list[float]:
        """Vectorize text using the embedding client.

        Parameters
        ----------
        text : str
            The text to be vectorized.

        Returns
        -------
        list[float]
            A list of floats representing the vectorized text.
        """
        response = self._embed_client.embeddings.create(
            input=text, model=self.embedding_model_name
        )
        return response.data[0].embedding


def get_weaviate_async_client(
    http_host: str | None = None,
    http_port: int | None = None,
    http_secure: bool = False,
    grpc_host: str | None = None,
    grpc_port: int | None = None,
    grpc_secure: bool = False,
    api_key: str | None = None,
    headers: dict[str, str] | None = None,
    additional_config: AdditionalConfig | None = None,
    skip_init_checks: bool = False,
) -> WeaviateAsyncClient:
    """Get an async Weaviate client.

    If no parameters are provided, the function will attempt to connect to a local
    Weaviate instance using environment variables.

    Parameters
    ----------
    http_host : str, optional, default=None
        The HTTP host for the Weaviate instance. If not provided, defaults to the
        `WEAVIATE_HTTP_HOST` environment variable or "localhost" if the environment
        variable is not set.
    http_port : int, optional, default=None
        The HTTP port for the Weaviate instance. If not provided, defaults to the
        `WEAVIATE_HTTP_PORT` environment variable or 8080 if the environment variable
        is not set.
    http_secure : bool, optional, default=False
        Whether to use HTTPS for the HTTP connection. Defaults to the
        `WEAVIATE_HTTP_SECURE` environment variable or `False` if the environment
        variable is not set.
    grpc_host : str, optional, default=None
        The gRPC host for the Weaviate instance. If not provided, defaults to the
        `WEAVIATE_GRPC_HOST` environment variable or "localhost" if the environment
        variable is not set.
    grpc_port : int, optional, default=None
        The gRPC port for the Weaviate instance. If not provided, defaults to the
        `WEAVIATE_GRPC_PORT` environment variable or 50051 if the environment variable
        is not set.
    grpc_secure : bool, optional, default=False
        Whether to use secure gRPC. Defaults to the `WEAVIATE_GRPC_SECURE` environment
        variable or `False` if the environment variable is not set.
    api_key : str, optional, default=None
        The API key for authentication with Weaviate. If not provided, defaults to the
        `WEAVIATE_API_KEY` environment variable.
    headers : dict[str, str], optional, default=None
        Additional headers to include in the request.
    additional_config : AdditionalConfig, optional, default=None
        Additional configuration for the Weaviate client.
    skip_init_checks : bool, optional, default=False
        Whether to skip initialization checks.

    Returns
    -------
    WeaviateAsyncClient
        An asynchronous Weaviate client configured with the provided parameters.
    """
    return weaviate.use_async_with_custom(
        http_host=http_host or os.getenv("WEAVIATE_HTTP_HOST", "localhost"),
        http_port=http_port or int(os.getenv("WEAVIATE_HTTP_PORT", "8080")),
        http_secure=http_secure
        or os.getenv("WEAVIATE_HTTP_SECURE", "false").lower() == "true",
        grpc_host=grpc_host or os.getenv("WEAVIATE_GRPC_HOST", "localhost"),
        grpc_port=grpc_port or int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
        grpc_secure=grpc_secure
        or os.getenv("WEAVIATE_GRPC_SECURE", "false").lower() == "true",
        auth_credentials=api_key or os.getenv("WEAVIATE_API_KEY"),
        headers=headers,
        additional_config=additional_config,
        skip_init_checks=skip_init_checks,
    )
