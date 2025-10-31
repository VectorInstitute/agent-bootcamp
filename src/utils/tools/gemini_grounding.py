"""Implements a tool to fetch Google Search grounded responses from Gemini."""

import asyncio
import os
from collections.abc import Mapping
from typing import Literal

import backoff
import httpx
from pydantic import BaseModel
from pydantic.fields import Field


RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class ModelSettings(BaseModel):
    """Configuration for the Gemini model used for web search."""

    model: Literal["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"] = (
        "gemini-2.5-flash"
    )
    temperature: float | None = Field(default=0.2, ge=0, le=2)
    max_output_tokens: int | None = Field(default=None, ge=1)
    seed: int | None = None
    thinking_budget: int | None = Field(default=-1, ge=-1)


class Response(BaseModel):
    """Response returned by Gemini."""

    text_with_citations: str
    web_search_queries: list[str]
    raw_response: Mapping[str, object]


class GeminiGroundingWithGoogleSearch:
    """Tool for fetching Google Search grounded responses from Gemini via a proxy.

    Parameters
    ----------
    base_url : str, optional, default=None
        Base URL for the Gemini proxy. Defaults to the value of the
        ``WEB_SEARCH_BASE_URL`` environment variable.
    api_key : str, optional, default=None
        API key for the Gemini proxy. Defaults to the value of the
        ``WEB_SEARCH_API_KEY`` environment variable.
    model_settings : ModelSettings, optional, default=None
        Settings for the Gemini model used for web search.
    max_concurrency : int, optional, default=5
        Maximum number of concurrent Gemini requests.
    timeout : int, optional, default=300
        Timeout for requests to the server.

    Raises
    ------
    ValueError
        If the ``WEB_SEARCH_API_KEY`` environment variable is not set or the
        ``WEB_SEARCH_BASE_URL`` environment variable is not set.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        model_settings: ModelSettings | None = None,
        max_concurrency: int = 5,
        timeout: int = 300,
    ) -> None:
        self.base_url = base_url or os.getenv("WEB_SEARCH_BASE_URL")
        self.api_key = api_key or os.getenv("WEB_SEARCH_API_KEY")
        self.model_settings = model_settings or ModelSettings()

        if self.api_key is None:
            raise ValueError("WEB_SEARCH_API_KEY environment variable is not set.")
        if self.base_url is None:
            raise ValueError("WEB_SEARCH_BASE_URL environment variable is not set.")

        self._semaphore = asyncio.Semaphore(max_concurrency)

        self._client = httpx.AsyncClient(
            timeout=timeout, headers={"X-API-Key": self.api_key}
        )
        self._endpoint = f"{self.base_url}/api/v1/grounding_with_search"

    async def get_web_search_grounded_response(self, query: str) -> Response:
        """Get Google Search grounded response to query from Gemini model.

        This function calls a Gemini model with Google Search tool enabled. How
        it works [1]_:
            - The model analyzes the input query and determines if a Google Search
              can improve the answer.
            - If needed, the model automatically generates one or multiple search
              queries and executes them.
            - The model processes the search results, synthesizes the information,
              and formulates a response.
            - The API returns a final, user-friendly response that is grounded in
              the search results.

        Parameters
        ----------
        query : str
            Query to pass to Gemini.

        Returns
        -------
        Response
            Response returned by Gemini. This includes the text with citations added,
            the web search queries executed (expanded from the input query), and the
            raw response object from the API.

        References
        ----------
        .. [1] https://ai.google.dev/gemini-api/docs/google-search#how_grounding_with_google_search_works
        """
        # Payload
        payload = self.model_settings.model_dump(exclude_unset=True)
        payload["query"] = query

        # Call Gemini
        response = await self._post_payload(payload)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Gemini call failed with status {exc.response.status_code}"
            ) from exc

        response_json = response.json()
        text_with_citations = add_citations(response_json)

        return Response(
            text_with_citations=text_with_citations,
            web_search_queries=response_json["candidates"][0]["grounding_metadata"][
                "web_search_queries"
            ],
            raw_response=response_json,
        )

    @backoff.on_exception(
        backoff.expo,
        (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.HTTPStatusError,  # only retry codes in RETRYABLE_STATUS
        ),
        giveup=lambda exc: (
            isinstance(exc, httpx.HTTPStatusError)
            and exc.response.status_code not in RETRYABLE_STATUS
        ),
        jitter=backoff.full_jitter,
        max_tries=5,
    )
    async def _post_payload(self, payload: dict[str, object]) -> httpx.Response:
        """Send a POST request to the endpoint with the given payload."""
        async with self._semaphore:
            return await self._client.post(self._endpoint, json=payload)


def add_citations(response: dict[str, object]) -> str:
    """Add citations to the Gemini response.

    Code based on example in [1]_.

    Parameters
    ----------
    response : dict of str to object
        JSON response returned by Gemini.

    Returns
    -------
    str
        Text with citations added.

    References
    ----------
    .. [1] https://ai.google.dev/gemini-api/docs/google-search#attributing_sources_with_inline_citations
    """
    text = response["candidates"][0]["content"]["parts"][0]["text"]
    supports = response["candidates"][0]["grounding_metadata"]["grounding_supports"]
    chunks = response["candidates"][0]["grounding_metadata"]["grounding_chunks"]

    # Sort supports by end_index in descending order to avoid shifting issues
    # when inserting.
    sorted_supports = sorted(
        supports, key=lambda s: s["segment"]["end_index"], reverse=True
    )

    for support in sorted_supports:
        end_index = support["segment"]["end_index"]
        if support["grounding_chunk_indices"]:
            # Create citation string like [1](link1)[2](link2)
            citation_links = []
            for i in support["grounding_chunk_indices"]:
                if i < len(chunks):
                    uri = chunks[i]["web"]["uri"]
                    citation_links.append(f"[{i + 1}]({uri})")

            citation_string = ", ".join(citation_links)
            text = text[:end_index] + citation_string + text[end_index:]

    return text
