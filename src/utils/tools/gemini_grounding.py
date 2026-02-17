"""Implements a tool to fetch Google Search grounded responses from Gemini."""

import asyncio
import os
from typing import Any, Literal
from urllib.parse import urlparse

import backoff
import httpx
from pydantic import BaseModel
from pydantic.fields import Field
import copy


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


class GroundedResponse(BaseModel):
    """Response returned by Gemini."""

    text_with_citations: str
    web_search_queries: list[str]
    citations: dict[int, str]


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
        self._endpoint = f"{self.base_url.strip('/')}/api/v1/grounding_with_search"

    async def get_web_search_grounded_response(self, query: str) -> GroundedResponse:
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
        GroundedResponse
            Response returned by Gemini. This includes the text with citations added,
            the web search queries executed (expanded from the input query), and a
            mapping of the citation ids to the website where the citation is from.

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
            raise exc from exc

        response_json = response.json()

        candidates: list[dict[str, Any]] | None = response_json.get("candidates")

        grounding_metadata: dict[str, Any] | None = (
            candidates[0].get("grounding_metadata") if candidates else None
        )
        web_search_queries: list[str] = (
            grounding_metadata["web_search_queries"] if grounding_metadata else []
        )

        text_with_citations, citations = add_citations(response_json)

        return GroundedResponse(
            text_with_citations=text_with_citations,
            web_search_queries=web_search_queries,
            citations=citations,
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


def add_citations(response: dict[str, object]) -> tuple[str, dict[int, str]]:
    """Add citations to the Gemini response.

    Code based on example in [1]_.

    Parameters
    ----------
    response : dict of str to object
        JSON response returned by Gemini.

    Returns
    -------
    tuple[str, dict[int, str]]
        The synthesized text and a mapping of citation ids to source labels.

    References
    ----------
    .. [1] https://ai.google.dev/gemini-api/docs/google-search#attributing_sources_with_inline_citations
    """
    candidates = response.get("candidates") if isinstance(response, dict) else None
    if not candidates:
        return "", {}

    candidate = candidates[0] or {}
    content = candidate.get("content") if isinstance(candidate, dict) else {}
    parts = content.get("parts") if isinstance(content, dict) else []

    text = ""
    for part in parts if isinstance(parts, list) else []:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            text = part["text"]
            break
    if not text:
        return "", {}

    meta = candidate.get("grounding_metadata") if isinstance(candidate, dict) else {}
    raw_supports = meta.get("grounding_supports") if isinstance(meta, dict) else []
    supports = raw_supports if isinstance(raw_supports, list) else []
    raw_chunks = meta.get("grounding_chunks") if isinstance(meta, dict) else []
    chunks = raw_chunks if isinstance(raw_chunks, list) else []

    filtered_candidate = filter_omers_websites(candidate)

    citations: dict[int, str] = {}
    chunk_to_id: dict[int, int] = {}

    if supports and chunks:
        citations, chunk_to_id = _collect_citations(filtered_candidate)

    # Sort supports by end_index in descending order to avoid shifting issues
    # when inserting.
    sorted_supports = sorted(
        (s for s in supports if isinstance(s, dict) and s.get("segment")),
        key=lambda s: s["segment"].get("end_index", 0),
        reverse=True,
    )

    for support in sorted_supports:
        segment = support.get("segment") or {}
        end_index = segment.get("end_index")
        if not isinstance(end_index, int) or end_index < 0 or end_index > len(text):
            continue
        indices = support.get("grounding_chunk_indices") or []
        citation_links: list[str] = []
        for idx in indices:
            if not isinstance(idx, int):
                continue
            citation_id = chunk_to_id.get(idx)
            if citation_id is None or idx >= len(chunks):
                continue
            web = chunks[idx].get("web") if isinstance(chunks[idx], dict) else {}
            uri = web.get("uri") if isinstance(web, dict) else None
            if uri:
                citation_links.append(f"[{citation_id}]({uri})")

        if citation_links:
            citation_string = ", ".join(citation_links)
            text = text[:end_index] + citation_string + text[end_index:]

    return text, citations


def filter_omers_websites(candidate: dict):

    filtered_candidate = copy.deepcopy(candidate)

    chunks = filtered_candidate.get("grounding_metadata", {}).get(
        "grounding_chunks", []
    )

    filtered_chunks = [
        chunk
        for chunk in chunks
        if "omers" in chunk.get("web", {}).get("title", "").lower()
    ]

    # Replace grounding_chunks with filtered version
    if "grounding_metadata" in filtered_candidate:
        filtered_candidate["grounding_metadata"]["grounding_chunks"] = filtered_chunks

    return filtered_candidate


def _collect_citations(candidate: dict) -> tuple[dict[int, str], dict[int, int]]:
    """Collect citation ids from a candidate dict."""
    supports = candidate["grounding_metadata"]["grounding_supports"]
    chunks = candidate["grounding_metadata"]["grounding_chunks"]

    citations: dict[int, str] = {}
    chunk_to_id: dict[int, int] = {}
    next_id = 1

    def label_for(chunk: dict) -> str:
        web = chunk.get("web") or {}
        title = web.get("title")
        uri = web.get("uri")
        if title:
            return title
        if uri:
            parsed = urlparse(uri)
            return parsed.hostname or parsed.netloc or uri
        return "unknown source"

    for support in supports:
        if not isinstance(support, dict):
            continue
        for chunk_idx in support.get("grounding_chunk_indices", []):
            if chunk_idx not in chunk_to_id and 0 <= chunk_idx < len(chunks):
                label = label_for(chunks[chunk_idx])
                chunk_to_id[chunk_idx] = next_id
                citations[next_id] = label
                next_id += 1

    return citations, chunk_to_id
