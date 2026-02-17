"""Knowledge Retrieval Agent – consults Weaviate for context enrichment.

Responsibilities:
  • Resolve entity aliases (e.g. "Google" → GOOGL)
  • Find prior top-lists or summaries relevant to the query
  • Provide entity hints the Orchestrator can use for planning

Non-blocking: failures return partial / empty results rather than crashing.
"""

from __future__ import annotations

import logging
from typing import Any

from ..config.settings import (
    WEAVIATE_COLLECTION, WEAVIATE_API_KEY, WEAVIATE_HTTP_HOST,
)
from ..models.schemas import TaskContext

logger = logging.getLogger(__name__)


def _weaviate_client():
    """Build a *sync* Weaviate client."""
    import weaviate
    from weaviate.auth import AuthApiKey

    if WEAVIATE_HTTP_HOST.endswith(".weaviate.cloud"):
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=f"https://{WEAVIATE_HTTP_HOST}",
            auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
        )
    return weaviate.connect_to_custom(
        http_host=WEAVIATE_HTTP_HOST,
        http_port=443, http_secure=True,
        grpc_host=WEAVIATE_HTTP_HOST, grpc_port=443, grpc_secure=True,
        auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
    )


class KnowledgeRetrievalAgent:
    """Look up aliases, entity hints, and prior summaries from the KB."""

    def run(self, ctx: TaskContext) -> dict[str, Any]:
        """Return ``{"aliases": {...}, "entity_hints": [...], "summaries": [...]}``."""
        result: dict[str, Any] = {"aliases": {}, "entity_hints": [], "summaries": []}
        try:
            client = _weaviate_client()
            try:
                col = client.collections.get(WEAVIATE_COLLECTION)

                # 1) BM25 search on the user query for entity hints / summaries
                resp = col.query.bm25(
                    query=ctx.user_query,
                    limit=8,
                    return_properties=[
                        "text", "title", "ticker", "company",
                        "dataset_source", "date",
                    ],
                )
                for obj in resp.objects:
                    p = {k: v for k, v in obj.properties.items() if v is not None}
                    ticker = p.get("ticker")
                    company = p.get("company")
                    if ticker and company:
                        result["aliases"][company] = ticker
                    if ticker and ticker not in result["entity_hints"]:
                        result["entity_hints"].append(ticker)
                    result["summaries"].append(
                        f"[{p.get('dataset_source','')} | {p.get('date','')}] "
                        f"{p.get('title','')}"
                    )

                # 2) If the context already has entities, resolve them
                for entity in ctx.entities:
                    if entity.upper() not in result["aliases"].values():
                        result["aliases"][entity] = entity.upper()

            finally:
                client.close()

        except Exception as exc:
            logger.warning("KnowledgeRetrievalAgent error: %s", exc)
            ctx.uncertainties.append(f"KB lookup failed: {exc}")

        return result
