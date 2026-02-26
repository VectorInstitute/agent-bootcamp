"""Sentiment analysis tool backed by Weaviate knowledge base.

Queries the Weaviate financial news collection for data related to a ticker,
year, or topic, then uses an LLM to produce a structured sentiment assessment.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env from project root (5 levels up from this file)
load_dotenv(Path(__file__).resolve().parents[4] / ".env")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.query import Filter
from openai import AsyncOpenAI

from utils.env_vars import Configs


# ---------------------------------------------------------------------------
# Weaviate helper (sync, for data retrieval)
# ---------------------------------------------------------------------------

WEAVIATE_COLLECTION = os.getenv("WEAVIATE_COLLECTION_NAME", "Hitachi_finance_news")


def _get_weaviate_sync_client():
    """Create a synchronous Weaviate client from environment variables."""
    http_host = os.getenv("WEAVIATE_HTTP_HOST", "localhost")
    api_key = os.getenv("WEAVIATE_API_KEY", "")

    if http_host.endswith(".weaviate.cloud"):
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=f"https://{http_host}",
            auth_credentials=AuthApiKey(api_key),
        )

    return weaviate.connect_to_custom(
        http_host=http_host,
        http_port=int(os.getenv("WEAVIATE_HTTP_PORT", "8080")),
        http_secure=os.getenv("WEAVIATE_HTTP_SECURE", "false").lower() == "true",
        grpc_host=os.getenv("WEAVIATE_GRPC_HOST", "localhost"),
        grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
        grpc_secure=os.getenv("WEAVIATE_GRPC_SECURE", "false").lower() == "true",
        auth_credentials=AuthApiKey(api_key),
    )


def query_weaviate_by_ticker(
    ticker: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Retrieve news, earnings, and price data for a ticker from Weaviate."""
    client = _get_weaviate_sync_client()
    try:
        col = client.collections.get(WEAVIATE_COLLECTION)
        ticker_filter = Filter.by_property("ticker").equal(ticker.upper())

        response = col.query.fetch_objects(
            filters=ticker_filter,
            limit=limit,
            return_properties=[
                "text", "title", "date", "category", "dataset_source",
                "ticker", "company", "quarter", "fiscal_year",
            ],
        )

        # Also grab news that *mention* this ticker
        mentioned_response = col.query.fetch_objects(
            filters=Filter.by_property("mentioned_companies").contains_any(
                [ticker.upper()]
            ),
            limit=limit,
            return_properties=[
                "text", "title", "date", "category", "dataset_source",
            ],
        )

        seen_titles: set[str] = set()
        results: list[dict] = []
        for obj in list(response.objects) + list(mentioned_response.objects):
            props = {k: v for k, v in obj.properties.items() if v is not None}
            title = props.get("title", "")
            if title not in seen_titles:
                seen_titles.add(title)
                results.append(props)

        return sorted(results, key=lambda d: d.get("date", ""))
    finally:
        client.close()


def query_weaviate_by_year(
    year: int,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Retrieve financial data for a specific year from Weaviate."""
    client = _get_weaviate_sync_client()
    try:
        col = client.collections.get(WEAVIATE_COLLECTION)

        results: list[dict] = []
        seen_titles: set[str] = set()

        for obj in col.iterator(
            include_vector=False,
            return_properties=[
                "text", "title", "date", "category", "dataset_source",
                "ticker", "company",
            ],
        ):
            date_str = obj.properties.get("date", "")
            if date_str and str(year) in str(date_str)[:4]:
                props = {k: v for k, v in obj.properties.items() if v is not None}
                title = props.get("title", "")
                if title not in seen_titles:
                    seen_titles.add(title)
                    results.append(props)
                    if len(results) >= limit:
                        break

        return sorted(results, key=lambda d: d.get("date", ""))
    finally:
        client.close()


def query_weaviate_by_topic(
    topic: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Retrieve articles matching a topic via keyword search in Weaviate."""
    client = _get_weaviate_sync_client()
    try:
        col = client.collections.get(WEAVIATE_COLLECTION)

        response = col.query.bm25(
            query=topic,
            limit=limit,
            return_properties=[
                "text", "title", "date", "category", "dataset_source",
                "ticker", "company",
            ],
        )

        results = []
        for obj in response.objects:
            props = {k: v for k, v in obj.properties.items() if v is not None}
            results.append(props)

        return sorted(results, key=lambda d: d.get("date", ""))
    finally:
        client.close()


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

# Rating scale used across all prompts:
#   1-4  → Negative
#   5    → Neutral
#   6-10 → Positive

_RATING_SCALE_TEXT = (
    "Use a 1-10 sentiment rating scale where: "
    "1-4 = Negative, 5 = Neutral, 6-10 = Positive. "
)

SYSTEM_PROMPT = (
    "You are a sentiment analysis agent. "
    + _RATING_SCALE_TEXT
    + "Return JSON with keys: rating (integer 1-10), "
    "rationale (short string explaining the score)."
)

NEWS_SYSTEM_PROMPT = (
    "You are a financial news sentiment analyst. "
    "Given financial data (news articles, earnings transcripts, price data), "
    + _RATING_SCALE_TEXT
    + "Return JSON with keys: rating (integer 1-10), "
    "rationale (short string explaining the score), "
    "references (array of short quoted phrases from the data)."
)

YEAR_SYSTEM_PROMPT = (
    "You are a financial news sentiment analyst. "
    "Given snippets from financial news in a specific year, "
    + _RATING_SCALE_TEXT
    + "Return JSON with keys: year (integer), rating (integer 1-10), "
    "rationale (short string explaining the score)."
)

TICKER_SYSTEM_PROMPT = (
    "You are a financial sentiment analyst. "
    "Given all available data (price history, earnings transcripts, news articles) "
    "for a specific stock ticker, provide an overall sentiment assessment. "
    + _RATING_SCALE_TEXT
    + "Return JSON with keys: ticker (string), rating (integer 1-10), "
    "rationale (short string explaining the score), "
    "references (array of short quoted phrases from the data)."
)

EXAMPLE_TEXT = (
    "The market showed resilience today despite inflation fears, "
    "with tech stocks leading the recovery."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run an async coroutine, handling nested event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def _derive_label(rating: int | float | None) -> str:
    """Derive a sentiment label from a 1-10 rating.

    Scale:
        1-4  → Negative
        5    → Neutral
        6-10 → Positive
    """
    if rating is None:
        return "unknown"
    r = int(round(rating))
    if r <= 4:
        return "Negative"
    elif r == 5:
        return "Neutral"
    else:
        return "Positive"


def _parse_json_response(content: str) -> dict[str, Any]:
    """Parse JSON from LLM response, handling code fences."""
    content = content.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    try:
        parsed = json.loads(content)
        # If the LLM returns a list, wrap the first element or flatten
        if isinstance(parsed, list):
            return parsed[0] if parsed else {"rating": None, "label": "unknown", "rationale": content}
        return parsed
    except json.JSONDecodeError:
        return {"rating": None, "label": "unknown", "rationale": content}


def _format_kb_data(records: list[dict[str, Any]], max_chars: int = 15000) -> str:
    """Format Weaviate records into a text block for the LLM."""
    parts: list[str] = []
    total = 0
    for rec in records:
        source = rec.get("dataset_source", "unknown")
        date = rec.get("date", "")
        title = rec.get("title", "")
        text = str(rec.get("text", ""))[:1000]
        entry = f"[{source} | {date}] {title}\n{text}"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)
    return "\n\n".join(parts)


async def analyze_sentiment(text: str, model: str | None = None) -> dict[str, Any]:
    """Analyze sentiment for arbitrary text."""
    configs = Configs()
    client = AsyncOpenAI(
        base_url=configs.openai_base_url,
        api_key=configs.openai_api_key,
    )

    try:
        response = await client.chat.completions.create(
            model=model or configs.default_worker_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        content = (response.choices[0].message.content or "").strip()
    finally:
        await client.close()

    data = _parse_json_response(content)
    # Normalise to rating-based output
    rating = data.get("rating")
    if rating is not None:
        rating = int(round(rating))
    data["rating"] = rating
    data["label"] = _derive_label(rating)
    data.setdefault("rationale", "")
    # Remove legacy score key if present
    data.pop("score", None)
    return data


async def analyze_ticker_sentiment(
    ticker: str,
    model: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Analyze sentiment for a stock ticker using Weaviate KB data.

    Retrieves all available data (price history, earnings, news) for the
    ticker from the Weaviate knowledge base and produces a sentiment rating.
    """
    configs = Configs()
    client = AsyncOpenAI(
        base_url=configs.openai_base_url,
        api_key=configs.openai_api_key,
    )

    records = query_weaviate_by_ticker(ticker, limit=limit)

    if not records:
        return {
            "ticker": ticker.upper(),
            "rating": None,
            "label": "unknown",
            "rationale": f"No data found for ticker {ticker} in the knowledge base.",
            "references": [],
        }

    combined = _format_kb_data(records)

    try:
        response = await client.chat.completions.create(
            model=model or configs.default_worker_model,
            messages=[
                {"role": "system", "content": TICKER_SYSTEM_PROMPT},
                {"role": "user", "content": combined},
            ],
            temperature=0,
        )
        content = (response.choices[0].message.content or "").strip()
    finally:
        await client.close()

    data = _parse_json_response(content)
    rating = data.get("rating")
    if rating is not None:
        rating = int(round(rating))
    data["ticker"] = ticker.upper()
    data["rating"] = rating
    data["label"] = _derive_label(rating)
    data.setdefault("rationale", "")
    data.setdefault("references", [])
    return data


async def analyze_financial_year_sentiment(
    year: int,
    model: str | None = None,
    max_results: int = 20,
) -> dict[str, Any]:
    """Analyze sentiment for financial news in a given year using Weaviate KB."""
    configs = Configs()
    client = AsyncOpenAI(
        base_url=configs.openai_base_url,
        api_key=configs.openai_api_key,
    )

    records = query_weaviate_by_year(year, limit=max_results)

    if not records:
        return {
            "year": year,
            "rating": None,
            "label": "unknown",
            "rationale": f"No data found for year {year} in the knowledge base.",
        }

    combined = _format_kb_data(records)

    try:
        response = await client.chat.completions.create(
            model=model or configs.default_worker_model,
            messages=[
                {"role": "system", "content": YEAR_SYSTEM_PROMPT},
                {"role": "user", "content": combined},
            ],
            temperature=0,
        )
        content = (response.choices[0].message.content or "").strip()
    finally:
        await client.close()

    data = _parse_json_response(content)
    rating = data.get("rating")
    if rating is not None:
        rating = int(round(rating))
    data["year"] = year
    data["rating"] = rating
    data["label"] = _derive_label(rating)
    data.setdefault("rationale", "")
    # Remove legacy score key if present
    data.pop("score", None)
    return data


async def analyze_financial_news_sentiment(
    query: str,
    model: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Analyze sentiment for financial news matching a topic query.

    Searches the Weaviate KB for articles matching the query and produces
    a sentiment rating based on the retrieved content.
    """
    configs = Configs()
    client = AsyncOpenAI(
        base_url=configs.openai_base_url,
        api_key=configs.openai_api_key,
    )

    records = query_weaviate_by_topic(query, limit=limit)

    if not records:
        return {
            "rating": None,
            "label": "unknown",
            "rationale": f"No articles found for query '{query}' in the knowledge base.",
            "references": [],
        }

    combined = _format_kb_data(records)

    try:
        response = await client.chat.completions.create(
            model=model or configs.default_worker_model,
            messages=[
                {"role": "system", "content": NEWS_SYSTEM_PROMPT},
                {"role": "user", "content": combined},
            ],
            temperature=0,
        )
        content = (response.choices[0].message.content or "").strip()
    finally:
        await client.close()

    data = _parse_json_response(content)
    rating = data.get("rating")
    if rating is not None:
        rating = int(round(rating))
    data["rating"] = rating
    data["label"] = _derive_label(rating)
    data.setdefault("rationale", "")
    data.setdefault("references", [])
    return data


# ---------------------------------------------------------------------------
# Synchronous wrappers for external callers
# ---------------------------------------------------------------------------

def analyze_sentiment_sync(text: str, model: str | None = None) -> dict[str, Any]:
    """Synchronous wrapper for ``analyze_sentiment``."""
    return _run_async(analyze_sentiment(text, model=model))


def analyze_ticker_sentiment_sync(
    ticker: str, model: str | None = None, limit: int = 20,
) -> dict[str, Any]:
    """Synchronous wrapper for ``analyze_ticker_sentiment``."""
    return _run_async(analyze_ticker_sentiment(ticker, model=model, limit=limit))


def analyze_year_sentiment_sync(
    year: int, model: str | None = None, max_results: int = 20,
) -> dict[str, Any]:
    """Synchronous wrapper for ``analyze_financial_year_sentiment``."""
    return _run_async(
        analyze_financial_year_sentiment(year, model=model, max_results=max_results)
    )


def analyze_news_sentiment_sync(
    query: str, model: str | None = None, limit: int = 10,
) -> dict[str, Any]:
    """Synchronous wrapper for ``analyze_financial_news_sentiment``."""
    return _run_async(
        analyze_financial_news_sentiment(query, model=model, limit=limit)
    )


# ---------------------------------------------------------------------------
# OpenAI tool schemas
# ---------------------------------------------------------------------------

TOOL_SCHEMA_TICKER = {
    "type": "function",
    "function": {
        "name": "analyze_ticker_sentiment",
        "description": (
            "Analyze the overall sentiment for a stock ticker using the "
            "Weaviate financial knowledge base. Returns a sentiment rating "
            "(1-10: 1-4 Negative, 5 Neutral, 6-10 Positive), "
            "label, rationale, and supporting references."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. 'AAPL', 'TSLA'.",
                },
            },
            "required": ["ticker"],
        },
    },
}

TOOL_SCHEMA_YEAR = {
    "type": "function",
    "function": {
        "name": "analyze_year_sentiment",
        "description": (
            "Analyze the overall financial-news sentiment for a given year "
            "using the Weaviate knowledge base. Returns a rating "
            "(1-10: 1-4 Negative, 5 Neutral, 6-10 Positive), label, and rationale."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "year": {
                    "type": "integer",
                    "description": "Four-digit year, e.g. 2024.",
                },
            },
            "required": ["year"],
        },
    },
}

TOOL_SCHEMA_NEWS = {
    "type": "function",
    "function": {
        "name": "analyze_news_sentiment",
        "description": (
            "Search the Weaviate knowledge base for financial news matching "
            "a topic query and return a sentiment rating "
            "(1-10: 1-4 Negative, 5 Neutral, 6-10 Positive), label, and rationale."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language topic query, e.g. 'tech earnings Q3'.",
                },
            },
            "required": ["query"],
        },
    },
}

TOOL_SCHEMA_TEXT = {
    "type": "function",
    "function": {
        "name": "analyze_sentiment",
        "description": (
            "Classify arbitrary text sentiment on a 1-10 scale: "
            "1-4 = Negative, 5 = Neutral, 6-10 = Positive. "
            "Returns a rating, label, and rationale."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Free-form text to analyze.",
                },
            },
            "required": ["text"],
        },
    },
}

# Convenience list of all schemas
TOOL_SCHEMAS = [
    TOOL_SCHEMA_TICKER,
    TOOL_SCHEMA_YEAR,
    TOOL_SCHEMA_NEWS,
    TOOL_SCHEMA_TEXT,
]

TOOL_IMPLEMENTATIONS = {
    "analyze_ticker_sentiment": analyze_ticker_sentiment_sync,
    "analyze_year_sentiment": analyze_year_sentiment_sync,
    "analyze_news_sentiment": analyze_news_sentiment_sync,
    "analyze_sentiment": analyze_sentiment_sync,
}


# ---------------------------------------------------------------------------
# CLI (for quick interactive testing)
# ---------------------------------------------------------------------------

def _format_output(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sentiment analysis tool (Weaviate KB)")
    parser.add_argument("text", nargs="?", default=None, help="Free text to analyze")
    parser.add_argument("--model", help="Optional model override")
    parser.add_argument("--ticker", help="Analyze sentiment for a stock ticker")
    parser.add_argument("--year", type=int, help="Analyze sentiment for a year")
    parser.add_argument("--query", help="Search KB for matching financial news")
    args = parser.parse_args()

    if args.ticker:
        result = analyze_ticker_sentiment_sync(args.ticker, model=args.model)
    elif args.year is not None:
        result = analyze_year_sentiment_sync(args.year, model=args.model)
    elif args.query:
        result = analyze_news_sentiment_sync(args.query, model=args.model)
    elif args.text:
        result = analyze_sentiment_sync(args.text, model=args.model)
    else:
        result = analyze_ticker_sentiment_sync("AAPL", model=args.model)

    print(_format_output(result))
