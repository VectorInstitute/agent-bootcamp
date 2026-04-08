"""
Tool for analysing stock performance using the Weaviate knowledge base.

Queries the Weaviate financial news collection for price history, earnings
transcripts, and financial news, then uses an LLM to produce a performance
rating score (1-10), future outlook, and justification.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, List

import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.query import Filter
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parents[4] / ".env")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils.client_manager import AsyncClientManager

# ---------------------------------------------------------------------------
# Weaviate helpers
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


# ---------------------------------------------------------------------------
# Data retrieval
# ---------------------------------------------------------------------------


def get_ticker_data(ticker: str) -> dict[str, list[dict]]:
    """Retrieve all Weaviate data for a given ticker, grouped by source.

    Returns a dict with keys ``price_data``, ``earnings``, ``news`` — each
    containing a list of property dicts.
    """
    client = _get_weaviate_sync_client()
    try:
        col = client.collections.get(WEAVIATE_COLLECTION)

        ticker_filter = Filter.by_property("ticker").equal(ticker.upper())

        # --- price data (stock_market) ---
        price_response = col.query.fetch_objects(
            filters=(
                ticker_filter
                & Filter.by_property("dataset_source").equal("stock_market")
            ),
            limit=100,
            return_properties=[
                "date", "open", "high", "low", "close", "volume", "text",
            ],
        )
        price_data = [
            {k: v for k, v in obj.properties.items() if v is not None}
            for obj in price_response.objects
        ]

        # --- earnings transcripts ---
        earnings_response = col.query.fetch_objects(
            filters=(
                ticker_filter
                & Filter.by_property("dataset_source").equal(
                    "sp500_earnings_transcripts"
                )
            ),
            limit=100,
            return_properties=[
                "date", "quarter", "fiscal_year", "fiscal_quarter", "text",
                "title",
            ],
        )
        earnings = [
            {k: v for k, v in obj.properties.items() if v is not None}
            for obj in earnings_response.objects
        ]

        # --- news (bloomberg + yahoo) ---
        news_response = col.query.fetch_objects(
            filters=(
                ticker_filter
                & (
                    Filter.by_property("dataset_source").equal(
                        "bloomberg_financial_news"
                    )
                    | Filter.by_property("dataset_source").equal(
                        "yahoo_finance_news"
                    )
                )
            ),
            limit=100,
            return_properties=["date", "title", "text", "category"],
        )
        # Also grab news that *mention* this ticker (mentioned_companies)
        mentioned_response = col.query.fetch_objects(
            filters=Filter.by_property("mentioned_companies").contains_any(
                [ticker.upper()]
            ),
            limit=50,
            return_properties=["date", "title", "text", "category"],
        )

        seen_titles: set[str] = set()
        news: list[dict] = []
        for obj in list(news_response.objects) + list(mentioned_response.objects):
            props = {k: v for k, v in obj.properties.items() if v is not None}
            title = props.get("title", "")
            if title not in seen_titles:
                seen_titles.add(title)
                news.append(props)

        return {
            "price_data": sorted(price_data, key=lambda d: d.get("date", "")),
            "earnings": sorted(earnings, key=lambda d: d.get("date", "")),
            "news": sorted(news, key=lambda d: d.get("date", "")),
        }

    finally:
        client.close()


# ---------------------------------------------------------------------------
# LLM-based performance scoring
# ---------------------------------------------------------------------------

_client_manager = None


def _get_client_manager() -> AsyncClientManager:
    global _client_manager
    if _client_manager is None:
        _client_manager = AsyncClientManager()
    return _client_manager


async def _analyse_with_llm(ticker: str, data: dict[str, list[dict]]) -> dict:
    """Send retrieved data to an LLM and get a structured performance analysis."""
    cm = _get_client_manager()

    # Build context sections
    price_summary = "\n".join(
        d.get("text", json.dumps(d)) for d in data["price_data"]
    ) or "No price data available."

    earnings_summary = "\n---\n".join(
        d.get("text", json.dumps(d)) for d in data["earnings"]
    ) or "No earnings data available."

    news_summary = "\n---\n".join(
        f"[{d.get('date','')}] {d.get('title','')}: {str(d.get('text',''))[:500]}"
        for d in data["news"]
    ) or "No news articles available."

    prompt = f"""You are a Stock Performance Analyst. Analyse the ticker "{ticker}" using ONLY the data provided below.

## Price History
{price_summary}

## Earnings Transcripts
{earnings_summary}

## News Articles
{news_summary}

Based on the data above, produce a JSON object (and NOTHING else) with exactly these keys:

{{
  "ticker": "{ticker}",
  "performance_score": <int 1-10>,
  "outlook": "<Bullish | Bearish | Volatile | Sideways>",
  "justification": "<2-4 sentence explanation citing specific data points>"
}}

Scoring guide:
  1-4  → Negative (declining price, poor earnings, negative news)
  5    → Neutral
  6-10 → Positive (rising price, strong earnings, positive sentiment)
"""

    response = await cm.openai_client.chat.completions.create(
        model=cm.configs.default_worker_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()

    # Extract JSON from potential markdown code fences
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    return json.loads(content)


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
# Public API
# ---------------------------------------------------------------------------


def analyse_stock_performance(ticker: str) -> dict:
    """Analyse a stock's performance using Weaviate knowledge base data.

    Retrieves price history, earnings transcripts, and news articles from the
    Weaviate financial news collection, then uses an LLM to produce a
    structured performance analysis.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. ``"AAPL"``, ``"TSLA"``).

    Returns
    -------
    dict
        A dictionary with keys:
        - ``ticker`` (str)
        - ``performance_score`` (int, 1–10)
        - ``outlook`` (str, one of Bullish/Bearish/Volatile/Sideways)
        - ``justification`` (str)
        - ``data_summary`` (dict with counts of price/earnings/news records)
    """
    ticker = ticker.upper().strip()
    data = get_ticker_data(ticker)

    if not any(data.values()):
        return {
            "ticker": ticker,
            "performance_score": None,
            "outlook": "Unknown",
            "justification": f"No data found for ticker {ticker} in the knowledge base.",
            "data_summary": {"price_records": 0, "earnings_records": 0, "news_records": 0},
        }

    analysis = _run_async(_analyse_with_llm(ticker, data))
    analysis["data_summary"] = {
        "price_records": len(data["price_data"]),
        "earnings_records": len(data["earnings"]),
        "news_records": len(data["news"]),
    }
    return analysis


# ---------------------------------------------------------------------------
# OpenAI tool schema
# ---------------------------------------------------------------------------

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "analyse_stock_performance",
        "description": (
            "Analyse a stock's performance using the Weaviate financial knowledge "
            "base. Returns a performance score (1-10), "
            "future outlook, and justification based on price history, earnings "
            "transcripts, and news articles."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. 'AAPL', 'TSLA', 'GOOGL'.",
                }
            },
            "required": ["ticker"],
        },
    },
}

TOOL_IMPLEMENTATIONS = {
    "analyse_stock_performance": analyse_stock_performance,
}