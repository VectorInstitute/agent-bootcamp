"""Researcher Agent – per-entity data fetch with parallel fan-out.

For each ticker the Researcher:
  1. Calls the sentiment analysis tool  → rating 1-10 + rationale
  2. Calls the performance analysis tool → score 1-10 + outlook
  3. Retries failed tool calls up to ``_MAX_RETRIES`` times with backoff
  4. Captures errors per entity (never crashes the whole run)

All tickers are researched **in parallel** using a thread pool, and the
two tool calls per ticker also run concurrently.  This cuts wall-clock
time from ``O(n × 2 × latency)`` to roughly ``O(latency)`` for typical
workloads (≤ 10 entities).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import Any

from ..models.schemas import CompanyResearch, TaskContext, ToolError

logger = logging.getLogger(__name__)

# Max parallel threads.  Keep modest to avoid API rate-limits.
_MAX_WORKERS = 8

# Per-tool timeout in seconds (prevents a single slow call from blocking)
_TOOL_TIMEOUT = 60

# Retry settings for transient errors (connection drops, gRPC resets, etc.)
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0  # seconds; doubles each attempt


# ── Lazy imports for the existing tools (avoid circular / heavy init) ──

def _sentiment(ticker: str) -> dict[str, Any]:
    from ..tools.sentiment_analysis_tool.tool import analyze_ticker_sentiment_sync
    return analyze_ticker_sentiment_sync(ticker)


def _performance(ticker: str) -> dict[str, Any]:
    from ..tools.performance_analysis_tool.tool import analyse_stock_performance
    return analyse_stock_performance(ticker)


def _call_with_retry(fn, ticker: str, tool_name: str) -> dict[str, Any]:
    """Call *fn(ticker)* with up to ``_MAX_RETRIES`` attempts on failure.

    Raises the last exception if all attempts fail.
    """
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return fn(ticker)
        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                wait = _RETRY_BACKOFF * (2 ** (attempt - 1))
                logger.warning(
                    "%s attempt %d/%d failed for %s: %s – retrying in %.1fs",
                    tool_name, attempt, _MAX_RETRIES, ticker, exc, wait,
                )
                time.sleep(wait)
            else:
                logger.warning(
                    "%s attempt %d/%d failed for %s: %s – giving up",
                    tool_name, attempt, _MAX_RETRIES, ticker, exc,
                )
    raise last_exc  # type: ignore[misc]


def _research_one(ticker: str) -> CompanyResearch:
    """Fetch sentiment + performance for one ticker **in parallel**."""
    cr = CompanyResearch(ticker=ticker)

    # Fire both tool calls concurrently within a small thread pool
    with ThreadPoolExecutor(max_workers=2) as inner:
        sent_future = inner.submit(_call_with_retry, _sentiment, ticker, "sentiment")
        perf_future = inner.submit(_call_with_retry, _performance, ticker, "performance")

        # Collect sentiment
        try:
            cr.sentiment = sent_future.result(timeout=_TOOL_TIMEOUT)
            refs = cr.sentiment.get("references", [])
            cr.news_snippets = [str(r) for r in refs][:5]
        except TimeoutError:
            logger.warning("Sentiment timed out for %s after %ds", ticker, _TOOL_TIMEOUT)
            cr.errors.append(ToolError(entity=ticker, tool="sentiment", error=f"timeout after {_TOOL_TIMEOUT}s"))
            sent_future.cancel()
        except Exception as exc:
            logger.warning("Sentiment error for %s: %s", ticker, exc)
            cr.errors.append(ToolError(entity=ticker, tool="sentiment", error=str(exc)))

        # Collect performance
        try:
            cr.performance = perf_future.result(timeout=_TOOL_TIMEOUT)
        except TimeoutError:
            logger.warning("Performance timed out for %s after %ds", ticker, _TOOL_TIMEOUT)
            cr.errors.append(ToolError(entity=ticker, tool="performance", error=f"timeout after {_TOOL_TIMEOUT}s"))
            perf_future.cancel()
        except Exception as exc:
            logger.warning("Performance error for %s: %s", ticker, exc)
            cr.errors.append(ToolError(entity=ticker, tool="performance", error=str(exc)))

    return cr


class ResearcherAgent:
    """Fan-out research across a list of entities **in parallel**."""

    def __init__(self, max_workers: int = _MAX_WORKERS):
        self.max_workers = max_workers

    def run(self, ctx: TaskContext, entities: list[str]) -> list[CompanyResearch]:
        """Research every entity concurrently; accumulate errors without crashing."""
        ctx.observations.append(
            f"Researching {len(entities)} entities in parallel "
            f"(max_workers={self.max_workers})…"
        )

        results_map: dict[str, CompanyResearch] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_ticker = {
                pool.submit(_research_one, ticker): ticker
                for ticker in entities
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    cr = future.result()
                except Exception as exc:
                    logger.error("Unhandled research error for %s: %s", ticker, exc)
                    cr = CompanyResearch(ticker=ticker)
                    cr.errors.append(
                        ToolError(entity=ticker, tool="research", error=str(exc))
                    )
                results_map[ticker] = cr

        # Preserve the original entity order in results
        results: list[CompanyResearch] = []
        for ticker in entities:
            cr = results_map.get(ticker, CompanyResearch(ticker=ticker))
            results.append(cr)
            if cr.errors:
                for e in cr.errors:
                    ctx.uncertainties.append(
                        f"{e.tool} failed for {e.entity}: {e.error}"
                    )

        return results
