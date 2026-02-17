"""Researcher Agent – per-entity data fetch with async fan-out.

For each ticker the Researcher:
  1. Calls the sentiment analysis tool  → rating 1-10 + rationale
  2. Calls the performance analysis tool → score 1-10 + outlook
  3. Captures errors per entity (never crashes the whole run)
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure the tools can import their helpers
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ..models.schemas import CompanyResearch, TaskContext, ToolError

logger = logging.getLogger(__name__)


# ── Lazy imports for the existing tools (avoid circular / heavy init) ──

def _sentiment(ticker: str) -> dict[str, Any]:
    from ..tools.sentiment_analysis_tool.tool import analyze_ticker_sentiment_sync
    return analyze_ticker_sentiment_sync(ticker)


def _performance(ticker: str) -> dict[str, Any]:
    from ..tools.performance_analysis_tool.tool import analyse_stock_performance
    return analyse_stock_performance(ticker)


def _research_one(ticker: str) -> CompanyResearch:
    """Fetch sentiment + performance for one ticker (sync)."""
    cr = CompanyResearch(ticker=ticker)

    # Sentiment
    try:
        cr.sentiment = _sentiment(ticker)
        refs = cr.sentiment.get("references", [])
        cr.news_snippets = [str(r) for r in refs][:5]
    except Exception as exc:
        logger.warning("Sentiment error for %s: %s", ticker, exc)
        cr.errors.append(ToolError(entity=ticker, tool="sentiment", error=str(exc)))

    # Performance
    try:
        cr.performance = _performance(ticker)
    except Exception as exc:
        logger.warning("Performance error for %s: %s", ticker, exc)
        cr.errors.append(ToolError(entity=ticker, tool="performance", error=str(exc)))

    return cr


class ResearcherAgent:
    """Fan-out research across a list of entities."""

    def run(self, ctx: TaskContext, entities: list[str]) -> list[CompanyResearch]:
        """Research every entity; accumulate errors without crashing."""
        results: list[CompanyResearch] = []
        for ticker in entities:
            ctx.observations.append(f"Researching {ticker}...")
            cr = _research_one(ticker)
            results.append(cr)
            if cr.errors:
                for e in cr.errors:
                    ctx.uncertainties.append(f"{e.tool} failed for {e.entity}: {e.error}")
        return results
