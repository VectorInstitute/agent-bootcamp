"""Tests for the Researcher agent with mocked tool calls."""

import time
from concurrent.futures import TimeoutError
from unittest.mock import patch, MagicMock, call

import pytest
from src.hitachione.agents.researcher import (
    ResearcherAgent, _research_one, _call_with_retry,
)
from src.hitachione.models.schemas import CompanyResearch, TaskContext, ToolError


# ── Fixtures / helpers ──────────────────────────────────────────────────

def _fake_sentiment(ticker: str) -> dict:
    return {"rating": 8, "label": "Positive", "rationale": f"{ticker} looks good",
            "references": [f"ref1-{ticker}", f"ref2-{ticker}"]}


def _fake_performance(ticker: str) -> dict:
    return {"performance_score": 7, "outlook": "Bullish",
            "justification": f"{ticker} strong growth"}


def _slow_sentiment(ticker: str) -> dict:
    """Simulates a 0.3s network call."""
    time.sleep(0.3)
    return _fake_sentiment(ticker)


def _slow_performance(ticker: str) -> dict:
    """Simulates a 0.3s network call."""
    time.sleep(0.3)
    return _fake_performance(ticker)


def _failing_sentiment(ticker: str) -> dict:
    raise RuntimeError(f"Sentiment API down for {ticker}")


def _failing_performance(ticker: str) -> dict:
    raise RuntimeError(f"Performance API down for {ticker}")


# ── _research_one ───────────────────────────────────────────────────────

# Disable retry delays / limit retries to 1 for fast failure tests
_fast_fail = [
    patch("src.hitachione.agents.researcher._RETRY_BACKOFF", 0.0),
    patch("src.hitachione.agents.researcher._MAX_RETRIES", 1),
]


def _apply_fast_fail(fn):
    """Stack _fast_fail patches onto a test function."""
    for p in reversed(_fast_fail):
        fn = p(fn)
    return fn


class TestResearchOne:
    @patch("src.hitachione.agents.researcher._performance", side_effect=_fake_performance)
    @patch("src.hitachione.agents.researcher._sentiment", side_effect=_fake_sentiment)
    def test_happy_path(self, mock_sent, mock_perf):
        cr = _research_one("AAPL")
        assert cr.ticker == "AAPL"
        assert cr.sentiment["rating"] == 8
        assert cr.performance["performance_score"] == 7
        assert cr.errors == []
        assert len(cr.news_snippets) == 2

    @_apply_fast_fail
    @patch("src.hitachione.agents.researcher._performance", side_effect=_fake_performance)
    @patch("src.hitachione.agents.researcher._sentiment", side_effect=_failing_sentiment)
    def test_sentiment_failure_captured(self, mock_sent, mock_perf):
        cr = _research_one("TSLA")
        assert cr.sentiment == {}
        assert cr.performance["performance_score"] == 7
        assert len(cr.errors) == 1
        assert cr.errors[0].tool == "sentiment"

    @_apply_fast_fail
    @patch("src.hitachione.agents.researcher._performance", side_effect=_failing_performance)
    @patch("src.hitachione.agents.researcher._sentiment", side_effect=_fake_sentiment)
    def test_performance_failure_captured(self, mock_sent, mock_perf):
        cr = _research_one("MSFT")
        assert cr.sentiment["rating"] == 8
        assert cr.performance == {}
        assert len(cr.errors) == 1
        assert cr.errors[0].tool == "performance"

    @_apply_fast_fail
    @patch("src.hitachione.agents.researcher._performance", side_effect=_failing_performance)
    @patch("src.hitachione.agents.researcher._sentiment", side_effect=_failing_sentiment)
    def test_both_failures(self, mock_sent, mock_perf):
        cr = _research_one("JPM")
        assert len(cr.errors) == 2


# ── ResearcherAgent.run (parallel fan-out) ──────────────────────────────

class TestResearcherAgentRun:
    @patch("src.hitachione.agents.researcher._performance", side_effect=_fake_performance)
    @patch("src.hitachione.agents.researcher._sentiment", side_effect=_fake_sentiment)
    def test_multiple_entities(self, mock_sent, mock_perf):
        agent = ResearcherAgent()
        ctx = TaskContext(user_query="test")
        results = agent.run(ctx, ["AAPL", "MSFT", "TSLA"])
        assert len(results) == 3
        assert [r.ticker for r in results] == ["AAPL", "MSFT", "TSLA"]
        assert all(r.errors == [] for r in results)

    @patch("src.hitachione.agents.researcher._performance", side_effect=_fake_performance)
    @patch("src.hitachione.agents.researcher._sentiment", side_effect=_fake_sentiment)
    def test_preserves_order(self, mock_sent, mock_perf):
        """Results should be in the same order as input entities."""
        agent = ResearcherAgent()
        ctx = TaskContext(user_query="test")
        order = ["NVDA", "AAPL", "GOOGL", "META"]
        results = agent.run(ctx, order)
        assert [r.ticker for r in results] == order

    @patch("src.hitachione.agents.researcher._performance", side_effect=_fake_performance)
    @patch("src.hitachione.agents.researcher._sentiment", side_effect=_fake_sentiment)
    def test_empty_entities(self, mock_sent, mock_perf):
        agent = ResearcherAgent()
        ctx = TaskContext(user_query="test")
        results = agent.run(ctx, [])
        assert results == []

    @patch("src.hitachione.agents.researcher._performance", side_effect=_slow_performance)
    @patch("src.hitachione.agents.researcher._sentiment", side_effect=_slow_sentiment)
    def test_parallel_is_faster_than_sequential(self, mock_sent, mock_perf):
        """With 4 tickers × 0.3s each, parallel should finish well under 4×0.6s."""
        agent = ResearcherAgent(max_workers=4)
        ctx = TaskContext(user_query="test")
        tickers = ["AAPL", "MSFT", "GOOGL", "NVDA"]

        start = time.time()
        results = agent.run(ctx, tickers)
        elapsed = time.time() - start

        assert len(results) == 4
        # Sequential would be ~4 × 0.6 = 2.4s.  Parallel should be < 1.5s.
        assert elapsed < 1.5, f"Parallel research took {elapsed:.2f}s – too slow"

    @_apply_fast_fail
    @patch("src.hitachione.agents.researcher._performance", side_effect=_failing_performance)
    @patch("src.hitachione.agents.researcher._sentiment", side_effect=_failing_sentiment)
    def test_errors_propagated_to_context(self, mock_sent, mock_perf):
        agent = ResearcherAgent()
        ctx = TaskContext(user_query="test")
        results = agent.run(ctx, ["AAPL"])
        assert len(ctx.uncertainties) >= 2
        assert any("sentiment" in u for u in ctx.uncertainties)
        assert any("performance" in u for u in ctx.uncertainties)


# ── Timeout behaviour ──────────────────────────────────────────────────

class TestResearcherTimeout:
    @patch("src.hitachione.agents.researcher._TOOL_TIMEOUT", 0.1)
    @patch("src.hitachione.agents.researcher._MAX_RETRIES", 1)
    @patch("src.hitachione.agents.researcher._RETRY_BACKOFF", 0.0)
    @patch("src.hitachione.agents.researcher._performance", side_effect=_fake_performance)
    @patch("src.hitachione.agents.researcher._sentiment", side_effect=lambda t: time.sleep(5) or _fake_sentiment(t))
    def test_timeout_captured_as_error(self, mock_sent, mock_perf):
        """If a tool exceeds _TOOL_TIMEOUT, a ToolError is recorded."""
        cr = _research_one("SLOW")
        assert any(e.tool == "sentiment" and "timeout" in e.error for e in cr.errors)
        # Performance should still succeed
        assert cr.performance.get("performance_score") == 7


# ── Retry behaviour ───────────────────────────────────────────────────

class TestRetryLogic:
    @patch("src.hitachione.agents.researcher._RETRY_BACKOFF", 0.0)
    @patch("src.hitachione.agents.researcher._MAX_RETRIES", 3)
    def test_retry_succeeds_on_second_attempt(self):
        """If a tool fails once then succeeds, the result is captured."""
        call_count = 0
        def flaky_sentiment(ticker: str) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Weaviate connection reset")
            return _fake_sentiment(ticker)

        result = _call_with_retry(flaky_sentiment, "META", "sentiment")
        assert result["rating"] == 8
        assert call_count == 2  # failed once, succeeded on retry

    @patch("src.hitachione.agents.researcher._RETRY_BACKOFF", 0.0)
    @patch("src.hitachione.agents.researcher._MAX_RETRIES", 3)
    def test_retry_exhausted_raises(self):
        """If all retries fail, the last exception is raised."""
        def always_failing(ticker: str) -> dict:
            raise ConnectionError("Weaviate down permanently")

        with pytest.raises(ConnectionError, match="permanently"):
            _call_with_retry(always_failing, "META", "sentiment")

    @patch("src.hitachione.agents.researcher._RETRY_BACKOFF", 0.0)
    @patch("src.hitachione.agents.researcher._MAX_RETRIES", 3)
    @patch("src.hitachione.agents.researcher._performance", side_effect=_fake_performance)
    def test_research_one_retries_transient_failure(self, mock_perf):
        """_research_one recovers from a transient sentiment failure."""
        call_count = 0
        def flaky(ticker):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("gRPC reset")
            return _fake_sentiment(ticker)

        with patch("src.hitachione.agents.researcher._sentiment", side_effect=flaky):
            cr = _research_one("META")

        assert cr.errors == []
        assert cr.sentiment["rating"] == 8
        assert cr.performance["performance_score"] == 7
        assert call_count == 3  # failed twice, succeeded on third
