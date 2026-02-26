"""Tests for the Synthesizer agent with mocked LLM calls."""

from unittest.mock import patch, MagicMock

import pytest

# Patch OpenAI before importing SynthesizerAgent so __init__ gets the mock
_mock_openai_cls = patch("src.hitachione.agents.synthesizer.OpenAI", MagicMock())
_mock_openai_cls.start()

from src.hitachione.agents.synthesizer import (  # noqa: E402
    SynthesizerAgent,
    _build_research_block,
    _estimate_confidence,
)
from src.hitachione.models.schemas import (  # noqa: E402
    CompanyResearch,
    Intent,
    SynthesizedAnswer,
    TaskContext,
    ToolError,
)


# ── Helpers ─────────────────────────────────────────────────────────────

def _cr(ticker: str, rating=8, score=7) -> CompanyResearch:
    return CompanyResearch(
        ticker=ticker,
        sentiment={"rating": rating, "label": "Positive", "rationale": "Good"},
        performance={"performance_score": score, "outlook": "Bullish",
                      "justification": "Strong growth"},
        news_snippets=[f"{ticker} news 1", f"{ticker} news 2"],
    )


def _ctx(query: str = "test", intent: Intent = Intent.COMPARE) -> TaskContext:
    ctx = TaskContext(user_query=query)
    ctx.intent = intent
    return ctx


# ── _estimate_confidence ────────────────────────────────────────────────

class TestConfidenceEstimation:
    def test_full_data(self):
        assert _estimate_confidence([_cr("AAPL"), _cr("MSFT")]) == 1.0

    def test_sentiment_only(self):
        cr = CompanyResearch(ticker="X", sentiment={"rating": 5})
        assert _estimate_confidence([cr]) == 0.5

    def test_performance_only(self):
        cr = CompanyResearch(ticker="X", performance={"performance_score": 5})
        assert _estimate_confidence([cr]) == 0.5

    def test_no_data(self):
        cr = CompanyResearch(ticker="X")
        assert _estimate_confidence([cr]) == 0.0

    def test_empty_list(self):
        assert _estimate_confidence([]) == 0.0

    def test_mixed_coverage(self):
        full = _cr("AAPL")
        empty = CompanyResearch(ticker="X")
        assert _estimate_confidence([full, empty]) == 0.5


# ── _build_research_block ──────────────────────────────────────────────

class TestBuildResearchBlock:
    def test_includes_ticker_heading(self):
        block = _build_research_block([_cr("AAPL")])
        assert "## AAPL" in block

    def test_includes_sentiment(self):
        block = _build_research_block([_cr("AAPL")])
        assert "Sentiment" in block
        assert "rating=8" in block

    def test_includes_performance(self):
        block = _build_research_block([_cr("AAPL")])
        assert "Performance" in block

    def test_includes_errors(self):
        cr = CompanyResearch(ticker="X")
        cr.errors.append(ToolError(entity="X", tool="sentiment", error="timeout"))
        block = _build_research_block([cr])
        assert "Data gaps" in block

    def test_multiple_companies(self):
        block = _build_research_block([_cr("AAPL"), _cr("MSFT")])
        assert "AAPL" in block
        assert "MSFT" in block


# ── SynthesizerAgent.run ────────────────────────────────────────────────

class TestSynthesizerAgent:
    def _make_agent(self, output_text: str = "# Answer\nLooks great!"):
        """Create a SynthesizerAgent with a mocked LLM that returns the given text."""
        agent = SynthesizerAgent()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = output_text
        agent._llm = MagicMock()
        agent._llm.chat.completions.create.return_value = mock_resp
        return agent

    def test_happy_path(self):
        agent = self._make_agent("# Answer\nLooks great!")
        ctx = _ctx()
        ans = agent.run(ctx, [_cr("AAPL")])

        assert isinstance(ans, SynthesizedAnswer)
        assert "Answer" in ans.markdown
        assert ans.confidence == 1.0
        assert ans.caveats == []

    def test_llm_failure_uses_fallback(self):
        agent = SynthesizerAgent()
        agent._llm = MagicMock()
        agent._llm.chat.completions.create.side_effect = RuntimeError("API down")
        ctx = _ctx()
        ans = agent.run(ctx, [_cr("AAPL")])

        assert "AAPL" in ans.markdown
        assert "sentiment=8" in ans.markdown
        assert "LLM unavailable" in ans.rationale

    def test_caveats_for_partial_data(self):
        agent = self._make_agent("# Partial")
        cr = CompanyResearch(ticker="X")
        cr.errors.append(ToolError(entity="X", tool="sentiment", error="timeout"))
        ans = agent.run(_ctx(), [cr])

        assert len(ans.caveats) >= 1
        assert ans.confidence < 1.0

    def test_citations_collected(self):
        agent = self._make_agent("# Answer")
        ans = agent.run(_ctx(), [_cr("AAPL")])
        assert len(ans.citations) >= 1
