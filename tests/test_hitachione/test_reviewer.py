"""Tests for the Reviewer agent (pure deterministic logic, no mocks needed)."""

import pytest
from src.hitachione.agents.reviewer import ReviewerAgent
from src.hitachione.models.schemas import (
    CompanyResearch,
    ReviewFeedback,
    SynthesizedAnswer,
    TaskContext,
    ToolError,
)


def _make_ctx(*entities: str) -> TaskContext:
    ctx = TaskContext(user_query="test")
    ctx.entities = list(entities)
    return ctx


def _make_answer(
    research: list[CompanyResearch],
    markdown: str = "x" * 100,
    confidence: float = 0.8,
) -> SynthesizedAnswer:
    return SynthesizedAnswer(
        markdown=markdown,
        confidence=confidence,
        raw_research=research,
    )


class TestReviewerAgent:
    def setup_method(self):
        self.reviewer = ReviewerAgent()

    # ── happy path ──────────────────────────────────────────────────────

    def test_all_checks_pass(self):
        """Complete data → ReviewFeedback.ok == True."""
        cr = CompanyResearch(
            ticker="AAPL",
            sentiment={"rating": 8, "label": "Positive"},
            performance={"performance_score": 7, "outlook": "Bullish"},
        )
        ctx = _make_ctx("AAPL")
        fb = self.reviewer.run(ctx, _make_answer([cr]))
        assert fb.ok is True
        assert fb.missing == []

    # ── entity coverage ─────────────────────────────────────────────────

    def test_missing_entity_detected(self):
        """If an entity in the context was not researched, flag it."""
        cr = CompanyResearch(ticker="AAPL", sentiment={"rating": 8},
                             performance={"performance_score": 7})
        ctx = _make_ctx("AAPL", "MSFT")
        fb = self.reviewer.run(ctx, _make_answer([cr]))
        assert fb.ok is False
        assert any("MSFT" in m for m in fb.missing)

    # ── score completeness ──────────────────────────────────────────────

    def test_missing_sentiment_rating(self):
        cr = CompanyResearch(ticker="AAPL", sentiment={},
                             performance={"performance_score": 7})
        ctx = _make_ctx("AAPL")
        fb = self.reviewer.run(ctx, _make_answer([cr]))
        assert fb.ok is False
        assert fb.retriable is True
        assert any("sentiment" in m.lower() for m in fb.missing)

    def test_missing_performance_score(self):
        cr = CompanyResearch(ticker="AAPL", sentiment={"rating": 8},
                             performance={})
        ctx = _make_ctx("AAPL")
        fb = self.reviewer.run(ctx, _make_answer([cr]))
        assert fb.ok is False
        assert fb.retriable is True
        assert any("performance" in m.lower() for m in fb.missing)

    # ── answer quality ──────────────────────────────────────────────────

    def test_short_answer_flagged(self):
        cr = CompanyResearch(ticker="AAPL", sentiment={"rating": 8},
                             performance={"performance_score": 7})
        ctx = _make_ctx("AAPL")
        fb = self.reviewer.run(ctx, _make_answer([cr], markdown="short"))
        assert fb.ok is False
        assert any("too short" in m.lower() for m in fb.missing)

    # ── confidence threshold ────────────────────────────────────────────

    def test_low_confidence_flagged(self):
        cr = CompanyResearch(ticker="AAPL", sentiment={"rating": 8},
                             performance={"performance_score": 7})
        ctx = _make_ctx("AAPL")
        fb = self.reviewer.run(ctx, _make_answer([cr], confidence=0.2))
        assert fb.ok is False
        assert any("confidence" in m.lower() for m in fb.missing)

    # ── multiple issues ─────────────────────────────────────────────────

    def test_multiple_issues_accumulated(self):
        cr = CompanyResearch(ticker="AAPL", sentiment={}, performance={})
        ctx = _make_ctx("AAPL", "MSFT")
        fb = self.reviewer.run(ctx, _make_answer([cr], markdown="x", confidence=0.1))
        assert fb.ok is False
        assert len(fb.missing) >= 4  # missing entity, sentiment, perf, short, confidence

    # ── broad-query entity coverage ─────────────────────────────────────

    def test_broad_query_passes_with_80_percent_coverage(self):
        """With >10 entities and ≥80% researched, entity coverage passes."""
        # 15 entities, 12 researched = 80%
        all_tickers = [f"T{i:03d}" for i in range(15)]
        researched = [
            CompanyResearch(
                ticker=t,
                sentiment={"rating": 7},
                performance={"performance_score": 6},
            )
            for t in all_tickers[:12]
        ]
        ctx = _make_ctx(*all_tickers)
        fb = self.reviewer.run(ctx, _make_answer(researched))
        # Should NOT have entity coverage issues
        coverage_issues = [m for m in fb.missing if "coverage" in m.lower() or "not researched" in m.lower()]
        assert coverage_issues == []

    def test_broad_query_fails_below_80_percent(self):
        """With >10 entities and <80% researched, entity coverage fails."""
        # 15 entities, only 5 researched = 33%
        all_tickers = [f"T{i:03d}" for i in range(15)]
        researched = [
            CompanyResearch(
                ticker=t,
                sentiment={"rating": 7},
                performance={"performance_score": 6},
            )
            for t in all_tickers[:5]
        ]
        ctx = _make_ctx(*all_tickers)
        fb = self.reviewer.run(ctx, _make_answer(researched))
        assert fb.ok is False
        assert any("coverage" in m.lower() for m in fb.missing)

    # ── no-KB-data detection (not retriable) ────────────────────────

    def test_no_kb_data_not_retriable(self):
        """When tools return 'No data found in the knowledge base', issues are not retriable."""
        cr = CompanyResearch(
            ticker="XOM",
            sentiment={
                "rating": None,
                "label": "unknown",
                "rationale": "No data found for ticker XOM in the knowledge base.",
            },
            performance={
                "performance_score": None,
                "outlook": "Unknown",
                "justification": "No data found for ticker XOM in the knowledge base.",
            },
        )
        ctx = _make_ctx("XOM")
        fb = self.reviewer.run(ctx, _make_answer([cr], confidence=0.0))
        assert fb.ok is False
        assert fb.retriable is False
        # Missing items should mention "knowledge base", NOT "missing ... rating"
        assert all("knowledge base" in m.lower() or "confidence" in m.lower()
                   for m in fb.missing)

    def test_mixed_kb_data_and_error_is_retriable(self):
        """If some tickers have no KB data and others have transient errors, retriable=True."""
        cr_ok = CompanyResearch(
            ticker="AAPL",
            sentiment={"rating": 8, "label": "Positive"},
            performance={"performance_score": 7, "outlook": "Bullish"},
        )
        cr_no_kb = CompanyResearch(
            ticker="XOM",
            sentiment={
                "rating": None,
                "rationale": "No data found for ticker XOM in the knowledge base.",
            },
            performance={
                "performance_score": None,
                "justification": "No data found for ticker XOM in the knowledge base.",
            },
        )
        cr_error = CompanyResearch(
            ticker="META",
            sentiment={},  # transient error – no rationale
            performance={"performance_score": 7, "outlook": "Bullish"},
        )
        ctx = _make_ctx("AAPL", "XOM", "META")
        fb = self.reviewer.run(ctx, _make_answer([cr_ok, cr_no_kb, cr_error]))
        assert fb.ok is False
        assert fb.retriable is True  # META’s missing sentiment IS retriable
