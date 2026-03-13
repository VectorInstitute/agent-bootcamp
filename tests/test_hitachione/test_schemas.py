"""Tests for shared data models (schemas.py)."""

import pytest
from src.hitachione.models.schemas import (
    CompanyResearch,
    Intent,
    ReviewFeedback,
    SynthesizedAnswer,
    TaskContext,
    ToolError,
)


# ── Intent enum ──────────────────────────────────────────────────────────

class TestIntent:
    def test_all_values_exist(self):
        """Every documented intent string should map to an enum member."""
        for val in ("rank", "compare", "snapshot", "event_reaction",
                     "fundamentals", "macro", "mixed"):
            assert Intent(val).value == val

    def test_invalid_intent_raises(self):
        with pytest.raises(ValueError):
            Intent("nonexistent")


# ── TaskContext ──────────────────────────────────────────────────────────

class TestTaskContext:
    def test_defaults(self):
        ctx = TaskContext(user_query="test")
        assert ctx.user_query == "test"
        assert ctx.intent == Intent.MIXED
        assert ctx.entities == []
        assert ctx.observations == []
        assert ctx.iteration == 0
        assert len(ctx.run_id) == 12

    def test_run_id_unique(self):
        a = TaskContext(user_query="a")
        b = TaskContext(user_query="b")
        assert a.run_id != b.run_id

    def test_mutable_collections_independent(self):
        """Default lists should not be shared across instances."""
        a = TaskContext(user_query="a")
        b = TaskContext(user_query="b")
        a.entities.append("AAPL")
        assert "AAPL" not in b.entities


# ── CompanyResearch ──────────────────────────────────────────────────────

class TestCompanyResearch:
    def test_defaults(self):
        cr = CompanyResearch(ticker="TSLA")
        assert cr.ticker == "TSLA"
        assert cr.sentiment == {}
        assert cr.performance == {}
        assert cr.errors == []

    def test_errors_accumulate(self):
        cr = CompanyResearch(ticker="AAPL")
        cr.errors.append(ToolError(entity="AAPL", tool="sentiment", error="timeout"))
        cr.errors.append(ToolError(entity="AAPL", tool="performance", error="404"))
        assert len(cr.errors) == 2


# ── SynthesizedAnswer ───────────────────────────────────────────────────

class TestSynthesizedAnswer:
    def test_defaults(self):
        ans = SynthesizedAnswer()
        assert ans.markdown == ""
        assert ans.confidence == 0.0
        assert ans.caveats == []
        assert ans.citations == []
        assert ans.raw_research == []


# ── ReviewFeedback ──────────────────────────────────────────────────────

class TestReviewFeedback:
    def test_defaults_not_ok(self):
        fb = ReviewFeedback()
        assert fb.ok is False
        assert fb.missing == []
