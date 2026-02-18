"""End-to-end tests for the Orchestrator ReAct loop (all externals mocked)."""

from unittest.mock import MagicMock, patch

import pytest
from src.hitachione.models.schemas import (
    CompanyResearch, Intent, ReviewFeedback, SynthesizedAnswer, TaskContext,
)

# Patch OpenAI before importing Orchestrator so __init__ gets the mock
_mock_openai_cls = patch("src.hitachione.agents.orchestrator.OpenAI", MagicMock())
_mock_openai_cls.start()

from src.hitachione.agents.orchestrator import Orchestrator  # noqa: E402


def _quick_answer(confidence: float = 0.85, markdown: str = "## Report\nAll good."):
    a = SynthesizedAnswer()
    a.markdown = markdown
    a.confidence = confidence
    a.caveats = []
    a.raw_research = []
    return a


def _ok_feedback():
    return ReviewFeedback(ok=True, missing=[], notes="All checks passed")


def _bad_feedback(missing=None):
    return ReviewFeedback(
        ok=False,
        missing=missing or ["AAPL not researched"],
        notes="Entity coverage incomplete",
    )


# ── Happy-path ──────────────────────────────────────────────────────────


class TestOrchestratorHappyPath:

    @patch.object(Orchestrator, "_parse_intent")
    def test_single_entity_happy_path(self, mock_parse):
        """Intent → KB → Research → Synthesize → Review OK → return."""
        # _parse_intent sets entities directly on ctx
        def parse_side_effect(ctx):
            ctx.intent = Intent.SNAPSHOT
            ctx.entities = ["AAPL"]
            ctx.timeframe = ""
            ctx.sector = ""

        mock_parse.side_effect = parse_side_effect

        orch = Orchestrator(max_iterations=2)

        # KB returns nothing new
        orch.kb_agent = MagicMock()
        orch.kb_agent.run.return_value = {
            "aliases": {},
            "entity_hints": [],
            "summaries": [],
        }

        # Researcher returns one CompanyResearch
        cr = CompanyResearch(ticker="AAPL")
        cr.sentiment = {"rating": 8, "label": "Positive", "rationale": "Good"}
        cr.performance = {"score": 9, "outlook": "Strong"}
        orch.researcher = MagicMock()
        orch.researcher.run.return_value = [cr]

        # Synthesizer returns a nice answer
        orch.synthesizer = MagicMock()
        orch.synthesizer.run.return_value = _quick_answer()

        # Reviewer says OK
        orch.reviewer = MagicMock()
        orch.reviewer.run.return_value = _ok_feedback()

        answer = orch.run("Tell me about AAPL")
        assert answer.confidence >= 0.8
        assert "Report" in answer.markdown


# ── No-entities path ────────────────────────────────────────────────────


class TestNoEntitiesPath:

    @patch.object(Orchestrator, "_parse_intent")
    @patch("src.hitachione.agents.orchestrator._find_symbols", return_value=[])
    def test_no_entities_returns_caveat(self, mock_find, mock_parse):
        """If no entities found at all, return user-facing caveat."""
        def parse_side_effect(ctx):
            ctx.intent = Intent.RANK
            ctx.entities = []
            ctx.timeframe = ""
            ctx.sector = ""

        mock_parse.side_effect = parse_side_effect

        orch = Orchestrator(max_iterations=1)
        orch.kb_agent = MagicMock()
        orch.kb_agent.run.return_value = {
            "aliases": {}, "entity_hints": [], "summaries": [],
        }

        answer = orch.run("random question with no tickers")
        assert answer.confidence == 0.0
        assert "tickers" in answer.markdown.lower() or "identify" in answer.markdown.lower()


# ── Reviewer rejection + reflection loop ────────────────────────────────


class TestReflectionLoop:

    @patch.object(Orchestrator, "_parse_intent")
    def test_reviewer_rejects_then_accepts(self, mock_parse):
        """Reviewer fails on iter 1, Orchestrator retries, reviewer passes on iter 2."""
        def parse_side_effect(ctx):
            ctx.intent = Intent.COMPARE
            ctx.entities = ["AAPL", "MSFT"]
            ctx.timeframe = ""
            ctx.sector = ""

        mock_parse.side_effect = parse_side_effect

        orch = Orchestrator(max_iterations=3)

        orch.kb_agent = MagicMock()
        orch.kb_agent.run.return_value = {
            "aliases": {}, "entity_hints": [], "summaries": [],
        }

        cr_a = CompanyResearch(ticker="AAPL")
        cr_a.sentiment = {"rating": 6, "label": "Positive"}
        cr_a.performance = {"score": 7}
        cr_m = CompanyResearch(ticker="MSFT")
        cr_m.sentiment = {"rating": 7, "label": "Positive"}
        cr_m.performance = {"score": 8}
        orch.researcher = MagicMock()
        orch.researcher.run.return_value = [cr_a, cr_m]

        # Synthesizer: weak on iter 1, strong on iter 2
        weak = _quick_answer(confidence=0.4, markdown="Meh")
        strong = _quick_answer(confidence=0.9, markdown="## Comprehensive\nGreat comparison.")
        orch.synthesizer = MagicMock()
        orch.synthesizer.run.side_effect = [weak, strong]

        # Reviewer: reject first, accept second
        orch.reviewer = MagicMock()
        orch.reviewer.run.side_effect = [
            _bad_feedback(["confidence below threshold"]),
            _ok_feedback(),
        ]

        answer = orch.run("Compare AAPL vs MSFT")
        assert answer.confidence == 0.9
        assert orch.reviewer.run.call_count == 2

    @patch.object(Orchestrator, "_parse_intent")
    def test_max_iterations_reached(self, mock_parse):
        """If reviewer never accepts, the answer is returned after max iterations with a caveat."""
        def parse_side_effect(ctx):
            ctx.intent = Intent.SNAPSHOT
            ctx.entities = ["TSLA"]
            ctx.timeframe = ""
            ctx.sector = ""

        mock_parse.side_effect = parse_side_effect

        orch = Orchestrator(max_iterations=2)
        orch.kb_agent = MagicMock()
        orch.kb_agent.run.return_value = {
            "aliases": {}, "entity_hints": [], "summaries": [],
        }

        cr = CompanyResearch(ticker="TSLA")
        cr.sentiment = {"rating": 3, "label": "Negative"}
        cr.performance = {"score": 4}
        orch.researcher = MagicMock()
        orch.researcher.run.return_value = [cr]

        orch.synthesizer = MagicMock()
        orch.synthesizer.run.return_value = _quick_answer(confidence=0.3, markdown="Meh")

        orch.reviewer = MagicMock()
        orch.reviewer.run.return_value = _bad_feedback()

        answer = orch.run("TSLA outlook")
        assert any("incomplete" in c.lower() or "iteration" in c.lower() for c in answer.caveats)
        assert orch.reviewer.run.call_count == 2


# ── Company retrieval gating ────────────────────────────────────────────


class TestCompanyRetrievalGating:

    @patch.object(Orchestrator, "_parse_intent")
    @patch("src.hitachione.agents.orchestrator._find_symbols")
    def test_broad_query_calls_company_filter(self, mock_find, mock_parse):
        """When intent has no explicit entities, company_filter is called."""
        def parse_side_effect(ctx):
            ctx.intent = Intent.RANK
            ctx.entities = []  # no explicit entities
            ctx.timeframe = ""
            ctx.sector = "tech"

        mock_parse.side_effect = parse_side_effect
        mock_find.return_value = ["AAPL", "MSFT", "GOOGL"]

        orch = Orchestrator(max_iterations=1)
        orch.kb_agent = MagicMock()
        orch.kb_agent.run.return_value = {
            "aliases": {}, "entity_hints": [], "summaries": [],
        }

        cr_a = CompanyResearch(ticker="AAPL")
        cr_a.sentiment = {"rating": 8, "label": "Positive"}
        cr_a.performance = {"score": 9}
        cr_m = CompanyResearch(ticker="MSFT")
        cr_m.sentiment = {"rating": 7, "label": "Positive"}
        cr_m.performance = {"score": 8}
        cr_g = CompanyResearch(ticker="GOOGL")
        cr_g.sentiment = {"rating": 7, "label": "Positive"}
        cr_g.performance = {"score": 7}
        orch.researcher = MagicMock()
        orch.researcher.run.return_value = [cr_a, cr_m, cr_g]
        orch.synthesizer = MagicMock()
        orch.synthesizer.run.return_value = _quick_answer()
        orch.reviewer = MagicMock()
        orch.reviewer.run.return_value = _ok_feedback()

        answer = orch.run("Top tech stocks")
        mock_find.assert_called_once()

    @patch.object(Orchestrator, "_parse_intent")
    @patch("src.hitachione.agents.orchestrator._find_symbols")
    def test_explicit_entities_skip_company_filter(self, mock_find, mock_parse):
        """When intent has explicit entities, company_filter is NOT called."""
        def parse_side_effect(ctx):
            ctx.intent = Intent.SNAPSHOT
            ctx.entities = ["TSLA"]
            ctx.timeframe = ""
            ctx.sector = ""

        mock_parse.side_effect = parse_side_effect

        orch = Orchestrator(max_iterations=1)
        orch.kb_agent = MagicMock()
        orch.kb_agent.run.return_value = {
            "aliases": {}, "entity_hints": [], "summaries": [],
        }

        cr = CompanyResearch(ticker="TSLA")
        cr.sentiment = {"rating": 5, "label": "Neutral"}
        cr.performance = {"score": 6}
        orch.researcher = MagicMock()
        orch.researcher.run.return_value = [cr]
        orch.synthesizer = MagicMock()
        orch.synthesizer.run.return_value = _quick_answer()
        orch.reviewer = MagicMock()
        orch.reviewer.run.return_value = _ok_feedback()

        answer = orch.run("TSLA outlook")
        mock_find.assert_not_called()


# ── Plan generation ─────────────────────────────────────────────────────


class TestPlanGeneration:

    def test_rank_intent_plan(self):
        orch = Orchestrator()
        ctx = TaskContext(user_query="Top 5 tech stocks")
        ctx.intent = Intent.RANK
        ctx.entities = ["AAPL", "MSFT"]
        plan = orch._plan(ctx)
        assert any("rank" in step.lower() or "score" in step.lower() for step in plan)

    def test_empty_entities_plan(self):
        orch = Orchestrator()
        ctx = TaskContext(user_query="Energy stocks")
        ctx.intent = Intent.RANK
        ctx.entities = []
        plan = orch._plan(ctx)
        assert any("discover" in step.lower() for step in plan)

    def test_snapshot_intent_plan(self):
        orch = Orchestrator()
        ctx = TaskContext(user_query="TSLA snapshot")
        ctx.intent = Intent.SNAPSHOT
        ctx.entities = ["TSLA"]
        plan = orch._plan(ctx)
        assert any("latest" in step.lower() or "fetch" in step.lower() for step in plan)


# ── Reflect ─────────────────────────────────────────────────────────────


class TestReflect:

    def test_reflect_missing_entities(self):
        orch = Orchestrator()
        ctx = TaskContext(user_query="test")
        ctx.entities = ["AAPL", "MSFT"]
        fb = ReviewFeedback(
            ok=False,
            missing=["Entity AAPL not researched"],
            notes="incomplete",
        )
        adj = orch._reflect(ctx, fb)
        assert adj["action"] == "retry_failed_entities"
        assert "AAPL" in adj["entities"]
        # ctx.entities narrowed to just the failed ticker
        assert ctx.entities == ["AAPL"]

    def test_reflect_missing_performance_narrows_entities(self):
        """If reviewer says 'META: missing performance score', retry just META."""
        orch = Orchestrator()
        ctx = TaskContext(user_query="test")
        ctx.entities = ["AAPL", "META", "MSFT"]
        fb = ReviewFeedback(
            ok=False,
            missing=["META: missing performance score"],
            notes="incomplete",
        )
        adj = orch._reflect(ctx, fb)
        assert adj["action"] == "retry_failed_entities"
        assert ctx.entities == ["META"]

    def test_reflect_adds_missing_entity_to_ctx(self):
        """_reflect should add the missing ticker back into ctx.entities."""
        orch = Orchestrator()
        ctx = TaskContext(user_query="test")
        ctx.entities = ["MSFT"]
        fb = ReviewFeedback(
            ok=False,
            missing=["Entity AAPL not researched"],
            notes="incomplete",
        )
        orch._reflect(ctx, fb)
        assert "AAPL" in ctx.entities

    def test_reflect_low_confidence(self):
        orch = Orchestrator()
        ctx = TaskContext(user_query="test")
        fb = ReviewFeedback(
            ok=False,
            missing=["confidence too low"],
            notes="bump up",
        )
        adj = orch._reflect(ctx, fb)
        assert adj["action"] == "broaden_search"

    def test_reflect_no_actionable_items(self):
        orch = Orchestrator()
        ctx = TaskContext(user_query="test")
        fb = ReviewFeedback(ok=False, missing=["answer is short"], notes="")
        adj = orch._reflect(ctx, fb)
        assert adj["action"] == "none"
