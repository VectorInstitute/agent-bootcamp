"""Tests for the Knowledge Retrieval agent with mocked Weaviate."""

from unittest.mock import patch, MagicMock

import pytest
from src.hitachione.agents.knowledge_retrieval import KnowledgeRetrievalAgent
from src.hitachione.models.schemas import TaskContext


def _make_weaviate_obj(ticker, company, title="News", source="bloomberg", date="2024-01-01"):
    """Create a mock Weaviate object with properties."""
    obj = MagicMock()
    obj.properties = {
        "ticker": ticker,
        "company": company,
        "title": title,
        "dataset_source": source,
        "date": date,
        "text": f"Article about {company}",
    }
    return obj


class TestKnowledgeRetrievalAgent:

    @patch("src.hitachione.agents.knowledge_retrieval._weaviate_client")
    def test_happy_path(self, mock_client_fn):
        """BM25 returns objects → aliases and hints are extracted."""
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client
        mock_col = MagicMock()
        mock_client.collections.get.return_value = mock_col

        mock_col.query.bm25.return_value.objects = [
            _make_weaviate_obj("AAPL", "Apple Inc."),
            _make_weaviate_obj("AAPL", "Apple Inc.", title="Second article"),
            _make_weaviate_obj("MSFT", "Microsoft Corporation"),
        ]

        agent = KnowledgeRetrievalAgent()
        ctx = TaskContext(user_query="tech stocks")
        result = agent.run(ctx)

        assert "Apple Inc." in result["aliases"]
        assert result["aliases"]["Apple Inc."] == "AAPL"
        assert "AAPL" in result["entity_hints"]
        assert "MSFT" in result["entity_hints"]
        assert len(result["summaries"]) == 3
        mock_client.close.assert_called_once()

    @patch("src.hitachione.agents.knowledge_retrieval._weaviate_client")
    def test_no_results(self, mock_client_fn):
        """Empty BM25 result → empty but valid return structure."""
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client
        mock_col = MagicMock()
        mock_client.collections.get.return_value = mock_col
        mock_col.query.bm25.return_value.objects = []

        agent = KnowledgeRetrievalAgent()
        ctx = TaskContext(user_query="unknown sector")
        result = agent.run(ctx)

        assert result["entity_hints"] == []
        assert result["aliases"] == {}
        assert result["summaries"] == []

    @patch("src.hitachione.agents.knowledge_retrieval._weaviate_client")
    def test_weaviate_failure_graceful(self, mock_client_fn):
        """If Weaviate throws, the agent returns partial results + logs uncertainty."""
        mock_client_fn.side_effect = RuntimeError("Connection refused")

        agent = KnowledgeRetrievalAgent()
        ctx = TaskContext(user_query="test")
        result = agent.run(ctx)

        assert result["entity_hints"] == []
        assert len(ctx.uncertainties) >= 1

    @patch("src.hitachione.agents.knowledge_retrieval._weaviate_client")
    def test_existing_entities_resolved(self, mock_client_fn):
        """Entities already in ctx get added to aliases."""
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client
        mock_col = MagicMock()
        mock_client.collections.get.return_value = mock_col
        mock_col.query.bm25.return_value.objects = []

        agent = KnowledgeRetrievalAgent()
        ctx = TaskContext(user_query="compare TSLA")
        ctx.entities = ["TSLA"]
        result = agent.run(ctx)

        assert result["aliases"]["TSLA"] == "TSLA"

    @patch("src.hitachione.agents.knowledge_retrieval._weaviate_client")
    def test_dedup_entity_hints(self, mock_client_fn):
        """Same ticker from multiple BM25 hits should appear only once."""
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client
        mock_col = MagicMock()
        mock_client.collections.get.return_value = mock_col
        mock_col.query.bm25.return_value.objects = [
            _make_weaviate_obj("AAPL", "Apple Inc.", title="Article 1"),
            _make_weaviate_obj("AAPL", "Apple Inc.", title="Article 2"),
        ]

        agent = KnowledgeRetrievalAgent()
        result = agent.run(TaskContext(user_query="apple"))
        assert result["entity_hints"].count("AAPL") == 1
