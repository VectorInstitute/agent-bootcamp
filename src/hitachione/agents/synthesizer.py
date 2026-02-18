"""Synthesizer Agent – composes a ranked / comparative answer from research.

Takes a list of ``CompanyResearch`` objects and produces a user-facing
``SynthesizedAnswer`` with:
  • Markdown answer (ranked list or comparison table)
  • Rationale explaining the scoring / ranking
  • Caveats for partial data
  • Citations from news snippets
  • Confidence estimate
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from ..config.settings import OPENAI_BASE_URL, OPENAI_API_KEY, WORKER_MODEL
from ..models.schemas import (
    CompanyResearch, Intent, SynthesizedAnswer, TaskContext,
)

logger = logging.getLogger(__name__)

# ── Prompt templates ─────────────────────────────────────────────────────

_SYSTEM = (
    "You are a financial intelligence synthesizer. "
    "Given structured research data for companies, compose a clear, "
    "well-structured Markdown answer for the user. "
    "Include a rationale section explaining your reasoning. "
    "If data is partial, add explicit caveats. "
    "If news references are available, include them as citations. "
    "Do NOT give investment advice – present analytics and insights only. "
    "Return raw Markdown text (not wrapped in JSON)."
)


def _build_research_block(research: list[CompanyResearch]) -> str:
    """Serialize research data into a text block for the LLM."""
    parts: list[str] = []
    for cr in research:
        lines = [f"## {cr.ticker}"]
        if cr.sentiment:
            lines.append(
                f"- Sentiment: rating={cr.sentiment.get('rating')}, "
                f"label={cr.sentiment.get('label')}, "
                f"rationale={cr.sentiment.get('rationale','')[:200]}"
            )
        if cr.performance:
            lines.append(
                f"- Performance: score={cr.performance.get('performance_score')}, "
                f"outlook={cr.performance.get('outlook')}, "
                f"justification={cr.performance.get('justification','')[:200]}"
            )
        if cr.news_snippets:
            lines.append("- News references: " + "; ".join(cr.news_snippets[:3]))
        if cr.errors:
            lines.append(
                "- ⚠ Data gaps: "
                + ", ".join(f"{e.tool} ({e.error[:60]})" for e in cr.errors)
            )
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _estimate_confidence(research: list[CompanyResearch]) -> float:
    """Heuristic confidence score 0-1 based on data completeness."""
    if not research:
        return 0.0
    scores: list[float] = []
    for cr in research:
        s = 0.0
        if cr.sentiment and cr.sentiment.get("rating") is not None:
            s += 0.5
        if cr.performance and cr.performance.get("performance_score") is not None:
            s += 0.5
        scores.append(s)
    return sum(scores) / len(scores)


class SynthesizerAgent:
    """Compose a user-facing answer from per-entity research."""

    def run(
        self,
        ctx: TaskContext,
        research: list[CompanyResearch],
    ) -> SynthesizedAnswer:
        answer = SynthesizedAnswer(raw_research=research)
        answer.confidence = _estimate_confidence(research)

        # Gather caveats
        for cr in research:
            for e in cr.errors:
                answer.caveats.append(f"{e.entity}: {e.tool} unavailable ({e.error[:80]})")
        if answer.confidence < 0.5:
            answer.caveats.append("Low overall data coverage – results are best-effort.")

        # Gather citations
        for cr in research:
            answer.citations.extend(cr.news_snippets[:3])

        # Build the research context for the LLM
        data_block = _build_research_block(research)
        user_msg = (
            f"User query: {ctx.user_query}\n"
            f"Intent: {ctx.intent.value}\n"
            f"Timeframe: {ctx.timeframe or 'not specified'}\n\n"
            f"Research data:\n{data_block}"
        )

        try:
            client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=WORKER_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
            )
            md = (resp.choices[0].message.content or "").strip()
            answer.markdown = md
            # Extract first paragraph as rationale summary
            answer.rationale = md.split("\n\n")[0] if md else ""
        except Exception as exc:
            logger.error("Synthesizer LLM error: %s", exc)
            # Fallback: build a simple text answer from raw data
            answer.markdown = self._fallback_markdown(ctx, research)
            answer.rationale = "Generated from raw data (LLM unavailable)."

        return answer

    @staticmethod
    def _fallback_markdown(
        ctx: TaskContext, research: list[CompanyResearch]
    ) -> str:
        lines = [f"## Results for: {ctx.user_query}\n"]
        for cr in research:
            sent = cr.sentiment.get("rating", "?") if cr.sentiment else "?"
            perf = cr.performance.get("performance_score", "?") if cr.performance else "?"
            lines.append(f"- **{cr.ticker}**: sentiment={sent}/10, performance={perf}/10")
        return "\n".join(lines)
