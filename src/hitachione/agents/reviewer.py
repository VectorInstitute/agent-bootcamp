"""Reviewer Agent – deterministic quality gate for synthesized answers.

Checks:
  1. Entity coverage: were all requested entities actually researched?
  2. Score completeness: does each entity have sentiment + performance?
  3. Answer quality: is the markdown non-empty and reasonably long?
  4. Confidence threshold: is overall confidence above the minimum?

Returns ``ReviewFeedback(ok=True/False, missing=[...], notes=...)``.
"""

from __future__ import annotations

import logging

from ..config.settings import QUALITY_THRESHOLD
from ..models.schemas import (
    ReviewFeedback, SynthesizedAnswer, TaskContext,
)

logger = logging.getLogger(__name__)


class ReviewerAgent:
    """Deterministic checks – no LLM calls, fast and predictable."""

    def run(self, ctx: TaskContext, answer: SynthesizedAnswer) -> ReviewFeedback:
        fb = ReviewFeedback()
        missing: list[str] = []

        # 1. Entity coverage
        researched = {cr.ticker for cr in answer.raw_research}
        for entity in ctx.entities:
            if entity.upper() not in researched:
                missing.append(f"Entity {entity} not researched")

        # 2. Score completeness
        for cr in answer.raw_research:
            if not cr.sentiment or cr.sentiment.get("rating") is None:
                missing.append(f"{cr.ticker}: missing sentiment rating")
            if not cr.performance or cr.performance.get("performance_score") is None:
                missing.append(f"{cr.ticker}: missing performance score")

        # 3. Answer quality
        if len(answer.markdown) < 50:
            missing.append("Answer text too short")

        # 4. Confidence
        if answer.confidence < QUALITY_THRESHOLD:
            missing.append(
                f"Confidence {answer.confidence:.2f} below threshold "
                f"{QUALITY_THRESHOLD:.2f}"
            )

        fb.missing = missing
        fb.ok = len(missing) == 0
        fb.notes = (
            "All checks passed." if fb.ok
            else f"{len(missing)} issue(s) found: {'; '.join(missing[:5])}"
        )

        logger.info("Reviewer: ok=%s, issues=%d", fb.ok, len(missing))
        return fb
