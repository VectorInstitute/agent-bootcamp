"""Reviewer Agent – deterministic quality gate for synthesized answers.

Checks:
  1. Entity coverage: were requested entities actually researched?
     For broad queries (>10 entities), requires ≥80% coverage instead of 100%.
  2. Score completeness: does each researched entity have sentiment + performance?
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

# When entity count exceeds this, switch from 100% to 80% coverage check
_BROAD_QUERY_THRESHOLD = 10
_BROAD_COVERAGE_RATIO = 0.8


# Sentinel phrases emitted by the tools when data is absent
_NO_DATA_PHRASES = ("no data found",)


def _is_no_kb_data(rationale: str | None) -> bool:
    """Return True when a tool rationale means the KB simply has no records."""
    if not rationale or not isinstance(rationale, str):
        return False
    lower = rationale.lower()
    return any(phrase in lower for phrase in _NO_DATA_PHRASES)


class ReviewerAgent:
    """Deterministic checks – no LLM calls, fast and predictable."""

    def run(self, ctx: TaskContext, answer: SynthesizedAnswer) -> ReviewFeedback:
        fb = ReviewFeedback()
        missing: list[str] = []
        retriable_issues = 0  # count issues that a retry could potentially fix

        # 1. Entity coverage
        researched = {cr.ticker for cr in answer.raw_research}
        total_entities = len(ctx.entities)
        not_researched = [e for e in ctx.entities if e.upper() not in researched]

        if total_entities > _BROAD_QUERY_THRESHOLD:
            # Broad query: require ≥80% coverage
            coverage_ratio = 1.0 - (len(not_researched) / total_entities) if total_entities else 1.0
            if coverage_ratio < _BROAD_COVERAGE_RATIO:
                missing.append(
                    f"Entity coverage {coverage_ratio:.0%} below "
                    f"{_BROAD_COVERAGE_RATIO:.0%} threshold "
                    f"({len(not_researched)}/{total_entities} not researched)"
                )
                retriable_issues += 1
        else:
            # Narrow query: require 100% coverage
            for entity in not_researched:
                missing.append(f"Entity {entity} not researched")
                retriable_issues += 1

        # 2. Score completeness
        #    Distinguish "no data in KB" (not retriable) from transient errors
        #    (retriable).  The tools set a rationale / justification that
        #    contains 'No data found' when the KB has no records.
        for cr in answer.raw_research:
            sent_rationale = cr.sentiment.get("rationale") if cr.sentiment else None
            perf_justification = cr.performance.get("justification") if cr.performance else None

            if not cr.sentiment or cr.sentiment.get("rating") is None:
                if _is_no_kb_data(sent_rationale):
                    missing.append(f"{cr.ticker}: no sentiment data in knowledge base")
                else:
                    missing.append(f"{cr.ticker}: missing sentiment rating")
                    retriable_issues += 1

            if not cr.performance or cr.performance.get("performance_score") is None:
                if _is_no_kb_data(perf_justification):
                    missing.append(f"{cr.ticker}: no performance data in knowledge base")
                else:
                    missing.append(f"{cr.ticker}: missing performance score")
                    retriable_issues += 1

        # 3. Answer quality
        if len(answer.markdown) < 50:
            missing.append("Answer text too short")
            retriable_issues += 1

        # 4. Confidence
        if answer.confidence < QUALITY_THRESHOLD:
            missing.append(
                f"Confidence {answer.confidence:.2f} below threshold "
                f"{QUALITY_THRESHOLD:.2f}"
            )
            # Low confidence by itself is not retriable – it's a consequence
            # of missing data.  Only if other retriable issues exist will
            # a retry improve confidence.

        fb.missing = missing
        fb.ok = len(missing) == 0
        fb.retriable = retriable_issues > 0
        fb.notes = (
            "All checks passed." if fb.ok
            else f"{len(missing)} issue(s) found: {'; '.join(missing[:5])}"
        )

        logger.info(
            "Reviewer: ok=%s, retriable=%s, issues=%d",
            fb.ok, fb.retriable, len(missing),
        )
        return fb
