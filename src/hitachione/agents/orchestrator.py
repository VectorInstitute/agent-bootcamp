"""Orchestrator – the agentic ReAct loop (Plan → Act → Observe → Reflect).

This is the top-level entry point.  Given a free-text user prompt it:

1. **Parses intent** (rank, compare, snapshot, event_reaction, …)
2. **Plans** subgoals and identifies information gaps
3. **Acts** by calling sub-agents (KB retrieval, company retrieval,
   researcher, synthesizer, reviewer)
4. **Observes** outputs and assesses sufficiency
5. **Reflects** — if the Reviewer flags issues and we have iterations left,
   revises the plan and loops
6. **Stops** when the Reviewer says OK, the budget is exhausted, or
   information gain is negligible
7. **Returns** a clean ``SynthesizedAnswer`` with rationale + caveats

Usage::

    from hitachione.agents.orchestrator import Orchestrator
    answer = Orchestrator().run("Top 3 tech stocks of 2024")
    print(answer.markdown)
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

# Ensure tools directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ..config.settings import (
    MAX_ITERATIONS, OPENAI_API_KEY, OPENAI_BASE_URL, PLANNER_MODEL,
)
from ..models.schemas import (
    Intent, SynthesizedAnswer, TaskContext,
)
from ..services.tracing import Tracer
from .knowledge_retrieval import KnowledgeRetrievalAgent
from .researcher import ResearcherAgent
from .reviewer import ReviewerAgent
from .synthesizer import SynthesizerAgent

logger = logging.getLogger(__name__)

# ── Intent parsing prompt ────────────────────────────────────────────────

_INTENT_PROMPT = """\
Classify the user's financial query into exactly one intent and extract
structured fields.  Return ONLY valid JSON (no markdown fences):

{
  "intent": "<rank|compare|snapshot|event_reaction|fundamentals|macro|mixed>",
  "entities": ["<TICKER1>", ...],
  "timeframe": "<e.g. last 3 years, 2024 Q3, or empty string>",
  "sector": "<e.g. automotive, tech, or empty string>"
}

Rules:
- "entities" must be uppercase ticker symbols when identifiable.
- If the user mentions company names instead of tickers, map them to tickers
  (e.g. "Tesla" → "TSLA", "Apple" → "AAPL", "Google" → "GOOGL").
- If unsure about tickers, leave entities empty.
- Keep it concise.
"""

# ── Company retrieval helper ─────────────────────────────────────────────

def _find_symbols(query: str) -> list[str]:
    """Call the company filtering tool to discover relevant tickers."""
    try:
        from ..tools.company_filtering_tool.tool import find_relevant_symbols
        return find_relevant_symbols(query, use_llm_filter=True)
    except Exception as exc:
        logger.warning("Company filtering tool error: %s", exc)
        return []


# ── Orchestrator ─────────────────────────────────────────────────────────

class Orchestrator:
    """Agentic ReAct orchestrator for financial intelligence queries."""

    def __init__(self, max_iterations: int = MAX_ITERATIONS):
        self.max_iter = max_iterations
        self.kb_agent = KnowledgeRetrievalAgent()
        self.researcher = ResearcherAgent()
        self.synthesizer = SynthesizerAgent()
        self.reviewer = ReviewerAgent()

    # ── public API ──────────────────────────────────────────────────────

    def run(
        self,
        user_query: str,
        *,
        default_timeframe: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> SynthesizedAnswer:
        """Execute the full plan-act-observe-reflect loop."""

        tracer = Tracer.start(
            "orchestrator_run",
            metadata={"query": user_query, **(metadata or {})},
        )

        ctx = TaskContext(user_query=user_query, timeframe=default_timeframe)

        # ── STEP 1: Parse intent ────────────────────────────────────────
        with tracer.span("intent_parse") as sp:
            self._parse_intent(ctx)
            sp.update(output={
                "intent": ctx.intent.value,
                "entities": ctx.entities,
                "timeframe": ctx.timeframe,
                "sector": ctx.sector,
            })
            ctx.observations.append(
                f"Intent={ctx.intent.value}, entities={ctx.entities}, "
                f"timeframe={ctx.timeframe}, sector={ctx.sector}"
            )

        answer = SynthesizedAnswer()

        for iteration in range(1, self.max_iter + 1):
            ctx.iteration = iteration
            logger.info("── Iteration %d/%d ──", iteration, self.max_iter)

            # ── STEP 2: Plan ────────────────────────────────────────────
            with tracer.span("planning", metadata={"iteration": iteration}) as sp:
                plan = self._plan(ctx)
                ctx.plan = plan
                sp.update(output=plan)
                ctx.observations.append(f"Plan (iter {iteration}): {plan}")

            # ── STEP 3: Act – KB retrieval ──────────────────────────────
            with tracer.span("knowledge_retrieval") as sp:
                kb_data = self.kb_agent.run(ctx)
                sp.update(output=kb_data)
                # Merge entity hints into context
                for hint in kb_data.get("entity_hints", []):
                    if hint not in ctx.entities:
                        ctx.entities.append(hint)

            # ── STEP 3b: Act – Company retrieval (if entities empty) ───
            if not ctx.entities:
                with tracer.span("company_retrieval") as sp:
                    ctx.observations.append("No entities yet – calling company filter")
                    symbols = _find_symbols(ctx.user_query)
                    ctx.entities = symbols
                    sp.update(output=symbols)

            if not ctx.entities:
                ctx.uncertainties.append("Could not identify any tickers")
                answer.caveats.append(
                    "No tickers could be identified for this query."
                )
                answer.markdown = (
                    f"I wasn't able to identify specific tickers for: "
                    f"*{user_query}*. Please try including ticker symbols "
                    f"(e.g. AAPL, TSLA)."
                )
                answer.confidence = 0.0
                tracer.end(output={"markdown": answer.markdown})
                return answer

            # ── STEP 4: Act – Research ──────────────────────────────────
            with tracer.span("research_fanout") as sp:
                research = self.researcher.run(ctx, ctx.entities)
                sp.update(output={
                    "count": len(research),
                    "tickers": [r.ticker for r in research],
                })

            # ── STEP 5: Act – Synthesize ────────────────────────────────
            with tracer.span("synthesizer") as sp:
                answer = self.synthesizer.run(ctx, research)
                sp.update(output={
                    "confidence": answer.confidence,
                    "caveats": answer.caveats,
                    "md_length": len(answer.markdown),
                })

            # ── STEP 6: Observe – Review ────────────────────────────────
            with tracer.span("reviewer") as sp:
                feedback = self.reviewer.run(ctx, answer)
                sp.update(output={
                    "ok": feedback.ok,
                    "missing": feedback.missing,
                    "notes": feedback.notes,
                })
                ctx.observations.append(
                    f"Reviewer (iter {iteration}): ok={feedback.ok}, "
                    f"notes={feedback.notes}"
                )

            # ── STEP 7: Reflect & decide to stop ───────────────────────
            if feedback.ok:
                logger.info("Reviewer OK – stopping")
                break

            if iteration < self.max_iter:
                # Reflect: try to address missing items
                with tracer.span("reflection") as sp:
                    adjustments = self._reflect(ctx, feedback)
                    sp.update(output=adjustments)
                    ctx.observations.append(f"Reflection: {adjustments}")
            else:
                logger.info("Max iterations reached – returning best effort")
                answer.caveats.append(
                    "Maximum analysis iterations reached; some data may be incomplete."
                )

        tracer.end(output={"confidence": answer.confidence, "caveats": answer.caveats})
        return answer

    # ── Private helpers ─────────────────────────────────────────────────

    def _parse_intent(self, ctx: TaskContext) -> None:
        """Use LLM to classify intent and extract entities / timeframe."""
        try:
            client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=PLANNER_MODEL,
                messages=[
                    {"role": "system", "content": _INTENT_PROMPT},
                    {"role": "user", "content": ctx.user_query},
                ],
                temperature=0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            # Strip markdown fences if present
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            data = json.loads(raw)
        except Exception as exc:
            logger.warning("Intent parse error: %s", exc)
            data = {}

        intent_str = data.get("intent", "mixed")
        try:
            ctx.intent = Intent(intent_str)
        except ValueError:
            ctx.intent = Intent.MIXED

        ctx.entities = [e.upper() for e in data.get("entities", [])]
        ctx.timeframe = data.get("timeframe", ctx.timeframe) or ""
        ctx.sector = data.get("sector", "") or ""

    def _plan(self, ctx: TaskContext) -> list[str]:
        """Generate a simple step plan from the current context."""
        steps = []
        if not ctx.entities:
            steps.append("Discover relevant entities via KB + company filter")
        else:
            steps.append(f"Research entities: {', '.join(ctx.entities)}")

        if ctx.intent in (Intent.RANK, Intent.COMPARE):
            steps.append("Fetch sentiment + performance for each entity")
            steps.append("Score & rank entities")
        elif ctx.intent == Intent.SNAPSHOT:
            steps.append("Fetch latest data for the entity")
        elif ctx.intent == Intent.EVENT_REACTION:
            steps.append("Fetch recent news + price reaction")
        else:
            steps.append("Fetch sentiment + performance data")

        steps.append("Synthesize answer with rationale + caveats")
        steps.append("Review for quality and completeness")

        return steps

    def _reflect(self, ctx: TaskContext, feedback) -> dict[str, Any]:
        """Adjust the context based on reviewer feedback."""
        adjustments: dict[str, Any] = {"action": "none"}

        missing = feedback.missing
        # If entities are missing research, they may need to be re-fetched
        missing_entities = [
            m for m in missing
            if "not researched" in m.lower()
        ]
        if missing_entities:
            adjustments["action"] = "retry_missing_entities"
            adjustments["details"] = missing_entities

        # If confidence is low, try broader KB search
        if any("confidence" in m.lower() for m in missing):
            adjustments["action"] = "broaden_search"
            ctx.observations.append("Broadening search due to low confidence")

        return adjustments
