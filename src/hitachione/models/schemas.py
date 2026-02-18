"""Shared data models for the multi-agent financial intelligence system.

All models are plain dataclasses – no heavy framework dependency.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ── Intent taxonomy ──────────────────────────────────────────────────────

class Intent(str, Enum):
    RANK = "rank"
    COMPARE = "compare"
    SNAPSHOT = "snapshot"
    EVENT_REACTION = "event_reaction"
    FUNDAMENTALS = "fundamentals"
    MACRO = "macro"
    MIXED = "mixed"


# ── Per-run context (scratchpad / blackboard) ────────────────────────────

@dataclass
class TaskContext:
    """Short-lived context for a single orchestrator run."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    user_query: str = ""
    intent: Intent = Intent.MIXED
    timeframe: str = ""            # e.g. "last 3 years", "2024 Q3"
    sector: str = ""               # e.g. "automotive"
    entities: list[str] = field(default_factory=list)   # ticker symbols
    constraints: dict[str, Any] = field(default_factory=dict)

    # Blackboard – accumulates across iterations
    plan: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    iteration: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


# ── Tool / agent outputs ────────────────────────────────────────────────

@dataclass
class ToolError:
    """Captures a non-fatal error from a tool call."""

    entity: str
    tool: str
    error: str


@dataclass
class CompanyResearch:
    """Research bundle for one company/ticker."""

    ticker: str
    sentiment: dict[str, Any] = field(default_factory=dict)
    performance: dict[str, Any] = field(default_factory=dict)
    news_snippets: list[str] = field(default_factory=list)
    errors: list[ToolError] = field(default_factory=list)


@dataclass
class SynthesizedAnswer:
    """Final user-facing answer produced by the Synthesizer."""

    markdown: str = ""
    rationale: str = ""
    caveats: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    confidence: float = 0.0        # 0-1
    raw_research: list[CompanyResearch] = field(default_factory=list)


@dataclass
class ReviewFeedback:
    """Output of the Reviewer agent."""

    ok: bool = False
    missing: list[str] = field(default_factory=list)
    notes: str = ""
