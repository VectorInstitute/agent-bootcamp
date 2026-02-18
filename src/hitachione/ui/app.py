"""Gradio UI for the multi-agent financial intelligence system.

Uses gr.ChatInterface (same pattern as the other bootcamp apps)
so the user types a financial prompt and gets a Markdown answer.
"""

from __future__ import annotations

import logging

import gradio as gr

from ..agents.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

_orchestrator = Orchestrator()


def _respond(
    message: str,
    history: list[dict],
) -> str:
    """Process one user query and return the assistant's answer.

    The orchestrator can take 30-60 s, during which Gradio shows a spinner.
    """
    if not message.strip():
        return "*Please enter a financial query.*"

    try:
        answer = _orchestrator.run(message.strip())
    except Exception as exc:
        logger.exception("Orchestrator error")
        return f"**Error:** {exc}"

    # Build final output
    parts = [answer.markdown]

    if answer.caveats:
        parts.append("\n---\n**Caveats:**")
        for c in answer.caveats:
            parts.append(f"- {c}")

    if answer.citations:
        parts.append("\n**Citations:**")
        for i, cit in enumerate(answer.citations[:5], 1):
            parts.append(f"{i}. {cit}")

    parts.append(f"\n*Confidence: {answer.confidence:.0%}*")

    return "\n".join(parts)


def build_app() -> gr.ChatInterface:
    """Build and return the Gradio ChatInterface app."""
    demo = gr.ChatInterface(
        fn=_respond,
        title="🏦 Financial Intelligence Agent",
        description=(
            "Ask any financial question – ranking, comparison, snapshot, "
            "event reaction, and more."
        ),
        examples=[
            "Top 3 tech stocks of 2024",
            "Compare TSLA vs AAPL vs NVDA",
            "What moved NVDA after last earnings?",
        ],
        chatbot=gr.Chatbot(height=600),
        textbox=gr.Textbox(lines=1, placeholder="Enter your financial query…"),
    )
    return demo
