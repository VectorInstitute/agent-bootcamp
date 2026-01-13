"""Session management utilities for agent conversations."""

import uuid
from typing import Any

import agents
from gradio.components.chatbot import ChatMessage


def get_or_create_session(
    history: list[ChatMessage], session_state: dict[str, Any]
) -> agents.SQLiteSession:
    """Get existing session or create a new one for conversation persistence."""
    if len(history) == 0:
        session = agents.SQLiteSession(session_id=str(uuid.uuid4()))
        session_state["session"] = session
    else:
        session = session_state["session"]
    return session
