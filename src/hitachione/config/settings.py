"""Centralised settings for the multi-agent financial intelligence system.

All secrets come from env-vars / .env – never hard-coded.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_PROJECT_ROOT / ".env")

# ── LLM ──────────────────────────────────────────────────────────────────
OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PLANNER_MODEL = os.getenv("DEFAULT_PLANNER_MODEL", "gemini-2.5-flash")
WORKER_MODEL = os.getenv("DEFAULT_WORKER_MODEL", "gemini-2.5-flash")

# ── Weaviate ─────────────────────────────────────────────────────────────
WEAVIATE_COLLECTION = os.getenv("WEAVIATE_COLLECTION_NAME", "Hitachi_finance_news")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
WEAVIATE_HTTP_HOST = os.getenv("WEAVIATE_HTTP_HOST", "localhost")
WEAVIATE_GRPC_HOST = os.getenv("WEAVIATE_GRPC_HOST", "localhost")

# ── Langfuse (optional) ─────────────────────────────────────────────────
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")

# ── Orchestrator defaults ────────────────────────────────────────────────
MAX_ITERATIONS = int(os.getenv("ORCHESTRATOR_MAX_ITERATIONS", "2"))
QUALITY_THRESHOLD = float(os.getenv("ORCHESTRATOR_QUALITY_THRESHOLD", "0.6"))
