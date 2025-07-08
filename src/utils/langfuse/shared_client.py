"""Shared instance of langfuse client."""

from langfuse import get_client


__all__ = ["langfuse"]


langfuse = get_client()
