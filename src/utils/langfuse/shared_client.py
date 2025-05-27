"""Shared instance of langfuse client."""

from langfuse import Langfuse


__all__ = ["langfuse"]


langfuse = Langfuse()
