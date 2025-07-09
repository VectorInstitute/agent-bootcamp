"""Shared instance of langfuse client."""

from os import getenv

from langfuse import Langfuse

from ..env_vars import Configs


__all__ = ["langfuse_client"]


config = Configs.from_env_var()
assert getenv("LANGFUSE_PUBLIC_KEY") is not None
langfuse_client = Langfuse(
    public_key=config.langfuse_public_key, secret_key=config.langfuse_secret_key
)
