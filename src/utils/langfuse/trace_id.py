"""
Obtain trace_id, required for linking trace to dataset row.

Full documentation:
langfuse.com/docs/integrations/openaiagentssdk/example-evaluating-openai-agents
running-the-agent-on-the-dataset
"""


def get_langfuse_trace_id():
    """Obtain "formatted" trace_id for LangFuse."""
