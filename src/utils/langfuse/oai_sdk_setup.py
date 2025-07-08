"""Utils for redirecting OpenAI Agent SDK traces to LangFuse via OpenTelemetry.

Full documentation:
langfuse.com/docs/integrations/openaiagentssdk/openai-agents
"""

import logfire
import nest_asyncio

from .otlp_env_setup import set_up_langfuse_otlp_env_vars


def configure_oai_agents_sdk(service_name: str) -> None:
    """Register Langfuse as tracing provider for OAI Agents SDK."""
    nest_asyncio.apply()
    logfire.configure(service_name=service_name, send_to_logfire=False, scrubbing=False)
    logfire.instrument_openai_agents()


def setup_langfuse_tracer(service_name: str = "agents_sdk") -> None:
    """Register Langfuse as the default tracing provider and return tracer.

    Returns
    -------
    tracer: OpenTelemetry Tracer
    """
    set_up_langfuse_otlp_env_vars()
    configure_oai_agents_sdk(service_name)
