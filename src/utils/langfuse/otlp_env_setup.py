"""Set up environment variables for LangFuse integration."""

import base64
import logging
import os

from ..env_vars import Configs


def set_up_langfuse_otlp_env_vars():
    """Set up environment variables for Langfuse OpenTelemetry integration.

    OTLP = OpenTelemetry Protocol.

    This function updates environment variables.

    Also refer to:
    langfuse.com/docs/integrations/openaiagentssdk/openai-agents
    """
    configs = Configs.from_env_var()

    langfuse_auth = base64.b64encode(
        f"{configs.langfuse_public_key}:{configs.langfuse_secret_key}".encode()
    ).decode()

    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = (
        configs.langfuse_host + "/api/public/otel"
    )
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"

    logging.info(f"Langfuse host: {configs.langfuse_host}")
