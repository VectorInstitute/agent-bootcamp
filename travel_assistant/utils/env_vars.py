"""Interface for storing and accessing config env vars."""

from os import environ

import pydantic
from dotenv import load_dotenv


class Configs(pydantic.BaseModel):
    """Type-friendly collection of env var configs."""

    # OpenAI-compatible LLM
    openai_base_url: str
    openai_api_key: str
    agent_llm_name: str

    # External tool APIs
    tavily_api_key: str

    # Embeddings
    embedding_base_url: str
    embedding_api_key: str

    # Weaviate
    weaviate_http_host: str
    weaviate_grpc_host: str
    weaviate_api_key: str
    weaviate_http_port: int = 443
    weaviate_grpc_port: int = 443
    weaviate_http_secure: bool = True
    weaviate_grpc_secure: bool = True

    # Langfuse
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str = "https://us.cloud.langfuse.com"

    def _check_langfuse(self):
        """Ensure that Langfuse pk and sk are in the right place."""
        if not self.langfuse_public_key.startswith("pk-lf-"):
            raise ValueError("LANGFUSE_PUBLIC_KEY should start with pk-lf-")

        if not self.langfuse_secret_key.startswith("sk-lf-"):
            raise ValueError("LANGFUSE_SECRET_KEY should start with sk-lf-")

    @classmethod
    def from_env_var(cls) -> "Configs":
        """Initialize from env vars."""
        # Load values from a local .env file if present
        load_dotenv()  # noqa: F401 - ensures .env is read when available

        # Only include keys defined in the Configs model
        data: dict[str, str] = {}
        for field in cls.model_fields:
            env_key = field.upper()
            value = environ.get(env_key)
            if value is not None:
                data[field] = value.strip().strip('"').strip("'")

        try:
            config = cls(**data)
            config._check_langfuse()
            return config

        except pydantic.ValidationError as e:
            raise ValueError(
                "Some ENV VARs are missing. See above for details. "
                "Try to load your .env file as follows: \n"
                "```\nuv run --env-file .env -m ...\n```"
            ) from e
