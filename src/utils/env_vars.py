"""Interface for storing and accessing config env vars."""

import pydantic


class Configs(pydantic.BaseModel):
    """Type-friendly collection of env var configs."""

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
    langfuse_host: str = "https://cloud.langfuse.com"

    def _check_langfuse(self):
        """Ensure that Langfuse pk and sk are in the right place."""
        if not self.langfuse_public_key.startswith("pk-lf-"):
            raise ValueError("LANGFUSE_PUBLIC_KEY should start with pk-lf-")

        if not self.langfuse_secret_key.startswith("sk-lf-"):
            raise ValueError("LANGFUSE_SECRET_KEY should start with sk-lf-")

    @staticmethod
    def from_env_var() -> "Configs":
        """Initialize from env vars."""
        from os import environ

        # Add only config line items defined in Configs.
        data: dict[str, str] = {}
        for k, v in environ.items():
            _key = k.lower()
            data[_key] = v

        config = Configs(**data)
        config._check_langfuse()
        return config
