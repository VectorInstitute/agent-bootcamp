"""Interface for storing and accessing config env vars."""

import pydantic


class Configs(pydantic.BaseModel):
    """Type-friendly collection of env var configs."""

    es_api_key: str
    es_host: str

    @staticmethod
    def from_env_var() -> "Configs":
        """Initialize from env vars."""
        from os import environ

        # Add only config line items defined in Configs.
        data: dict[str, str] = {}
        for k, v in environ.items():
            _key = k.lower()
            data[_key] = v

        return Configs(**data)
