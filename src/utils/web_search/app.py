"""FastAPI application exposing the Gemini grounding proxy."""

import asyncio
import inspect
import logging
import os
import random
from datetime import datetime
from typing import Annotated, Literal, Optional

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, status
from google import genai
from google.api_core import exceptions as google_exceptions
from google.auth.credentials import AnonymousCredentials
from google.genai import types
from pydantic import BaseModel, Field

from .auth import (
    APIKeyAuthenticator,
    ExpiredAPIKeyError,
    InactiveAPIKeyError,
    InvalidAPIKeyError,
)
from .daily_usage import DailyUsageRepository
from .db import APIKeyRecord, APIKeyRepository, UsageLimitExceededError


try:
    from google.cloud import firestore
except ImportError:  # pragma: no cover - dependency required in production
    firestore = None


logger = logging.getLogger(__name__)

MAX_GEMINI_ATTEMPTS = int(os.getenv("GEMINI_MAX_ATTEMPTS", "5"))
MAX_BACKOFF_SECONDS = float(os.getenv("GEMINI_MAX_BACKOFF_SECONDS", "10"))
FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION", "apiKeys")
API_KEY_CACHE_TTL = int(os.getenv("API_KEY_CACHE_TTL", "30"))
API_KEY_CACHE_MAX_ITEMS = int(os.getenv("API_KEY_CACHE_MAX_ITEMS", "1024"))
FREE_LIMIT_DEFAULT_PRO = 1500
FREE_LIMIT_DEFAULT_FLASH = 1500


def _parse_free_limit(env_var: str, default: int) -> int:
    """Parse a non-negative integer from the environment with logging."""
    value = os.getenv(env_var)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning(
            "Invalid value '%s' for %s; falling back to %d",
            value,
            env_var,
            default,
        )
        return default
    if parsed < 0:
        logger.warning(
            "Negative value '%s' for %s; treating as 0",
            value,
            env_var,
        )
        return 0
    return parsed


MODEL_TO_USAGE_BUCKET: dict[str, str] = {
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.5-flash": "gemini-2.5-flash-family",
    "gemini-2.5-flash-lite": "gemini-2.5-flash-family",
}

BUCKET_FREE_LIMITS: dict[str, int] = {
    "gemini-2.5-pro": _parse_free_limit(
        "GEMINI_GROUNDING_FREE_LIMIT_PRO",
        FREE_LIMIT_DEFAULT_PRO,
    ),
    "gemini-2.5-flash-family": _parse_free_limit(
        "GEMINI_GROUNDING_FREE_LIMIT_FLASH",
        FREE_LIMIT_DEFAULT_FLASH,
    ),
}


def _resolve_usage_bucket(model: str) -> tuple[str, int]:
    """Return the usage bucket and free allowance for the given model."""
    bucket = MODEL_TO_USAGE_BUCKET.get(model, model)
    return bucket, BUCKET_FREE_LIMITS.get(bucket, 0)


RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    google_exceptions.ResourceExhausted,
    google_exceptions.ServiceUnavailable,
    google_exceptions.InternalServerError,
    google_exceptions.DeadlineExceeded,
)

if hasattr(google_exceptions, "TooManyRequests"):
    RETRYABLE_EXCEPTIONS = RETRYABLE_EXCEPTIONS + (google_exceptions.TooManyRequests,)  # type: ignore[assignment]


app = FastAPI()
router = APIRouter()

grounding_tool = types.Tool(google_search=types.GoogleSearch())


class RequestBody(BaseModel):
    """Request payload accepted by the grounding proxy."""

    query: str
    model: Literal["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"] = (
        "gemini-2.5-flash"
    )
    temperature: float | None = Field(default=0.2, ge=0, le=2)
    max_output_tokens: int | None = Field(default=None, ge=1)
    seed: int | None = None
    thinking_budget: int | None = Field(default=-1, ge=-1)


class APIKeySummary(BaseModel):
    """Serializable summary of an API key record."""

    lookup_hash: str
    display_prefix: str
    role: Literal["admin", "user"]
    owner: str | None
    status: Literal["active", "suspended"]
    usage_count: int
    usage_limit: int
    last_used_at: datetime | None
    created_at: datetime
    created_by: str
    metadata: dict[str, object]
    expires_at: datetime | None


class AdminCreateKeyRequest(BaseModel):
    """Payload accepted when administrators create new API keys."""

    role: Literal["admin", "user"] = "user"
    owner: str | None = None
    usage_limit: int = Field(default=0, ge=0)
    metadata: dict[str, str] = Field(default_factory=dict)
    expires_at: datetime | None = None


class AdminCreateKeyResponse(BaseModel):
    """Response returned after creating a new API key."""

    api_key: str
    record: APIKeySummary


class AdminUpdateKeyRequest(BaseModel):
    """Payload used to adjust usage limit and expiration."""

    usage_limit: int | None = Field(default=None, ge=0)
    expires_at: datetime | None = None


class APIKeyUsageResponse(BaseModel):
    """User-facing usage information for the current API key."""

    usage_count: int
    usage_limit: int
    expires_at: datetime | None


def _ensure_firestore_dependency() -> None:
    """Validate that the Firestore dependency is available.

    Raises
    ------
    RuntimeError
        Raised when ``google-cloud-firestore`` has not been installed.
    """
    if firestore is None:
        raise RuntimeError(
            "google-cloud-firestore must be installed to run the grounding proxy",
        )


def _build_api_key_summary(record: APIKeyRecord) -> APIKeySummary:
    """Convert an ``APIKeyRecord`` to ``APIKeySummary`` for responses.

    Parameters
    ----------
    record : APIKeyRecord
        Record retrieved from the data store.

    Returns
    -------
    APIKeySummary
        Serializable representation used by administrative endpoints.
    """
    return APIKeySummary(
        lookup_hash=record.lookup_hash,
        display_prefix=record.display_prefix,
        role=record.role,
        owner=record.owner,
        status=record.status,
        usage_count=record.usage_count,
        usage_limit=record.usage_limit,
        last_used_at=record.last_used_at,
        created_at=record.created_at,
        created_by=record.created_by,
        metadata=record.metadata,
        expires_at=record.expires_at,
    )


async def startup_event() -> None:
    """Initialise Firestore client and authentication dependencies.

    Raises
    ------
    RuntimeError
        Raised when the Firestore dependency is not available.
    """
    _ensure_firestore_dependency()

    project_id = os.getenv("FIRESTORE_PROJECT_ID")
    client_kwargs = {}
    if os.getenv("FIRESTORE_EMULATOR_HOST"):
        client_kwargs["credentials"] = AnonymousCredentials()
    else:
        client_kwargs["database"] = os.getenv("FIRESTORE_DATABASE_NAME")

    firestore_client = firestore.AsyncClient(project=project_id, **client_kwargs)

    repository = APIKeyRepository(
        firestore_client,
        collection_name=FIRESTORE_COLLECTION,
    )
    app.state.firestore_client = firestore_client
    app.state.authenticator = APIKeyAuthenticator(
        repository,
        cache_ttl_seconds=API_KEY_CACHE_TTL,
        cache_max_items=API_KEY_CACHE_MAX_ITEMS,
    )
    app.state.daily_usage_repository = DailyUsageRepository(
        firestore_client,
        collection_name=os.getenv(
            "DAILY_USAGE_COLLECTION",
            "dailyUsageCounters",
        ),
    )


async def shutdown_event() -> None:
    """Release Firestore resources during application shutdown.

    Returns
    -------
    None
        This function performs a best-effort shutdown of Firestore resources.
    """
    firestore_client: firestore.AsyncClient = getattr(
        app.state, "firestore_client", None
    )
    if firestore_client:
        close_callable = getattr(firestore_client, "close", None)
        if callable(close_callable):
            close_result = close_callable()
            if inspect.isawaitable(close_result):
                await close_result


app.add_event_handler("startup", startup_event)
app.add_event_handler("shutdown", shutdown_event)


def get_authenticator() -> APIKeyAuthenticator:
    """Return the singleton authenticator stored on the app state.

    Returns
    -------
    APIKeyAuthenticator
        Shared authenticator instance used by request dependencies.

    Raises
    ------
    RuntimeError
        Raised when the application startup hook has not executed.
    """
    authenticator: APIKeyAuthenticator | None = getattr(
        app.state, "authenticator", None
    )
    if authenticator is None:
        raise RuntimeError("Authenticator has not been initialised")
    return authenticator


def get_daily_usage_repository() -> DailyUsageRepository:
    """Return the daily usage repository stored on the app state."""
    repository: DailyUsageRepository | None = getattr(
        app.state,
        "daily_usage_repository",
        None,
    )
    if repository is None:
        raise RuntimeError("Daily usage repository has not been initialised")
    return repository


async def _authenticate_request(
    api_key_header: str,
    authenticator: APIKeyAuthenticator,
    *,
    consume_usage: bool,
) -> APIKeyRecord:
    """Authenticate API keys with consistent error handling."""
    try:
        return await authenticator.reserve_usage(
            api_key_header,
            consume_usage=consume_usage,
        )
    except InvalidAPIKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key provided",
        ) from exc
    except InactiveAPIKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key is inactive",
        ) from exc
    except ExpiredAPIKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key has expired",
        ) from exc
    except UsageLimitExceededError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key usage limit exceeded",
        ) from exc


async def require_api_key_without_consumption(
    api_key_header: Annotated[str, Header(alias="X-API-Key")],
    authenticator: Annotated[APIKeyAuthenticator, Depends(get_authenticator)],
) -> APIKeyRecord:
    """Validate the user's API key without decrementing the usage counter."""
    return await _authenticate_request(
        api_key_header,
        authenticator,
        consume_usage=False,
    )


async def require_admin_api_key(
    api_key_header: Annotated[str, Header(alias="X-API-Key")],
    authenticator: Annotated[APIKeyAuthenticator, Depends(get_authenticator)],
) -> APIKeyRecord:
    """Ensure that the request is authorised with an admin-level API key.

    Parameters
    ----------
    api_key_header : str
        API key supplied in the ``X-API-Key`` header.
    authenticator : APIKeyAuthenticator
        Authenticator responsible for validating the key.

    Returns
    -------
    APIKeyRecord
        The same record when it corresponds to an admin key.

    Raises
    ------
    HTTPException
        Raised when the key does not grant administrative privileges.
    """
    record = await _authenticate_request(
        api_key_header,
        authenticator,
        consume_usage=False,
    )
    if record.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return record


async def call_gemini_with_retry(request: RequestBody) -> types.GenerateContentResponse:
    """Invoke Gemini with retries to respect Google rate limits.

    Parameters
    ----------
    request : RequestBody
        Payload to forward to Gemini.

    Returns
    -------
    google.genai.types.GenerateContentResponse
        Response returned by the Gemini model upon success.

    Raises
    ------
    HTTPException
        Raised with status ``502`` when Gemini cannot service the request
        after exhausting retries.
    """
    attempt = 0
    while attempt < MAX_GEMINI_ATTEMPTS:
        attempt += 1
        try:
            async with genai.Client().aio as client:
                return await client.models.generate_content(
                    model=request.model,
                    contents=request.query,
                    config=types.GenerateContentConfig(
                        temperature=request.temperature,
                        max_output_tokens=request.max_output_tokens,
                        seed=request.seed,
                        safety_settings=[
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                            ),
                        ],
                        tools=[grounding_tool],
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=request.thinking_budget,
                        ),
                    ),
                )
        except RETRYABLE_EXCEPTIONS as exc:
            if attempt >= MAX_GEMINI_ATTEMPTS:
                logger.exception(
                    "Gemini request failed after %d retries", MAX_GEMINI_ATTEMPTS
                )
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Gemini is currently unavailable",
                ) from exc

            backoff = min(2 ** (attempt - 1), MAX_BACKOFF_SECONDS) + random.uniform(
                0,
                0.5,
            )
            await asyncio.sleep(backoff)
        except google_exceptions.GoogleAPICallError as exc:
            logger.exception("Gemini request failed with unrecoverable error")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Gemini call failed",
            ) from exc

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Gemini call failed",
    )


@app.get("/healthz")
async def health() -> dict[str, str]:
    """Return health information for readiness probes.

    Returns
    -------
    dict of str to str
        Health status indicator consumed by platform checks.
    """
    return {"ok": "true"}


@router.post("/v1/grounding_with_search")
async def search(
    request: RequestBody,
    record: Annotated[
        APIKeyRecord,
        Depends(require_api_key_without_consumption),
    ],
    authenticator: Annotated[APIKeyAuthenticator, Depends(get_authenticator)],
    daily_usage: Annotated[
        DailyUsageRepository,
        Depends(get_daily_usage_repository),
    ],
) -> dict[str, object]:
    """Proxy Gemini grounding requests with quota enforcement.

    Parameters
    ----------
    request : RequestBody
        Payload describing the Gemini call.
    record : APIKeyRecord
        API key record produced by ``require_api_key``.
    authenticator : APIKeyAuthenticator
        Authenticator dependency used to roll back usage reservations on error.

    Returns
    -------
    dict of str to object
        JSON serialisable response returned by the Gemini model.
    """
    bucket, free_limit = _resolve_usage_bucket(request.model)
    consumed_api_quota = False
    reservation = await daily_usage.reserve(bucket, free_limit)

    if not reservation.consumed_free:
        try:
            updated_record = await authenticator.consume_usage(record.lookup_hash)
        except UsageLimitExceededError as exc:
            await daily_usage.release(reservation)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="API key usage limit exceeded",
            ) from exc
        except InvalidAPIKeyError as exc:
            await daily_usage.release(reservation)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key provided",
            ) from exc
        except InactiveAPIKeyError as exc:
            await daily_usage.release(reservation)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="API key is inactive",
            ) from exc
        except ExpiredAPIKeyError as exc:
            await daily_usage.release(reservation)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="API key has expired",
            ) from exc

        record = updated_record
        consumed_api_quota = True

    try:
        response = await call_gemini_with_retry(request)
    except Exception:
        try:
            await daily_usage.release(reservation)
        except Exception:  # pragma: no cover - defensive logging for rollbacks
            logger.exception(
                "Failed to roll back daily usage for bucket %s",
                bucket,
            )

        if consumed_api_quota:
            try:
                await authenticator.release_usage(record.lookup_hash)
            except Exception:  # pragma: no cover - defensive logging for rollbacks
                logger.exception(
                    "Failed to roll back usage for API key %s", record.lookup_hash
                )
        raise

    logger.info(
        "Gemini request completed for model %s (bucket=%s, consumed_free=%s)",
        request.model,
        bucket,
        reservation.consumed_free if reservation else False,
    )
    return response.to_json_dict()


@router.get("/usage")
async def usage(
    record: Annotated[APIKeyRecord, Depends(require_api_key_without_consumption)],
) -> APIKeyUsageResponse:
    """Return current usage information for the caller's API key.

    Parameters
    ----------
    record : APIKeyRecord
        Updated API key record produced by ``require_api_key_without_consumption``.

    Returns
    -------
    APIKeyUsageResponse
        Usage statistics scoped to the caller's API key.
    """
    return APIKeyUsageResponse(
        usage_count=record.usage_count,
        usage_limit=record.usage_limit,
        expires_at=record.expires_at,
    )


@router.get("/admin/api-keys")
async def list_api_keys(
    _: Annotated[APIKeyRecord, Depends(require_admin_api_key)],
    authenticator: Annotated[APIKeyAuthenticator, Depends(get_authenticator)],
    status_filter: Optional[Literal["active", "suspended"]] = None,
) -> list[APIKeySummary]:
    """List API keys for administrative purposes.

    Parameters
    ----------
    _ : APIKeyRecord
        Verified administrative API key record (discarded after verification).
    authenticator : APIKeyAuthenticator
        Authenticator instance used to access the repository.
    status_filter : {"active", "suspended"}, optional
        Optional filter restricting which keys are returned.

    Returns
    -------
    list of APIKeySummary
        Collection of API key summaries.
    """
    records = await authenticator.list_keys(status=status_filter)
    return [_build_api_key_summary(record) for record in records]


@router.post("/admin/api-keys", status_code=status.HTTP_201_CREATED)
async def create_api_key(
    payload: AdminCreateKeyRequest,
    admin: Annotated[APIKeyRecord, Depends(require_admin_api_key)],
    authenticator: Annotated[APIKeyAuthenticator, Depends(get_authenticator)],
) -> AdminCreateKeyResponse:
    """Create a new API key and return the plaintext token once.

    Parameters
    ----------
    payload : AdminCreateKeyRequest
        Details used to create the new API key.
    admin : APIKeyRecord
        Record for the administrator creating the key.
    authenticator : APIKeyAuthenticator
        Authenticator instance used to create and persist the key.

    Returns
    -------
    AdminCreateKeyResponse
        Response containing the plaintext API key and summary metadata.
    """
    api_key, record = await authenticator.create_api_key(
        role=payload.role,
        owner=payload.owner,
        usage_limit=payload.usage_limit,
        created_by=admin.owner or admin.lookup_hash,
        metadata=payload.metadata,
        expires_at=payload.expires_at,
    )
    return AdminCreateKeyResponse(
        api_key=api_key, record=_build_api_key_summary(record)
    )


@router.post("/admin/api-keys/{lookup_hash}/activate")
async def activate_api_key(
    lookup_hash: str,
    _: Annotated[APIKeyRecord, Depends(require_admin_api_key)],
    authenticator: Annotated[APIKeyAuthenticator, Depends(get_authenticator)],
) -> dict[str, str]:
    """Activate a suspended API key.

    Parameters
    ----------
    lookup_hash : str
        Identifier for the API key that should be activated.
    _ : APIKeyRecord
        Verified administrative API key record (discarded after verification).
    authenticator : APIKeyAuthenticator
        Authenticator instance used to mutate the key.

    Returns
    -------
    dict of str to str
        Message confirming the new status.
    """
    await authenticator.activate(lookup_hash)
    return {"status": "active"}


@router.post("/admin/api-keys/{lookup_hash}/deactivate")
async def deactivate_api_key(
    lookup_hash: str,
    _: Annotated[APIKeyRecord, Depends(require_admin_api_key)],
    authenticator: Annotated[APIKeyAuthenticator, Depends(get_authenticator)],
) -> dict[str, str]:
    """Suspend an API key to block further requests.

    Parameters
    ----------
    lookup_hash : str
        Identifier for the API key that should be suspended.
    _ : APIKeyRecord
        Verified administrative API key record (discarded after verification).
    authenticator : APIKeyAuthenticator
        Authenticator instance used to mutate the key.

    Returns
    -------
    dict of str to str
        Message confirming the new status.
    """
    await authenticator.deactivate(lookup_hash)
    return {"status": "suspended"}


@router.patch("/admin/api-keys/{lookup_hash}")
async def update_api_key(
    lookup_hash: str,
    payload: AdminUpdateKeyRequest,
    _: Annotated[APIKeyRecord, Depends(require_admin_api_key)],
    authenticator: Annotated[APIKeyAuthenticator, Depends(get_authenticator)],
) -> APIKeySummary:
    """Update mutable metadata for an API key.

    Parameters
    ----------
    lookup_hash : str
        Identifier for the API key whose limit should change.
    payload : AdminUpdateKeyRequest
        Payload containing the fields to update.
    _ : APIKeyRecord
        Verified administrative API key record (discarded after verification).
    authenticator : APIKeyAuthenticator
        Authenticator instance used to mutate the key.

    Returns
    -------
    APIKeySummary
        Updated API key summary reflecting the applied changes.
    """
    updates = payload.model_dump(exclude_unset=True)

    if "usage_limit" in updates:
        await authenticator.adjust_usage_limit(lookup_hash, updates["usage_limit"])
    if "expires_at" in updates:
        await authenticator.adjust_expiration(lookup_hash, updates["expires_at"])

    record = await authenticator.get_api_key(lookup_hash)
    return _build_api_key_summary(record)


@router.delete("/admin/api-keys/{lookup_hash}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_api_key(
    lookup_hash: str,
    _: Annotated[APIKeyRecord, Depends(require_admin_api_key)],
    authenticator: Annotated[APIKeyAuthenticator, Depends(get_authenticator)],
) -> None:
    """Delete an API key permanently.

    Parameters
    ----------
    lookup_hash : str
        Identifier for the API key that should be removed.
    _ : APIKeyRecord
        Verified administrative API key record (discarded after verification).
    authenticator : APIKeyAuthenticator
        Authenticator instance used to mutate the key.

    Returns
    -------
    None
        This endpoint does not return a body.
    """
    await authenticator.delete_key(lookup_hash)


app.include_router(router, prefix="/api")
