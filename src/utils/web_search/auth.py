"""Authentication helpers for the Gemini grounding proxy."""

import base64
import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from .db import APIKeyNotFoundError, APIKeyRecord, APIKeyRepository, Role, Status


DEFAULT_CACHE_TTL_SECONDS = int(os.getenv("API_KEY_CACHE_TTL", "30"))
DEFAULT_CACHE_MAX_ITEMS = int(os.getenv("API_KEY_CACHE_MAX_ITEMS", "1024"))
PBKDF2_ITERATIONS = int(os.getenv("API_KEY_PBKDF2_ITERATIONS", "200000"))
PBKDF2_SALT_BYTES = int(os.getenv("API_KEY_PBKDF2_SALT_BYTES", "16"))


class InvalidAPIKeyError(Exception):
    """Raised when the provided API key cannot be verified."""


class InactiveAPIKeyError(Exception):
    """Raised when an API key is valid but not currently active."""


class ExpiredAPIKeyError(Exception):
    """Raised when an API key has passed its expiration timestamp."""


def _now() -> datetime:
    """Return the current UTC time.

    Returns
    -------
    datetime
        Current time in UTC with timezone information.
    """
    return datetime.now(tz=timezone.utc)


def generate_api_key() -> str:
    """Generate a random API key string.

    Returns
    -------
    str
        Newly generated API key safe for display.
    """
    return secrets.token_urlsafe(32)


def generate_salt() -> str:
    """Generate a cryptographically secure salt encoded in base64.

    Returns
    -------
    str
        Base64 encoded salt suitable for PBKDF2.
    """
    return base64.b64encode(secrets.token_bytes(PBKDF2_SALT_BYTES)).decode("ascii")


def hash_api_key(api_key: str, salt: str) -> str:
    """Derive a PBKDF2-HMAC hash for the provided API key.

    Parameters
    ----------
    api_key : str
        Raw API key that needs to be stored securely.
    salt : str
        Base64 encoded salt to combine with the API key.

    Returns
    -------
    str
        Base64 encoded PBKDF2-HMAC digest.
    """
    salt_bytes = base64.b64decode(salt.encode("ascii"))
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        api_key.encode("utf-8"),
        salt_bytes,
        PBKDF2_ITERATIONS,
    )
    return base64.b64encode(derived).decode("ascii")


def derive_lookup_hash(api_key: str) -> str:
    """Compute a stable SHA-256 digest for document lookup.

    Parameters
    ----------
    api_key : str
        Raw API key generated for the client.

    Returns
    -------
    str
        Lowercase hex string that can be used as document id.
    """
    digest = hashlib.sha256(api_key.encode("utf-8")).digest()
    return base64.b16encode(digest).decode("ascii").lower()


def verify_api_key(api_key: str, salt: str, hashed_key: str) -> bool:
    """Verify that the supplied API key matches the stored hash.

    Parameters
    ----------
    api_key : str
        Raw API key provided by the caller.
    salt : str
        Base64 encoded salt stored alongside the key.
    hashed_key : str
        Stored PBKDF2-HMAC hash retrieved from Firestore.

    Returns
    -------
    bool
        ``True`` when the API key can be verified.
    """
    expected = hash_api_key(api_key, salt)
    return hmac.compare_digest(expected, hashed_key)


def _normalise_datetime(value: Optional[datetime]) -> Optional[datetime]:
    """Ensure datetimes are timezone-aware and expressed in UTC."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(slots=True)
class CacheEntry:
    """Cache structure used to reduce Firestore lookups."""

    record: APIKeyRecord
    expires_at: datetime

    def is_expired(self, *, clock: Callable[[], datetime]) -> bool:
        """Return ``True`` when the cache entry is past its TTL.

        Parameters
        ----------
        clock : callable
            Clock function returning current time for comparisons.

        Returns
        -------
        bool
            ``True`` when the entry should be discarded.
        """
        return clock() >= self.expires_at


class APIKeyAuthenticator:
    """Authenticate API keys and enforce usage limits."""

    def __init__(
        self,
        repository: APIKeyRepository,
        *,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
        cache_max_items: int = DEFAULT_CACHE_MAX_ITEMS,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        """Initialise the authenticator with repository and cache settings.

        Parameters
        ----------
        repository : APIKeyRepository
            Repository used to interact with persistent storage.
        cache_ttl_seconds : int, default=30
            Time-to-live for memory cache entries.
        cache_max_items : int, default=1024
            Maximum number of cache entries retained in memory.
        clock : callable, default=_now
            Testable clock returning the current UTC datetime.
        """
        self._repository = repository
        self._cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._cache_max_items = cache_max_items
        self._clock = clock
        self._cache: dict[str, CacheEntry] = {}

    def _cache_lookup(self, lookup_hash: str) -> Optional[APIKeyRecord]:
        """Retrieve a cached record when it exists and is still valid.

        Parameters
        ----------
        lookup_hash : str
            Lookup hash derived from the caller's API key.

        Returns
        -------
        APIKeyRecord or None
            Cached record when available, otherwise ``None``.
        """
        entry = self._cache.get(lookup_hash)
        if not entry:
            return None
        if entry.is_expired(clock=self._clock):
            self._cache.pop(lookup_hash, None)
            return None
        return entry.record

    def _cache_store(self, record: APIKeyRecord) -> None:
        """Insert a record into the cache, evicting the oldest when needed.

        Parameters
        ----------
        record : APIKeyRecord
            Record that should be cached for subsequent lookups.
        """
        if len(self._cache) >= self._cache_max_items:
            # Simple eviction policy: remove the first inserted key.
            oldest_key = next(iter(self._cache))
            self._cache.pop(oldest_key, None)

        self._cache[record.lookup_hash] = CacheEntry(
            record=record,
            expires_at=self._clock() + self._cache_ttl,
        )

    async def reserve_usage(
        self,
        api_key: str,
        *,
        consume_usage: bool = True,
    ) -> APIKeyRecord:
        """Verify an API key and optionally reserve one unit of usage.

        Parameters
        ----------
        api_key : str
            Raw API key provided in the ``X-API-Key`` header.
        consume_usage : bool, default=True
            When ``True`` the usage counter is incremented atomically. When
            ``False`` the API key is only validated and cached.

        Returns
        -------
        APIKeyRecord
            Updated record containing the new usage counter when
            ``consume_usage`` is ``True``; otherwise the cached record.

        Raises
        ------
        InvalidAPIKeyError
            Raised when the API key cannot be found or verified.
        InactiveAPIKeyError
            Raised when the API key is not currently active.
        UsageLimitExceededError
            Propagated when the call would exceed the configured quota.
        """
        lookup_hash = derive_lookup_hash(api_key)
        record = self._cache_lookup(lookup_hash)

        if not record:
            try:
                record = await self._repository.get_api_key(lookup_hash)
            except APIKeyNotFoundError as exc:
                raise InvalidAPIKeyError("API key not recognised") from exc

            if not verify_api_key(api_key, record.salt, record.hashed_key):
                raise InvalidAPIKeyError("API key signature invalid")

            self._cache_store(record)

        if record.status != "active":
            raise InactiveAPIKeyError("API key has been suspended")

        if record.expires_at and self._clock() >= record.expires_at:
            self._cache.pop(lookup_hash, None)
            raise ExpiredAPIKeyError("API key has expired")

        if consume_usage:
            updated_record = await self._repository.update_usage_counter(lookup_hash)
            self._cache_store(updated_record)
            return updated_record

        return record

    async def consume_usage(self, lookup_hash: str) -> APIKeyRecord:
        """Increment usage counter for a previously validated API key."""
        record = self._cache_lookup(lookup_hash)

        if not record:
            try:
                record = await self._repository.get_api_key(lookup_hash)
            except APIKeyNotFoundError as exc:
                raise InvalidAPIKeyError("API key not recognised") from exc
            self._cache_store(record)

        if record.status != "active":
            raise InactiveAPIKeyError("API key has been suspended")

        if record.expires_at and self._clock() >= record.expires_at:
            self._cache.pop(lookup_hash, None)
            raise ExpiredAPIKeyError("API key has expired")

        try:
            updated_record = await self._repository.update_usage_counter(lookup_hash)
        except APIKeyNotFoundError as exc:
            self._cache.pop(lookup_hash, None)
            raise InvalidAPIKeyError("API key not recognised") from exc

        self._cache_store(updated_record)
        return updated_record

    async def release_usage(self, lookup_hash: str) -> APIKeyRecord:
        """Rollback a previously reserved usage slot.

        Parameters
        ----------
        lookup_hash : str
            Lookup hash corresponding to the API key whose usage should be
            decremented.

        Returns
        -------
        APIKeyRecord
            Updated record containing the decremented usage counter.
        """
        try:
            updated_record = await self._repository.decrement_usage_counter(
                lookup_hash,
            )
        except APIKeyNotFoundError as exc:  # pragma: no cover - defensive branch
            self._cache.pop(lookup_hash, None)
            raise InvalidAPIKeyError("API key not recognised") from exc

        self._cache_store(updated_record)
        return updated_record

    async def create_api_key(
        self,
        *,
        role: Role,
        owner: Optional[str],
        usage_limit: int,
        created_by: str,
        metadata: Optional[dict[str, str]] = None,
        expires_at: Optional[datetime] = None,
    ) -> tuple[str, APIKeyRecord]:
        r"""Create a new API key and persist it to Firestore.

        Parameters
        ----------
        role : Role
            Role assigned to the API key (for example ``\"user\"`` or
            ``\"admin\"``).
        owner : str, optional
            Optional identifier describing the owner of the API key.
        usage_limit : int
            Maximum total number of requests allowed. ``0`` indicates
            unlimited lifetime usage.
        created_by : str
            Identifier for the administrator creating the key.
        metadata : dict of str to str, optional
            Additional metadata stored alongside the record.
        expires_at : datetime, optional
            Optional absolute expiration timestamp. ``None`` keeps the key
            valid indefinitely.

        Returns
        -------
        tuple of (str, APIKeyRecord)
            The plaintext API key and the persisted record.
        """
        api_key = generate_api_key()
        lookup_hash = derive_lookup_hash(api_key)
        salt = generate_salt()
        hashed_key = hash_api_key(api_key, salt)
        normalised_expires_at = _normalise_datetime(expires_at)

        record = APIKeyRecord(
            lookup_hash=lookup_hash,
            hashed_key=hashed_key,
            salt=salt,
            display_prefix=api_key[:8],
            role=role,
            owner=owner,
            status="active",
            usage_count=0,
            usage_limit=usage_limit,
            last_used_at=None,
            created_at=self._clock(),
            created_by=created_by,
            metadata=metadata or {},
            expires_at=normalised_expires_at,
        )

        await self._repository.create_api_key(record)
        self._cache_store(record)
        return api_key, record

    async def deactivate(self, lookup_hash: str) -> None:
        """Suspend an API key from further usage.

        Parameters
        ----------
        lookup_hash : str
            Lookup hash representing the API key that should be suspended.
        """
        await self._update_status(lookup_hash, "suspended")

    async def activate(self, lookup_hash: str) -> None:
        """Mark a previously suspended API key as active.

        Parameters
        ----------
        lookup_hash : str
            Lookup hash representing the API key that should be activated.
        """
        await self._update_status(lookup_hash, "active")

    async def _update_status(self, lookup_hash: str, status: Status) -> None:
        """Update status and cache.

        Parameters
        ----------
        lookup_hash : str
            Lookup hash representing the API key to update.
        status : Status
            New status value that should be stored.
        """
        await self._repository.set_status(lookup_hash, status)
        self._cache.pop(lookup_hash, None)

    async def adjust_usage_limit(self, lookup_hash: str, usage_limit: int) -> None:
        """Change the usage limit for a given API key.

        Parameters
        ----------
        lookup_hash : str
            Lookup hash for the API key to modify.
        usage_limit : int
            New usage limit. ``0`` indicates unlimited usage.
        """
        await self._repository.update_usage_limit(lookup_hash, usage_limit)
        self._cache.pop(lookup_hash, None)

    async def delete_key(self, lookup_hash: str) -> None:
        """Delete an API key from Firestore and clear cache.

        Parameters
        ----------
        lookup_hash : str
            Lookup hash for the API key to remove permanently.
        """
        await self._repository.delete_api_key(lookup_hash)
        self._cache.pop(lookup_hash, None)

    async def list_keys(
        self,
        *,
        status: Optional[Status] = None,
        limit: int = 100,
    ) -> list[APIKeyRecord]:
        """Return a list of API keys for administrative use.

        Parameters
        ----------
        status : Status, optional
            Optional filter restricting the status of returned keys.
        limit : int, default=100
            Maximum number of records to retrieve from storage.

        Returns
        -------
        list of APIKeyRecord
            Collection of API key records matching the filter.
        """
        return await self._repository.list_api_keys(status=status, limit=limit)

    async def adjust_expiration(
        self,
        lookup_hash: str,
        expires_at: Optional[datetime],
    ) -> None:
        """Update the expiration timestamp for a given API key.

        Parameters
        ----------
        lookup_hash : str
            Lookup hash for the API key to modify.
        expires_at : datetime, optional
            New absolute expiration timestamp. ``None`` removes the expiration.
        """
        await self._repository.update_expiration(
            lookup_hash,
            _normalise_datetime(expires_at),
        )
        self._cache.pop(lookup_hash, None)

    async def get_api_key(self, lookup_hash: str) -> APIKeyRecord:
        """Fetch an API key record and refresh the cache."""
        record = await self._repository.get_api_key(lookup_hash)
        self._cache_store(record)
        return record
