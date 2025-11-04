"""Helpers for interacting with the Firestore-backed API key store."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Optional


try:
    from google.cloud.firestore_v1 import (
        SERVER_TIMESTAMP,
        AsyncClient,
        AsyncDocumentReference,
        AsyncTransaction,
        DocumentSnapshot,
        async_transactional,
    )
except ImportError:  # pragma: no cover - imported at runtime in production
    AsyncClient = Any  # type: ignore
    AsyncDocumentReference = Any  # type: ignore
    DocumentSnapshot = Any  # type: ignore
    AsyncTransaction = Any  # type: ignore
    SERVER_TIMESTAMP = None  # type: ignore

    def async_transactional(func):  # type: ignore
        """Passthrough decorator used when Firestore is not available."""
        return func


class APIKeyNotFoundError(Exception):
    """Raised when an API key document cannot be found."""


class UsageLimitExceededError(Exception):
    """Raised when an API key has exceeded its usage limit."""


Role = Literal["admin", "user"]
Status = Literal["active", "suspended"]


def _ensure_timezone(value: Optional[datetime]) -> Optional[datetime]:
    """Return timezone-aware datetime."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


@dataclass(slots=True)
class APIKeyRecord:
    """Represents a single API key record stored in Firestore."""

    lookup_hash: str
    hashed_key: str
    salt: str
    display_prefix: str
    role: Role
    owner: Optional[str]
    status: Status
    usage_count: int
    usage_limit: int
    last_used_at: Optional[datetime]
    created_at: datetime
    created_by: str
    metadata: dict[str, Any]
    expires_at: Optional[datetime]

    @classmethod
    def from_snapshot(
        cls,
        lookup_hash: str,
        snapshot: DocumentSnapshot,
    ) -> "APIKeyRecord":
        """Create an ``APIKeyRecord`` from a Firestore document snapshot.

        Parameters
        ----------
        lookup_hash : str
            The SHA-256 digest used as the document id for fast lookups.
        snapshot : google.cloud.firestore_v1.DocumentSnaptshot
            Firestore document containing the API key record data.

        Returns
        -------
        APIKeyRecord
            Parsed record with strict defaults for missing fields.
        """
        data = snapshot.to_dict() or {}
        last_used_at = _ensure_timezone(data.get("last_used_at"))
        created_at = _ensure_timezone(data.get("created_at")) or datetime.now(
            tz=timezone.utc
        )
        expires_at = _ensure_timezone(data.get("expires_at"))
        return cls(
            lookup_hash=lookup_hash,
            hashed_key=data["hashed_key"],
            salt=data["salt"],
            display_prefix=data.get("display_prefix", ""),
            role=data.get("role", "user"),
            owner=data.get("owner"),
            status=data.get("status", "active"),
            usage_count=int(data.get("usage_count", 0)),
            usage_limit=int(data.get("usage_limit", 0)),
            last_used_at=last_used_at,
            created_at=created_at,
            created_by=data.get("created_by", "system"),
            metadata=data.get("metadata", {}),
            expires_at=expires_at,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the record to a Firestore-compatible dictionary.

        Returns
        -------
        dict of str to Any
            Mapping representing the Firestore document payload.
        """
        return {
            "hashed_key": self.hashed_key,
            "salt": self.salt,
            "display_prefix": self.display_prefix,
            "role": self.role,
            "owner": self.owner,
            "status": self.status,
            "usage_count": self.usage_count,
            "usage_limit": self.usage_limit,
            "last_used_at": self.last_used_at,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "metadata": self.metadata,
            "expires_at": self.expires_at,
        }


class APIKeyRepository:
    """Repository abstraction around the Firestore collection."""

    def __init__(self, client: AsyncClient, collection_name: str = "apiKeys") -> None:
        """Initialise the repository with a Firestore client and collection.

        Parameters
        ----------
        client : AsyncClient
            Asynchronous Firestore client bound to the desired GCP project.
        collection_name : str, default="apiKeys"
            Name of the collection storing API keys.
        """
        self._client = client
        self._collection = collection_name

    def _document(self, lookup_hash: str) -> AsyncDocumentReference:
        """Return the document reference for a given lookup hash.

        Parameters
        ----------
        lookup_hash : str
            SHA-256 digest used as a stable document identifier.

        Returns
        -------
        AsyncDocumentReference
            Document reference inside the configured collection.
        """
        return self._client.collection(self._collection).document(lookup_hash)

    async def create_api_key(self, record: APIKeyRecord) -> None:
        """Persist a new API key record.

        Parameters
        ----------
        record : APIKeyRecord
            Fully populated API key record to store.
        """
        await self._document(record.lookup_hash).set(record.to_dict())

    async def get_api_key(self, lookup_hash: str) -> APIKeyRecord:
        """Fetch an API key record by its lookup hash.

        Parameters
        ----------
        lookup_hash : str
            SHA-256 digest used to locate the API key document.

        Returns
        -------
        APIKeyRecord
            Parsed record for the requested API key.

        Raises
        ------
        APIKeyNotFoundError
            Raised when no document matches the provided lookup hash.
        """
        doc_ref = self._document(lookup_hash)
        snapshot = await doc_ref.get()
        if not snapshot.exists:
            raise APIKeyNotFoundError(lookup_hash)
        return APIKeyRecord.from_snapshot(lookup_hash, snapshot)

    async def delete_api_key(self, lookup_hash: str) -> None:
        """Delete an API key document.

        Parameters
        ----------
        lookup_hash : str
            SHA-256 digest corresponding to the key to delete.
        """
        await self._document(lookup_hash).delete()

    async def list_api_keys(
        self,
        *,
        status: Optional[Status] = None,
        limit: int = 100,
    ) -> list[APIKeyRecord]:
        """List API keys, optionally filtered by status.

        Parameters
        ----------
        status : Status, optional
            Optional status filter. When omitted all keys are returned.
        limit : int, default=100
            Maximum number of records to return.

        Returns
        -------
        list of APIKeyRecord
            Collection of API key records.
        """
        query = self._client.collection(self._collection)
        if status:
            query = query.where("status", "==", status)
        query = query.limit(limit)

        records: list[APIKeyRecord] = []
        async for snapshot in query.stream():
            records.append(APIKeyRecord.from_snapshot(snapshot.id, snapshot))
        return records

    async def update_usage_counter(self, lookup_hash: str) -> APIKeyRecord:
        """Atomically increment usage count for an API key.

        The operation ensures that the usage limit is never exceeded by
        reading the current value inside a Firestore transaction before
        applying the increment.

        Parameters
        ----------
        lookup_hash : str
            SHA-256 digest corresponding to the key to update.

        Returns
        -------
        APIKeyRecord
            The API key record containing the updated usage counter.

        Raises
        ------
        APIKeyNotFoundError
            Raised when no document matches ``lookup_hash``.
        UsageLimitExceededError
            Raised when the increment would exceed the configured limit.
        """
        doc_ref = self._document(lookup_hash)

        @async_transactional
        async def _increment(
            transaction: AsyncTransaction,
            reference: AsyncDocumentReference,
        ) -> APIKeyRecord:
            snapshot = await reference.get(transaction=transaction)
            if not snapshot.exists:
                raise APIKeyNotFoundError(lookup_hash)

            data: dict[str, Any] = snapshot.to_dict() or {}
            usage_count = int(data.get("usage_count", 0))
            usage_limit = int(data.get("usage_limit", 0))

            # Reject requests that would exceed the assigned quota.
            if usage_limit and usage_count >= usage_limit:
                raise UsageLimitExceededError(lookup_hash)

            server_timestamp = SERVER_TIMESTAMP or datetime.now(tz=timezone.utc)  # type: ignore[arg-type]

            transaction.update(
                reference,
                {"usage_count": usage_count + 1, "last_used_at": server_timestamp},
            )

            return APIKeyRecord(
                lookup_hash=lookup_hash,
                hashed_key=data["hashed_key"],
                salt=data["salt"],
                display_prefix=data.get("display_prefix", ""),
                role=data.get("role", "user"),
                owner=data.get("owner"),
                status=data.get("status", "active"),
                usage_count=usage_count + 1,
                usage_limit=usage_limit,
                last_used_at=datetime.now(tz=timezone.utc),
                created_at=_ensure_timezone(data.get("created_at"))
                or datetime.now(tz=timezone.utc),
                created_by=data.get("created_by", "system"),
                metadata=data.get("metadata", {}),
                expires_at=_ensure_timezone(data.get("expires_at")),
            )

        return await _increment(self._client.transaction(), doc_ref)

    async def set_status(self, lookup_hash: str, status: Status) -> None:
        """Update the ``status`` field for an API key record.

        Parameters
        ----------
        lookup_hash : str
            SHA-256 digest corresponding to the key to update.
        status : Status
            New status string to persist.
        """
        await self._document(lookup_hash).update({"status": status})

    async def update_usage_limit(self, lookup_hash: str, usage_limit: int) -> None:
        """Update the ``usage_limit`` for an API key record.

        Parameters
        ----------
        lookup_hash : str
            SHA-256 digest corresponding to the key to update.
        usage_limit : int
            New usage limit to store. ``0`` indicates unlimited usage.
        """
        await self._document(lookup_hash).update({"usage_limit": usage_limit})

    async def update_expiration(
        self,
        lookup_hash: str,
        expires_at: Optional[datetime],
    ) -> None:
        """Update the ``expires_at`` timestamp for an API key record.

        Parameters
        ----------
        lookup_hash : str
            SHA-256 digest corresponding to the key to update.
        expires_at : datetime or None
            When provided, absolute expiration timestamp stored in UTC. ``None``
            removes the expiration deadline.
        """
        await self._document(lookup_hash).update({"expires_at": expires_at})
