"""Track daily usage for Gemini models to account for free-tier allowances."""

import asyncio
import os
import random
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Callable, Optional


try:
    from google.api_core.exceptions import Aborted
    from google.cloud.firestore_v1 import (
        SERVER_TIMESTAMP,
        AsyncClient,
        AsyncDocumentReference,
        AsyncTransaction,
        async_transactional,
    )
except ImportError:  # pragma: no cover - imported dynamically in production
    Aborted = RuntimeError  # type: ignore
    AsyncClient = Any  # type: ignore
    AsyncDocumentReference = Any  # type: ignore
    AsyncTransaction = Any  # type: ignore
    SERVER_TIMESTAMP = None  # type: ignore

    def async_transactional(func):  # type: ignore
        """Passthrough decorator used when Firestore is not installed."""
        return func


DAILY_USAGE_COLLECTION = os.getenv("DAILY_USAGE_COLLECTION", "dailyUsageCounters")
DAILY_USAGE_MAX_RETRIES = int(os.getenv("DAILY_USAGE_MAX_RETRIES", "8"))
DAILY_USAGE_BASE_DELAY = float(os.getenv("DAILY_USAGE_BASE_DELAY", "0.05"))
DAILY_USAGE_MAX_DELAY = float(os.getenv("DAILY_USAGE_MAX_DELAY", "1.0"))


def _now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(tz=timezone.utc)


def _retry_delay(attempt: int) -> float:
    """Calculate a jittered exponential backoff for retries."""
    delay = DAILY_USAGE_BASE_DELAY * (2**attempt)
    capped = min(delay, DAILY_USAGE_MAX_DELAY)
    if capped <= 0:
        return 0.0
    jitter = random.uniform(0, capped / 2)
    return capped + jitter


@dataclass(slots=True)
class UsageReservation:
    """Represents a single reserved request in the daily counter."""

    bucket: str
    day: date
    consumed_free: bool


def _ensure_utc(value: Optional[datetime]) -> Optional[datetime]:
    """Return a timezone-aware UTC datetime."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class DailyUsageRepository:
    """Persist and update daily usage counters for Gemini buckets."""

    def __init__(
        self,
        client: AsyncClient,
        *,
        collection_name: str = DAILY_USAGE_COLLECTION,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        """Initialise the repository with a Firestore client."""
        self._client = client
        self._collection = collection_name
        self._clock = clock

    def _document(self, bucket: str, day: date) -> AsyncDocumentReference:
        """Return the document reference for the given bucket/day pair."""
        identifier = f"{bucket}:{day.isoformat()}"
        return self._client.collection(self._collection).document(identifier)

    async def reserve(self, bucket: str, free_limit: int) -> UsageReservation:
        """Reserve a usage slot for the given bucket.

        Parameters
        ----------
        bucket : str
            Logical identifier grouping the models that share a free allowance.
        free_limit : int
            Daily number of free requests for this bucket. ``0`` disables the
            free tier so every call should fall through to API-key accounting.

        Returns
        -------
        UsageReservation
            Reservation describing whether the free allowance was consumed.
        """
        free_limit = max(free_limit, 0)
        today = self._clock().date()
        doc_ref = self._document(bucket, today)

        @async_transactional
        async def _increment(
            transaction: AsyncTransaction,
            reference: AsyncDocumentReference,
        ) -> UsageReservation:
            snapshot = await reference.get(transaction=transaction)
            current_total = 0
            if snapshot.exists:
                data: dict[str, Any] = snapshot.to_dict() or {}
                current_total = int(data.get("total_count", 0))

                transaction.update(
                    reference,
                    {
                        "total_count": current_total + 1,
                        "updated_at": SERVER_TIMESTAMP or _ensure_utc(self._clock()),
                    },
                )
            else:
                transaction.set(
                    reference,
                    {
                        "bucket": bucket,
                        "date": today.isoformat(),
                        "total_count": 1,
                        "created_at": SERVER_TIMESTAMP or _ensure_utc(self._clock()),
                        "updated_at": SERVER_TIMESTAMP or _ensure_utc(self._clock()),
                    },
                )

            consumed_free = free_limit > 0 and current_total < free_limit

            return UsageReservation(
                bucket=bucket, day=today, consumed_free=consumed_free
            )

        attempts = 0
        while True:
            try:
                transaction = self._client.transaction()
                return await _increment(transaction, doc_ref)
            except (Aborted, ValueError):
                if attempts >= DAILY_USAGE_MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(_retry_delay(attempts))
                attempts += 1

    async def release(self, reservation: UsageReservation) -> None:
        """Rollback a reservation when the downstream call fails."""
        doc_ref = self._document(reservation.bucket, reservation.day)

        @async_transactional
        async def _decrement(
            transaction: AsyncTransaction,
            reference: AsyncDocumentReference,
        ) -> None:
            snapshot = await reference.get(transaction=transaction)
            if not snapshot.exists:
                return

            data: dict[str, Any] = snapshot.to_dict() or {}
            current_total = int(data.get("total_count", 0))
            new_total = max(current_total - 1, 0)

            transaction.update(
                reference,
                {
                    "total_count": new_total,
                    "updated_at": SERVER_TIMESTAMP or _ensure_utc(self._clock()),
                },
            )

        attempts = 0
        while True:
            try:
                transaction = self._client.transaction()
                await _decrement(transaction, doc_ref)
                return
            except (Aborted, ValueError):
                if attempts >= DAILY_USAGE_MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(_retry_delay(attempts))
                attempts += 1
