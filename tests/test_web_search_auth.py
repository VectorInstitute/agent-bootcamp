"""Unit tests for the Gemini grounding proxy authentication helpers."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from src.utils.web_search import auth
from src.utils.web_search.db import (
    APIKeyNotFoundError,
    APIKeyRecord,
    Status,
    UsageLimitExceededError,
)


class FakeRepository:
    """In-memory stand-in for ``APIKeyRepository`` used in tests."""

    def __init__(self) -> None:
        """Initialise the repository with an empty in-memory store."""
        self.records: dict[str, APIKeyRecord] = {}

    async def create_api_key(self, record: APIKeyRecord) -> None:
        """Create API key."""
        self.records[record.lookup_hash] = record

    async def get_api_key(self, lookup_hash: str) -> APIKeyRecord:
        """Get API key."""
        try:
            return self.records[lookup_hash]
        except KeyError as exc:
            raise APIKeyNotFoundError(lookup_hash) from exc

    async def delete_api_key(self, lookup_hash: str) -> None:
        """Delete API key."""
        self.records.pop(lookup_hash, None)

    async def list_api_keys(
        self,
        *,
        status: Status | None = None,
        limit: int = 100,
    ) -> list[APIKeyRecord]:
        """List API keys."""
        records = list[APIKeyRecord](self.records.values())
        if status:
            records = [record for record in records if record.status == status]
        return records[:limit]

    async def update_usage_counter(self, lookup_hash: str) -> APIKeyRecord:
        """Update usage counter."""
        if lookup_hash not in self.records:
            raise APIKeyNotFoundError(lookup_hash)

        record = self.records[lookup_hash]
        if record.usage_limit and record.usage_count >= record.usage_limit:
            raise UsageLimitExceededError(lookup_hash)

        updated = replace(
            record,
            usage_count=record.usage_count + 1,
            last_used_at=datetime.now(tz=timezone.utc),
        )
        self.records[lookup_hash] = updated
        return updated

    async def decrement_usage_counter(self, lookup_hash: str) -> APIKeyRecord:
        """Decrement usage counter."""
        if lookup_hash not in self.records:
            raise APIKeyNotFoundError(lookup_hash)

        record = self.records[lookup_hash]
        new_count = max(record.usage_count - 1, 0)
        updated = replace(record, usage_count=new_count)
        self.records[lookup_hash] = updated
        return updated

    async def set_status(self, lookup_hash: str, status: Status) -> None:
        """Set status."""
        record = self.records[lookup_hash]
        self.records[lookup_hash] = replace(record, status=status)

    async def update_usage_limit(self, lookup_hash: str, usage_limit: int) -> None:
        """Update usage limit."""
        record = self.records[lookup_hash]
        self.records[lookup_hash] = replace(record, usage_limit=usage_limit)

    async def update_expiration(
        self,
        lookup_hash: str,
        expires_at: Optional[datetime],
    ) -> None:
        """Update expiration."""
        record = self.records[lookup_hash]
        self.records[lookup_hash] = replace(record, expires_at=expires_at)


def fixed_clock() -> datetime:
    """Return a deterministic timestamp for testing."""
    return datetime(2025, 1, 1, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_reserve_usage_increments_counter() -> None:
    """Ensure usage counts increment after reserving a request."""
    repository = FakeRepository()
    authenticator = auth.APIKeyAuthenticator(repository, clock=fixed_clock)

    api_key, record = await authenticator.create_api_key(
        role="user",
        owner="owner-1",
        usage_limit=5,
        created_by="admin",
    )

    updated_record = await authenticator.reserve_usage(api_key)

    assert updated_record.usage_count == 1
    assert repository.records[record.lookup_hash].usage_count == 1


@pytest.mark.asyncio
async def test_release_usage_rolls_back_counter() -> None:
    """Ensure usage reservations can be rolled back after failures."""
    repository = FakeRepository()
    authenticator = auth.APIKeyAuthenticator(repository, clock=fixed_clock)

    api_key, record = await authenticator.create_api_key(
        role="user",
        owner="owner-rollback",
        usage_limit=5,
        created_by="admin",
    )

    await authenticator.reserve_usage(api_key)
    await authenticator.release_usage(record.lookup_hash)

    assert repository.records[record.lookup_hash].usage_count == 0


@pytest.mark.asyncio
async def test_reserve_usage_without_consuming() -> None:
    """Ensure validation can occur without incrementing the counter."""
    repository = FakeRepository()
    authenticator = auth.APIKeyAuthenticator(repository, clock=fixed_clock)

    api_key, record = await authenticator.create_api_key(
        role="admin",
        owner="owner-no-consume",
        usage_limit=10,
        created_by="admin",
    )

    returned_record = await authenticator.reserve_usage(
        api_key,
        consume_usage=False,
    )

    assert returned_record.usage_count == 0
    assert repository.records[record.lookup_hash].usage_count == 0


@pytest.mark.asyncio
async def test_reserve_usage_rejects_invalid_key() -> None:
    """Ensure invalid API keys raise ``InvalidAPIKeyError``."""
    repository = FakeRepository()
    authenticator = auth.APIKeyAuthenticator(repository, clock=fixed_clock)

    await authenticator.create_api_key(
        role="user",
        owner=None,
        usage_limit=0,
        created_by="admin",
    )

    with pytest.raises(auth.InvalidAPIKeyError):
        await authenticator.reserve_usage("invalid-key")


@pytest.mark.asyncio
async def test_reserve_usage_respects_status() -> None:
    """Ensure suspended keys cannot be used."""
    repository = FakeRepository()
    authenticator = auth.APIKeyAuthenticator(repository, clock=fixed_clock)

    api_key, record = await authenticator.create_api_key(
        role="user",
        owner=None,
        usage_limit=1,
        created_by="admin",
    )

    await authenticator.deactivate(record.lookup_hash)

    with pytest.raises(auth.InactiveAPIKeyError):
        await authenticator.reserve_usage(api_key)


@pytest.mark.asyncio
async def test_reserve_usage_respects_usage_limit() -> None:
    """Ensure usage limits are enforced."""
    repository = FakeRepository()
    authenticator = auth.APIKeyAuthenticator(repository, clock=fixed_clock)

    api_key, _ = await authenticator.create_api_key(
        role="user",
        owner=None,
        usage_limit=1,
        created_by="admin",
    )

    await authenticator.reserve_usage(api_key)

    with pytest.raises(UsageLimitExceededError):
        await authenticator.reserve_usage(api_key)


@pytest.mark.asyncio
async def test_reserve_usage_rejects_expired_key() -> None:
    """Ensure expired keys are rejected even before Firestore TTL cleanup."""
    repository = FakeRepository()
    authenticator = auth.APIKeyAuthenticator(repository, clock=fixed_clock)

    expires_at = fixed_clock() - timedelta(seconds=1)
    api_key, record = await authenticator.create_api_key(
        role="user",
        owner=None,
        usage_limit=5,
        created_by="admin",
        expires_at=expires_at,
    )

    repository.records[record.lookup_hash] = replace(
        repository.records[record.lookup_hash],
        expires_at=expires_at,
    )

    with pytest.raises(auth.ExpiredAPIKeyError):
        await authenticator.reserve_usage(api_key)


@pytest.mark.asyncio
async def test_adjust_expiration_updates_repository() -> None:
    """Ensure adjusting expiration propagates to the backing store."""
    repository = FakeRepository()
    authenticator = auth.APIKeyAuthenticator(repository, clock=fixed_clock)

    api_key, record = await authenticator.create_api_key(
        role="admin",
        owner="owner-expire",
        usage_limit=0,
        created_by="admin",
    )

    new_expiry = datetime(2025, 1, 2, tzinfo=timezone.utc)
    await authenticator.adjust_expiration(record.lookup_hash, new_expiry)

    stored = repository.records[record.lookup_hash]
    assert stored.expires_at == new_expiry

    await authenticator.adjust_expiration(record.lookup_hash, None)
    assert repository.records[record.lookup_hash].expires_at is None

    assert api_key  # avoid unused variable warning


@pytest.mark.asyncio
async def test_create_api_key_persists_metadata_and_expiry() -> None:
    """Ensure metadata and expiration are stored and retrievable."""
    repository = FakeRepository()
    authenticator = auth.APIKeyAuthenticator(repository, clock=fixed_clock)

    expiry = fixed_clock() + timedelta(days=7)
    api_key, record = await authenticator.create_api_key(
        role="admin",
        owner="owner-2",
        usage_limit=0,
        created_by="seed-admin",
        metadata={"team": "platform"},
        expires_at=expiry,
    )

    lookup_hash = auth.derive_lookup_hash(api_key)
    stored = repository.records[lookup_hash]

    assert stored.metadata == {"team": "platform"}
    assert stored.expires_at == expiry
    assert record.display_prefix == api_key[:8]


def test_hash_round_trip() -> None:
    """Ensure hashing and verification behave as expected."""
    api_key = "test-key"
    salt = auth.generate_salt()
    hashed = auth.hash_api_key(api_key, salt)

    assert auth.verify_api_key(api_key, salt, hashed)
    assert not auth.verify_api_key(api_key + "x", salt, hashed)
