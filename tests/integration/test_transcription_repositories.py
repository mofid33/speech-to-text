from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.infrastructure.models.transcription import (
    Transcription,
    TranscriptionStatus,
)
from src.infrastructure.models.transcription_segment import (
    TranscriptionSegment,
)
from src.infrastructure.repositories.transcription.transcription_repository import (
    TranscriptionRepository,
)
from src.infrastructure.repositories.transcription.transcription_segment_repository import (
    TranscriptionSegmentRepository,
)
from src.infrastructure.utils.config_reader import (
    ConfigReader,
)


class IntegrationDbContext:
    def __init__(
        self,
        database_url: str,
    ):
        self._engine = create_async_engine(
            database_url,
            pool_pre_ping=True,
        )

        self._session_factory = (
            async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
        )

    def session(
        self,
    ) -> AsyncSession:
        return self._session_factory()

    async def dispose_async(
        self,
    ) -> None:
        await self._engine.dispose()


def _normalize_database_url(
    database_url: str,
) -> str:
    if database_url.startswith(
        "postgresql://"
    ):
        return database_url.replace(
            "postgresql://",
            "postgresql+asyncpg://",
            1,
        )

    return database_url


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def db_context(
) -> AsyncIterator[IntegrationDbContext]:
    database_url = ConfigReader().get(
        "database.url"
    )

    if (
        not database_url
        or not isinstance(
            database_url,
            str,
        )
    ):
        raise RuntimeError(
            "Configuration key "
            "'database.url' "
            "is missing or invalid"
        )

    context = IntegrationDbContext(
        _normalize_database_url(
            database_url
        )
    )

    try:
        yield context
    finally:
        await context.dispose_async()


@pytest.mark.anyio
async def test_transcription_repository_crud(
    db_context: IntegrationDbContext,
) -> None:
    repository = TranscriptionRepository(
        db_context=db_context
    )

    transcription = Transcription(
        status=(
            TranscriptionStatus
            .CREATED
            .value
        ),
        requested_language="fa",
    )

    inserted = await repository.insert_async(
        transcription
    )

    assert inserted.id is not None

    assert (
        inserted.status
        == TranscriptionStatus.CREATED.value
    )

    loaded = await repository.get_by_id_async(
        inserted.id
    )

    assert loaded is not None
    assert loaded.id == inserted.id

    loaded.status = (
        TranscriptionStatus
        .UPLOADED
        .value
    )

    updated = await repository.update_async(
        loaded
    )

    assert (
        updated.status
        == TranscriptionStatus.UPLOADED.value
    )

    deleted = await repository.delete_async(
        inserted.id
    )

    assert deleted is True

    assert (
        await repository.get_by_id_async(
            inserted.id
        )
        is None
    )


@pytest.mark.anyio
async def test_transcription_segment_repository_orders_and_cascades(
    db_context: IntegrationDbContext,
) -> None:
    transcription_repository = (
        TranscriptionRepository(
            db_context=db_context
        )
    )

    segment_repository = (
        TranscriptionSegmentRepository(
            db_context=db_context
        )
    )

    transcription = (
        await transcription_repository.insert_async(
            Transcription(
                status=(
                    TranscriptionStatus
                    .CREATED
                    .value
                ),
                requested_language="fa",
            )
        )
    )

    await segment_repository.insert_async(
        TranscriptionSegment(
            transcription_id=transcription.id,
            sequence=1,
            start_time=2.5,
            end_time=5.0,
            raw_text="بخش دوم",
        )
    )

    await segment_repository.insert_async(
        TranscriptionSegment(
            transcription_id=transcription.id,
            sequence=0,
            start_time=0.0,
            end_time=2.5,
            raw_text="بخش اول",
        )
    )

    segments = (
        await segment_repository
        .get_by_transcription_id_async(
            transcription.id
        )
    )

    assert [
        segment.sequence
        for segment in segments
    ] == [
        0,
        1,
    ]

    assert [
        segment.raw_text
        for segment in segments
    ] == [
        "بخش اول",
        "بخش دوم",
    ]

    assert (
        await transcription_repository
        .delete_async(
            transcription.id
        )
        is True
    )

    assert (
        await segment_repository
        .get_by_transcription_id_async(
            transcription.id
        )
        == []
    )