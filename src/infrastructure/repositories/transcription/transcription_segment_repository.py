from sqlalchemy import delete, select

from src.infrastructure.context.sql_db.psql_dbcontext import (
    PsqlDbContext,
)
from src.infrastructure.di.inject import inject
from src.infrastructure.models.transcription_segment import (
    TranscriptionSegment,
)
from src.infrastructure.repositories.base.base_repository import (
    BaseRepository,
)


@inject
class TranscriptionSegmentRepository(
    BaseRepository[TranscriptionSegment]
):
    def __init__(
        self,
        db_context: PsqlDbContext,
    ):
        super().__init__(
            db_context=db_context,
            model=TranscriptionSegment,
        )

    async def insert_many_async(
        self,
        segments: list[
            TranscriptionSegment
        ],
    ) -> list[
        TranscriptionSegment
    ]:
        if not segments:
            return []

        async with self._db_context.session() as session:
            try:
                session.add_all(
                    segments
                )

                await session.commit()

                for segment in segments:
                    await session.refresh(
                        segment
                    )

                return segments

            except Exception:
                await session.rollback()
                raise

    async def replace_for_transcription_async(
        self,
        *,
        transcription_id: str,
        segments: list[
            TranscriptionSegment
        ],
    ) -> list[
        TranscriptionSegment
    ]:
        for segment in segments:
            if (
                str(
                    segment.transcription_id
                )
                != transcription_id
            ):
                raise ValueError(
                    "All segments must belong "
                    "to the requested transcription"
                )

        async with self._db_context.session() as session:
            try:
                await session.execute(
                    delete(
                        TranscriptionSegment
                    ).where(
                        TranscriptionSegment
                        .transcription_id
                        == transcription_id
                    )
                )

                if segments:
                    session.add_all(
                        segments
                    )

                await session.commit()

                for segment in segments:
                    await session.refresh(
                        segment
                    )

                return segments

            except Exception:
                await session.rollback()
                raise

    async def update_many_async(
        self,
        segments: list[
            TranscriptionSegment
        ],
    ) -> list[
        TranscriptionSegment
    ]:
        if not segments:
            return []

        async with self._db_context.session() as session:
            try:
                merged_segments = [
                    await session.merge(
                        segment
                    )
                    for segment in segments
                ]

                await session.commit()

                for segment in merged_segments:
                    await session.refresh(
                        segment
                    )

                return merged_segments

            except Exception:
                await session.rollback()
                raise

    async def get_by_transcription_id_async(
        self,
        transcription_id: str,
    ) -> list[
        TranscriptionSegment
    ]:
        async with self._db_context.session() as session:
            statement = (
                select(
                    TranscriptionSegment
                )
                .where(
                    TranscriptionSegment
                    .transcription_id
                    == transcription_id
                )
                .order_by(
                    TranscriptionSegment
                    .sequence
                    .asc()
                )
            )

            result = await session.scalars(
                statement
            )

            return list(
                result.all()
            )