from src.infrastructure.context.sql_db.psql_dbcontext import PsqlDbContext
from src.infrastructure.di.inject import inject
from src.infrastructure.models.transcription import Transcription
from src.infrastructure.repositories.base.base_repository import BaseRepository


@inject
class TranscriptionRepository(
    BaseRepository[Transcription]
):
    def __init__(
        self,
        db_context: PsqlDbContext,
    ):
        super().__init__(
            db_context=db_context,
            model=Transcription,
        )