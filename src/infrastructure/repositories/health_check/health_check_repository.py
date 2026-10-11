from __future__ import annotations

from sqlalchemy import text

from src.infrastructure.context.sql_db.psql_dbcontext import (
    PsqlDbContext,
)
from src.infrastructure.di.inject import inject


@inject
class HealthCheckRepository:
    def __init__(
        self,
        db_context: PsqlDbContext,
    ) -> None:
        self._db_context = db_context

    async def check_database_async(self) -> bool:
        try:
            async with self._db_context.session() as session:
                await session.execute(
                    text("SELECT 1")
                )

            return True

        except Exception:
            return False