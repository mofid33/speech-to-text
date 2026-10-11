from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from src.infrastructure.di.inject import inject
from src.infrastructure.utils.config_reader import (
    ConfigReader,
)


class Base(DeclarativeBase):
    pass


@inject
class PsqlDbContext:
    __di_singleton__ = True

    def __init__(
        self,
        config_reader: ConfigReader,
    ) -> None:
        database_url = (
            config_reader.get_database_url()
        )

        self._engine: AsyncEngine = (
            create_async_engine(
                database_url,
                pool_pre_ping=True,
            )
        )

        self._session_factory = (
            async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
        )

    def session(self) -> AsyncSession:
        return self._session_factory()

    async def dispose_async(self) -> None:
        await self._engine.dispose()