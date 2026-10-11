from typing import Generic, TypeVar

from sqlalchemy import select

from src.infrastructure.context.sql_db.psql_dbcontext import PsqlDbContext


TModel = TypeVar("TModel")


class BaseRepository(Generic[TModel]):
    def __init__(
        self,
        db_context: PsqlDbContext,
        model: type[TModel],
    ):
        self._db_context = db_context
        self._model = model

    async def insert_async(
        self,
        entity: TModel,
    ) -> TModel:
        async with self._db_context.session() as session:
            try:
                session.add(entity)

                await session.commit()
                await session.refresh(entity)

                return entity
            except Exception:
                await session.rollback()
                raise

    async def insert_and_get_id_async(
        self,
        entity: TModel,
    ) -> str:
        inserted_entity = await self.insert_async(entity)

        entity_id = getattr(
            inserted_entity,
            "id",
            None,
        )

        if entity_id is None:
            raise AttributeError(
                f"{self._model.__name__} does not expose an 'id' attribute"
            )

        return str(entity_id)

    async def get_by_id_async(
        self,
        entity_id: str,
    ) -> TModel | None:
        async with self._db_context.session() as session:
            return await session.get(
                self._model,
                entity_id,
            )

    async def get_all_async(
        self,
    ) -> list[TModel]:
        async with self._db_context.session() as session:
            result = await session.scalars(
                select(self._model)
            )

            return list(result.all())

    async def update_async(
        self,
        entity: TModel,
    ) -> TModel:
        async with self._db_context.session() as session:
            try:
                merged_entity = await session.merge(entity)

                await session.commit()
                await session.refresh(merged_entity)

                return merged_entity
            except Exception:
                await session.rollback()
                raise

    async def delete_async(
        self,
        entity_id: str,
    ) -> bool:
        async with self._db_context.session() as session:
            try:
                entity = await session.get(
                    self._model,
                    entity_id,
                )

                if entity is None:
                    return False

                await session.delete(entity)
                await session.commit()

                return True
            except Exception:
                await session.rollback()
                raise