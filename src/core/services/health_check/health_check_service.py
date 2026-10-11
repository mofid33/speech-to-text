from __future__ import annotations

from typing import Any

from src.infrastructure.di.inject import inject
from src.infrastructure.repositories.health_check.health_check_repository import (
    HealthCheckRepository,
)
from src.infrastructure.utils.config_reader import (
    ConfigReader,
)


@inject
class HealthCheckService:
    def __init__(
        self,
        config_reader: ConfigReader,
        health_check_repository: HealthCheckRepository,
    ) -> None:
        self._config_reader = config_reader
        self._health_check_repository = (
            health_check_repository
        )

    def get_liveness(
        self,
    ) -> dict[str, Any]:
        return {
            "status": "ok",
            "service": self._config_reader.get_app_name(),
            "version": self._config_reader.get_app_version(),
            "environment": (
                self._config_reader.get_environment()
            ),
        }

    async def check_database_async(
        self,
    ) -> bool:
        return await (
            self._health_check_repository
            .check_database_async()
        )