from fastapi import (
    APIRouter,
    HTTPException,
    status,
)

from src.application.health_check.dtos.health_check_dto import (
    HealthCheckResponseDto,
    ReadinessResponseDto,
)
from src.core.services.health_check.health_check_service import (
    HealthCheckService,
)
from src.infrastructure.di.inject import inject


@inject
class HealthCheckController:
    def __init__(
        self,
        health_check_service: HealthCheckService,
    ) -> None:
        self._health_check_service = (
            health_check_service
        )

    def api(self) -> APIRouter:
        router = APIRouter(
            prefix="",
            tags=["Health Check"],
        )

        @router.get(
            "/",
            response_model=HealthCheckResponseDto,
            status_code=status.HTTP_200_OK,
            summary="Service liveness",
        )
        async def get_liveness(
        ) -> HealthCheckResponseDto:
            result = (
                self._health_check_service
                .get_liveness()
            )

            return HealthCheckResponseDto(
                **result
            )

        @router.get(
            "/ready",
            response_model=ReadinessResponseDto,
            status_code=status.HTTP_200_OK,
            summary="Service readiness",
        )
        async def get_readiness(
        ) -> ReadinessResponseDto:
            database_is_ready = await (
                self._health_check_service
                .check_database_async()
            )

            if not database_is_ready:
                raise HTTPException(
                    status_code=(
                        status.HTTP_503_SERVICE_UNAVAILABLE
                    ),
                    detail="PostgreSQL is not ready.",
                )

            return ReadinessResponseDto(
                status="ok",
                database="ok",
            )

        return router