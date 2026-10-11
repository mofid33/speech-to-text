from __future__ import annotations

import importlib
import logging
import re
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import AsyncIterator

from fastapi import (
    APIRouter,
    FastAPI,
    Request,
    Response,
)

from src.infrastructure.context.sql_db.psql_dbcontext import (
    PsqlDbContext,
)
from src.infrastructure.di.inject import inject
from src.infrastructure.observability.telemetry import (
    extract_transcription_id_from_path,
    log_event,
    observe_api_request,
    prometheus_content_type,
    render_prometheus_metrics,
    transcription_context,
)
from src.infrastructure.utils.config_reader import (
    ConfigReader,
)


_LOGGER = logging.getLogger(
    "speech_to_text.api"
)

_TRANSCRIPTION_ROUTE_PARAMETER_PATTERN = re.compile(
    r"(?<=/transcription/)[^/]+",
    re.IGNORECASE,
)


@inject
class WebService:
    def __init__(
        self,
        config_reader: ConfigReader,
        db_context: PsqlDbContext,
    ) -> None:
        self._config_reader = config_reader
        self._db_context = db_context

        self._metrics_enabled = (
            self._read_bool(
                self._config_reader.get(
                    (
                        "observability."
                        "api_metrics.enabled"
                    ),
                    True,
                )
            )
        )

        self._metrics_path = str(
            self._config_reader.get(
                (
                    "observability."
                    "api_metrics.path"
                ),
                "/metrics",
            )
        ).strip()

        if (
            not self._metrics_path
            or not self._metrics_path
            .startswith("/")
        ):
            raise ValueError(
                "observability.api_metrics.path "
                "must start with '/'"
            )

        self._app = FastAPI(
            title=(
                self._config_reader
                .get_app_name()
            ),
            version=(
                self._config_reader
                .get_app_version()
            ),
            lifespan=self._lifespan,
        )

        self._app.middleware(
            "http"
        )(
            self._observe_request_async
        )

        if self._metrics_enabled:
            self._register_metrics_route()

        self._discover_controllers()

    @asynccontextmanager
    async def _lifespan(
        self,
        _: FastAPI,
    ) -> AsyncIterator[None]:
        yield

        await (
            self._db_context
            .dispose_async()
        )

    async def _observe_request_async(
        self,
        request: Request,
        call_next: Callable[
            [Request],
            Awaitable[Response],
        ],
    ) -> Response:
        if (
            self._metrics_enabled
            and request.url.path
            == self._metrics_path
        ):
            return await call_next(
                request
            )

        started_at = perf_counter()
        status_code = 500

        transcription_id = (
            extract_transcription_id_from_path(
                request.url.path
            )
        )

        with transcription_context(
            transcription_id
        ):
            try:
                response = (
                    await call_next(
                        request
                    )
                )

                status_code = (
                    response.status_code
                )

                return response

            finally:
                duration_seconds = (
                    perf_counter()
                    - started_at
                )

                route = (
                    self._resolve_route_template(
                        request
                    )
                )

                observe_api_request(
                    method=request.method,
                    route=route,
                    status_code=status_code,
                    duration_seconds=(
                        duration_seconds
                    ),
                )

                log_level = logging.INFO

                if status_code >= 500:
                    log_level = logging.ERROR

                elif status_code >= 400:
                    log_level = logging.WARNING

                log_event(
                    _LOGGER,
                    level=log_level,
                    event=(
                        "api_request_completed"
                    ),
                    message=(
                        "API request completed"
                    ),
                    method=request.method,
                    route=route,
                    status_code=(
                        status_code
                    ),
                    duration_seconds=round(
                        duration_seconds,
                        6,
                    ),
                )

    def _register_metrics_route(
        self,
    ) -> None:
        async def metrics() -> Response:
            return Response(
                content=(
                    render_prometheus_metrics()
                ),
                media_type=(
                    prometheus_content_type()
                ),
            )

        self._app.add_api_route(
            self._metrics_path,
            metrics,
            methods=[
                "GET"
            ],
            include_in_schema=False,
            name="prometheus_metrics",
        )

    def _discover_controllers(
        self,
    ) -> None:
        application_directory = (
            Path(__file__)
            .resolve()
            .parent
        )

        api_prefix = (
            self._config_reader
            .get_api_prefix()
        )

        for feature_directory in sorted(
            application_directory
            .iterdir()
        ):
            if (
                not feature_directory
                .is_dir()
            ):
                continue

            feature_name = (
                feature_directory.name
            )

            if (
                feature_name
                .startswith("_")
                or feature_name
                == "__pycache__"
            ):
                continue

            controller_file = (
                feature_directory
                / (
                    f"{feature_name}"
                    "_controller.py"
                )
            )

            if (
                not controller_file
                .is_file()
            ):
                continue

            module_name = (
                "src.application."
                f"{feature_name}."
                f"{feature_name}"
                "_controller"
            )

            module = (
                importlib.import_module(
                    module_name
                )
            )

            class_name = (
                "".join(
                    part.capitalize()
                    for part
                    in feature_name
                    .split("_")
                )
                + "Controller"
            )

            controller_class = getattr(
                module,
                class_name,
                None,
            )

            if controller_class is None:
                raise RuntimeError(
                    "Controller class "
                    f"'{class_name}' "
                    "was not found in "
                    f"'{module_name}'."
                )

            controller = (
                controller_class()
            )

            router = (
                controller.api()
            )

            if not isinstance(
                router,
                APIRouter,
            ):
                raise TypeError(
                    f"'{class_name}.api()' "
                    "must return "
                    "fastapi.APIRouter."
                )

            self._app.include_router(
                router,
                prefix=(
                    f"{api_prefix}/"
                    f"{feature_name}"
                ),
            )

    def get_app(
        self,
    ) -> FastAPI:
        return self._app

    @staticmethod
    def _resolve_route_template(
        request: Request,
    ) -> str:
        route_path = str(
            request.url.path
        )

        return (
            _TRANSCRIPTION_ROUTE_PARAMETER_PATTERN
            .sub(
                "{transcription_id}",
                route_path,
                count=1,
            )
        )

    @staticmethod
    def _read_bool(
        raw_value: object,
    ) -> bool:
        if isinstance(
            raw_value,
            bool,
        ):
            return raw_value

        raise ValueError(
            "observability."
            "api_metrics.enabled "
            "must be a boolean"
        )