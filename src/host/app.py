from fastapi import FastAPI

from src.application.web import WebService
from src.infrastructure.di.bootstrap import (
    bootstrap_di,
)
from src.infrastructure.di.inject import resolve
from src.infrastructure.observability.telemetry import (
    configure_structured_logging,
)
from src.infrastructure.utils.config_reader import (
    ConfigReader,
)


bootstrap_di()

_config_reader = resolve(
    ConfigReader
)

configure_structured_logging(
    log_level=str(
        _config_reader.get(
            "observability.log_level",
            "INFO",
        )
    )
)

_web_service = resolve(
    WebService
)

app: FastAPI = (
    _web_service.get_app()
)