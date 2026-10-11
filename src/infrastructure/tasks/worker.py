import logging
import sys

from src.core.services.transcription.transcription_processing_service import (
    TranscriptionProcessingService,
)
from src.infrastructure.di.bootstrap import (
    bootstrap_di,
)
from src.infrastructure.di.inject import (
    resolve,
)
from src.infrastructure.observability.telemetry import (
    configure_structured_logging,
    log_event,
    start_metrics_http_server,
)
from src.infrastructure.providers.queue.celery_queue_provider import (
    CeleryQueueProvider,
)
from src.infrastructure.tasks.transcription_task import (
    register_transcription_task,
)
from src.infrastructure.utils.config_reader import (
    ConfigReader,
)


_LOGGER = logging.getLogger(
    "speech_to_text.worker"
)


def main() -> None:
    bootstrap_di()

    config_reader = resolve(
        ConfigReader
    )

    configure_structured_logging(
        log_level=str(
            config_reader.get(
                "observability.log_level",
                "INFO",
            )
        )
    )

    worker_metrics_enabled = (
        _read_bool(
            config_reader.get(
                "observability."
                "worker_metrics.enabled",
                True,
            ),
            key=(
                "observability."
                "worker_metrics.enabled"
            ),
        )
    )

    if worker_metrics_enabled:
        metrics_host = str(
            config_reader.get(
                "observability."
                "worker_metrics.host",
                "127.0.0.1",
            )
        ).strip()

        metrics_port = (
            _read_port(
                config_reader.get(
                    "observability."
                    "worker_metrics.port",
                    9101,
                ),
                key=(
                    "observability."
                    "worker_metrics.port"
                ),
            )
        )

        if not metrics_host:
            raise ValueError(
                "observability."
                "worker_metrics.host "
                "must not be empty"
            )

        start_metrics_http_server(
            host=metrics_host,
            port=metrics_port,
        )

        log_event(
            _LOGGER,
            level=logging.INFO,
            event=(
                "worker_metrics_started"
            ),
            message=(
                "Worker metrics "
                "endpoint started"
            ),
            host=metrics_host,
            port=metrics_port,
        )

    celery_queue_provider = resolve(
        CeleryQueueProvider
    )

    transcription_processing_service = (
        resolve(
            TranscriptionProcessingService
        )
    )

    register_transcription_task(
        celery_app=(
            celery_queue_provider
            .celery_app
        ),
        transcription_processing_service=(
            transcription_processing_service
        ),
    )

    worker_arguments = [
        "worker",
        "-Q",
        celery_queue_provider
        .queue_name,
        *sys.argv[1:],
    ]

    log_event(
        _LOGGER,
        level=logging.INFO,
        event="worker_starting",
        message=(
            "Transcription Worker "
            "starting"
        ),
        queue=(
            celery_queue_provider
            .queue_name
        ),
    )

    celery_queue_provider.celery_app.worker_main(
        worker_arguments
    )


def _read_bool(
    raw_value: object,
    *,
    key: str,
) -> bool:
    if isinstance(
        raw_value,
        bool,
    ):
        return raw_value

    raise ValueError(
        f"Configuration key "
        f"'{key}' must be "
        "a boolean"
    )


def _read_port(
    raw_value: object,
    *,
    key: str,
) -> int:
    if isinstance(
        raw_value,
        bool,
    ):
        raise ValueError(
            f"Configuration key "
            f"'{key}' must be "
            "a valid TCP port"
        )

    try:
        port = int(
            raw_value
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"Configuration key "
            f"'{key}' must be "
            "a valid TCP port"
        ) from exc

    if not 1 <= port <= 65535:
        raise ValueError(
            f"Configuration key "
            f"'{key}' must be "
            "between 1 and 65535"
        )

    return port


if __name__ == "__main__":
    main()