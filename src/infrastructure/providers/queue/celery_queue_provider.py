import asyncio
import logging
from datetime import UTC, datetime
from time import perf_counter

from celery import Celery

from src.infrastructure.di.inject import inject
from src.infrastructure.observability.telemetry import (
    log_event,
    observe_queue_publish,
    record_transcription_failure,
    transcription_context,
)
from src.infrastructure.utils.config_reader import (
    ConfigReader,
)


TRANSCRIPTION_TASK_NAME = (
    "speech_to_text."
    "transcription.process"
)

_LOGGER = logging.getLogger(
    "speech_to_text.queue"
)


@inject
class CeleryQueueProvider:
    __di_singleton__ = True

    def __init__(
        self,
        config_reader: ConfigReader,
    ):
        broker_url = str(
            config_reader.get_required(
                "queue.broker_url"
            )
        ).strip()

        queue_name = str(
            config_reader.get_required(
                "queue.transcription_queue"
            )
        ).strip()

        if not broker_url:
            raise ValueError(
                "queue.broker_url "
                "must not be empty"
            )

        if not queue_name:
            raise ValueError(
                "queue.transcription_queue "
                "must not be empty"
            )

        self._queue_name = (
            queue_name
        )

        self._celery_app = Celery(
            "speech_to_text",
            broker=broker_url,
        )

        self._celery_app.conf.update(
            task_serializer="json",
            accept_content=[
                "json"
            ],
            result_serializer="json",
            enable_utc=True,
            timezone="UTC",
            task_default_queue=(
                self._queue_name
            ),
            task_acks_late=True,
            task_reject_on_worker_lost=True,
            worker_prefetch_multiplier=1,
            broker_connection_retry_on_startup=(
                True
            ),
            task_publish_retry=True,
            task_publish_retry_policy={
                "max_retries": 3,
                "interval_start": 0,
                "interval_step": 0.5,
                "interval_max": 2,
            },
        )

    @property
    def celery_app(
        self,
    ) -> Celery:
        return self._celery_app

    @property
    def queue_name(
        self,
    ) -> str:
        return self._queue_name

    async def enqueue_transcription_async(
        self,
        transcription_id: str,
    ) -> None:
        started_at = (
            perf_counter()
        )

        enqueued_at = (
            datetime.now(
                UTC
            ).isoformat()
        )

        with transcription_context(
            transcription_id
        ):
            try:
                await asyncio.to_thread(
                    self._celery_app
                    .send_task,
                    TRANSCRIPTION_TASK_NAME,
                    args=[
                        transcription_id,
                        enqueued_at,
                    ],
                    queue=(
                        self._queue_name
                    ),
                    ignore_result=True,
                    retry=True,
                    retry_policy={
                        "max_retries": 3,
                        "interval_start": 0,
                        "interval_step": 0.5,
                        "interval_max": 2,
                    },
                )

            except Exception:
                duration_seconds = (
                    perf_counter()
                    - started_at
                )

                observe_queue_publish(
                    outcome="failed",
                    duration_seconds=(
                        duration_seconds
                    ),
                )

                record_transcription_failure(
                    stage="queue",
                    error_code=(
                        "queue_publish_failed"
                    ),
                )

                log_event(
                    _LOGGER,
                    level=logging.ERROR,
                    event=(
                        "queue_publish_failed"
                    ),
                    message=(
                        "Transcription queue "
                        "publish failed"
                    ),
                    queue=(
                        self._queue_name
                    ),
                    duration_seconds=round(
                        duration_seconds,
                        6,
                    ),
                )

                raise

            duration_seconds = (
                perf_counter()
                - started_at
            )

            observe_queue_publish(
                outcome="success",
                duration_seconds=(
                    duration_seconds
                ),
            )

            log_event(
                _LOGGER,
                level=logging.INFO,
                event=(
                    "queue_publish_completed"
                ),
                message=(
                    "Transcription queued "
                    "successfully"
                ),
                queue=(
                    self._queue_name
                ),
                duration_seconds=round(
                    duration_seconds,
                    6,
                ),
            )