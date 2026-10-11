import asyncio
import logging
import os
from collections.abc import Coroutine
from time import perf_counter
from typing import Any, TypeVar

from celery import Celery

from src.core.services.transcription.transcription_processing_service import (
    TranscriptionProcessingService,
)
from src.infrastructure.observability.telemetry import (
    log_event,
    observe_queue_wait_from_enqueued_at,
    observe_worker_exception,
    observe_worker_result,
    record_worker_retry,
    transcription_context,
)
from src.infrastructure.providers.queue.celery_queue_provider import (
    TRANSCRIPTION_TASK_NAME,
)


T = TypeVar("T")

_LOGGER = logging.getLogger(
    "speech_to_text.worker.transcription"
)


class _PersistentAsyncRunner:
    def __init__(self):
        self._process_id: int | None = None
        self._runner: asyncio.Runner | None = None

    def run(
        self,
        coroutine: Coroutine[
            Any,
            Any,
            T,
        ],
    ) -> T:
        current_process_id = os.getpid()

        if (
            self._runner is None
            or self._process_id
            != current_process_id
        ):
            if self._runner is not None:
                self._runner.close()

            self._runner = asyncio.Runner()
            self._process_id = current_process_id

        return self._runner.run(
            coroutine
        )


_async_runner = _PersistentAsyncRunner()


def _get_retry_countdown_seconds(
    retry_number: int,
) -> int:
    return min(
        5
        * (
            2
            ** retry_number
        ),
        60,
    )


def register_transcription_task(
    celery_app: Celery,
    transcription_processing_service: (
        TranscriptionProcessingService
    ),
) -> None:
    if (
        TRANSCRIPTION_TASK_NAME
        in celery_app.tasks
    ):
        celery_app.tasks.unregister(
            TRANSCRIPTION_TASK_NAME
        )

    @celery_app.task(
        bind=True,
        name=TRANSCRIPTION_TASK_NAME,
        max_retries=3,
        acks_late=True,
        reject_on_worker_lost=True,
    )
    def process_transcription(
        task,
        transcription_id: str,
        enqueued_at: str | None = None,
    ) -> None:
        with transcription_context(
            transcription_id
        ):
            queue_wait_seconds = None

            if (
                int(
                    task.request.retries
                )
                == 0
            ):
                queue_wait_seconds = (
                    observe_queue_wait_from_enqueued_at(
                        enqueued_at
                    )
                )

            started_at = perf_counter()

            log_event(
                _LOGGER,
                level=logging.INFO,
                event="worker_job_started",
                message=(
                    "Transcription Worker "
                    "job started"
                ),
                queue_wait_seconds=(
                    round(
                        queue_wait_seconds,
                        6,
                    )
                    if (
                        queue_wait_seconds
                        is not None
                    )
                    else None
                ),
            )

            try:
                transcription = (
                    _async_runner.run(
                        transcription_processing_service
                        .process_job_async(
                            transcription_id
                        )
                    )
                )

                processing_duration_seconds = (
                    perf_counter()
                    - started_at
                )

                (
                    outcome,
                    rtf,
                ) = observe_worker_result(
                    transcription,
                    processing_duration_seconds=(
                        processing_duration_seconds
                    ),
                )

                log_event(
                    _LOGGER,
                    level=(
                        logging.ERROR
                        if outcome == "failed"
                        else logging.INFO
                    ),
                    event="worker_job_finished",
                    message=(
                        "Transcription Worker "
                        "job finished"
                    ),
                    stage=outcome,
                    error_code=getattr(
                        transcription,
                        "error_code",
                        None,
                    ),
                    duration_seconds=round(
                        processing_duration_seconds,
                        6,
                    ),
                    rtf=(
                        round(
                            rtf,
                            6,
                        )
                        if rtf is not None
                        else None
                    ),
                )

            except Exception as exc:
                processing_duration_seconds = (
                    perf_counter()
                    - started_at
                )

                observe_worker_exception(
                    processing_duration_seconds=(
                        processing_duration_seconds
                    )
                )

                retry_number = int(
                    task.request.retries
                )

                max_retries = int(
                    task.max_retries
                )

                if (
                    retry_number
                    >= max_retries
                ):
                    failed_transcription = (
                        _async_runner.run(
                            transcription_processing_service
                            .mark_processing_retry_exhausted_async(
                                transcription_id=(
                                    transcription_id
                                ),
                                error=exc,
                            )
                        )
                    )

                    (
                        outcome,
                        rtf,
                    ) = observe_worker_result(
                        failed_transcription,
                        processing_duration_seconds=(
                            processing_duration_seconds
                        ),
                    )

                    log_event(
                        _LOGGER,
                        level=logging.ERROR,
                        event=(
                            "worker_retry_"
                            "exhausted"
                        ),
                        message=(
                            "Transcription Worker "
                            "retry limit exhausted"
                        ),
                        stage=outcome,
                        error_code=getattr(
                            failed_transcription,
                            "error_code",
                            (
                                "processing_"
                                "retry_exhausted"
                            ),
                        ),
                        exception_type=(
                            type(exc).__name__
                        ),
                        duration_seconds=round(
                            processing_duration_seconds,
                            6,
                        ),
                        rtf=(
                            round(
                                rtf,
                                6,
                            )
                            if rtf is not None
                            else None
                        ),
                        retry_number=retry_number,
                        max_retries=max_retries,
                    )

                    return

                countdown = (
                    _get_retry_countdown_seconds(
                        retry_number
                    )
                )

                record_worker_retry()

                log_event(
                    _LOGGER,
                    level=logging.WARNING,
                    event=(
                        "worker_retry_scheduled"
                    ),
                    message=(
                        "Transcription Worker "
                        "retry scheduled"
                    ),
                    exception_type=(
                        type(exc).__name__
                    ),
                    duration_seconds=round(
                        processing_duration_seconds,
                        6,
                    ),
                    retry_number=(
                        retry_number
                        + 1
                    ),
                    max_retries=max_retries,
                    countdown_seconds=countdown,
                )

                raise task.retry(
                    exc=exc,
                    countdown=countdown,
                )