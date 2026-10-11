import asyncio
import json
import logging
from datetime import (
    UTC,
    datetime,
    timedelta,
)
from io import (
    BytesIO,
    StringIO,
)
from types import SimpleNamespace
from uuid import uuid4

from celery import Celery
from fastapi.testclient import (
    TestClient,
)
from prometheus_client import REGISTRY

from src.core.services.transcription.transcription_service import (
    TranscriptionService,
)
from src.host.app import app
from src.infrastructure.observability.telemetry import (
    JsonLogFormatter,
    log_event,
    transcription_context,
)
from src.infrastructure.providers.queue.celery_queue_provider import (
    TRANSCRIPTION_TASK_NAME,
)
from src.infrastructure.tasks.transcription_task import (
    register_transcription_task,
)


class FakeConfigReader:
    def get_required(
        self,
        key: str,
    ) -> object:
        values = {
            (
                "transcription.upload."
                "max_file_size_mb"
            ): 500,
            (
                "transcription.upload."
                "allowed_extensions"
            ): [
                ".wav",
            ],
            (
                "transcription.upload."
                "allowed_content_types"
            ): [
                "audio/wav",
                "application/octet-stream",
            ],
        }

        return values[
            key
        ]


class FakeTranscriptionRepository:
    def __init__(
        self,
    ):
        self.inserted = []
        self.updated = []

    async def insert_async(
        self,
        entity,
    ):
        if entity.id is None:
            entity.id = str(
                uuid4()
            )

        self.inserted.append(
            entity
        )

        return entity

    async def update_async(
        self,
        entity,
    ):
        self.updated.append(
            entity
        )

        return entity


class FakeSegmentRepository:
    async def get_by_transcription_id_async(
        self,
        transcription_id: str,
    ):
        return []


class FakeStorageProvider:
    def __init__(
        self,
    ):
        self.uploaded = []

    async def upload_file_async(
        self,
        file_obj,
        object_key: str,
        content_type: (
            str
            | None
        ) = None,
    ) -> str:
        self.uploaded.append(
            object_key
        )

        return object_key

    async def delete_file_async(
        self,
        object_key: str,
    ) -> None:
        return None


class FakeQueueProvider:
    async def enqueue_transcription_async(
        self,
        transcription_id: str,
    ) -> None:
        return None


class ObservableProcessingService:
    def __init__(
        self,
        *,
        final_status: str,
        error_code: (
            str
            | None
        ) = None,
    ):
        self.transcription_id = (
            "77777777-7777-7777-"
            "7777-777777777777"
        )

        self._initial = (
            SimpleNamespace(
                id=(
                    self.transcription_id
                ),
                status="queued",
                modification_datetime=(
                    datetime.now(UTC)
                    - timedelta(
                        seconds=2
                    )
                ),
                duration_seconds=10.0,
                error_code=None,
            )
        )

        self._final = (
            SimpleNamespace(
                id=(
                    self.transcription_id
                ),
                status=(
                    final_status
                ),
                modification_datetime=(
                    datetime.now(
                        UTC
                    )
                ),
                duration_seconds=10.0,
                error_code=(
                    error_code
                ),
            )
        )

    async def get_by_id_async(
        self,
        transcription_id: str,
    ):
        assert (
            transcription_id
            == self.transcription_id
        )

        return self._initial

    async def process_job_async(
        self,
        transcription_id: str,
    ):
        assert (
            transcription_id
            == self.transcription_id
        )

        return self._final

    async def mark_processing_retry_exhausted_async(
        self,
        transcription_id: str,
        error: Exception,
    ):
        return SimpleNamespace(
            id=transcription_id,
            status="failed",
            duration_seconds=10.0,
            error_code=(
                "processing_"
                "retry_exhausted"
            ),
        )


def _create_upload_service(
    *,
    repository: (
        FakeTranscriptionRepository
    ),
    storage: FakeStorageProvider,
) -> TranscriptionService:
    return TranscriptionService(
        transcription_repository=(
            repository
        ),
        transcription_segment_repository=(
            FakeSegmentRepository()
        ),
        storage_provider=(
            storage
        ),
        celery_queue_provider=(
            FakeQueueProvider()
        ),
        config_reader=(
            FakeConfigReader()
        ),
    )


def _sample_value(
    name: str,
    labels: (
        dict[str, str]
        | None
    ) = None,
) -> float:
    value = (
        REGISTRY.get_sample_value(
            name,
            labels or {},
        )
    )

    return float(
        value
        or 0.0
    )


def test_upload_filename_is_not_used_as_storage_path(
) -> None:
    async def run_test(
    ) -> None:
        repository = (
            FakeTranscriptionRepository()
        )

        storage = (
            FakeStorageProvider()
        )

        service = (
            _create_upload_service(
                repository=(
                    repository
                ),
                storage=storage,
            )
        )

        transcription = (
            await service
            .create_from_upload_async(
                filename=(
                    "../../../../"
                    "tmp/evil.wav"
                ),
                content_type=(
                    "audio/wav"
                ),
                file_size_bytes=5,
                file_obj=BytesIO(
                    b"audio"
                ),
            )
        )

        expected_key = (
            "transcriptions/"
            f"{transcription.id}/"
            "source.wav"
        )

        assert (
            transcription.storage_key
            == expected_key
        )

        assert (
            storage.uploaded
            == [
                expected_key
            ]
        )

        assert (
            ".."
            not in expected_key
        )

        assert (
            "evil"
            not in expected_key
        )

    asyncio.run(
        run_test()
    )


def test_structured_log_contains_transcription_id(
) -> None:
    stream = StringIO()

    handler = (
        logging.StreamHandler(
            stream
        )
    )

    handler.setFormatter(
        JsonLogFormatter()
    )

    logger = logging.getLogger(
        "speech_to_text."
        "stage12_test"
    )

    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(
        logging.INFO
    )

    logger.addHandler(
        handler
    )

    transcription_id = (
        "88888888-8888-8888-"
        "8888-888888888888"
    )

    try:
        with transcription_context(
            transcription_id
        ):
            log_event(
                logger,
                level=logging.INFO,
                event=(
                    "stage12_test_event"
                ),
                message=(
                    "Stage 12 structured log"
                ),
                stage="transcribing",
            )

        payload = json.loads(
            stream.getvalue()
            .strip()
        )

        assert (
            payload[
                "transcription_id"
            ]
            == transcription_id
        )

        assert (
            payload[
                "event"
            ]
            == "stage12_test_event"
        )

        assert (
            payload[
                "stage"
            ]
            == "transcribing"
        )

        assert (
            payload[
                "level"
            ]
            == "INFO"
        )

    finally:
        logger.removeHandler(
            handler
        )

        handler.close()


def test_worker_records_queue_wait_processing_rtf_and_completed_outcome(
) -> None:
    processing_service = (
        ObservableProcessingService(
            final_status=(
                "completed"
            )
        )
    )

    queue_wait_before = (
        _sample_value(
            "speech_to_text_"
            "queue_wait_seconds_count"
        )
    )

    processing_before = (
        _sample_value(
            (
                "speech_to_text_"
                "worker_processing_"
                "duration_seconds_count"
            ),
            {
                "outcome": (
                    "completed"
                )
            },
        )
    )

    rtf_before = (
        _sample_value(
            (
                "speech_to_text_"
                "worker_processing_"
                "realtime_factor_count"
            )
        )
    )

    outcome_before = (
        _sample_value(
            (
                "speech_to_text_"
                "worker_jobs_total"
            ),
            {
                "outcome": (
                    "completed"
                )
            },
        )
    )

    celery_app = Celery(
        "stage12_success",
        broker="memory://",
    )

    register_transcription_task(
        celery_app=celery_app,
        transcription_processing_service=(
            processing_service
        ),
    )

    result = (
        celery_app.tasks[
            TRANSCRIPTION_TASK_NAME
        ]
        .apply(
            args=[
                (
                    processing_service
                    .transcription_id
                ),
                (
                    datetime.now(UTC)
                    - timedelta(
                        seconds=2
                    )
                ).isoformat(),
            ]
        )
    )

    assert result.successful()

    assert (
        _sample_value(
            "speech_to_text_"
            "queue_wait_seconds_count"
        )
        == queue_wait_before
        + 1
    )

    assert (
        _sample_value(
            (
                "speech_to_text_"
                "worker_processing_"
                "duration_seconds_count"
            ),
            {
                "outcome": (
                    "completed"
                )
            },
        )
        == processing_before
        + 1
    )

    assert (
        _sample_value(
            (
                "speech_to_text_"
                "worker_processing_"
                "realtime_factor_count"
            )
        )
        == rtf_before
        + 1
    )

    assert (
        _sample_value(
            (
                "speech_to_text_"
                "worker_jobs_total"
            ),
            {
                "outcome": (
                    "completed"
                )
            },
        )
        == outcome_before
        + 1
    )


def test_worker_records_failed_outcome_by_stage_and_error_code(
) -> None:
    processing_service = (
        ObservableProcessingService(
            final_status="failed",
            error_code=(
                "asr_transcription_failed"
            ),
        )
    )

    failure_labels = {
        "stage": "transcribing",
        "error_code": (
            "asr_transcription_failed"
        ),
    }

    failure_before = (
        _sample_value(
            (
                "speech_to_text_"
                "transcription_"
                "failures_total"
            ),
            failure_labels,
        )
    )

    outcome_before = (
        _sample_value(
            (
                "speech_to_text_"
                "worker_jobs_total"
            ),
            {
                "outcome": (
                    "failed"
                )
            },
        )
    )

    celery_app = Celery(
        "stage12_failure",
        broker="memory://",
    )

    register_transcription_task(
        celery_app=celery_app,
        transcription_processing_service=(
            processing_service
        ),
    )

    result = (
        celery_app.tasks[
            TRANSCRIPTION_TASK_NAME
        ]
        .apply(
            args=[
                (
                    processing_service
                    .transcription_id
                )
            ]
        )
    )

    assert result.successful()

    assert (
        _sample_value(
            (
                "speech_to_text_"
                "transcription_"
                "failures_total"
            ),
            failure_labels,
        )
        == failure_before
        + 1
    )

    assert (
        _sample_value(
            (
                "speech_to_text_"
                "worker_jobs_total"
            ),
            {
                "outcome": (
                    "failed"
                )
            },
        )
        == outcome_before
        + 1
    )


def test_invalid_upload_does_not_expose_stack_trace(
) -> None:
    with TestClient(
        app
    ) as client:
        response = client.post(
            "/api/v1/"
            "transcription/",
            files={
                "file": (
                    "attack.exe",
                    b"not-audio",
                    (
                        "application/"
                        "octet-stream"
                    ),
                )
            },
        )

    assert (
        response.status_code
        == 415
    )

    response_text = (
        json.dumps(
            response.json()
        )
        .lower()
    )

    assert (
        "traceback"
        not in response_text
    )

    assert (
        "stack trace"
        not in response_text
    )


def test_metrics_endpoint_exposes_metrics_and_is_hidden_from_openapi(
) -> None:
    with TestClient(
        app
    ) as client:
        liveness_response = (
            client.get(
                "/api/v1/"
                "health_check/"
            )
        )

        metrics_response = (
            client.get(
                "/metrics"
            )
        )

        openapi_response = (
            client.get(
                "/openapi.json"
            )
        )

    assert (
        liveness_response.status_code
        == 200
    )

    assert (
        metrics_response.status_code
        == 200
    )

    assert (
        "speech_to_text_"
        "api_request_"
        "duration_seconds"
        in metrics_response.text
    )

    assert (
        "speech_to_text_"
        "transcription_"
        "failures_total"
        in metrics_response.text
    )

    assert (
        "/metrics"
        not in (
            openapi_response
            .json()[
                "paths"
            ]
        )
    )