from datetime import (
    UTC,
    datetime,
)
from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import (
    TestClient,
)

from src.application.transcription.transcription_controller import (
    TranscriptionController,
)
from src.core.services.transcription.transcription_service import (
    TranscriptionInvalidStateError,
    TranscriptionQueuePublishError,
    TranscriptionResultNotReadyError,
    TranscriptionRetryUnavailableError,
    TranscriptionService,
)
from src.infrastructure.providers.storage.s3_storage_provider import (
    StorageProviderError,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


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
                ".mp3",
                ".wav",
            ],
            (
                "transcription.upload."
                "allowed_content_types"
            ): [
                "application/octet-stream",
                "audio/mpeg",
                "audio/wav",
            ],
        }

        return values[
            key
        ]


class FakeTranscriptionRepository:
    def __init__(
        self,
        transcriptions=None,
    ):
        self.transcriptions = {
            str(
                item.id
            ): item
            for item
            in (
                transcriptions
                or []
            )
        }

        self.status_history: list[
            str
        ] = []

        self.deleted_ids: list[
            str
        ] = []

    async def insert_async(
        self,
        entity,
    ):
        if entity.id is None:
            entity.id = str(
                uuid4()
            )

        now = datetime.now(
            UTC
        )

        if (
            getattr(
                entity,
                "requested_language",
                None,
            )
            is None
        ):
            entity.requested_language = (
                "fa"
            )

        if (
            getattr(
                entity,
                "creation_datetime",
                None,
            )
            is None
        ):
            entity.creation_datetime = now

        if (
            getattr(
                entity,
                "modification_datetime",
                None,
            )
            is None
        ):
            entity.modification_datetime = (
                now
            )

        self.transcriptions[
            str(
                entity.id
            )
        ] = entity

        return entity

    async def get_by_id_async(
        self,
        transcription_id: str,
    ):
        return self.transcriptions.get(
            transcription_id
        )

    async def update_async(
        self,
        entity,
    ):
        entity.modification_datetime = (
            datetime.now(
                UTC
            )
        )

        self.transcriptions[
            str(
                entity.id
            )
        ] = entity

        status_value = getattr(
            entity.status,
            "value",
            entity.status,
        )

        self.status_history.append(
            str(
                status_value
            )
        )

        return entity

    async def delete_async(
        self,
        transcription_id: str,
    ) -> bool:
        if (
            transcription_id
            not in self.transcriptions
        ):
            return False

        del self.transcriptions[
            transcription_id
        ]

        self.deleted_ids.append(
            transcription_id
        )

        return True


class FakeTranscriptionSegmentRepository:
    def __init__(
        self,
        segments_by_transcription=None,
    ):
        self.segments_by_transcription = (
            segments_by_transcription
            or {}
        )

    async def get_by_transcription_id_async(
        self,
        transcription_id: str,
    ):
        return list(
            self.segments_by_transcription.get(
                transcription_id,
                [],
            )
        )


class FakeStorageProvider:
    def __init__(self):
        self.uploaded = []
        self.deleted = []

        self.fail_delete = (
            False
        )

    async def upload_file_async(
        self,
        file_obj,
        object_key: str,
        content_type: str | None = None,
    ) -> str:
        file_obj.seek(
            0
        )

        self.uploaded.append(
            {
                "object_key": (
                    object_key
                ),
                "content_type": (
                    content_type
                ),
                "content": (
                    file_obj.read()
                ),
            }
        )

        return object_key

    async def delete_file_async(
        self,
        object_key: str,
    ) -> None:
        if self.fail_delete:
            raise StorageProviderError(
                "simulated delete failure"
            )

        self.deleted.append(
            object_key
        )


class FakeQueueProvider:
    def __init__(self):
        self.published_ids: list[
            str
        ] = []

        self.should_fail = (
            False
        )

    async def enqueue_transcription_async(
        self,
        transcription_id: str,
    ) -> None:
        if self.should_fail:
            raise RuntimeError(
                "simulated queue failure"
            )

        self.published_ids.append(
            transcription_id
        )


def create_transcription(
    *,
    status: str,
    transcription_id: str | None = None,
    storage_key: str | None = None,
    raw_text: str | None = None,
    normalized_text: str | None = None,
    processed_text: str | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
):
    now = datetime.now(
        UTC
    )

    resolved_id = (
        transcription_id
        or str(
            uuid4()
        )
    )

    return SimpleNamespace(
        id=resolved_id,
        status=status,
        original_filename=(
            "sample.wav"
        ),
        content_type="audio/wav",
        file_size_bytes=1024,
        duration_seconds=12.5,
        storage_key=(
            storage_key
            if storage_key is not None
            else (
                "transcriptions/"
                f"{resolved_id}/"
                "source.wav"
            )
        ),
        requested_language="fa",
        detected_language="fa",
        asr_engine=(
            "faster-whisper"
        ),
        asr_model="large-v3",
        raw_text=raw_text,
        normalized_text=(
            normalized_text
        ),
        processed_text=(
            processed_text
        ),
        post_processing_model=(
            "test-llm"
        ),
        post_processing_version=(
            "persian-transcript-"
            "postprocess-v1"
        ),
        error_code=error_code,
        error_message=error_message,
        creation_datetime=now,
        started_datetime=now,
        completed_datetime=(
            now
            if status
            in {
                "completed",
                "failed",
                "cancelled",
            }
            else None
        ),
        modification_datetime=now,
    )


def create_segment(
    *,
    transcription_id: str,
    sequence: int,
    raw_text: str,
):
    return SimpleNamespace(
        transcription_id=(
            transcription_id
        ),
        sequence=sequence,
        start_time=float(
            sequence
        ),
        end_time=float(
            sequence
            + 1
        ),
        raw_text=raw_text,
        normalized_text=(
            raw_text
        ),
        processed_text=(
            raw_text
        ),
        speaker_label=None,
    )


def create_service(
    *,
    transcriptions=None,
    segments_by_transcription=None,
):
    repository = (
        FakeTranscriptionRepository(
            transcriptions
        )
    )

    segment_repository = (
        FakeTranscriptionSegmentRepository(
            segments_by_transcription
        )
    )

    storage = (
        FakeStorageProvider()
    )

    queue = (
        FakeQueueProvider()
    )

    service = TranscriptionService(
        transcription_repository=(
            repository
        ),
        transcription_segment_repository=(
            segment_repository
        ),
        storage_provider=storage,
        celery_queue_provider=queue,
        config_reader=(
            FakeConfigReader()
        ),
    )

    return (
        service,
        repository,
        storage,
        queue,
    )


@pytest.mark.anyio
async def test_create_upload_queues_in_one_workflow():
    (
        service,
        repository,
        storage,
        queue,
    ) = create_service()

    transcription = (
        await service
        .create_and_enqueue_from_upload_async(
            filename="sample.wav",
            content_type="audio/wav",
            file_size_bytes=5,
            file_obj=BytesIO(
                b"audio"
            ),
        )
    )

    assert (
        getattr(
            transcription.status,
            "value",
            transcription.status,
        )
        == "queued"
    )

    assert queue.published_ids == [
        str(
            transcription.id
        )
    ]

    assert (
        storage.uploaded[
            0
        ][
            "content"
        ]
        == b"audio"
    )

    assert (
        repository.status_history[
            -2:
        ]
        == [
            "uploaded",
            "queued",
        ]
    )


@pytest.mark.anyio
async def test_queue_failure_restores_uploaded_state():
    (
        service,
        repository,
        _,
        queue,
    ) = create_service()

    queue.should_fail = (
        True
    )

    with pytest.raises(
        TranscriptionQueuePublishError
    ) as error_info:
        await (
            service
            .create_and_enqueue_from_upload_async(
                filename="sample.wav",
                content_type="audio/wav",
                file_size_bytes=5,
                file_obj=BytesIO(
                    b"audio"
                ),
            )
        )

    transcription = (
        repository.transcriptions[
            error_info.value
            .transcription_id
        ]
    )

    assert (
        getattr(
            transcription.status,
            "value",
            transcription.status,
        )
        == "uploaded"
    )

    assert (
        repository.status_history[
            -2:
        ]
        == [
            "queued",
            "uploaded",
        ]
    )


@pytest.mark.anyio
async def test_retry_resumes_from_normalized_text():
    transcription = (
        create_transcription(
            status="failed",
            raw_text="متن خام",
            normalized_text=(
                "متن نرمال"
            ),
            error_code=(
                "llm_post_processing_failed"
            ),
            error_message=(
                "temporary failure"
            ),
        )
    )

    (
        service,
        _,
        _,
        queue,
    ) = create_service(
        transcriptions=[
            transcription
        ]
    )

    result = (
        await service.retry_async(
            str(
                transcription.id
            )
        )
    )

    assert (
        result.status
        == "post_processing"
    )

    assert (
        result.raw_text
        == "متن خام"
    )

    assert (
        result.normalized_text
        == "متن نرمال"
    )

    assert result.error_code is None
    assert result.error_message is None

    assert (
        result.completed_datetime
        is None
    )

    assert queue.published_ids == [
        str(
            transcription.id
        )
    ]


@pytest.mark.anyio
async def test_retry_rejects_failed_job_without_source_audio():
    transcription = (
        create_transcription(
            status="failed",
            storage_key="",
            error_code=(
                "storage_error"
            ),
        )
    )

    (
        service,
        _,
        _,
        _,
    ) = create_service(
        transcriptions=[
            transcription
        ]
    )

    with pytest.raises(
        TranscriptionRetryUnavailableError
    ):
        await service.retry_async(
            str(
                transcription.id
            )
        )


@pytest.mark.anyio
async def test_result_requires_terminal_state_and_returns_segments():
    queued = (
        create_transcription(
            status="queued"
        )
    )

    completed = (
        create_transcription(
            status="completed",
            raw_text="سلام دنیا",
            normalized_text=(
                "سلام دنیا"
            ),
            processed_text=(
                "سلام دنیا."
            ),
        )
    )

    segments = [
        create_segment(
            transcription_id=str(
                completed.id
            ),
            sequence=0,
            raw_text="سلام",
        ),
        create_segment(
            transcription_id=str(
                completed.id
            ),
            sequence=1,
            raw_text="دنیا",
        ),
    ]

    (
        service,
        _,
        _,
        _,
    ) = create_service(
        transcriptions=[
            queued,
            completed,
        ],
        segments_by_transcription={
            str(
                completed.id
            ): segments,
        },
    )

    with pytest.raises(
        TranscriptionResultNotReadyError
    ):
        await service.get_result_async(
            str(
                queued.id
            )
        )

    (
        result,
        returned_segments,
    ) = (
        await service
        .get_result_async(
            str(
                completed.id
            )
        )
    )

    assert (
        result.processed_text
        == "سلام دنیا."
    )

    assert [
        item.sequence
        for item
        in returned_segments
    ] == [
        0,
        1,
    ]


@pytest.mark.anyio
async def test_delete_rejects_active_and_deletes_terminal_job():
    active = (
        create_transcription(
            status="transcribing"
        )
    )

    completed = (
        create_transcription(
            status="completed"
        )
    )

    (
        service,
        repository,
        storage,
        _,
    ) = create_service(
        transcriptions=[
            active,
            completed,
        ]
    )

    with pytest.raises(
        TranscriptionInvalidStateError
    ):
        await service.delete_async(
            str(
                active.id
            )
        )

    await service.delete_async(
        str(
            completed.id
        )
    )

    assert (
        str(
            completed.id
        )
        not in repository.transcriptions
    )

    assert storage.deleted == [
        completed.storage_key
    ]


def test_final_http_contract():
    completed = (
        create_transcription(
            status="completed",
            raw_text="سلام دنیا",
            normalized_text=(
                "سلام دنیا"
            ),
            processed_text=(
                "سلام دنیا."
            ),
        )
    )

    queued = (
        create_transcription(
            status="queued"
        )
    )

    segments = [
        create_segment(
            transcription_id=str(
                completed.id
            ),
            sequence=0,
            raw_text="سلام دنیا",
        )
    ]

    (
        service,
        _,
        _,
        _,
    ) = create_service(
        transcriptions=[
            completed,
            queued,
        ],
        segments_by_transcription={
            str(
                completed.id
            ): segments,
        },
    )

    app = FastAPI()

    controller = TranscriptionController(
        transcription_service=(
            service
        )
    )

    app.include_router(
        controller.api(),
        prefix=(
            "/api/v1/transcription"
        ),
    )

    with TestClient(
        app
    ) as client:
        upload_response = (
            client.post(
                (
                    "/api/v1/"
                    "transcription/"
                ),
                files={
                    "file": (
                        "upload.wav",
                        b"audio",
                        "audio/wav",
                    )
                },
            )
        )

        assert (
            upload_response.status_code
            == 202
        )

        assert (
            upload_response.json()[
                "status"
            ]
            == "queued"
        )

        invalid_id_response = (
            client.get(
                (
                    "/api/v1/"
                    "transcription/"
                    "not-a-uuid"
                )
            )
        )

        assert (
            invalid_id_response
            .status_code
            == 422
        )

        status_response = (
            client.get(
                (
                    "/api/v1/"
                    "transcription/"
                    f"{completed.id}"
                )
            )
        )

        assert (
            status_response.status_code
            == 200
        )

        result_response = (
            client.get(
                (
                    "/api/v1/"
                    "transcription/"
                    f"{completed.id}/"
                    "result"
                )
            )
        )

        assert (
            result_response.status_code
            == 200
        )

        assert (
            result_response.json()[
                "processed_text"
            ]
            == "سلام دنیا."
        )

        assert (
            result_response.json()[
                "segments"
            ][
                0
            ][
                "sequence"
            ]
            == 0
        )

        not_ready_response = (
            client.get(
                (
                    "/api/v1/"
                    "transcription/"
                    f"{queued.id}/"
                    "result"
                )
            )
        )

        assert (
            not_ready_response.status_code
            == 409
        )

        assert (
            not_ready_response.json()[
                "detail"
            ][
                "code"
            ]
            == (
                "transcription_result_"
                "not_ready"
            )
        )

        retry_response = (
            client.post(
                (
                    "/api/v1/"
                    "transcription/"
                    f"{completed.id}/"
                    "retry"
                )
            )
        )

        assert (
            retry_response.status_code
            == 409
        )

        delete_response = (
            client.delete(
                (
                    "/api/v1/"
                    "transcription/"
                    f"{completed.id}"
                )
            )
        )

        assert (
            delete_response.status_code
            == 204
        )

        assert (
            delete_response.content
            == b""
        )