import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from celery import Celery

from src.core.services.transcription.transcription_processing_service import (
    TranscriptionProcessingService,
    TranscriptionQueuePublishError,
)
from src.infrastructure.providers.media.ffmpeg_provider import (
    AudioMediaInfo,
)
from src.infrastructure.providers.queue.celery_queue_provider import (
    TRANSCRIPTION_TASK_NAME,
)
from src.infrastructure.tasks.transcription_task import (
    register_transcription_task,
)


class FakeTranscriptionRepository:
    def __init__(
        self,
        transcription: Any,
    ):
        self.transcription = transcription
        self.update_count = 0

    async def get_by_id_async(
        self,
        transcription_id: str,
    ) -> Any:
        if (
            str(
                self.transcription.id
            )
            != transcription_id
        ):
            return None

        return self.transcription

    async def update_async(
        self,
        transcription: Any,
    ) -> Any:
        self.transcription = transcription
        self.update_count += 1

        return transcription


class FakeCeleryQueueProvider:
    def __init__(
        self,
        should_fail: bool = False,
    ):
        self.should_fail = should_fail
        self.enqueued_ids: list[str] = []

    async def enqueue_transcription_async(
        self,
        transcription_id: str,
    ) -> None:
        if self.should_fail:
            raise RuntimeError(
                "RabbitMQ unavailable"
            )

        self.enqueued_ids.append(
            transcription_id
        )


class FakeStorageProvider:
    async def download_file_async(
        self,
        object_key: str,
        destination_path: str | Path,
    ) -> Path:
        path = Path(
            destination_path
        )

        path.write_bytes(
            b"fake-audio"
        )

        return path


class FakeFfmpegProvider:
    async def probe_audio_async(
        self,
        file_path: str | Path,
    ) -> AudioMediaInfo:
        return AudioMediaInfo(
            duration_seconds=10.0,
            codec_name="pcm_s16le",
            sample_rate_hz=16000,
            channels=1,
        )


class FakeConfigReader:
    def get_required(
        self,
        key: str,
    ) -> object:
        if (
            key
            == "transcription.upload.max_duration_seconds"
        ):
            return 14400

        raise KeyError(
            key
        )


class FakeProcessingService:
    def __init__(self):
        self.processed_ids: list[str] = []

    async def process_job_async(
        self,
        transcription_id: str,
    ) -> None:
        self.processed_ids.append(
            transcription_id
        )


def create_transcription(
    status: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=(
            "11111111-1111-1111-"
            "1111-111111111111"
        ),
        status=status,
        storage_key=(
            "transcriptions/"
            "11111111-1111-1111-"
            "1111-111111111111/"
            "source.wav"
        ),
        duration_seconds=None,
        raw_text=None,
        error_code=None,
        error_message=None,
    )


def create_processing_service(
    *,
    repository: FakeTranscriptionRepository,
    queue_provider: FakeCeleryQueueProvider,
) -> TranscriptionProcessingService:
    return TranscriptionProcessingService(
        transcription_repository=repository,
        transcription_segment_repository=object(),
        celery_queue_provider=queue_provider,
        storage_provider=FakeStorageProvider(),
        ffmpeg_provider=FakeFfmpegProvider(),
        faster_whisper_provider=object(),
        persian_text_normalization_service=object(),
        text_post_processing_service=object(),
        config_reader=FakeConfigReader(),
    )


def test_enqueue_marks_transcription_as_queued() -> None:
    async def run_test() -> None:
        transcription = create_transcription(
            "uploaded"
        )

        repository = FakeTranscriptionRepository(
            transcription
        )

        queue_provider = (
            FakeCeleryQueueProvider()
        )

        service = create_processing_service(
            repository=repository,
            queue_provider=queue_provider,
        )

        result = await service.enqueue_async(
            str(
                transcription.id
            )
        )

        assert result.status == "queued"
        assert repository.update_count == 1

        assert (
            queue_provider.enqueued_ids
            == [
                str(
                    transcription.id
                )
            ]
        )

    asyncio.run(
        run_test()
    )


def test_enqueue_restores_uploaded_when_publish_fails() -> None:
    async def run_test() -> None:
        transcription = create_transcription(
            "uploaded"
        )

        repository = FakeTranscriptionRepository(
            transcription
        )

        queue_provider = (
            FakeCeleryQueueProvider(
                should_fail=True
            )
        )

        service = create_processing_service(
            repository=repository,
            queue_provider=queue_provider,
        )

        with pytest.raises(
            TranscriptionQueuePublishError
        ):
            await service.enqueue_async(
                str(
                    transcription.id
                )
            )

        assert (
            transcription.status
            == "uploaded"
        )

        assert repository.update_count == 2

    asyncio.run(
        run_test()
    )


def test_processing_validation_moves_to_transcribing() -> None:
    async def run_test() -> None:
        transcription = create_transcription(
            "queued"
        )

        repository = FakeTranscriptionRepository(
            transcription
        )

        service = create_processing_service(
            repository=repository,
            queue_provider=(
                FakeCeleryQueueProvider()
            ),
        )

        result = await service.process_async(
            str(
                transcription.id
            )
        )

        assert (
            result.status
            == "transcribing"
        )

        assert (
            result.duration_seconds
            == 10.0
        )

        assert repository.update_count == 2

    asyncio.run(
        run_test()
    )


def test_duplicate_delivery_is_safe_while_validating() -> None:
    async def run_test() -> None:
        transcription = create_transcription(
            "validating"
        )

        repository = FakeTranscriptionRepository(
            transcription
        )

        service = create_processing_service(
            repository=repository,
            queue_provider=(
                FakeCeleryQueueProvider()
            ),
        )

        result = await service.process_async(
            str(
                transcription.id
            )
        )

        assert (
            result.status
            == "validating"
        )

        assert repository.update_count == 0

    asyncio.run(
        run_test()
    )


def test_celery_task_only_calls_processing_service() -> None:
    celery_app = Celery(
        "test_worker",
        broker="memory://",
    )

    processing_service = (
        FakeProcessingService()
    )

    register_transcription_task(
        celery_app=celery_app,
        transcription_processing_service=(
            processing_service
        ),
    )

    transcription_id = (
        "22222222-2222-2222-"
        "2222-222222222222"
    )

    result = (
        celery_app.tasks[
            TRANSCRIPTION_TASK_NAME
        ]
        .apply(
            args=[
                transcription_id
            ]
        )
    )

    assert result.successful()

    assert (
        processing_service.processed_ids
        == [
            transcription_id
        ]
    )