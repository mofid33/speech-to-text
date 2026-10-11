from io import BytesIO
from uuid import uuid4

import pytest
from botocore.exceptions import (
    EndpointConnectionError,
)

from src.core.services.transcription.transcription_service import (
    AudioFileTooLargeError,
    TranscriptionService,
    TranscriptionStorageError,
    UnsupportedAudioFileError,
)
from src.infrastructure.models.transcription import (
    TranscriptionStatus,
)
from src.infrastructure.providers.storage.s3_storage_provider import (
    S3StorageProvider,
    StorageProviderError,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeConfigReader:
    def __init__(
        self,
        overrides=None,
    ):
        self._values = {
            (
                "transcription.upload."
                "max_file_size_mb"
            ): 500,
            (
                "transcription.upload."
                "allowed_extensions"
            ): [
                ".aac",
                ".flac",
                ".m4a",
                ".mp3",
                ".ogg",
                ".opus",
                ".wav",
                ".webm",
            ],
            (
                "transcription.upload."
                "allowed_content_types"
            ): [
                "application/octet-stream",
                "audio/aac",
                "audio/flac",
                "audio/mp4",
                "audio/mpeg",
                "audio/ogg",
                "audio/opus",
                "audio/wav",
                "audio/webm",
                "audio/x-flac",
                "audio/x-m4a",
                "audio/x-wav",
                "binary/octet-stream",
            ],
            "storage.bucket_name": (
                "transcriptions"
            ),
            "storage.endpoint_url": (
                "http://localhost:19000"
            ),
            "storage.region_name": (
                "us-east-1"
            ),
            "storage.access_key_id": (
                "speech_to_text"
            ),
            "storage.secret_access_key": (
                "speech_to_text_secret"
            ),
            "storage.addressing_style": (
                "path"
            ),
            "storage.auto_create_bucket": (
                True
            ),
            "storage.verify_ssl": False,
        }

        if overrides:
            self._values.update(
                overrides
            )

    def get(
        self,
        key,
        default=None,
    ):
        return self._values.get(
            key,
            default,
        )

    def get_required(
        self,
        key,
    ):
        if key not in self._values:
            raise KeyError(
                key
            )

        return self._values[
            key
        ]


class FakeTranscriptionRepository:
    def __init__(self):
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


class FakeTranscriptionSegmentRepository:
    async def get_by_transcription_id_async(
        self,
        transcription_id,
    ):
        return []


class FakeQueueProvider:
    async def enqueue_transcription_async(
        self,
        transcription_id,
    ):
        return None


class FakeS3StorageProvider:
    def __init__(self):
        self.uploaded = []
        self.deleted = []

        self.should_fail_upload = (
            False
        )

    async def upload_file_async(
        self,
        file_obj,
        object_key,
        content_type=None,
    ):
        if self.should_fail_upload:
            raise StorageProviderError(
                "Simulated storage failure"
            )

        file_obj.seek(0)

        content = (
            file_obj.read()
        )

        self.uploaded.append(
            {
                "object_key": (
                    object_key
                ),
                "content_type": (
                    content_type
                ),
                "content": content,
            }
        )

        return object_key

    async def delete_file_async(
        self,
        object_key,
    ):
        self.deleted.append(
            object_key
        )


def _create_service(
    *,
    repository=None,
    storage=None,
    config=None,
):
    return TranscriptionService(
        transcription_repository=(
            repository
            or FakeTranscriptionRepository()
        ),
        transcription_segment_repository=(
            FakeTranscriptionSegmentRepository()
        ),
        storage_provider=(
            storage
            or FakeS3StorageProvider()
        ),
        celery_queue_provider=(
            FakeQueueProvider()
        ),
        config_reader=(
            config
            or FakeConfigReader()
        ),
    )


@pytest.mark.anyio
async def test_create_from_upload_saves_file_and_marks_uploaded():
    repository = (
        FakeTranscriptionRepository()
    )

    storage = (
        FakeS3StorageProvider()
    )

    service = _create_service(
        repository=repository,
        storage=storage,
    )

    transcription = (
        await service
        .create_from_upload_async(
            filename="sample.mp3",
            content_type="audio/mpeg",
            file_size_bytes=6,
            file_obj=BytesIO(
                b"audio!"
            ),
        )
    )

    assert len(
        repository.inserted
    ) == 1

    assert len(
        storage.uploaded
    ) == 1

    assert (
        transcription.status
        == TranscriptionStatus.UPLOADED
    )

    assert (
        transcription.original_filename
        == "sample.mp3"
    )

    assert (
        transcription.file_size_bytes
        == 6
    )

    assert (
        transcription.storage_key
        == (
            "transcriptions/"
            f"{transcription.id}/"
            "source.mp3"
        )
    )

    assert (
        storage.uploaded[
            0
        ][
            "content"
        ]
        == b"audio!"
    )


@pytest.mark.anyio
async def test_rejects_unsupported_extension_before_db_insert():
    repository = (
        FakeTranscriptionRepository()
    )

    storage = (
        FakeS3StorageProvider()
    )

    service = _create_service(
        repository=repository,
        storage=storage,
    )

    with pytest.raises(
        UnsupportedAudioFileError
    ):
        await service.create_from_upload_async(
            filename="sample.exe",
            content_type=(
                "application/octet-stream"
            ),
            file_size_bytes=10,
            file_obj=BytesIO(
                b"0123456789"
            ),
        )

    assert repository.inserted == []
    assert storage.uploaded == []


@pytest.mark.anyio
async def test_file_size_limit_is_loaded_from_configuration():
    repository = (
        FakeTranscriptionRepository()
    )

    storage = (
        FakeS3StorageProvider()
    )

    config = FakeConfigReader(
        {
            (
                "transcription.upload."
                "max_file_size_mb"
            ): 1,
        }
    )

    service = _create_service(
        repository=repository,
        storage=storage,
        config=config,
    )

    with pytest.raises(
        AudioFileTooLargeError,
        match="1 MB upload limit",
    ):
        await service.create_from_upload_async(
            filename="large.wav",
            content_type="audio/wav",
            file_size_bytes=(
                1024
                * 1024
            )
            + 1,
            file_obj=BytesIO(
                b"x"
            ),
        )

    assert repository.inserted == []
    assert storage.uploaded == []


@pytest.mark.anyio
async def test_allowed_extensions_are_loaded_from_configuration():
    repository = (
        FakeTranscriptionRepository()
    )

    storage = (
        FakeS3StorageProvider()
    )

    config = FakeConfigReader(
        {
            (
                "transcription.upload."
                "allowed_extensions"
            ): [
                ".wav",
            ],
        }
    )

    service = _create_service(
        repository=repository,
        storage=storage,
        config=config,
    )

    with pytest.raises(
        UnsupportedAudioFileError
    ):
        await service.create_from_upload_async(
            filename="sample.mp3",
            content_type="audio/mpeg",
            file_size_bytes=4,
            file_obj=BytesIO(
                b"test"
            ),
        )

    assert repository.inserted == []
    assert storage.uploaded == []


@pytest.mark.anyio
async def test_storage_failure_marks_transcription_failed():
    repository = (
        FakeTranscriptionRepository()
    )

    storage = (
        FakeS3StorageProvider()
    )

    storage.should_fail_upload = (
        True
    )

    service = _create_service(
        repository=repository,
        storage=storage,
    )

    with pytest.raises(
        TranscriptionStorageError
    ):
        await service.create_from_upload_async(
            filename="sample.flac",
            content_type="audio/flac",
            file_size_bytes=4,
            file_obj=BytesIO(
                b"test"
            ),
        )

    assert len(
        repository.inserted
    ) == 1

    assert len(
        repository.updated
    ) == 1

    transcription = (
        repository.updated[
            -1
        ]
    )

    assert (
        transcription.status
        == TranscriptionStatus.FAILED
    )

    assert (
        transcription.storage_key
        is None
    )


@pytest.mark.anyio
async def test_s3_bucket_connection_failure_is_normalized():
    provider = S3StorageProvider(
        config_reader=(
            FakeConfigReader()
        )
    )

    def fail_head_bucket(
        **kwargs,
    ):
        raise EndpointConnectionError(
            endpoint_url=(
                "http://localhost:19000"
            )
        )

    provider._client.head_bucket = (
        fail_head_bucket
    )

    with pytest.raises(
        StorageProviderError,
        match=(
            "Failed to access bucket "
            "'transcriptions'"
        ),
    ):
        await provider.upload_file_async(
            file_obj=BytesIO(
                b"test"
            ),
            object_key=(
                "transcriptions/"
                "test/source.wav"
            ),
            content_type="audio/wav",
        )