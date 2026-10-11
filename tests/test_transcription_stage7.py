import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.core.services.transcription.persian_text_normalization_service import (
    PersianTextNormalizationService,
)
from src.core.services.transcription.text_post_processing_service import (
    TextPostProcessingResult,
)
from src.core.services.transcription.transcription_processing_service import (
    TranscriptionProcessingService,
)
from src.infrastructure.providers.asr.faster_whisper_provider import (
    AsrSegmentResult,
    AsrTranscriptionError,
    AsrTranscriptionResult,
)
from src.infrastructure.providers.media.ffmpeg_provider import (
    AudioMediaInfo,
)


class FakeTranscriptionRepository:
    def __init__(
        self,
        transcription: Any,
    ):
        self.transcription = transcription
        self.update_count = 0
        self.status_history: list[str] = []

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

        self.status_history.append(
            str(
                transcription.status
            )
        )

        return transcription


class FakeTranscriptionSegmentRepository:
    def __init__(self):
        self.inserted_segments: list[Any] = []
        self.updated_segments: list[Any] = []

    async def insert_many_async(
        self,
        segments: list[Any],
    ) -> list[Any]:
        self.inserted_segments.extend(
            segments
        )

        return segments

    async def get_by_transcription_id_async(
        self,
        transcription_id: str,
    ) -> list[Any]:
        return [
            segment
            for segment
            in self.inserted_segments
            if (
                segment.transcription_id
                == transcription_id
            )
        ]

    async def update_many_async(
        self,
        segments: list[Any],
    ) -> list[Any]:
        self.updated_segments.extend(
            segments
        )

        return segments


class FakeQueueProvider:
    async def enqueue_transcription_async(
        self,
        transcription_id: str,
    ) -> None:
        return None


class FakeStorageProvider:
    def __init__(self):
        self.downloaded_keys: list[str] = []

    async def download_file_async(
        self,
        object_key: str,
        destination_path: str | Path,
    ) -> Path:
        self.downloaded_keys.append(
            object_key
        )

        path = Path(
            destination_path
        )

        path.write_bytes(
            b"fake-audio"
        )

        return path


class FakeFfmpegProvider:
    def __init__(self):
        self.probed_paths: list[Path] = []

    async def probe_audio_async(
        self,
        file_path: str | Path,
    ) -> AudioMediaInfo:
        self.probed_paths.append(
            Path(
                file_path
            )
        )

        return AudioMediaInfo(
            duration_seconds=12.5,
            codec_name="pcm_s16le",
            sample_rate_hz=16000,
            channels=1,
        )


class FakeAsrProvider:
    def __init__(
        self,
        *,
        result: AsrTranscriptionResult | None = None,
        exception: Exception | None = None,
    ):
        self.result = (
            result
            or create_asr_result()
        )

        self.exception = exception
        self.transcribed_paths: list[Path] = []

    async def transcribe_async(
        self,
        audio_path: str | Path,
        model_reference: str | None = None,
    ) -> AsrTranscriptionResult:
        self.transcribed_paths.append(
            Path(
                audio_path
            )
        )

        if self.exception is not None:
            raise self.exception

        return self.result


class FakeTextPostProcessingService:
    async def process_text_async(
        self,
        normalized_text: str,
    ) -> TextPostProcessingResult:
        return TextPostProcessingResult(
            processed_text=normalized_text,
            model=None,
            version="llm-disabled-v1",
            llm_used=False,
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


def create_asr_result() -> AsrTranscriptionResult:
    return AsrTranscriptionResult(
        engine="faster-whisper",
        model="large-v3",
        text="سلام دنیا. این یک تست است.",
        detected_language="fa",
        language_probability=0.99,
        duration_seconds=12.5,
        duration_after_vad_seconds=12.1,
        segments=(
            AsrSegmentResult(
                sequence=0,
                start_time=0.0,
                end_time=4.2,
                text="سلام دنیا.",
                avg_logprob=-0.1,
                no_speech_probability=0.01,
            ),
            AsrSegmentResult(
                sequence=1,
                start_time=4.2,
                end_time=12.5,
                text="این یک تست است.",
                avg_logprob=-0.2,
                no_speech_probability=0.02,
            ),
        ),
    )


def create_transcription(
    *,
    status: str = "queued",
    raw_text: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=(
            "77777777-7777-7777-"
            "7777-777777777777"
        ),
        status=status,
        storage_key=(
            "transcriptions/"
            "77777777-7777-7777-"
            "7777-777777777777/"
            "source.wav"
        ),
        duration_seconds=None,
        requested_language="fa",
        detected_language=None,
        asr_engine=None,
        asr_model=None,
        raw_text=raw_text,
        normalized_text=None,
        processed_text=None,
        post_processing_model=None,
        post_processing_version=None,
        completed_datetime=None,
        error_code=None,
        error_message=None,
    )


def create_service(
    *,
    transcription: Any,
    asr_provider: Any | None = None,
) -> tuple[
    TranscriptionProcessingService,
    FakeTranscriptionRepository,
    FakeTranscriptionSegmentRepository,
    FakeStorageProvider,
    FakeFfmpegProvider,
    FakeAsrProvider,
]:
    transcription_repository = (
        FakeTranscriptionRepository(
            transcription
        )
    )

    segment_repository = (
        FakeTranscriptionSegmentRepository()
    )

    storage_provider = (
        FakeStorageProvider()
    )

    ffmpeg_provider = (
        FakeFfmpegProvider()
    )

    resolved_asr_provider = (
        asr_provider
        if asr_provider is not None
        else FakeAsrProvider()
    )

    service = (
        TranscriptionProcessingService(
            transcription_repository=(
                transcription_repository
            ),
            transcription_segment_repository=(
                segment_repository
            ),
            celery_queue_provider=(
                FakeQueueProvider()
            ),
            storage_provider=(
                storage_provider
            ),
            ffmpeg_provider=(
                ffmpeg_provider
            ),
            faster_whisper_provider=(
                resolved_asr_provider
            ),
            persian_text_normalization_service=(
                PersianTextNormalizationService()
            ),
            text_post_processing_service=(
                FakeTextPostProcessingService()
            ),
            config_reader=(
                FakeConfigReader()
            ),
        )
    )

    return (
        service,
        transcription_repository,
        segment_repository,
        storage_provider,
        ffmpeg_provider,
        resolved_asr_provider,
    )


def test_worker_persists_raw_text_and_ordered_segments() -> None:
    async def run_test() -> None:
        transcription = (
            create_transcription()
        )

        (
            service,
            repository,
            segment_repository,
            storage_provider,
            ffmpeg_provider,
            asr_provider,
        ) = create_service(
            transcription=transcription
        )

        result = (
            await service.process_job_async(
                str(
                    transcription.id
                )
            )
        )

        assert result.status == "completed"

        assert (
            result.duration_seconds
            == 12.5
        )

        assert (
            result.detected_language
            == "fa"
        )

        assert (
            result.asr_engine
            == "faster-whisper"
        )

        assert (
            result.asr_model
            == "large-v3"
        )

        assert (
            result.raw_text
            == "سلام دنیا. این یک تست است."
        )

        assert (
            result.normalized_text
            == "سلام دنیا. این یک تست است."
        )

        assert (
            result.processed_text
            == "سلام دنیا. این یک تست است."
        )

        assert (
            result.post_processing_model
            is None
        )

        assert (
            result.post_processing_version
            == "llm-disabled-v1"
        )

        assert (
            result.completed_datetime
            is not None
        )

        assert result.error_code is None
        assert result.error_message is None

        assert (
            repository.status_history
            == [
                "validating",
                "transcribing",
                "normalizing",
                "post_processing",
                "completed",
            ]
        )

        assert (
            storage_provider.downloaded_keys
            == [
                transcription.storage_key
            ]
        )

        assert (
            len(
                ffmpeg_provider.probed_paths
            )
            == 1
        )

        assert (
            len(
                asr_provider.transcribed_paths
            )
            == 1
        )

        segments = (
            segment_repository
            .inserted_segments
        )

        assert [
            segment.sequence
            for segment in segments
        ] == [
            0,
            1,
        ]

        assert [
            segment.start_time
            for segment in segments
        ] == [
            0.0,
            4.2,
        ]

        assert [
            segment.end_time
            for segment in segments
        ] == [
            4.2,
            12.5,
        ]

        assert [
            segment.raw_text
            for segment in segments
        ] == [
            "سلام دنیا.",
            "این یک تست است.",
        ]

        assert [
            segment.normalized_text
            for segment in segments
        ] == [
            "سلام دنیا.",
            "این یک تست است.",
        ]

        assert all(
            segment.processed_text
            is None
            for segment in segments
        )

    asyncio.run(
        run_test()
    )


def test_asr_failure_does_not_create_raw_transcript() -> None:
    async def run_test() -> None:
        transcription = (
            create_transcription()
        )

        asr_provider = FakeAsrProvider(
            exception=(
                AsrTranscriptionError(
                    "synthetic ASR failure"
                )
            )
        )

        (
            service,
            repository,
            segment_repository,
            _,
            _,
            _,
        ) = create_service(
            transcription=transcription,
            asr_provider=asr_provider,
        )

        result = (
            await service.process_job_async(
                str(
                    transcription.id
                )
            )
        )

        assert result.status == "failed"

        assert (
            result.error_code
            == "asr_transcription_failed"
        )

        assert result.raw_text is None

        assert (
            segment_repository.inserted_segments
            == []
        )

        assert (
            repository.status_history
            == [
                "validating",
                "transcribing",
                "failed",
            ]
        )

    asyncio.run(
        run_test()
    )


def test_existing_raw_text_is_never_overwritten() -> None:
    async def run_test() -> None:
        transcription = create_transcription(
            status="transcribing",
            raw_text="متن خام قبلی",
        )

        (
            service,
            repository,
            segment_repository,
            storage_provider,
            _,
            asr_provider,
        ) = create_service(
            transcription=transcription
        )

        result = (
            await service.process_job_async(
                str(
                    transcription.id
                )
            )
        )

        assert result.status == "completed"

        assert (
            result.raw_text
            == "متن خام قبلی"
        )

        assert (
            result.normalized_text
            == "متن خام قبلی"
        )

        assert (
            result.processed_text
            == "متن خام قبلی"
        )

        assert repository.update_count == 3

        assert (
            repository.status_history
            == [
                "normalizing",
                "post_processing",
                "completed",
            ]
        )

        assert (
            segment_repository.inserted_segments
            == []
        )

        assert (
            storage_provider.downloaded_keys
            == []
        )

        assert (
            asr_provider.transcribed_paths
            == []
        )

    asyncio.run(
        run_test()
    )