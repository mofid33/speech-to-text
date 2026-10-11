import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.core.services.transcription.persian_text_normalization_service import (
    PersianTextNormalizationService,
)
from src.core.services.transcription.text_post_processing_service import (
    TextPostProcessingProviderError,
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
from src.infrastructure.providers.storage.s3_storage_provider import (
    StorageProviderError,
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


class FakeSegmentRepository:
    def __init__(
        self,
        segments: list[Any] | None = None,
    ):
        self.segments = list(
            segments
            or []
        )

        self.replace_count = 0

    async def insert_many_async(
        self,
        segments: list[Any],
    ) -> list[Any]:
        self.segments.extend(
            segments
        )
        return segments

    async def replace_for_transcription_async(
        self,
        *,
        transcription_id: str,
        segments: list[Any],
    ) -> list[Any]:
        self.replace_count += 1

        self.segments = [
            segment
            for segment in segments
            if (
                str(
                    segment.transcription_id
                )
                == transcription_id
            )
        ]

        return self.segments

    async def get_by_transcription_id_async(
        self,
        transcription_id: str,
    ) -> list[Any]:
        return [
            segment
            for segment in self.segments
            if (
                str(
                    segment.transcription_id
                )
                == transcription_id
            )
        ]

    async def update_many_async(
        self,
        segments: list[Any],
    ) -> list[Any]:
        return segments


class FakeQueueProvider:
    async def enqueue_transcription_async(
        self,
        transcription_id: str,
    ) -> None:
        return None


class FlakyStorageProvider:
    def __init__(
        self,
        fail_times: int = 0,
    ):
        self.fail_times = fail_times
        self.calls = 0

    async def download_file_async(
        self,
        object_key: str,
        destination_path: str | Path,
    ) -> Path:
        self.calls += 1

        if self.calls <= self.fail_times:
            raise StorageProviderError(
                "temporary storage failure"
            )

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
            duration_seconds=12.5,
            codec_name="pcm_s16le",
            sample_rate_hz=16000,
            channels=1,
        )


class FlakyAsrProvider:
    def __init__(
        self,
        fail_times: int = 0,
    ):
        self.fail_times = fail_times
        self.calls = 0

    async def transcribe_async(
        self,
        audio_path: str | Path,
        model_reference: str | None = None,
    ) -> AsrTranscriptionResult:
        self.calls += 1

        if self.calls <= self.fail_times:
            raise AsrTranscriptionError(
                "temporary ASR failure"
            )

        return create_asr_result()


class FlakyPostProcessingService:
    def __init__(
        self,
        fail_times: int = 0,
    ):
        self.fail_times = fail_times
        self.calls = 0

    async def process_text_async(
        self,
        normalized_text: str,
    ) -> TextPostProcessingResult:
        self.calls += 1

        if self.calls <= self.fail_times:
            raise TextPostProcessingProviderError(
                "temporary vLLM failure"
            )

        return TextPostProcessingResult(
            processed_text=normalized_text,
            model="test-llm",
            version=(
                "persian-transcript-"
                "postprocess-v1"
            ),
            llm_used=True,
        )


class FakeConfigReader:
    def __init__(
        self,
        *,
        max_attempts: int = 1,
    ):
        self.max_attempts = (
            max_attempts
        )

    def get_required(
        self,
        key: str,
    ) -> object:
        values = {
            (
                "transcription.upload."
                "max_duration_seconds"
            ): 14400,
            (
                "transcription.processing."
                "provider_retry.max_attempts"
            ): self.max_attempts,
            (
                "transcription.processing."
                "provider_retry.backoff_seconds"
            ): 0,
            (
                "transcription.processing."
                "provider_retry.max_backoff_seconds"
            ): 0,
        }

        return values[
            key
        ]


def create_asr_result() -> AsrTranscriptionResult:
    return AsrTranscriptionResult(
        engine="faster-whisper",
        model="large-v3",
        text="سلام دنیا",
        detected_language="fa",
        language_probability=0.99,
        duration_seconds=12.5,
        duration_after_vad_seconds=12.0,
        segments=(
            AsrSegmentResult(
                sequence=0,
                start_time=0.0,
                end_time=5.0,
                text="سلام",
                avg_logprob=-0.1,
                no_speech_probability=0.01,
            ),
            AsrSegmentResult(
                sequence=1,
                start_time=5.0,
                end_time=12.5,
                text="دنیا",
                avg_logprob=-0.1,
                no_speech_probability=0.01,
            ),
        ),
    )


def create_transcription(
    *,
    status: str,
    raw_text: str | None = None,
    normalized_text: str | None = None,
    processed_text: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=(
            "10101010-1010-1010-"
            "1010-101010101010"
        ),
        status=status,
        storage_key=(
            "transcriptions/"
            "10101010-1010-1010-"
            "1010-101010101010/"
            "source.wav"
        ),
        duration_seconds=None,
        requested_language="fa",
        detected_language=None,
        asr_engine=None,
        asr_model=None,
        raw_text=raw_text,
        normalized_text=normalized_text,
        processed_text=processed_text,
        post_processing_model=None,
        post_processing_version=None,
        started_datetime=None,
        completed_datetime=None,
        error_code=None,
        error_message=None,
    )


def create_service(
    *,
    transcription: Any,
    max_attempts: int = 1,
    storage_provider: Any | None = None,
    asr_provider: Any | None = None,
    post_processing_service: Any | None = None,
    segment_repository: FakeSegmentRepository | None = None,
) -> tuple[
    TranscriptionProcessingService,
    FakeTranscriptionRepository,
    FakeSegmentRepository,
    FlakyStorageProvider,
    FlakyAsrProvider,
    FlakyPostProcessingService,
]:
    transcription_repository = (
        FakeTranscriptionRepository(
            transcription
        )
    )

    resolved_segment_repository = (
        segment_repository
        or FakeSegmentRepository()
    )

    resolved_storage_provider = (
        storage_provider
        or FlakyStorageProvider()
    )

    resolved_asr_provider = (
        asr_provider
        or FlakyAsrProvider()
    )

    resolved_post_processing_service = (
        post_processing_service
        or FlakyPostProcessingService()
    )

    service = TranscriptionProcessingService(
        transcription_repository=(
            transcription_repository
        ),
        transcription_segment_repository=(
            resolved_segment_repository
        ),
        celery_queue_provider=(
            FakeQueueProvider()
        ),
        storage_provider=(
            resolved_storage_provider
        ),
        ffmpeg_provider=(
            FakeFfmpegProvider()
        ),
        faster_whisper_provider=(
            resolved_asr_provider
        ),
        persian_text_normalization_service=(
            PersianTextNormalizationService()
        ),
        text_post_processing_service=(
            resolved_post_processing_service
        ),
        config_reader=(
            FakeConfigReader(
                max_attempts=max_attempts
            )
        ),
    )

    return (
        service,
        transcription_repository,
        resolved_segment_repository,
        resolved_storage_provider,
        resolved_asr_provider,
        resolved_post_processing_service,
    )


def test_completed_job_is_idempotent_no_op() -> None:
    async def run_test() -> None:
        transcription = (
            create_transcription(
                status="completed",
                raw_text="خام",
                normalized_text="نرمال",
                processed_text="نهایی",
            )
        )

        (
            service,
            repository,
            _,
            storage,
            asr,
            post_processing,
        ) = create_service(
            transcription=transcription
        )

        result = await service.process_job_async(
            str(
                transcription.id
            )
        )

        assert result is transcription
        assert result.status == "completed"

        assert repository.update_count == 0
        assert storage.calls == 0
        assert asr.calls == 0
        assert post_processing.calls == 0

    asyncio.run(
        run_test()
    )


def test_validating_job_resumes_after_worker_redelivery() -> None:
    async def run_test() -> None:
        transcription = (
            create_transcription(
                status="validating"
            )
        )

        (
            service,
            repository,
            _,
            storage,
            asr,
            post_processing,
        ) = create_service(
            transcription=transcription
        )

        result = await service.process_job_async(
            str(
                transcription.id
            )
        )

        assert result.status == "completed"

        assert (
            repository.status_history
            == [
                "transcribing",
                "normalizing",
                "post_processing",
                "completed",
            ]
        )

        assert storage.calls == 1
        assert asr.calls == 1
        assert post_processing.calls == 1

    asyncio.run(
        run_test()
    )


def test_storage_operation_retries_and_recovers() -> None:
    async def run_test() -> None:
        transcription = (
            create_transcription(
                status="queued"
            )
        )

        storage = FlakyStorageProvider(
            fail_times=1
        )

        (
            service,
            _,
            _,
            _,
            _,
            _,
        ) = create_service(
            transcription=transcription,
            max_attempts=2,
            storage_provider=storage,
        )

        result = await service.process_job_async(
            str(
                transcription.id
            )
        )

        assert result.status == "completed"
        assert storage.calls == 2

    asyncio.run(
        run_test()
    )


def test_asr_operation_retries_and_recovers() -> None:
    async def run_test() -> None:
        transcription = (
            create_transcription(
                status="queued"
            )
        )

        asr = FlakyAsrProvider(
            fail_times=1
        )

        (
            service,
            _,
            _,
            _,
            _,
            _,
        ) = create_service(
            transcription=transcription,
            max_attempts=2,
            asr_provider=asr,
        )

        result = await service.process_job_async(
            str(
                transcription.id
            )
        )

        assert result.status == "completed"
        assert asr.calls == 2

    asyncio.run(
        run_test()
    )


def test_llm_provider_failure_retries_without_losing_text() -> None:
    async def run_test() -> None:
        transcription = (
            create_transcription(
                status="post_processing",
                raw_text="متن خام",
                normalized_text="متن نرمال",
            )
        )

        post_processing = (
            FlakyPostProcessingService(
                fail_times=1
            )
        )

        (
            service,
            _,
            _,
            _,
            _,
            _,
        ) = create_service(
            transcription=transcription,
            max_attempts=2,
            post_processing_service=(
                post_processing
            ),
        )

        result = await service.process_job_async(
            str(
                transcription.id
            )
        )

        assert result.status == "completed"

        assert result.raw_text == "متن خام"

        assert (
            result.normalized_text
            == "متن نرمال"
        )

        assert (
            result.processed_text
            == "متن نرمال"
        )

        assert post_processing.calls == 2

    asyncio.run(
        run_test()
    )


def test_partial_raw_segments_are_replaced_on_retry() -> None:
    async def run_test() -> None:
        transcription = (
            create_transcription(
                status="transcribing"
            )
        )

        stale_segment = SimpleNamespace(
            transcription_id=str(
                transcription.id
            ),
            sequence=0,
            start_time=0.0,
            end_time=1.0,
            raw_text="قدیمی",
            normalized_text=None,
            processed_text=None,
        )

        segment_repository = (
            FakeSegmentRepository(
                segments=[
                    stale_segment
                ]
            )
        )

        (
            service,
            _,
            _,
            _,
            _,
            _,
        ) = create_service(
            transcription=transcription,
            segment_repository=(
                segment_repository
            ),
        )

        result = await service.process_job_async(
            str(
                transcription.id
            )
        )

        assert result.status == "completed"

        assert (
            segment_repository.replace_count
            == 1
        )

        assert [
            segment.sequence
            for segment
            in segment_repository.segments
        ] == [
            0,
            1,
        ]

        assert [
            segment.raw_text
            for segment
            in segment_repository.segments
        ] == [
            "سلام",
            "دنیا",
        ]

    asyncio.run(
        run_test()
    )


def test_retry_exhaustion_preserves_raw_and_normalized_text() -> None:
    async def run_test() -> None:
        transcription = (
            create_transcription(
                status="post_processing",
                raw_text="متن خام",
                normalized_text="متن نرمال",
            )
        )

        (
            service,
            _,
            _,
            _,
            _,
            _,
        ) = create_service(
            transcription=transcription
        )

        result = (
            await service
            .mark_processing_retry_exhausted_async(
                transcription_id=str(
                    transcription.id
                ),
                error=RuntimeError(
                    "database temporarily unavailable"
                ),
            )
        )

        assert result.status == "failed"

        assert (
            result.error_code
            == "processing_retry_exhausted"
        )

        assert result.raw_text == "متن خام"

        assert (
            result.normalized_text
            == "متن نرمال"
        )

        assert result.processed_text is None

    asyncio.run(
        run_test()
    )