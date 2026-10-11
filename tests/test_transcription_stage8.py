import asyncio
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
    def __init__(
        self,
        segments: list[Any] | None = None,
    ):
        self.segments = (
            segments
            or []
        )

        self.update_many_count = 0
        self.updated_segments: list[Any] = []

    async def insert_many_async(
        self,
        segments: list[Any],
    ) -> list[Any]:
        self.segments.extend(
            segments
        )

        return segments

    async def get_by_transcription_id_async(
        self,
        transcription_id: str,
    ) -> list[Any]:
        return [
            segment
            for segment in self.segments
            if str(
                segment.transcription_id
            )
            == transcription_id
        ]

    async def update_many_async(
        self,
        segments: list[Any],
    ) -> list[Any]:
        self.update_many_count += 1

        self.updated_segments.extend(
            segments
        )

        return segments


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


def create_service(
    *,
    transcription: Any,
    segments: list[Any] | None = None,
) -> tuple[
    TranscriptionProcessingService,
    FakeTranscriptionRepository,
    FakeTranscriptionSegmentRepository,
]:
    transcription_repository = (
        FakeTranscriptionRepository(
            transcription
        )
    )

    segment_repository = (
        FakeTranscriptionSegmentRepository(
            segments
        )
    )

    service = (
        TranscriptionProcessingService(
            transcription_repository=(
                transcription_repository
            ),
            transcription_segment_repository=(
                segment_repository
            ),
            celery_queue_provider=object(),
            storage_provider=object(),
            ffmpeg_provider=object(),
            faster_whisper_provider=object(),
            persian_text_normalization_service=(
                PersianTextNormalizationService()
            ),
            text_post_processing_service=(
                FakeTextPostProcessingService()
            ),
            config_reader=FakeConfigReader(),
        )
    )

    return (
        service,
        transcription_repository,
        segment_repository,
    )


def test_persian_normalizer_standardizes_unicode_space_and_half_space() -> None:
    service = (
        PersianTextNormalizationService()
    )

    raw_text = (
        "\ufeff  علي\tو كيان  \r\n"
        "مي رود\u200d   خانه ها\n\n\n"
        "نمي شود  "
    )

    result = service.normalize_text(
        raw_text
    )

    assert result == (
        "علی و کیان\n"
        "می‌رود خانه‌ها\n\n"
        "نمی‌شود"
    )


def test_worker_normalizes_without_overwriting_raw() -> None:
    async def run_test() -> None:
        raw_text = (
            "  كد 123 تغيير نكند\r\n"
            "مي رود  "
        )

        transcription = SimpleNamespace(
            id=(
                "88888888-8888-8888-"
                "8888-888888888888"
            ),
            status="normalizing",
            raw_text=raw_text,
            normalized_text=None,
            processed_text=None,
            post_processing_model=None,
            post_processing_version=None,
            completed_datetime=None,
            error_code=None,
            error_message=None,
        )

        segments = [
            SimpleNamespace(
                transcription_id=str(
                    transcription.id
                ),
                sequence=0,
                raw_text="مي رود",
                normalized_text=None,
                processed_text=None,
            ),
            SimpleNamespace(
                transcription_id=str(
                    transcription.id
                ),
                sequence=1,
                raw_text="كتاب ها",
                normalized_text=None,
                processed_text=None,
            ),
        ]

        (
            service,
            repository,
            segment_repository,
        ) = create_service(
            transcription=transcription,
            segments=segments,
        )

        result = (
            await service.process_job_async(
                str(
                    transcription.id
                )
            )
        )

        assert result.status == "completed"

        assert result.raw_text == raw_text

        assert (
            result.normalized_text
            == (
                "کد 123 تغییر نکند\n"
                "می‌رود"
            )
        )

        assert (
            result.processed_text
            == result.normalized_text
        )

        assert (
            repository.status_history
            == [
                "post_processing",
                "completed",
            ]
        )

        assert [
            segment.raw_text
            for segment in segments
        ] == [
            "مي رود",
            "كتاب ها",
        ]

        assert [
            segment.normalized_text
            for segment in segments
        ] == [
            "می‌رود",
            "کتاب‌ها",
        ]

        assert (
            segment_repository.update_many_count
            == 1
        )

    asyncio.run(
        run_test()
    )


def test_normalizing_retry_preserves_existing_normalized_values() -> None:
    async def run_test() -> None:
        transcription = SimpleNamespace(
            id=(
                "99999999-9999-9999-"
                "9999-999999999999"
            ),
            status="normalizing",
            raw_text="مي رود",
            normalized_text="مقدار قبلی",
            processed_text=None,
            post_processing_model=None,
            post_processing_version=None,
            completed_datetime=None,
            error_code=None,
            error_message=None,
        )

        segment = SimpleNamespace(
            transcription_id=str(
                transcription.id
            ),
            sequence=0,
            raw_text="كتاب ها",
            normalized_text="سگمنت قبلی",
            processed_text=None,
        )

        (
            service,
            repository,
            segment_repository,
        ) = create_service(
            transcription=transcription,
            segments=[
                segment
            ],
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
            == "مي رود"
        )

        assert (
            result.normalized_text
            == "مقدار قبلی"
        )

        assert (
            result.processed_text
            == "مقدار قبلی"
        )

        assert (
            segment.raw_text
            == "كتاب ها"
        )

        assert (
            segment.normalized_text
            == "سگمنت قبلی"
        )

        assert (
            segment_repository.update_many_count
            == 0
        )

        assert repository.update_count == 2

    asyncio.run(
        run_test()
    )


def test_post_processing_retry_completes_when_normalization_is_complete() -> None:
    async def run_test() -> None:
        transcription = SimpleNamespace(
            id=(
                "aaaaaaaa-aaaa-aaaa-"
                "aaaa-aaaaaaaaaaaa"
            ),
            status="post_processing",
            raw_text="مي رود",
            normalized_text="می‌رود",
            processed_text=None,
            post_processing_model=None,
            post_processing_version=None,
            completed_datetime=None,
            error_code=None,
            error_message=None,
        )

        (
            service,
            repository,
            segment_repository,
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
            == "مي رود"
        )

        assert (
            result.normalized_text
            == "می‌رود"
        )

        assert (
            result.processed_text
            == "می‌رود"
        )

        assert repository.update_count == 1

        assert (
            segment_repository.update_many_count
            == 0
        )

    asyncio.run(
        run_test()
    )