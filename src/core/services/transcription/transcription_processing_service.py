import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeVar

from src.core.services.transcription.persian_text_normalization_service import (
    PersianTextNormalizationService,
)
from src.core.services.transcription.text_post_processing_service import (
    TextPostProcessingError,
    TextPostProcessingProviderError,
    TextPostProcessingService,
    TextPostProcessingValidationError,
)
from src.infrastructure.di.inject import inject
from src.infrastructure.models.transcription import Transcription
from src.infrastructure.models.transcription_segment import (
    TranscriptionSegment,
)
from src.infrastructure.providers.asr.faster_whisper_provider import (
    AsrConfigurationError,
    AsrModelLoadError,
    AsrProviderError,
    AsrTranscriptionError,
    AsrTranscriptionResult,
    FasterWhisperProvider,
)
from src.infrastructure.providers.media.ffmpeg_provider import (
    FfmpegProvider,
    InvalidMediaError,
    MediaProbeError,
)
from src.infrastructure.providers.queue.celery_queue_provider import (
    CeleryQueueProvider,
)
from src.infrastructure.providers.storage.s3_storage_provider import (
    S3StorageProvider,
    StorageProviderError,
)
from src.infrastructure.repositories.transcription.transcription_repository import (
    TranscriptionRepository,
)
from src.infrastructure.repositories.transcription.transcription_segment_repository import (
    TranscriptionSegmentRepository,
)
from src.infrastructure.utils.config_reader import ConfigReader


T = TypeVar("T")


class TranscriptionNotFoundError(Exception):
    def __init__(self, transcription_id: str):
        self.transcription_id = transcription_id
        super().__init__(
            f"Transcription '{transcription_id}' was not found"
        )


class TranscriptionInvalidStateError(Exception):
    def __init__(
        self,
        transcription_id: str,
        current_status: str,
        expected_status: str,
    ):
        self.transcription_id = transcription_id
        self.current_status = current_status
        self.expected_status = expected_status

        super().__init__(
            f"Transcription '{transcription_id}' "
            f"is in status '{current_status}', "
            f"expected '{expected_status}'"
        )


class TranscriptionQueuePublishError(Exception):
    def __init__(self, transcription_id: str):
        self.transcription_id = transcription_id

        super().__init__(
            f"Could not enqueue transcription '{transcription_id}'"
        )


@inject
class TranscriptionProcessingService:
    def __init__(
        self,
        transcription_repository: TranscriptionRepository,
        transcription_segment_repository: TranscriptionSegmentRepository,
        celery_queue_provider: CeleryQueueProvider,
        storage_provider: S3StorageProvider,
        ffmpeg_provider: FfmpegProvider,
        faster_whisper_provider: FasterWhisperProvider,
        persian_text_normalization_service: PersianTextNormalizationService,
        text_post_processing_service: TextPostProcessingService,
        config_reader: ConfigReader,
    ):
        self._transcription_repository = transcription_repository
        self._transcription_segment_repository = (
            transcription_segment_repository
        )
        self._celery_queue_provider = celery_queue_provider
        self._storage_provider = storage_provider
        self._ffmpeg_provider = ffmpeg_provider
        self._faster_whisper_provider = faster_whisper_provider
        self._persian_text_normalization_service = (
            persian_text_normalization_service
        )
        self._text_post_processing_service = (
            text_post_processing_service
        )

        self._max_duration_seconds = self._read_positive_float(
            config_reader=config_reader,
            key="transcription.upload.max_duration_seconds",
        )

        self._provider_retry_max_attempts = (
            self._read_optional_positive_int(
                config_reader=config_reader,
                key=(
                    "transcription.processing."
                    "provider_retry.max_attempts"
                ),
                default=1,
            )
        )

        self._provider_retry_backoff_seconds = (
            self._read_optional_non_negative_float(
                config_reader=config_reader,
                key=(
                    "transcription.processing."
                    "provider_retry.backoff_seconds"
                ),
                default=0.0,
            )
        )

        self._provider_retry_max_backoff_seconds = (
            self._read_optional_non_negative_float(
                config_reader=config_reader,
                key=(
                    "transcription.processing."
                    "provider_retry.max_backoff_seconds"
                ),
                default=self._provider_retry_backoff_seconds,
            )
        )

        if (
            self._provider_retry_max_backoff_seconds
            < self._provider_retry_backoff_seconds
        ):
            raise ValueError(
                "transcription.processing.provider_retry."
                "max_backoff_seconds must be greater than or "
                "equal to backoff_seconds"
            )

    async def get_by_id_async(
        self,
        transcription_id: str,
    ) -> Transcription:
        transcription = (
            await self._transcription_repository.get_by_id_async(
                transcription_id
            )
        )

        if transcription is None:
            raise TranscriptionNotFoundError(
                transcription_id
            )

        return transcription

    async def enqueue_async(
        self,
        transcription_id: str,
    ) -> Transcription:
        transcription = await self.get_by_id_async(
            transcription_id
        )

        current_status = self._get_status_value(
            transcription.status
        )

        if current_status != "uploaded":
            raise TranscriptionInvalidStateError(
                transcription_id=transcription_id,
                current_status=current_status,
                expected_status="uploaded",
            )

        self._set_status(
            transcription=transcription,
            status_value="queued",
        )

        transcription = (
            await self._transcription_repository.update_async(
                transcription
            )
        )

        try:
            await (
                self._celery_queue_provider
                .enqueue_transcription_async(
                    transcription_id=str(
                        transcription.id
                    )
                )
            )
        except Exception as exc:
            self._set_status(
                transcription=transcription,
                status_value="uploaded",
            )

            await self._transcription_repository.update_async(
                transcription
            )

            raise TranscriptionQueuePublishError(
                transcription_id
            ) from exc

        return transcription

    async def process_async(
        self,
        transcription_id: str,
    ) -> Transcription:
        """
        Validation-only contract used by the earlier stages.

        queued
            -> validating
            -> transcribing

        A duplicate delivery already in validating remains a no-op
        here because process_async is not the full Worker workflow.
        """

        transcription = await self.get_by_id_async(
            transcription_id
        )

        current_status = self._get_status_value(
            transcription.status
        )

        if current_status == "validating":
            return transcription

        if current_status != "queued":
            raise TranscriptionInvalidStateError(
                transcription_id=transcription_id,
                current_status=current_status,
                expected_status="queued",
            )

        transcription = await self._mark_validating_async(
            transcription
        )

        return await self._validate_audio_async(
            transcription=transcription,
            continue_to_asr=False,
        )

    async def process_job_async(
        self,
        transcription_id: str,
    ) -> Transcription:
        """
        Idempotent Worker workflow.

        queued
            -> validating
            -> transcribing
            -> normalizing
            -> post_processing
            -> completed

        Every persisted intermediate state is resumable.

        Terminal states are no-ops so RabbitMQ redelivery cannot
        process an already completed/failed/cancelled job again.
        """

        transcription = await self.get_by_id_async(
            transcription_id
        )

        current_status = self._get_status_value(
            transcription.status
        )

        if current_status in {
            "completed",
            "failed",
            "cancelled",
        }:
            return transcription

        if current_status == "validating":
            return await self._validate_audio_async(
                transcription=transcription,
                continue_to_asr=True,
            )

        if current_status == "post_processing":
            if (
                self._has_raw_text(transcription)
                and self._has_normalized_text(
                    transcription
                )
            ):
                return (
                    await self
                    ._post_process_transcription_async(
                        transcription
                    )
                )

            raise TranscriptionInvalidStateError(
                transcription_id=transcription_id,
                current_status=current_status,
                expected_status="normalizing",
            )

        if current_status == "normalizing":
            if not self._has_raw_text(
                transcription
            ):
                raise TranscriptionInvalidStateError(
                    transcription_id=transcription_id,
                    current_status=current_status,
                    expected_status="transcribing",
                )

            return (
                await self
                ._normalize_and_post_process_async(
                    transcription
                )
            )

        if current_status == "transcribing":
            if self._has_raw_text(
                transcription
            ):
                transcription = (
                    await self
                    ._move_to_normalizing_async(
                        transcription
                    )
                )

                return (
                    await self
                    ._normalize_and_post_process_async(
                        transcription
                    )
                )

            return (
                await self
                ._transcribe_from_storage_async(
                    transcription
                )
            )

        if current_status != "queued":
            raise TranscriptionInvalidStateError(
                transcription_id=transcription_id,
                current_status=current_status,
                expected_status="queued",
            )

        transcription = await self._mark_validating_async(
            transcription
        )

        return await self._validate_audio_async(
            transcription=transcription,
            continue_to_asr=True,
        )

    async def mark_processing_retry_exhausted_async(
        self,
        transcription_id: str,
        error: Exception,
    ) -> Transcription:
        """
        Called only when Celery has exhausted retries for an
        unexpected Worker-level exception.

        Existing raw_text/normalized_text/processed_text are never
        cleared here.
        """

        transcription = await self.get_by_id_async(
            transcription_id
        )

        current_status = self._get_status_value(
            transcription.status
        )

        if current_status in {
            "completed",
            "failed",
            "cancelled",
        }:
            return transcription

        error_details = str(error).strip()

        error_message = (
            "Worker retry limit was exhausted"
        )

        if error_details:
            error_message = (
                f"{error_message}: "
                f"{type(error).__name__}: "
                f"{error_details}"
            )

        return await self._mark_failed_async(
            transcription=transcription,
            error_code="processing_retry_exhausted",
            error_message=error_message,
        )

    async def _mark_validating_async(
        self,
        transcription: Transcription,
    ) -> Transcription:
        self._set_status(
            transcription=transcription,
            status_value="validating",
        )

        if (
            getattr(
                transcription,
                "started_datetime",
                None,
            )
            is None
        ):
            transcription.started_datetime = (
                datetime.now(UTC)
            )

        transcription.error_code = None
        transcription.error_message = None

        return (
            await self._transcription_repository
            .update_async(
                transcription
            )
        )

    async def _validate_audio_async(
        self,
        *,
        transcription: Transcription,
        continue_to_asr: bool,
    ) -> Transcription:
        storage_key = self._get_storage_key(
            transcription
        )

        if not storage_key:
            return await self._mark_failed_async(
                transcription=transcription,
                error_code="audio_validation_failed",
                error_message=(
                    "Audio object storage key is missing"
                ),
            )

        suffix = Path(storage_key).suffix

        try:
            with TemporaryDirectory(
                prefix="speech_to_text_processing_"
            ) as temporary_directory:
                local_path = (
                    Path(temporary_directory)
                    / f"source{suffix}"
                )

                await self._run_provider_operation_with_retry_async(
                    operation=lambda: (
                        self._storage_provider
                        .download_file_async(
                            object_key=storage_key,
                            destination_path=local_path,
                        )
                    ),
                    retryable_exceptions=(
                        StorageProviderError,
                    ),
                )

                media_info = (
                    await self
                    ._run_provider_operation_with_retry_async(
                        operation=lambda: (
                            self._ffmpeg_provider
                            .probe_audio_async(
                                local_path
                            )
                        ),
                        retryable_exceptions=(
                            MediaProbeError,
                        ),
                    )
                )

                transcription.duration_seconds = (
                    media_info.duration_seconds
                )

                if (
                    media_info.duration_seconds
                    > self._max_duration_seconds
                ):
                    return await self._mark_failed_async(
                        transcription=transcription,
                        error_code=(
                            "audio_duration_exceeded"
                        ),
                        error_message=(
                            "Audio duration exceeds the "
                            "configured limit of "
                            f"{self._max_duration_seconds:g} "
                            "seconds"
                        ),
                    )

                transcription.error_code = None
                transcription.error_message = None

                self._set_status(
                    transcription=transcription,
                    status_value="transcribing",
                )

                transcription = (
                    await self
                    ._transcription_repository
                    .update_async(
                        transcription
                    )
                )

                if not continue_to_asr:
                    return transcription

                return (
                    await self
                    ._transcribe_local_audio_async(
                        transcription=transcription,
                        local_path=local_path,
                    )
                )

        except StorageProviderError:
            return await self._mark_failed_async(
                transcription=transcription,
                error_code="storage_error",
                error_message=(
                    "Audio could not be read "
                    "from object storage"
                ),
            )

        except InvalidMediaError:
            return await self._mark_failed_async(
                transcription=transcription,
                error_code="invalid_media",
                error_message=(
                    "Uploaded file is not a valid "
                    "supported audio file"
                ),
            )

        except MediaProbeError:
            return await self._mark_failed_async(
                transcription=transcription,
                error_code="audio_validation_failed",
                error_message=(
                    "Audio validation could not "
                    "be completed"
                ),
            )

    async def _transcribe_from_storage_async(
        self,
        transcription: Transcription,
    ) -> Transcription:
        storage_key = self._get_storage_key(
            transcription
        )

        if not storage_key:
            return await self._mark_failed_async(
                transcription=transcription,
                error_code="storage_error",
                error_message=(
                    "Audio object storage key is missing"
                ),
            )

        suffix = Path(storage_key).suffix

        try:
            with TemporaryDirectory(
                prefix="speech_to_text_asr_"
            ) as temporary_directory:
                local_path = (
                    Path(temporary_directory)
                    / f"source{suffix}"
                )

                await self._run_provider_operation_with_retry_async(
                    operation=lambda: (
                        self._storage_provider
                        .download_file_async(
                            object_key=storage_key,
                            destination_path=local_path,
                        )
                    ),
                    retryable_exceptions=(
                        StorageProviderError,
                    ),
                )

                return (
                    await self
                    ._transcribe_local_audio_async(
                        transcription=transcription,
                        local_path=local_path,
                    )
                )

        except StorageProviderError:
            return await self._mark_failed_async(
                transcription=transcription,
                error_code="storage_error",
                error_message=(
                    "Audio could not be read "
                    "from object storage"
                ),
            )

    async def _transcribe_local_audio_async(
        self,
        *,
        transcription: Transcription,
        local_path: Path,
    ) -> Transcription:
        if self._has_raw_text(
            transcription
        ):
            transcription = (
                await self
                ._move_to_normalizing_async(
                    transcription
                )
            )

            return (
                await self
                ._normalize_and_post_process_async(
                    transcription
                )
            )

        try:
            asr_result = (
                await self
                ._run_provider_operation_with_retry_async(
                    operation=lambda: (
                        self._faster_whisper_provider
                        .transcribe_async(
                            local_path
                        )
                    ),
                    retryable_exceptions=(
                        AsrProviderError,
                    ),
                    non_retryable_exceptions=(
                        AsrConfigurationError,
                    ),
                )
            )

        except AsrConfigurationError:
            return await self._mark_failed_async(
                transcription=transcription,
                error_code="asr_configuration_error",
                error_message=(
                    "ASR model is not configured"
                ),
            )

        except AsrModelLoadError:
            return await self._mark_failed_async(
                transcription=transcription,
                error_code="asr_model_load_error",
                error_message=(
                    "ASR model could not be loaded"
                ),
            )

        except AsrTranscriptionError:
            return await self._mark_failed_async(
                transcription=transcription,
                error_code="asr_transcription_failed",
                error_message=(
                    "Audio transcription could "
                    "not be completed"
                ),
            )

        except AsrProviderError:
            return await self._mark_failed_async(
                transcription=transcription,
                error_code="asr_failed",
                error_message=(
                    "ASR processing could not "
                    "be completed"
                ),
            )

        transcription = (
            await self
            ._persist_raw_transcription_async(
                transcription=transcription,
                asr_result=asr_result,
            )
        )

        return (
            await self
            ._normalize_and_post_process_async(
                transcription
            )
        )

    async def _persist_raw_transcription_async(
        self,
        *,
        transcription: Transcription,
        asr_result: AsrTranscriptionResult,
    ) -> Transcription:
        if self._has_raw_text(
            transcription
        ):
            return (
                await self
                ._move_to_normalizing_async(
                    transcription
                )
            )

        transcription_id = str(
            transcription.id
        )

        segments = [
            TranscriptionSegment(
                transcription_id=transcription_id,
                sequence=segment.sequence,
                start_time=segment.start_time,
                end_time=segment.end_time,
                raw_text=segment.text,
            )
            for segment in asr_result.segments
        ]

        existing_segments = (
            await self
            ._transcription_segment_repository
            .get_by_transcription_id_async(
                transcription_id
            )
        )

        if existing_segments:
            await (
                self
                ._transcription_segment_repository
                .replace_for_transcription_async(
                    transcription_id=(
                        transcription_id
                    ),
                    segments=segments,
                )
            )

        elif segments:
            await (
                self
                ._transcription_segment_repository
                .insert_many_async(
                    segments
                )
            )

        transcription.detected_language = (
            asr_result.detected_language
        )
        transcription.asr_engine = (
            asr_result.engine
        )
        transcription.asr_model = (
            asr_result.model
        )
        transcription.raw_text = (
            asr_result.text
        )

        transcription.error_code = None
        transcription.error_message = None

        self._set_status(
            transcription=transcription,
            status_value="normalizing",
        )

        return (
            await self._transcription_repository
            .update_async(
                transcription
            )
        )

    async def _move_to_normalizing_async(
        self,
        transcription: Transcription,
    ) -> Transcription:
        current_status = self._get_status_value(
            transcription.status
        )

        if current_status == "normalizing":
            return transcription

        self._set_status(
            transcription=transcription,
            status_value="normalizing",
        )

        transcription.error_code = None
        transcription.error_message = None

        return (
            await self._transcription_repository
            .update_async(
                transcription
            )
        )

    async def _normalize_and_post_process_async(
        self,
        transcription: Transcription,
    ) -> Transcription:
        transcription = (
            await self._normalize_transcription_async(
                transcription
            )
        )

        return (
            await self
            ._post_process_transcription_async(
                transcription
            )
        )

    async def _normalize_transcription_async(
        self,
        transcription: Transcription,
    ) -> Transcription:
        current_status = self._get_status_value(
            transcription.status
        )

        if current_status != "normalizing":
            raise TranscriptionInvalidStateError(
                transcription_id=str(
                    transcription.id
                ),
                current_status=current_status,
                expected_status="normalizing",
            )

        if not self._has_raw_text(
            transcription
        ):
            raise TranscriptionInvalidStateError(
                transcription_id=str(
                    transcription.id
                ),
                current_status=current_status,
                expected_status="transcribing",
            )

        segments = (
            await self
            ._transcription_segment_repository
            .get_by_transcription_id_async(
                str(transcription.id)
            )
        )

        segments_to_update: list[
            TranscriptionSegment
        ] = []

        for segment in segments:
            if (
                getattr(
                    segment,
                    "normalized_text",
                    None,
                )
                is not None
            ):
                continue

            segment.normalized_text = (
                self
                ._persian_text_normalization_service
                .normalize_text(
                    segment.raw_text
                )
            )

            segments_to_update.append(
                segment
            )

        if segments_to_update:
            await (
                self
                ._transcription_segment_repository
                .update_many_async(
                    segments_to_update
                )
            )

        if not self._has_normalized_text(
            transcription
        ):
            transcription.normalized_text = (
                self
                ._persian_text_normalization_service
                .normalize_text(
                    str(
                        transcription.raw_text
                    )
                )
            )

        transcription.error_code = None
        transcription.error_message = None

        self._set_status(
            transcription=transcription,
            status_value="post_processing",
        )

        return (
            await self._transcription_repository
            .update_async(
                transcription
            )
        )

    async def _post_process_transcription_async(
        self,
        transcription: Transcription,
    ) -> Transcription:
        current_status = self._get_status_value(
            transcription.status
        )

        if current_status != "post_processing":
            raise TranscriptionInvalidStateError(
                transcription_id=str(
                    transcription.id
                ),
                current_status=current_status,
                expected_status="post_processing",
            )

        if not self._has_normalized_text(
            transcription
        ):
            raise TranscriptionInvalidStateError(
                transcription_id=str(
                    transcription.id
                ),
                current_status=current_status,
                expected_status="normalizing",
            )

        if not self._has_processed_text(
            transcription
        ):
            try:
                normalized_text = str(
                    transcription.normalized_text
                )

                post_processing_result = (
                    await self
                    ._run_provider_operation_with_retry_async(
                        operation=lambda: (
                            self
                            ._text_post_processing_service
                            .process_text_async(
                                normalized_text
                            )
                        ),
                        retryable_exceptions=(
                            TextPostProcessingProviderError,
                        ),
                    )
                )

            except TextPostProcessingValidationError:
                return await self._mark_failed_async(
                    transcription=transcription,
                    error_code=(
                        "post_processing_validation_failed"
                    ),
                    error_message=(
                        "LLM output failed "
                        "semantic-preservation validation"
                    ),
                )

            except TextPostProcessingProviderError:
                return await self._mark_failed_async(
                    transcription=transcription,
                    error_code=(
                        "llm_post_processing_failed"
                    ),
                    error_message=(
                        "LLM post-processing could "
                        "not be completed"
                    ),
                )

            except TextPostProcessingError:
                return await self._mark_failed_async(
                    transcription=transcription,
                    error_code="post_processing_failed",
                    error_message=(
                        "Text post-processing could "
                        "not be completed"
                    ),
                )

            transcription.processed_text = (
                post_processing_result.processed_text
            )

            transcription.post_processing_model = (
                post_processing_result.model
            )

            transcription.post_processing_version = (
                post_processing_result.version
            )

        transcription.error_code = None
        transcription.error_message = None

        if (
            getattr(
                transcription,
                "completed_datetime",
                None,
            )
            is None
        ):
            transcription.completed_datetime = (
                datetime.now(UTC)
            )

        self._set_status(
            transcription=transcription,
            status_value="completed",
        )

        return (
            await self._transcription_repository
            .update_async(
                transcription
            )
        )

    async def _mark_failed_async(
        self,
        *,
        transcription: Transcription,
        error_code: str,
        error_message: str,
    ) -> Transcription:
        self._set_status(
            transcription=transcription,
            status_value="failed",
        )

        transcription.error_code = error_code
        transcription.error_message = (
            error_message
        )

        return (
            await self._transcription_repository
            .update_async(
                transcription
            )
        )

    async def _run_provider_operation_with_retry_async(
        self,
        *,
        operation: Callable[
            [],
            Awaitable[T],
        ],
        retryable_exceptions: tuple[
            type[BaseException],
            ...,
        ],
        non_retryable_exceptions: tuple[
            type[BaseException],
            ...,
        ] = (),
    ) -> T:
        for attempt_number in range(
            1,
            self._provider_retry_max_attempts
            + 1,
        ):
            try:
                return await operation()

            except non_retryable_exceptions:
                raise

            except retryable_exceptions:
                if (
                    attempt_number
                    >= self
                    ._provider_retry_max_attempts
                ):
                    raise

                delay_seconds = (
                    self._get_provider_retry_delay(
                        failed_attempt_number=(
                            attempt_number
                        )
                    )
                )

                if delay_seconds > 0:
                    await asyncio.sleep(
                        delay_seconds
                    )

        raise RuntimeError(
            "Provider retry loop terminated "
            "unexpectedly"
        )

    def _get_provider_retry_delay(
        self,
        *,
        failed_attempt_number: int,
    ) -> float:
        delay_seconds = (
            self._provider_retry_backoff_seconds
            * (
                2
                ** (
                    failed_attempt_number
                    - 1
                )
            )
        )

        return min(
            delay_seconds,
            self
            ._provider_retry_max_backoff_seconds,
        )

    @staticmethod
    def _get_storage_key(
        transcription: Transcription,
    ) -> str:
        return str(
            getattr(
                transcription,
                "storage_key",
                "",
            )
            or ""
        ).strip()

    @staticmethod
    def _has_raw_text(
        transcription: Transcription,
    ) -> bool:
        return (
            getattr(
                transcription,
                "raw_text",
                None,
            )
            is not None
        )

    @staticmethod
    def _has_normalized_text(
        transcription: Transcription,
    ) -> bool:
        return (
            getattr(
                transcription,
                "normalized_text",
                None,
            )
            is not None
        )

    @staticmethod
    def _has_processed_text(
        transcription: Transcription,
    ) -> bool:
        return (
            getattr(
                transcription,
                "processed_text",
                None,
            )
            is not None
        )

    @staticmethod
    def _read_positive_float(
        *,
        config_reader: ConfigReader,
        key: str,
    ) -> float:
        raw_value = (
            config_reader.get_required(
                key
            )
        )

        if isinstance(
            raw_value,
            bool,
        ):
            raise ValueError(
                f"Configuration key '{key}' "
                "must be a positive number"
            )

        try:
            value = float(
                raw_value
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Configuration key '{key}' "
                "must be a positive number"
            ) from exc

        if value <= 0:
            raise ValueError(
                f"Configuration key '{key}' "
                "must be a positive number"
            )

        return value

    @staticmethod
    def _read_optional_positive_int(
        *,
        config_reader: ConfigReader,
        key: str,
        default: int,
    ) -> int:
        try:
            raw_value = (
                config_reader.get_required(
                    key
                )
            )
        except (
            KeyError,
            AttributeError,
        ):
            return default

        if isinstance(
            raw_value,
            bool,
        ):
            raise ValueError(
                f"Configuration key '{key}' "
                "must be a positive integer"
            )

        try:
            value = int(
                raw_value
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Configuration key '{key}' "
                "must be a positive integer"
            ) from exc

        if value <= 0:
            raise ValueError(
                f"Configuration key '{key}' "
                "must be a positive integer"
            )

        return value

    @staticmethod
    def _read_optional_non_negative_float(
        *,
        config_reader: ConfigReader,
        key: str,
        default: float,
    ) -> float:
        try:
            raw_value = (
                config_reader.get_required(
                    key
                )
            )
        except (
            KeyError,
            AttributeError,
        ):
            return default

        if isinstance(
            raw_value,
            bool,
        ):
            raise ValueError(
                f"Configuration key '{key}' "
                "must be a non-negative number"
            )

        try:
            value = float(
                raw_value
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Configuration key '{key}' "
                "must be a non-negative number"
            ) from exc

        if value < 0:
            raise ValueError(
                f"Configuration key '{key}' "
                "must be a non-negative number"
            )

        return value

    @staticmethod
    def _get_status_value(
        status: object,
    ) -> str:
        if isinstance(
            status,
            Enum,
        ):
            return str(
                status.value
            )

        return str(
            status
        )

    @staticmethod
    def _set_status(
        transcription: Transcription,
        status_value: str,
    ) -> None:
        current_status = (
            transcription.status
        )

        if isinstance(
            current_status,
            Enum,
        ):
            enum_type = type(
                current_status
            )

            transcription.status = (
                enum_type(
                    status_value
                )
            )

            return

        transcription.status = status_value