from enum import Enum
from pathlib import Path
from typing import BinaryIO

from src.infrastructure.di.inject import inject
from src.infrastructure.models.transcription import (
    Transcription,
    TranscriptionStatus,
)
from src.infrastructure.models.transcription_segment import (
    TranscriptionSegment,
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


class TranscriptionUploadValidationError(ValueError):
    """Base validation exception for transcription uploads."""


class EmptyAudioFileError(TranscriptionUploadValidationError):
    pass


class AudioFileTooLargeError(TranscriptionUploadValidationError):
    pass


class UnsupportedAudioFileError(TranscriptionUploadValidationError):
    pass


class TranscriptionStorageError(RuntimeError):
    pass


class TranscriptionNotFoundError(Exception):
    def __init__(self, transcription_id: str):
        self.transcription_id = transcription_id

        super().__init__(
            f"Transcription '{transcription_id}' was not found"
        )


class TranscriptionInvalidStateError(Exception):
    def __init__(
        self,
        *,
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


class TranscriptionQueuePublishError(RuntimeError):
    def __init__(self, transcription_id: str):
        self.transcription_id = transcription_id

        super().__init__(
            f"Could not enqueue transcription "
            f"'{transcription_id}'"
        )


class TranscriptionResultNotReadyError(Exception):
    def __init__(
        self,
        *,
        transcription_id: str,
        current_status: str,
    ):
        self.transcription_id = transcription_id
        self.current_status = current_status

        super().__init__(
            f"Transcription '{transcription_id}' result "
            f"is not available while status is "
            f"'{current_status}'"
        )


class TranscriptionRetryUnavailableError(Exception):
    def __init__(
        self,
        *,
        transcription_id: str,
        reason: str,
    ):
        self.transcription_id = transcription_id
        self.reason = reason

        super().__init__(
            f"Transcription '{transcription_id}' "
            f"cannot be retried: {reason}"
        )


@inject
class TranscriptionService:
    BYTES_PER_MEBIBYTE = 1024 * 1024

    _RESULT_STATUSES = {
        TranscriptionStatus.COMPLETED.value,
        TranscriptionStatus.FAILED.value,
        TranscriptionStatus.CANCELLED.value,
    }

    _DELETABLE_STATUSES = {
        TranscriptionStatus.CREATED.value,
        TranscriptionStatus.UPLOADED.value,
        TranscriptionStatus.COMPLETED.value,
        TranscriptionStatus.FAILED.value,
        TranscriptionStatus.CANCELLED.value,
    }

    def __init__(
        self,
        transcription_repository: TranscriptionRepository,
        transcription_segment_repository: TranscriptionSegmentRepository,
        storage_provider: S3StorageProvider,
        celery_queue_provider: CeleryQueueProvider,
        config_reader: ConfigReader,
    ):
        self._transcription_repository = (
            transcription_repository
        )
        self._transcription_segment_repository = (
            transcription_segment_repository
        )
        self._storage_provider = storage_provider
        self._celery_queue_provider = celery_queue_provider

        self._max_file_size_mb = self._read_positive_int(
            config_reader=config_reader,
            key="transcription.upload.max_file_size_mb",
        )

        self._max_file_size_bytes = (
            self._max_file_size_mb
            * self.BYTES_PER_MEBIBYTE
        )

        self._allowed_extensions = self._read_string_set(
            config_reader=config_reader,
            key="transcription.upload.allowed_extensions",
        )

        self._allowed_content_types = self._read_string_set(
            config_reader=config_reader,
            key=(
                "transcription.upload."
                "allowed_content_types"
            ),
        )

        invalid_extensions = {
            extension
            for extension
            in self._allowed_extensions
            if not extension.startswith(".")
        }

        if invalid_extensions:
            raise ValueError(
                "transcription.upload.allowed_extensions "
                "entries must start with '.'"
            )

    async def create_from_upload_async(
        self,
        *,
        filename: str,
        content_type: str | None,
        file_size_bytes: int,
        file_obj: BinaryIO,
    ) -> Transcription:
        normalized_filename = (
            filename.strip()
        )

        normalized_content_type = (
            content_type.strip().lower()
            if content_type
            else None
        )

        extension = self._validate_upload(
            filename=normalized_filename,
            content_type=normalized_content_type,
            file_size_bytes=file_size_bytes,
        )

        transcription = Transcription(
            original_filename=(
                normalized_filename
            ),
            content_type=(
                normalized_content_type
            ),
            file_size_bytes=(
                file_size_bytes
            ),
            storage_key=None,
            status=TranscriptionStatus.CREATED,
        )

        transcription = (
            await self._transcription_repository
            .insert_async(
                transcription
            )
        )

        storage_key = self._build_storage_key(
            transcription_id=str(
                transcription.id
            ),
            extension=extension,
        )

        try:
            await self._storage_provider.upload_file_async(
                file_obj=file_obj,
                object_key=storage_key,
                content_type=(
                    normalized_content_type
                ),
            )

        except StorageProviderError as exc:
            self._set_status(
                transcription=transcription,
                status_value=(
                    TranscriptionStatus.FAILED.value
                ),
            )

            try:
                await (
                    self._transcription_repository
                    .update_async(
                        transcription
                    )
                )
            except Exception:
                pass

            raise TranscriptionStorageError(
                "Audio upload to object storage failed"
            ) from exc

        transcription.storage_key = (
            storage_key
        )

        self._set_status(
            transcription=transcription,
            status_value=(
                TranscriptionStatus.UPLOADED.value
            ),
        )

        try:
            return (
                await self
                ._transcription_repository
                .update_async(
                    transcription
                )
            )

        except Exception:
            await (
                self
                ._compensate_storage_upload_async(
                    storage_key
                )
            )

            transcription.storage_key = None

            self._set_status(
                transcription=transcription,
                status_value=(
                    TranscriptionStatus.FAILED.value
                ),
            )

            try:
                await (
                    self._transcription_repository
                    .update_async(
                        transcription
                    )
                )
            except Exception:
                pass

            raise

    async def create_and_enqueue_from_upload_async(
        self,
        *,
        filename: str,
        content_type: str | None,
        file_size_bytes: int,
        file_obj: BinaryIO,
    ) -> Transcription:
        transcription = (
            await self.create_from_upload_async(
                filename=filename,
                content_type=content_type,
                file_size_bytes=file_size_bytes,
                file_obj=file_obj,
            )
        )

        return await self._enqueue_uploaded_async(
            transcription
        )

    async def get_by_id_async(
        self,
        transcription_id: str,
    ) -> Transcription:
        transcription = (
            await self._transcription_repository
            .get_by_id_async(
                transcription_id
            )
        )

        if transcription is None:
            raise TranscriptionNotFoundError(
                transcription_id
            )

        return transcription

    async def get_result_async(
        self,
        transcription_id: str,
    ) -> tuple[
        Transcription,
        list[TranscriptionSegment],
    ]:
        transcription = (
            await self.get_by_id_async(
                transcription_id
            )
        )

        current_status = (
            self._get_status_value(
                transcription.status
            )
        )

        if (
            current_status
            not in self._RESULT_STATUSES
        ):
            raise TranscriptionResultNotReadyError(
                transcription_id=(
                    transcription_id
                ),
                current_status=current_status,
            )

        segments = (
            await self
            ._transcription_segment_repository
            .get_by_transcription_id_async(
                transcription_id
            )
        )

        return (
            transcription,
            segments,
        )

    async def retry_async(
        self,
        transcription_id: str,
    ) -> Transcription:
        transcription = (
            await self.get_by_id_async(
                transcription_id
            )
        )

        current_status = (
            self._get_status_value(
                transcription.status
            )
        )

        if (
            current_status
            == TranscriptionStatus.UPLOADED.value
        ):
            return (
                await self
                ._enqueue_uploaded_async(
                    transcription
                )
            )

        if (
            current_status
            != TranscriptionStatus.FAILED.value
        ):
            raise TranscriptionInvalidStateError(
                transcription_id=(
                    transcription_id
                ),
                current_status=current_status,
                expected_status=(
                    "failed or uploaded"
                ),
            )

        if not self._get_storage_key(
            transcription
        ):
            raise (
                TranscriptionRetryUnavailableError(
                    transcription_id=(
                        transcription_id
                    ),
                    reason=(
                        "the source audio is not "
                        "available in object storage"
                    ),
                )
            )

        previous_error_code = getattr(
            transcription,
            "error_code",
            None,
        )

        previous_error_message = getattr(
            transcription,
            "error_message",
            None,
        )

        previous_completed_datetime = getattr(
            transcription,
            "completed_datetime",
            None,
        )

        retry_status = (
            self._get_retry_status(
                transcription
            )
        )

        self._set_status(
            transcription=transcription,
            status_value=retry_status,
        )

        transcription.error_code = None
        transcription.error_message = None
        transcription.completed_datetime = None

        transcription = (
            await self._transcription_repository
            .update_async(
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
                status_value=current_status,
            )

            transcription.error_code = (
                previous_error_code
            )

            transcription.error_message = (
                previous_error_message
            )

            transcription.completed_datetime = (
                previous_completed_datetime
            )

            await (
                self._transcription_repository
                .update_async(
                    transcription
                )
            )

            raise (
                TranscriptionQueuePublishError(
                    transcription_id
                )
            ) from exc

        return transcription

    async def delete_async(
        self,
        transcription_id: str,
    ) -> None:
        transcription = (
            await self.get_by_id_async(
                transcription_id
            )
        )

        current_status = (
            self._get_status_value(
                transcription.status
            )
        )

        if (
            current_status
            not in self._DELETABLE_STATUSES
        ):
            raise TranscriptionInvalidStateError(
                transcription_id=(
                    transcription_id
                ),
                current_status=current_status,
                expected_status=(
                    "created, uploaded, completed, "
                    "failed, or cancelled"
                ),
            )

        storage_key = (
            self._get_storage_key(
                transcription
            )
        )

        if storage_key:
            try:
                await (
                    self._storage_provider
                    .delete_file_async(
                        storage_key
                    )
                )

            except StorageProviderError as exc:
                raise TranscriptionStorageError(
                    "Audio deletion from "
                    "object storage failed"
                ) from exc

        deleted = (
            await self._transcription_repository
            .delete_async(
                transcription_id
            )
        )

        if not deleted:
            raise TranscriptionNotFoundError(
                transcription_id
            )

    async def _enqueue_uploaded_async(
        self,
        transcription: Transcription,
    ) -> Transcription:
        transcription_id = str(
            transcription.id
        )

        current_status = (
            self._get_status_value(
                transcription.status
            )
        )

        if (
            current_status
            != TranscriptionStatus.UPLOADED.value
        ):
            raise TranscriptionInvalidStateError(
                transcription_id=(
                    transcription_id
                ),
                current_status=current_status,
                expected_status=(
                    TranscriptionStatus
                    .UPLOADED.value
                ),
            )

        self._set_status(
            transcription=transcription,
            status_value=(
                TranscriptionStatus.QUEUED.value
            ),
        )

        transcription.error_code = None
        transcription.error_message = None

        transcription = (
            await self._transcription_repository
            .update_async(
                transcription
            )
        )

        try:
            await (
                self._celery_queue_provider
                .enqueue_transcription_async(
                    transcription_id=(
                        transcription_id
                    )
                )
            )

        except Exception as exc:
            self._set_status(
                transcription=transcription,
                status_value=(
                    TranscriptionStatus
                    .UPLOADED.value
                ),
            )

            await (
                self._transcription_repository
                .update_async(
                    transcription
                )
            )

            raise (
                TranscriptionQueuePublishError(
                    transcription_id
                )
            ) from exc

        return transcription

    def _validate_upload(
        self,
        *,
        filename: str,
        content_type: str | None,
        file_size_bytes: int,
    ) -> str:
        if not filename:
            raise UnsupportedAudioFileError(
                "Uploaded file must have a filename"
            )

        if file_size_bytes <= 0:
            raise EmptyAudioFileError(
                "Uploaded audio file is empty"
            )

        if (
            file_size_bytes
            > self._max_file_size_bytes
        ):
            raise AudioFileTooLargeError(
                "Audio file exceeds the "
                f"{self._max_file_size_mb} "
                "MB upload limit"
            )

        extension = (
            Path(filename)
            .suffix
            .lower()
        )

        if (
            extension
            not in self._allowed_extensions
        ):
            raise UnsupportedAudioFileError(
                "Unsupported audio extension "
                f"'{extension or '<none>'}'"
            )

        if (
            content_type
            and content_type
            not in self._allowed_content_types
        ):
            raise UnsupportedAudioFileError(
                "Unsupported content type "
                f"'{content_type}'"
            )

        return extension

    @staticmethod
    def _build_storage_key(
        *,
        transcription_id: str,
        extension: str,
    ) -> str:
        return (
            "transcriptions/"
            f"{transcription_id}/"
            f"source{extension}"
        )

    async def _compensate_storage_upload_async(
        self,
        storage_key: str,
    ) -> None:
        try:
            await (
                self._storage_provider
                .delete_file_async(
                    storage_key
                )
            )
        except StorageProviderError:
            pass

    @staticmethod
    def _get_retry_status(
        transcription: Transcription,
    ) -> str:
        if (
            getattr(
                transcription,
                "normalized_text",
                None,
            )
            is not None
        ):
            return (
                TranscriptionStatus
                .POST_PROCESSING.value
            )

        if (
            getattr(
                transcription,
                "raw_text",
                None,
            )
            is not None
        ):
            return (
                TranscriptionStatus
                .NORMALIZING.value
            )

        return (
            TranscriptionStatus
            .QUEUED.value
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
        *,
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
            transcription.status = (
                type(
                    current_status
                )(
                    status_value
                )
            )
            return

        transcription.status = (
            status_value
        )

    @staticmethod
    def _read_positive_int(
        *,
        config_reader: ConfigReader,
        key: str,
    ) -> int:
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
    def _read_string_set(
        *,
        config_reader: ConfigReader,
        key: str,
    ) -> set[str]:
        raw_value = (
            config_reader.get_required(
                key
            )
        )

        if (
            not isinstance(
                raw_value,
                list,
            )
            or any(
                not isinstance(
                    item,
                    str,
                )
                for item
                in raw_value
            )
        ):
            raise ValueError(
                f"Configuration key '{key}' "
                "must be a YAML list of strings"
            )

        values = {
            item.strip().lower()
            for item
            in raw_value
            if item.strip()
        }

        if not values:
            raise ValueError(
                f"Configuration key '{key}' "
                "must not be empty"
            )

        return values