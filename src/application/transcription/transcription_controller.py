import asyncio
from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Response,
    UploadFile,
    status,
)

from src.application.transcription.dtos.transcription_dto import (
    TranscriptionAcceptedDto,
    TranscriptionErrorDetailDto,
    TranscriptionErrorResponseDto,
    TranscriptionResultDto,
    TranscriptionSegmentResultDto,
    TranscriptionStatusDto,
)
from src.core.services.transcription.transcription_service import (
    AudioFileTooLargeError,
    EmptyAudioFileError,
    TranscriptionInvalidStateError,
    TranscriptionNotFoundError,
    TranscriptionQueuePublishError,
    TranscriptionResultNotReadyError,
    TranscriptionRetryUnavailableError,
    TranscriptionService,
    TranscriptionStorageError,
    UnsupportedAudioFileError,
)
from src.infrastructure.di.inject import inject


@inject
class TranscriptionController:
    def __init__(
        self,
        transcription_service: TranscriptionService,
    ):
        self._transcription_service = (
            transcription_service
        )

    def api(self):
        router = APIRouter(
            prefix="",
            tags=["Transcription"],
        )

        @router.post(
            "/",
            response_model=TranscriptionAcceptedDto,
            status_code=status.HTTP_202_ACCEPTED,
            summary=(
                "Upload and queue transcription"
            ),
            responses={
                status.HTTP_400_BAD_REQUEST: {
                    "model": (
                        TranscriptionErrorResponseDto
                    ),
                },
                status.HTTP_413_CONTENT_TOO_LARGE: {
                    "model": (
                        TranscriptionErrorResponseDto
                    ),
                },
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {
                    "model": (
                        TranscriptionErrorResponseDto
                    ),
                },
                status.HTTP_503_SERVICE_UNAVAILABLE: {
                    "model": (
                        TranscriptionErrorResponseDto
                    ),
                },
            },
        )
        async def create_transcription(
            file: Annotated[
                UploadFile,
                File(
                    description=(
                        "Audio file to transcribe"
                    )
                ),
            ],
        ) -> TranscriptionAcceptedDto:
            try:
                file_size_bytes = (
                    file.size
                    if file.size is not None
                    else await (
                        self._get_file_size_async(
                            file
                        )
                    )
                )

                await file.seek(0)

                transcription = (
                    await self
                    ._transcription_service
                    .create_and_enqueue_from_upload_async(
                        filename=(
                            file.filename
                            or ""
                        ),
                        content_type=(
                            file.content_type
                        ),
                        file_size_bytes=(
                            file_size_bytes
                        ),
                        file_obj=file.file,
                    )
                )

                return (
                    TranscriptionAcceptedDto
                    .model_validate(
                        transcription
                    )
                )

            except EmptyAudioFileError as exc:
                raise HTTPException(
                    status_code=(
                        status.HTTP_400_BAD_REQUEST
                    ),
                    detail=self._error_detail(
                        code="empty_audio_file",
                        message=str(exc),
                    ),
                ) from exc

            except AudioFileTooLargeError as exc:
                raise HTTPException(
                    status_code=(
                        status
                        .HTTP_413_CONTENT_TOO_LARGE
                    ),
                    detail=self._error_detail(
                        code=(
                            "audio_file_too_large"
                        ),
                        message=str(exc),
                    ),
                ) from exc

            except UnsupportedAudioFileError as exc:
                raise HTTPException(
                    status_code=(
                        status
                        .HTTP_415_UNSUPPORTED_MEDIA_TYPE
                    ),
                    detail=self._error_detail(
                        code=(
                            "unsupported_audio_file"
                        ),
                        message=str(exc),
                    ),
                ) from exc

            except TranscriptionStorageError as exc:
                raise HTTPException(
                    status_code=(
                        status
                        .HTTP_503_SERVICE_UNAVAILABLE
                    ),
                    detail=self._error_detail(
                        code=(
                            "storage_unavailable"
                        ),
                        message=(
                            "Object storage is "
                            "temporarily unavailable"
                        ),
                    ),
                ) from exc

            except TranscriptionQueuePublishError as exc:
                raise HTTPException(
                    status_code=(
                        status
                        .HTTP_503_SERVICE_UNAVAILABLE
                    ),
                    detail=self._error_detail(
                        code="queue_unavailable",
                        message=str(exc),
                        transcription_id=(
                            exc.transcription_id
                        ),
                    ),
                ) from exc

            finally:
                await file.close()

        @router.get(
            "/{transcription_id}",
            response_model=TranscriptionStatusDto,
            summary=(
                "Get transcription status"
            ),
            responses={
                status.HTTP_404_NOT_FOUND: {
                    "model": (
                        TranscriptionErrorResponseDto
                    ),
                },
            },
        )
        @router.get(
            "/{transcription_id}/status",
            response_model=TranscriptionStatusDto,
            include_in_schema=False,
        )
        async def get_transcription_status(
            transcription_id: UUID,
        ) -> TranscriptionStatusDto:
            resolved_id = str(
                transcription_id
            )

            try:
                transcription = (
                    await self
                    ._transcription_service
                    .get_by_id_async(
                        resolved_id
                    )
                )

            except TranscriptionNotFoundError as exc:
                raise HTTPException(
                    status_code=(
                        status.HTTP_404_NOT_FOUND
                    ),
                    detail=self._error_detail(
                        code=(
                            "transcription_not_found"
                        ),
                        message=str(exc),
                        transcription_id=(
                            exc.transcription_id
                        ),
                    ),
                ) from exc

            return (
                TranscriptionStatusDto
                .model_validate(
                    transcription
                )
            )

        @router.get(
            "/{transcription_id}/result",
            response_model=TranscriptionResultDto,
            summary=(
                "Get transcription result"
            ),
            responses={
                status.HTTP_404_NOT_FOUND: {
                    "model": (
                        TranscriptionErrorResponseDto
                    ),
                },
                status.HTTP_409_CONFLICT: {
                    "model": (
                        TranscriptionErrorResponseDto
                    ),
                },
            },
        )
        async def get_transcription_result(
            transcription_id: UUID,
        ) -> TranscriptionResultDto:
            resolved_id = str(
                transcription_id
            )

            try:
                (
                    transcription,
                    segments,
                ) = (
                    await self
                    ._transcription_service
                    .get_result_async(
                        resolved_id
                    )
                )

            except TranscriptionNotFoundError as exc:
                raise HTTPException(
                    status_code=(
                        status.HTTP_404_NOT_FOUND
                    ),
                    detail=self._error_detail(
                        code=(
                            "transcription_not_found"
                        ),
                        message=str(exc),
                        transcription_id=(
                            exc.transcription_id
                        ),
                    ),
                ) from exc

            except TranscriptionResultNotReadyError as exc:
                raise HTTPException(
                    status_code=(
                        status.HTTP_409_CONFLICT
                    ),
                    detail=self._error_detail(
                        code=(
                            "transcription_result_"
                            "not_ready"
                        ),
                        message=str(exc),
                        transcription_id=(
                            exc.transcription_id
                        ),
                    ),
                ) from exc

            result = (
                TranscriptionResultDto
                .model_validate(
                    transcription
                )
            )

            return result.model_copy(
                update={
                    "segments": [
                        (
                            TranscriptionSegmentResultDto
                            .model_validate(
                                segment
                            )
                        )
                        for segment
                        in segments
                    ],
                }
            )

        @router.post(
            "/{transcription_id}/retry",
            response_model=TranscriptionAcceptedDto,
            status_code=status.HTTP_202_ACCEPTED,
            summary="Retry transcription",
            responses={
                status.HTTP_404_NOT_FOUND: {
                    "model": (
                        TranscriptionErrorResponseDto
                    ),
                },
                status.HTTP_409_CONFLICT: {
                    "model": (
                        TranscriptionErrorResponseDto
                    ),
                },
                status.HTTP_503_SERVICE_UNAVAILABLE: {
                    "model": (
                        TranscriptionErrorResponseDto
                    ),
                },
            },
        )
        async def retry_transcription(
            transcription_id: UUID,
        ) -> TranscriptionAcceptedDto:
            resolved_id = str(
                transcription_id
            )

            try:
                transcription = (
                    await self
                    ._transcription_service
                    .retry_async(
                        resolved_id
                    )
                )

            except TranscriptionNotFoundError as exc:
                raise HTTPException(
                    status_code=(
                        status.HTTP_404_NOT_FOUND
                    ),
                    detail=self._error_detail(
                        code=(
                            "transcription_not_found"
                        ),
                        message=str(exc),
                        transcription_id=(
                            exc.transcription_id
                        ),
                    ),
                ) from exc

            except TranscriptionRetryUnavailableError as exc:
                raise HTTPException(
                    status_code=(
                        status.HTTP_409_CONFLICT
                    ),
                    detail=self._error_detail(
                        code=(
                            "transcription_retry_"
                            "unavailable"
                        ),
                        message=str(exc),
                        transcription_id=(
                            exc.transcription_id
                        ),
                    ),
                ) from exc

            except TranscriptionInvalidStateError as exc:
                raise HTTPException(
                    status_code=(
                        status.HTTP_409_CONFLICT
                    ),
                    detail=self._error_detail(
                        code=(
                            "transcription_invalid_"
                            "state"
                        ),
                        message=str(exc),
                        transcription_id=(
                            exc.transcription_id
                        ),
                    ),
                ) from exc

            except TranscriptionQueuePublishError as exc:
                raise HTTPException(
                    status_code=(
                        status
                        .HTTP_503_SERVICE_UNAVAILABLE
                    ),
                    detail=self._error_detail(
                        code="queue_unavailable",
                        message=str(exc),
                        transcription_id=(
                            exc.transcription_id
                        ),
                    ),
                ) from exc

            return (
                TranscriptionAcceptedDto
                .model_validate(
                    transcription
                )
            )

        @router.delete(
            "/{transcription_id}",
            status_code=(
                status.HTTP_204_NO_CONTENT
            ),
            summary="Delete transcription",
            responses={
                status.HTTP_404_NOT_FOUND: {
                    "model": (
                        TranscriptionErrorResponseDto
                    ),
                },
                status.HTTP_409_CONFLICT: {
                    "model": (
                        TranscriptionErrorResponseDto
                    ),
                },
                status.HTTP_503_SERVICE_UNAVAILABLE: {
                    "model": (
                        TranscriptionErrorResponseDto
                    ),
                },
            },
        )
        async def delete_transcription(
            transcription_id: UUID,
        ) -> Response:
            resolved_id = str(
                transcription_id
            )

            try:
                await (
                    self._transcription_service
                    .delete_async(
                        resolved_id
                    )
                )

            except TranscriptionNotFoundError as exc:
                raise HTTPException(
                    status_code=(
                        status.HTTP_404_NOT_FOUND
                    ),
                    detail=self._error_detail(
                        code=(
                            "transcription_not_found"
                        ),
                        message=str(exc),
                        transcription_id=(
                            exc.transcription_id
                        ),
                    ),
                ) from exc

            except TranscriptionInvalidStateError as exc:
                raise HTTPException(
                    status_code=(
                        status.HTTP_409_CONFLICT
                    ),
                    detail=self._error_detail(
                        code=(
                            "transcription_invalid_"
                            "state"
                        ),
                        message=str(exc),
                        transcription_id=(
                            exc.transcription_id
                        ),
                    ),
                ) from exc

            except TranscriptionStorageError as exc:
                raise HTTPException(
                    status_code=(
                        status
                        .HTTP_503_SERVICE_UNAVAILABLE
                    ),
                    detail=self._error_detail(
                        code=(
                            "storage_unavailable"
                        ),
                        message=(
                            "Object storage is "
                            "temporarily unavailable"
                        ),
                        transcription_id=(
                            resolved_id
                        ),
                    ),
                ) from exc

            return Response(
                status_code=(
                    status.HTTP_204_NO_CONTENT
                )
            )

        return router

    @staticmethod
    def _error_detail(
        *,
        code: str,
        message: str,
        transcription_id: str | None = None,
    ) -> dict[str, str]:
        detail = TranscriptionErrorDetailDto(
            code=code,
            message=message,
            transcription_id=transcription_id,
        )

        return detail.model_dump(
            exclude_none=True
        )

    @staticmethod
    async def _get_file_size_async(
        file: UploadFile,
    ) -> int:
        current_position = (
            await asyncio.to_thread(
                file.file.tell
            )
        )

        try:
            await asyncio.to_thread(
                file.file.seek,
                0,
                2,
            )

            return await asyncio.to_thread(
                file.file.tell
            )

        finally:
            await asyncio.to_thread(
                file.file.seek,
                current_position,
            )