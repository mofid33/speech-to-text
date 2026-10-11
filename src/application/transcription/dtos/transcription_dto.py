from datetime import datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


TranscriptionStatusValue = Literal[
    "created",
    "uploaded",
    "queued",
    "validating",
    "transcribing",
    "normalizing",
    "post_processing",
    "completed",
    "failed",
    "cancelled",
]


class TranscriptionAcceptedDto(BaseModel):
    model_config = ConfigDict(
        from_attributes=True
    )

    id: str
    status: TranscriptionStatusValue

    original_filename: str | None = None
    content_type: str | None = None
    file_size_bytes: int | None = None

    creation_datetime: datetime
    modification_datetime: (
        datetime | None
    ) = None


class TranscriptionStatusDto(BaseModel):
    model_config = ConfigDict(
        from_attributes=True
    )

    id: str
    status: TranscriptionStatusValue

    original_filename: str | None = None
    content_type: str | None = None
    file_size_bytes: int | None = None
    duration_seconds: float | None = None

    requested_language: str
    detected_language: str | None = None

    error_code: str | None = None
    error_message: str | None = None

    creation_datetime: datetime
    started_datetime: datetime | None = None
    completed_datetime: datetime | None = None
    modification_datetime: (
        datetime | None
    ) = None


class TranscriptionSegmentResultDto(
    BaseModel
):
    model_config = ConfigDict(
        from_attributes=True
    )

    sequence: int

    start_time: float
    end_time: float

    raw_text: str
    normalized_text: str | None = None
    processed_text: str | None = None

    speaker_label: str | None = None


class TranscriptionResultDto(BaseModel):
    model_config = ConfigDict(
        from_attributes=True
    )

    id: str
    status: TranscriptionStatusValue

    requested_language: str
    detected_language: str | None = None

    asr_engine: str | None = None
    asr_model: str | None = None

    raw_text: str | None = None
    normalized_text: str | None = None
    processed_text: str | None = None

    post_processing_model: str | None = None
    post_processing_version: str | None = None

    error_code: str | None = None
    error_message: str | None = None

    creation_datetime: datetime
    started_datetime: datetime | None = None
    completed_datetime: datetime | None = None
    modification_datetime: (
        datetime | None
    ) = None

    segments: list[
        TranscriptionSegmentResultDto
    ] = Field(
        default_factory=list
    )


class TranscriptionErrorDetailDto(
    BaseModel
):
    code: str
    message: str
    transcription_id: str | None = None


class TranscriptionErrorResponseDto(
    BaseModel
):
    detail: TranscriptionErrorDetailDto


class TranscriptionUploadResponseDto(
    BaseModel
):
    """
    Backward-compatible DTO from earlier stages.
    """

    model_config = ConfigDict(
        from_attributes=True
    )

    id: str
    original_filename: str
    content_type: str
    file_size_bytes: int
    storage_key: str
    status: str
    creation_datetime: datetime
    modification_datetime: (
        datetime | None
    ) = None


class TranscriptionQueueStatusDto(
    BaseModel
):
    """
    Backward-compatible DTO from earlier stages.
    """

    model_config = ConfigDict(
        from_attributes=True
    )

    id: str
    status: str
    creation_datetime: datetime
    modification_datetime: (
        datetime | None
    ) = None