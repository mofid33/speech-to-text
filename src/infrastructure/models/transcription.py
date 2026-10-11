from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    String,
    Text,
)
from sqlalchemy.sql import func

from src.infrastructure.context.sql_db.psql_dbcontext import Base


class TranscriptionStatus(str, Enum):
    CREATED = "created"
    UPLOADED = "uploaded"
    QUEUED = "queued"
    VALIDATING = "validating"
    TRANSCRIBING = "transcribing"
    NORMALIZING = "normalizing"
    POST_PROCESSING = "post_processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Transcription(Base):
    __tablename__ = "transcription"

    __table_args__ = (
        CheckConstraint(
            "status IN ("
            "'created', 'uploaded', 'queued', 'validating', 'transcribing', "
            "'normalizing', 'post_processing', 'completed', 'failed', 'cancelled'"
            ")",
            name="ck_transcription_status",
        ),
        CheckConstraint(
            "file_size_bytes IS NULL OR file_size_bytes >= 0",
            name="ck_transcription_file_size_non_negative",
        ),
        CheckConstraint(
            "duration_seconds IS NULL OR duration_seconds >= 0",
            name="ck_transcription_duration_non_negative",
        ),
    )

    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )

    status = Column(
        String(32),
        nullable=False,
        index=True,
        default=TranscriptionStatus.CREATED.value,
        server_default=TranscriptionStatus.CREATED.value,
    )

    original_filename = Column(
        String(512),
        nullable=True,
    )

    content_type = Column(
        String(255),
        nullable=True,
    )

    file_size_bytes = Column(
        BigInteger,
        nullable=True,
    )

    duration_seconds = Column(
        Float,
        nullable=True,
    )

    storage_key = Column(
        String(1024),
        nullable=True,
        unique=True,
    )

    requested_language = Column(
        String(32),
        nullable=False,
        default="fa",
        server_default="fa",
    )

    detected_language = Column(
        String(32),
        nullable=True,
    )

    asr_engine = Column(
        String(100),
        nullable=True,
    )

    asr_model = Column(
        String(255),
        nullable=True,
    )

    raw_text = Column(
        Text,
        nullable=True,
    )

    normalized_text = Column(
        Text,
        nullable=True,
    )

    processed_text = Column(
        Text,
        nullable=True,
    )

    post_processing_model = Column(
        String(255),
        nullable=True,
    )

    post_processing_version = Column(
        String(100),
        nullable=True,
    )

    error_code = Column(
        String(100),
        nullable=True,
    )

    error_message = Column(
        Text,
        nullable=True,
    )

    creation_datetime = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
    )

    started_datetime = Column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_datetime = Column(
        DateTime(timezone=True),
        nullable=True,
    )

    modification_datetime = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        onupdate=lambda: datetime.now(UTC),
    )