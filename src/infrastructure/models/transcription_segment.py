from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.sql import func

from src.infrastructure.context.sql_db.psql_dbcontext import Base


class TranscriptionSegment(Base):
    __tablename__ = "transcription_segment"

    __table_args__ = (
        UniqueConstraint(
            "transcription_id",
            "sequence",
            name="uq_transcription_segment_transcription_sequence",
        ),
        CheckConstraint(
            "sequence >= 0",
            name="ck_transcription_segment_sequence_non_negative",
        ),
        CheckConstraint(
            "start_time >= 0",
            name="ck_transcription_segment_start_time_non_negative",
        ),
        CheckConstraint(
            "end_time >= start_time",
            name="ck_transcription_segment_time_range",
        ),
    )

    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )

    transcription_id = Column(
        String(36),
        ForeignKey(
            "transcription.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    sequence = Column(
        Integer,
        nullable=False,
    )

    start_time = Column(
        Float,
        nullable=False,
    )

    end_time = Column(
        Float,
        nullable=False,
    )

    raw_text = Column(
        Text,
        nullable=False,
    )

    normalized_text = Column(
        Text,
        nullable=True,
    )

    processed_text = Column(
        Text,
        nullable=True,
    )

    speaker_label = Column(
        String(100),
        nullable=True,
    )

    creation_datetime = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
    )

    modification_datetime = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        onupdate=lambda: datetime.now(UTC),
    )