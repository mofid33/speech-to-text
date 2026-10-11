"""create transcription tables

Revision ID: 20260830_0001
Revises:
Create Date: 2026-08-30
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260830_0001"

down_revision: Union[
    str,
    Sequence[str],
    None,
] = None

branch_labels: Union[
    str,
    Sequence[str],
    None,
] = None

depends_on: Union[
    str,
    Sequence[str],
    None,
] = None


def upgrade() -> None:
    op.create_table(
        "transcription",
        sa.Column(
            "id",
            sa.String(length=36),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(length=32),
            server_default=sa.text("'created'"),
            nullable=False,
        ),
        sa.Column(
            "original_filename",
            sa.String(length=512),
            nullable=True,
        ),
        sa.Column(
            "content_type",
            sa.String(length=255),
            nullable=True,
        ),
        sa.Column(
            "file_size_bytes",
            sa.BigInteger(),
            nullable=True,
        ),
        sa.Column(
            "duration_seconds",
            sa.Float(),
            nullable=True,
        ),
        sa.Column(
            "storage_key",
            sa.String(length=1024),
            nullable=True,
        ),
        sa.Column(
            "requested_language",
            sa.String(length=32),
            server_default=sa.text("'fa'"),
            nullable=False,
        ),
        sa.Column(
            "detected_language",
            sa.String(length=32),
            nullable=True,
        ),
        sa.Column(
            "asr_engine",
            sa.String(length=100),
            nullable=True,
        ),
        sa.Column(
            "asr_model",
            sa.String(length=255),
            nullable=True,
        ),
        sa.Column(
            "raw_text",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "normalized_text",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "processed_text",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "post_processing_model",
            sa.String(length=255),
            nullable=True,
        ),
        sa.Column(
            "post_processing_version",
            sa.String(length=100),
            nullable=True,
        ),
        sa.Column(
            "error_code",
            sa.String(length=100),
            nullable=True,
        ),
        sa.Column(
            "error_message",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "creation_datetime",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "started_datetime",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "completed_datetime",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "modification_datetime",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ("
            "'created', 'uploaded', 'queued', "
            "'validating', 'transcribing', "
            "'normalizing', 'post_processing', "
            "'completed', 'failed', 'cancelled'"
            ")",
            name="ck_transcription_status",
        ),
        sa.CheckConstraint(
            "file_size_bytes IS NULL "
            "OR file_size_bytes >= 0",
            name=(
                "ck_transcription_"
                "file_size_non_negative"
            ),
        ),
        sa.CheckConstraint(
            "duration_seconds IS NULL "
            "OR duration_seconds >= 0",
            name=(
                "ck_transcription_"
                "duration_non_negative"
            ),
        ),
        sa.PrimaryKeyConstraint(
            "id",
        ),
        sa.UniqueConstraint(
            "storage_key",
            name="uq_transcription_storage_key",
        ),
    )

    op.create_index(
        "ix_transcription_creation_datetime",
        "transcription",
        ["creation_datetime"],
        unique=False,
    )

    op.create_index(
        "ix_transcription_status",
        "transcription",
        ["status"],
        unique=False,
    )

    op.create_table(
        "transcription_segment",
        sa.Column(
            "id",
            sa.String(length=36),
            nullable=False,
        ),
        sa.Column(
            "transcription_id",
            sa.String(length=36),
            nullable=False,
        ),
        sa.Column(
            "sequence",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "start_time",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "end_time",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "raw_text",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "normalized_text",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "processed_text",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "speaker_label",
            sa.String(length=100),
            nullable=True,
        ),
        sa.Column(
            "creation_datetime",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "modification_datetime",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "sequence >= 0",
            name=(
                "ck_transcription_segment_"
                "sequence_non_negative"
            ),
        ),
        sa.CheckConstraint(
            "start_time >= 0",
            name=(
                "ck_transcription_segment_"
                "start_time_non_negative"
            ),
        ),
        sa.CheckConstraint(
            "end_time >= start_time",
            name=(
                "ck_transcription_segment_"
                "time_range"
            ),
        ),
        sa.ForeignKeyConstraint(
            ["transcription_id"],
            ["transcription.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "id",
        ),
        sa.UniqueConstraint(
            "transcription_id",
            "sequence",
            name=(
                "uq_transcription_segment_"
                "transcription_sequence"
            ),
        ),
    )

    op.create_index(
        "ix_transcription_segment_transcription_id",
        "transcription_segment",
        ["transcription_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_transcription_segment_transcription_id",
        table_name="transcription_segment",
    )

    op.drop_table(
        "transcription_segment"
    )

    op.drop_index(
        "ix_transcription_status",
        table_name="transcription",
    )

    op.drop_index(
        "ix_transcription_creation_datetime",
        table_name="transcription",
    )

    op.drop_table(
        "transcription"
    )