from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import mimetypes
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import httpx

from src.core.services.transcription.persian_text_normalization_service import (
    PersianTextNormalizationService,
)
from src.core.services.transcription.text_post_processing_service import (
    TextPostProcessingService,
)
from src.infrastructure.di.bootstrap import bootstrap_di
from src.infrastructure.di.inject import resolve
from src.infrastructure.providers.queue.celery_queue_provider import (
    CeleryQueueProvider,
)
from src.infrastructure.providers.storage.s3_storage_provider import (
    S3StorageProvider,
)
from src.infrastructure.repositories.transcription.transcription_repository import (
    TranscriptionRepository,
)
from src.infrastructure.repositories.transcription.transcription_segment_repository import (
    TranscriptionSegmentRepository,
)
from src.infrastructure.utils.config_reader import ConfigReader


TERMINAL_STATUSES = {
    "completed",
    "failed",
    "cancelled",
}


class FinalVerificationError(RuntimeError):
    pass


def _utc_now_iso() -> str:
    return datetime.now(
        UTC
    ).isoformat()


def _sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as stream:
        for chunk in iter(
            lambda: stream.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def _resolve_api_base_url(
    config_reader: ConfigReader,
    override: str | None,
) -> str:
    if override:
        value = (
            override
            .strip()
            .rstrip("/")
        )

        if not value:
            raise FinalVerificationError(
                "--api-base-url must not be empty"
            )

        return value

    host = str(
        config_reader.get_required(
            "server.host"
        )
    ).strip()

    port = int(
        config_reader.get_required(
            "server.port"
        )
    )

    if host in {
        "0.0.0.0",
        "::",
        "[::]",
    }:
        host = "127.0.0.1"

    return (
        f"http://{host}:{port}"
    )


def _require_status(
    response: httpx.Response,
    expected: int,
    operation: str,
) -> None:
    if (
        response.status_code
        == expected
    ):
        return

    body = (
        response.text
        .strip()
    )

    if len(
        body
    ) > 2000:
        body = (
            body[:2000]
            + "..."
        )

    raise FinalVerificationError(
        f"{operation} failed: "
        f"expected HTTP {expected}, "
        f"got HTTP {response.status_code}. "
        f"Body: {body}"
    )


def _json_object(
    response: httpx.Response,
    operation: str,
) -> dict[
    str,
    Any,
]:
    try:
        payload = (
            response.json()
        )

    except ValueError as exc:
        raise FinalVerificationError(
            f"{operation} returned invalid JSON"
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        raise FinalVerificationError(
            f"{operation} did not return "
            "a JSON object"
        )

    return payload


async def _upload_audio_async(
    client: httpx.AsyncClient,
    transcription_url: str,
    audio_path: Path,
    content_type: str | None = None,
) -> dict[
    str,
    Any,
]:
    resolved_content_type = (
        content_type
        or mimetypes.guess_type(
            audio_path.name
        )[0]
        or "application/octet-stream"
    )

    with audio_path.open(
        "rb"
    ) as stream:
        response = (
            await client.post(
                f"{transcription_url}/",
                files={
                    "file": (
                        audio_path.name,
                        stream,
                        resolved_content_type,
                    ),
                },
            )
        )

    _require_status(
        response,
        202,
        "Transcription upload",
    )

    payload = _json_object(
        response,
        "Transcription upload",
    )

    if not str(
        payload.get(
            "id"
        )
        or ""
    ).strip():
        raise FinalVerificationError(
            "Transcription upload "
            "did not return an id"
        )

    return payload


async def _wait_for_terminal_status_async(
    client: httpx.AsyncClient,
    transcription_url: str,
    transcription_id: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
    initial_status: str | None = None,
) -> tuple[
    dict[
        str,
        Any,
    ],
    list[str],
]:
    deadline = (
        asyncio
        .get_running_loop()
        .time()
        + timeout_seconds
    )

    history = (
        [
            initial_status
        ]
        if initial_status
        else []
    )

    while True:
        response = (
            await client.get(
                f"{transcription_url}/"
                f"{transcription_id}"
            )
        )

        _require_status(
            response,
            200,
            "Transcription status",
        )

        payload = _json_object(
            response,
            "Transcription status",
        )

        status = str(
            payload.get(
                "status"
            )
            or ""
        ).strip()

        if not status:
            raise FinalVerificationError(
                "Status response did not "
                "contain status"
            )

        if (
            not history
            or history[-1]
            != status
        ):
            history.append(
                status
            )

        if (
            status
            in TERMINAL_STATUSES
        ):
            return (
                payload,
                history,
            )

        if (
            asyncio
            .get_running_loop()
            .time()
            >= deadline
        ):
            raise FinalVerificationError(
                "Timed out waiting for "
                f"'{transcription_id}'. "
                f"Statuses: {history}"
            )

        await asyncio.sleep(
            poll_interval_seconds
        )


async def _get_result_async(
    client: httpx.AsyncClient,
    transcription_url: str,
    transcription_id: str,
) -> dict[
    str,
    Any,
]:
    response = (
        await client.get(
            f"{transcription_url}/"
            f"{transcription_id}/"
            "result"
        )
    )

    _require_status(
        response,
        200,
        "Transcription result",
    )

    return _json_object(
        response,
        "Transcription result",
    )


def _validate_completed_result(
    result: dict[
        str,
        Any,
    ],
    llm_enabled: bool,
) -> dict[
    str,
    Any,
]:
    if (
        result.get(
            "status"
        )
        != "completed"
    ):
        raise FinalVerificationError(
            "Main transcription did not "
            "complete: "
            "status="
            f"{result.get('status')}, "
            "error_code="
            f"{result.get('error_code')}"
        )

    texts: dict[
        str,
        str,
    ] = {}

    for name in (
        "raw_text",
        "normalized_text",
        "processed_text",
    ):
        value = result.get(
            name
        )

        if (
            not isinstance(
                value,
                str,
            )
            or not value.strip()
        ):
            raise FinalVerificationError(
                "Completed result contains "
                f"empty {name}; "
                "use a real speech "
                "audio file"
            )

        texts[
            name
        ] = value

    for name in (
        "asr_engine",
        "asr_model",
        "post_processing_version",
    ):
        if not str(
            result.get(
                name
            )
            or ""
        ).strip():
            raise FinalVerificationError(
                "Provenance field "
                f"'{name}' "
                "was not persisted"
            )

    version = str(
        result[
            "post_processing_version"
        ]
    )

    model = result.get(
        "post_processing_model"
    )

    if llm_enabled:
        if not str(
            model
            or ""
        ).strip():
            raise FinalVerificationError(
                "LLM is enabled but "
                "post_processing_model "
                "is empty"
            )

        if (
            version
            != (
                TextPostProcessingService
                .PROMPT_VERSION
            )
        ):
            raise FinalVerificationError(
                "Unexpected prompt version: "
                f"{version}"
            )

    else:
        if (
            model
            is not None
        ):
            raise FinalVerificationError(
                "LLM is disabled but "
                "post_processing_model "
                "is not null"
            )

        if (
            version
            != (
                TextPostProcessingService
                .DISABLED_VERSION
            )
        ):
            raise FinalVerificationError(
                "Unexpected disabled "
                "post-processing version: "
                f"{version}"
            )

        if (
            texts[
                "processed_text"
            ]
            != texts[
                "normalized_text"
            ]
        ):
            raise FinalVerificationError(
                "LLM is disabled but "
                "processed_text != "
                "normalized_text"
            )

    segments = result.get(
        "segments"
    )

    if (
        not isinstance(
            segments,
            list,
        )
        or not segments
    ):
        raise FinalVerificationError(
            "Completed result does not "
            "contain segments"
        )

    return {
        **texts,
        "raw_equals_normalized": (
            texts[
                "raw_text"
            ]
            == texts[
                "normalized_text"
            ]
        ),
        "normalized_equals_processed": (
            texts[
                "normalized_text"
            ]
            == texts[
                "processed_text"
            ]
        ),
        "raw_length": len(
            texts[
                "raw_text"
            ]
        ),
        "normalized_length": len(
            texts[
                "normalized_text"
            ]
        ),
        "processed_length": len(
            texts[
                "processed_text"
            ]
        ),
        "segment_count": len(
            segments
        ),
    }


async def _verify_database_async(
    transcription_repository: (
        TranscriptionRepository
    ),
    segment_repository: (
        TranscriptionSegmentRepository
    ),
    transcription_id: str,
    api_result: dict[
        str,
        Any,
    ],
) -> tuple[
    dict[
        str,
        Any,
    ],
    Any,
]:
    transcription = (
        await transcription_repository
        .get_by_id_async(
            transcription_id
        )
    )

    if (
        transcription
        is None
    ):
        raise FinalVerificationError(
            "Transcription was not found "
            "in PostgreSQL"
        )

    if (
        str(
            transcription.status
        )
        != "completed"
    ):
        raise FinalVerificationError(
            "PostgreSQL status is not "
            "completed: "
            f"{transcription.status}"
        )

    fields = (
        "raw_text",
        "normalized_text",
        "processed_text",
        "asr_engine",
        "asr_model",
        "post_processing_model",
        "post_processing_version",
    )

    mismatches = [
        name
        for name in fields
        if getattr(
            transcription,
            name,
            None,
        )
        != api_result.get(
            name
        )
    ]

    if mismatches:
        raise FinalVerificationError(
            "API and PostgreSQL differ for: "
            + ", ".join(
                mismatches
            )
        )

    db_segments = (
        await segment_repository
        .get_by_transcription_id_async(
            transcription_id
        )
    )

    api_segments = (
        api_result.get(
            "segments"
        )
    )

    if (
        not isinstance(
            api_segments,
            list,
        )
        or len(
            db_segments
        )
        != len(
            api_segments
        )
    ):
        raise FinalVerificationError(
            "PostgreSQL/API segment "
            "counts differ"
        )

    for (
        index,
        (
            db_segment,
            api_segment,
        ),
    ) in enumerate(
        zip(
            db_segments,
            api_segments,
            strict=True,
        )
    ):
        for name in (
            "sequence",
            "start_time",
            "end_time",
            "raw_text",
            "normalized_text",
            "processed_text",
            "speaker_label",
        ):
            if (
                getattr(
                    db_segment,
                    name,
                    None,
                )
                != api_segment.get(
                    name
                )
            ):
                raise FinalVerificationError(
                    "PostgreSQL/API segment "
                    "mismatch: "
                    f"index={index}, "
                    f"field={name}"
                )

    storage_key = str(
        getattr(
            transcription,
            "storage_key",
            "",
        )
        or ""
    ).strip()

    if not storage_key:
        raise FinalVerificationError(
            "PostgreSQL record does not "
            "contain storage_key"
        )

    return (
        {
            "status": str(
                transcription.status
            ),
            "storage_key": (
                storage_key
            ),
            "segment_count": len(
                db_segments
            ),
            "asr_engine": (
                transcription
                .asr_engine
            ),
            "asr_model": (
                transcription
                .asr_model
            ),
            "post_processing_model": (
                transcription
                .post_processing_model
            ),
            "post_processing_version": (
                transcription
                .post_processing_version
            ),
        },
        transcription,
    )


async def _verify_storage_async(
    storage_provider: (
        S3StorageProvider
    ),
    storage_key: str,
    source_audio_path: Path,
) -> dict[
    str,
    Any,
]:
    source_sha256 = (
        await asyncio.to_thread(
            _sha256_file,
            source_audio_path,
        )
    )

    source_size = (
        source_audio_path
        .stat()
        .st_size
    )

    with TemporaryDirectory(
        prefix=(
            "speech_to_text_"
            "stage14_storage_"
        )
    ) as temp_dir:
        destination = (
            Path(
                temp_dir
            )
            / (
                "source"
                f"{source_audio_path.suffix.lower()}"
            )
        )

        await (
            storage_provider
            .download_file_async(
                storage_key,
                destination,
            )
        )

        downloaded_sha256 = (
            await asyncio.to_thread(
                _sha256_file,
                destination,
            )
        )

        downloaded_size = (
            destination
            .stat()
            .st_size
        )

    if (
        downloaded_size
        != source_size
        or downloaded_sha256
        != source_sha256
    ):
        raise FinalVerificationError(
            "Object Storage content "
            "differs from the "
            "uploaded source"
        )

    return {
        "storage_key": (
            storage_key
        ),
        "size_bytes": (
            source_size
        ),
        "sha256": (
            source_sha256
        ),
        "content_match": True,
    }


def _queue_preflight_sync(
    queue_provider: (
        CeleryQueueProvider
    ),
    worker_ping_timeout_seconds: float,
) -> dict[
    str,
    Any,
]:
    with (
        queue_provider
        .celery_app
        .connection_for_read()
    ) as connection:
        connection.ensure_connection(
            max_retries=1
        )

    replies = (
        queue_provider
        .celery_app
        .control
        .ping(
            timeout=(
                worker_ping_timeout_seconds
            )
        )
    )

    if not replies:
        raise FinalVerificationError(
            "RabbitMQ is reachable, "
            "but no Celery Worker "
            "responded to ping"
        )

    workers = sorted(
        {
            str(
                name
            )
            for reply in replies
            if isinstance(
                reply,
                dict,
            )
            for name
            in reply.keys()
        }
    )

    return {
        "broker_connected": True,
        "queue_name": (
            queue_provider
            .queue_name
        ),
        "worker_ping_replies": len(
            replies
        ),
        "workers": workers,
    }


async def _verify_retry_flow_async(
    client: httpx.AsyncClient,
    transcription_url: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
    cleanup: bool,
) -> dict[
    str,
    Any,
]:
    with TemporaryDirectory(
        prefix=(
            "speech_to_text_"
            "stage14_retry_"
        )
    ) as temp_dir:
        invalid_audio = (
            Path(
                temp_dir
            )
            / "stage14-invalid.wav"
        )

        invalid_audio.write_bytes(
            b"speech-to-text-"
            b"stage14-invalid-audio"
        )

        accepted = (
            await _upload_audio_async(
                client,
                transcription_url,
                invalid_audio,
                "audio/wav",
            )
        )

    transcription_id = str(
        accepted[
            "id"
        ]
    )

    (
        first_terminal,
        first_history,
    ) = (
        await _wait_for_terminal_status_async(
            client,
            transcription_url,
            transcription_id,
            timeout_seconds,
            poll_interval_seconds,
            str(
                accepted.get(
                    "status"
                )
                or ""
            ),
        )
    )

    if (
        first_terminal.get(
            "status"
        )
        != "failed"
    ):
        raise FinalVerificationError(
            "Controlled invalid-audio "
            "job did not fail"
        )

    first_error = str(
        first_terminal.get(
            "error_code"
        )
        or ""
    ).strip()

    if not first_error:
        raise FinalVerificationError(
            "Controlled failure did not "
            "persist error_code"
        )

    retry_response = (
        await client.post(
            f"{transcription_url}/"
            f"{transcription_id}/"
            "retry"
        )
    )

    _require_status(
        retry_response,
        202,
        "Transcription retry",
    )

    retry_accepted = (
        _json_object(
            retry_response,
            "Transcription retry",
        )
    )

    retry_status = str(
        retry_accepted.get(
            "status"
        )
        or ""
    ).strip()

    if (
        not retry_status
        or retry_status
        == "failed"
    ):
        raise FinalVerificationError(
            "Retry endpoint did not "
            "resume the failed job"
        )

    (
        second_terminal,
        second_history,
    ) = (
        await _wait_for_terminal_status_async(
            client,
            transcription_url,
            transcription_id,
            timeout_seconds,
            poll_interval_seconds,
            retry_status,
        )
    )

    if (
        second_terminal.get(
            "status"
        )
        != "failed"
    ):
        raise FinalVerificationError(
            "Retried invalid-audio job "
            "did not fail again"
        )

    second_error = str(
        second_terminal.get(
            "error_code"
        )
        or ""
    ).strip()

    if not second_error:
        raise FinalVerificationError(
            "Retried failure did not "
            "persist error_code"
        )

    failed_result = (
        await _get_result_async(
            client,
            transcription_url,
            transcription_id,
        )
    )

    cleanup_status = (
        "kept"
    )

    if cleanup:
        delete_response = (
            await client.delete(
                f"{transcription_url}/"
                f"{transcription_id}"
            )
        )

        _require_status(
            delete_response,
            204,
            "Retry verification cleanup",
        )

        cleanup_status = (
            "deleted"
        )

    return {
        "transcription_id": (
            transcription_id
        ),
        "first_status_history": (
            first_history
        ),
        "first_error_code": (
            first_error
        ),
        "retry_accepted_status": (
            retry_status
        ),
        "second_status_history": (
            second_history
        ),
        "second_error_code": (
            second_error
        ),
        "failed_result_status": (
            failed_result.get(
                "status"
            )
        ),
        "cleanup": (
            cleanup_status
        ),
    }


def _read_json_log_entries(
    path: Path,
    transcription_id: str,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    if not path.is_file():
        raise FinalVerificationError(
            "Log file was not found: "
            f"{path}"
        )

    target = (
        transcription_id
        .strip()
        .lower()
    )

    matches: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for raw_line in (
        path.read_text(
            encoding="utf-8",
            errors="replace",
        )
        .splitlines()
    ):
        line = (
            raw_line
            .strip()
        )

        json_start = (
            line.find(
                "{"
            )
        )

        if json_start < 0:
            continue

        try:
            payload = json.loads(
                line[
                    json_start:
                ]
            )

        except json.JSONDecodeError:
            continue

        if not isinstance(
            payload,
            dict,
        ):
            continue

        current = str(
            payload.get(
                "transcription_id"
            )
            or ""
        ).strip().lower()

        if (
            current
            == target
        ):
            matches.append(
                payload
            )

    return matches


def _event_count(
    entries: list[
        dict[
            str,
            Any,
        ]
    ],
    event_name: str,
) -> int:
    return sum(
        1
        for entry in entries
        if entry.get(
            "event"
        )
        == event_name
    )


def _verify_logs(
    api_log_file: Path,
    worker_log_file: Path,
    main_id: str,
    retry_id: str,
) -> dict[
    str,
    Any,
]:
    main_api = (
        _read_json_log_entries(
            api_log_file,
            main_id,
        )
    )

    main_worker = (
        _read_json_log_entries(
            worker_log_file,
            main_id,
        )
    )

    retry_api = (
        _read_json_log_entries(
            api_log_file,
            retry_id,
        )
    )

    retry_worker = (
        _read_json_log_entries(
            worker_log_file,
            retry_id,
        )
    )

    checks = {
        "main_api_request_completed": (
            _event_count(
                main_api,
                "api_request_completed",
            )
            >= 1
        ),
        "main_queue_publish_completed": (
            _event_count(
                main_api,
                "queue_publish_completed",
            )
            >= 1
        ),
        "main_worker_job_started": (
            _event_count(
                main_worker,
                "worker_job_started",
            )
            >= 1
        ),
        "main_worker_job_finished": (
            _event_count(
                main_worker,
                "worker_job_finished",
            )
            >= 1
        ),
        "retry_queue_published_twice": (
            _event_count(
                retry_api,
                "queue_publish_completed",
            )
            >= 2
        ),
        "retry_worker_started_twice": (
            _event_count(
                retry_worker,
                "worker_job_started",
            )
            >= 2
        ),
        "retry_worker_finished_twice": (
            _event_count(
                retry_worker,
                "worker_job_finished",
            )
            >= 2
        ),
    }

    failed = [
        name
        for (
            name,
            passed,
        ) in checks.items()
        if not passed
    ]

    if failed:
        raise FinalVerificationError(
            "Structured log verification "
            "failed: "
            + ", ".join(
                failed
            )
        )

    return {
        "status": "verified",
        "api_log_file": str(
            api_log_file
        ),
        "worker_log_file": str(
            worker_log_file
        ),
        "checks": checks,
    }


async def _wait_for_log_verification_async(
    api_log_file: Path,
    worker_log_file: Path,
    main_id: str,
    retry_id: str,
    timeout_seconds: float,
) -> dict[
    str,
    Any,
]:
    deadline = (
        asyncio
        .get_running_loop()
        .time()
        + timeout_seconds
    )

    last_error: (
        FinalVerificationError
        | None
    ) = None

    while True:
        try:
            return _verify_logs(
                api_log_file,
                worker_log_file,
                main_id,
                retry_id,
            )

        except FinalVerificationError as exc:
            last_error = (
                exc
            )

        if (
            asyncio
            .get_running_loop()
            .time()
            >= deadline
        ):
            assert (
                last_error
                is not None
            )

            raise last_error

        await asyncio.sleep(
            0.25
        )


def _git_provenance() -> dict[
    str,
    Any,
]:
    def run(
        *args: str,
    ) -> subprocess.CompletedProcess[
        str
    ]:
        return subprocess.run(
            [
                "git",
                *args,
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    try:
        commit = run(
            "rev-parse",
            "HEAD",
        )

    except OSError:
        return {
            "available": False,
        }

    if (
        commit.returncode
        != 0
    ):
        return {
            "available": False,
        }

    status = run(
        "status",
        "--porcelain",
    )

    return {
        "available": True,
        "commit": (
            commit.stdout
            .strip()
        ),
        "working_tree_dirty": bool(
            status.stdout
            .strip()
        ),
    }


def _package_version(
    distribution_name: str,
) -> str | None:
    try:
        return metadata.version(
            distribution_name
        )

    except metadata.PackageNotFoundError:
        return None


def _prompt_sha256() -> str:
    prompt = str(
        getattr(
            TextPostProcessingService,
            "_SYSTEM_PROMPT",
            "",
        )
    )

    return hashlib.sha256(
        prompt.encode(
            "utf-8"
        )
    ).hexdigest()


def _read_bool(
    config_reader: ConfigReader,
    key: str,
) -> bool:
    value = (
        config_reader
        .get_required(
            key
        )
    )

    if isinstance(
        value,
        bool,
    ):
        return value

    raise FinalVerificationError(
        "Configuration key "
        f"'{key}' "
        "must be boolean"
    )


def _write_report(
    path: Path,
    report: dict[
        str,
        Any,
    ],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


async def run_final_verification_async(
    args: argparse.Namespace,
) -> dict[
    str,
    Any,
]:
    audio_path = (
        Path(
            args.audio
        )
        .expanduser()
        .resolve()
    )

    if (
        not audio_path.is_file()
        or (
            audio_path
            .stat()
            .st_size
            <= 0
        )
    ):
        raise FinalVerificationError(
            "Real audio file is "
            "missing or empty: "
            f"{audio_path}"
        )

    bootstrap_di()

    config = resolve(
        ConfigReader
    )

    transcription_repository = resolve(
        TranscriptionRepository
    )

    segment_repository = resolve(
        TranscriptionSegmentRepository
    )

    storage_provider = resolve(
        S3StorageProvider
    )

    queue_provider = resolve(
        CeleryQueueProvider
    )

    normalization_service = resolve(
        PersianTextNormalizationService
    )

    base_url = (
        _resolve_api_base_url(
            config,
            args.api_base_url,
        )
    )

    api_prefix = str(
        config.get_api_prefix()
    ).rstrip("/")

    transcription_url = (
        f"{base_url}"
        f"{api_prefix}"
        "/transcription"
    )

    llm_enabled = (
        _read_bool(
            config,
            "llm.enabled",
        )
    )

    if (
        args.require_llm
        and not llm_enabled
    ):
        raise FinalVerificationError(
            "--require-llm was requested "
            "but llm.enabled is false"
        )

    report: dict[
        str,
        Any,
    ] = {
        "stage": 14,
        "started_at": (
            _utc_now_iso()
        ),
        "overall_status": (
            "running"
        ),
        "environment": (
            config
            .get_environment()
        ),
        "api_base_url": (
            base_url
        ),
        "source_audio": {
            "path": str(
                audio_path
            ),
            "size_bytes": (
                audio_path
                .stat()
                .st_size
            ),
            "sha256": (
                await asyncio.to_thread(
                    _sha256_file,
                    audio_path,
                )
            ),
        },
        "runtime": {
            "python": (
                sys.version
            ),
            "platform": (
                platform
                .platform()
            ),
            "packages": {
                "faster-whisper": (
                    _package_version(
                        "faster-whisper"
                    )
                ),
                "ctranslate2": (
                    _package_version(
                        "ctranslate2"
                    )
                ),
                "celery": (
                    _package_version(
                        "celery"
                    )
                ),
                "httpx": (
                    _package_version(
                        "httpx"
                    )
                ),
            },
        },
        "git": (
            _git_provenance()
        ),
        "queue": (
            await asyncio.to_thread(
                _queue_preflight_sync,
                queue_provider,
                (
                    args
                    .worker_ping_timeout_seconds
                ),
            )
        ),
    }

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(
            args.http_timeout_seconds
        ),
        trust_env=False,
    ) as client:
        accepted = (
            await _upload_audio_async(
                client,
                transcription_url,
                audio_path,
            )
        )

        transcription_id = str(
            accepted[
                "id"
            ]
        )

        (
            terminal,
            history,
        ) = (
            await _wait_for_terminal_status_async(
                client,
                transcription_url,
                transcription_id,
                args.timeout_seconds,
                args.poll_interval_seconds,
                str(
                    accepted.get(
                        "status"
                    )
                    or ""
                ),
            )
        )

        if (
            terminal.get(
                "status"
            )
            != "completed"
        ):
            failed = (
                await _get_result_async(
                    client,
                    transcription_url,
                    transcription_id,
                )
            )

            raise FinalVerificationError(
                "Main real-audio job "
                "failed: "
                "error_code="
                f"{failed.get('error_code')}, "
                "error_message="
                f"{failed.get('error_message')}"
            )

        api_result = (
            await _get_result_async(
                client,
                transcription_url,
                transcription_id,
            )
        )

        text_comparison = (
            _validate_completed_result(
                api_result,
                llm_enabled,
            )
        )

        expected_normalized_text = (
            normalization_service
            .normalize_text(
                str(
                    api_result[
                        "raw_text"
                    ]
                )
            )
        )

        if (
            expected_normalized_text
            != api_result[
                "normalized_text"
            ]
        ):
            raise FinalVerificationError(
                "normalized_text does not "
                "match deterministic "
                "Persian normalization"
            )

        text_comparison[
            "deterministic_normalization_match"
        ] = True

        (
            db_report,
            db_transcription,
        ) = (
            await _verify_database_async(
                transcription_repository,
                segment_repository,
                transcription_id,
                api_result,
            )
        )

        storage_report = (
            await _verify_storage_async(
                storage_provider,
                str(
                    db_transcription
                    .storage_key
                ),
                audio_path,
            )
        )

        report[
            "main_transcription"
        ] = {
            "id": (
                transcription_id
            ),
            "status_history": (
                history
            ),
            "text_comparison": (
                text_comparison
            ),
            "database": (
                db_report
            ),
            "storage": (
                storage_report
            ),
            "provenance": {
                "configured_asr_model": (
                    config.get_required(
                        "asr.faster_whisper.model"
                    )
                ),
                "recorded_asr_engine": (
                    api_result.get(
                        "asr_engine"
                    )
                ),
                "recorded_asr_model": (
                    api_result.get(
                        "asr_model"
                    )
                ),
                "llm_enabled": (
                    llm_enabled
                ),
                "configured_llm_model": (
                    config.get_required(
                        "llm.model"
                    )
                ),
                "recorded_llm_model": (
                    api_result.get(
                        "post_processing_model"
                    )
                ),
                "recorded_prompt_version": (
                    api_result.get(
                        "post_processing_version"
                    )
                ),
                "prompt_sha256": (
                    _prompt_sha256()
                ),
                "expected_prompt_version": (
                    (
                        TextPostProcessingService
                        .PROMPT_VERSION
                    )
                    if llm_enabled
                    else (
                        TextPostProcessingService
                        .DISABLED_VERSION
                    )
                ),
            },
        }

        report[
            "retry_verification"
        ] = (
            await _verify_retry_flow_async(
                client,
                transcription_url,
                args.timeout_seconds,
                args.poll_interval_seconds,
                cleanup=(
                    not args
                    .keep_retry_job
                ),
            )
        )

    if (
        args
        .skip_log_verification
    ):
        report[
            "logs"
        ] = {
            "status": (
                "skipped_by_user"
            ),
            "main_transcription_id": (
                report[
                    "main_transcription"
                ][
                    "id"
                ]
            ),
            "retry_transcription_id": (
                report[
                    "retry_verification"
                ][
                    "transcription_id"
                ]
            ),
        }

        report[
            "overall_status"
        ] = (
            "passed_with_"
            "log_verification_skipped"
        )

    else:
        if (
            not args.api_log_file
            or not args.worker_log_file
        ):
            raise FinalVerificationError(
                "Provide --api-log-file "
                "and --worker-log-file, "
                "or use "
                "--skip-log-verification "
                "only for a non-final "
                "diagnostic run"
            )

        report[
            "logs"
        ] = (
            await (
                _wait_for_log_verification_async(
                    Path(
                        args.api_log_file
                    )
                    .expanduser()
                    .resolve(),
                    Path(
                        args.worker_log_file
                    )
                    .expanduser()
                    .resolve(),
                    report[
                        "main_transcription"
                    ][
                        "id"
                    ],
                    report[
                        "retry_verification"
                    ][
                        "transcription_id"
                    ],
                    args.log_timeout_seconds,
                )
            )
        )

        report[
            "overall_status"
        ] = "passed"

    report[
        "completed_at"
    ] = (
        _utc_now_iso()
    )

    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run stage 14 final "
            "verification against "
            "the live service."
        )
    )

    parser.add_argument(
        "--audio",
        required=True,
        help=(
            "Real speech audio "
            "file path"
        ),
    )

    parser.add_argument(
        "--api-base-url",
        default=None,
    )

    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=1800.0,
    )

    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--http-timeout-seconds",
        type=float,
        default=60.0,
    )

    parser.add_argument(
        "--worker-ping-timeout-seconds",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--log-timeout-seconds",
        type=float,
        default=15.0,
    )

    parser.add_argument(
        "--api-log-file",
        default=None,
    )

    parser.add_argument(
        "--worker-log-file",
        default=None,
    )

    parser.add_argument(
        "--skip-log-verification",
        action="store_true",
    )

    parser.add_argument(
        "--require-llm",
        action="store_true",
        help=(
            "Fail unless LLM "
            "post-processing is enabled "
            "(recommended in production)"
        ),
    )

    parser.add_argument(
        "--keep-retry-job",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        default=(
            "diagnostics/"
            "final_verification_report.json"
        ),
    )

    args = (
        parser.parse_args()
    )

    for name in (
        "timeout_seconds",
        "poll_interval_seconds",
        "http_timeout_seconds",
        "worker_ping_timeout_seconds",
        "log_timeout_seconds",
    ):
        if float(
            getattr(
                args,
                name,
            )
        ) <= 0:
            parser.error(
                "--"
                f"{name.replace('_', '-')}"
                " must be greater "
                "than zero"
            )

    return args


def main() -> None:
    args = (
        _parse_args()
    )

    output_path = (
        Path(
            args.output
        )
        .expanduser()
        .resolve()
    )

    try:
        report = asyncio.run(
            run_final_verification_async(
                args
            )
        )

    except Exception as exc:
        failure = {
            "stage": 14,
            "completed_at": (
                _utc_now_iso()
            ),
            "overall_status": (
                "failed"
            ),
            "error_type": (
                type(
                    exc
                ).__name__
            ),
            "error_message": str(
                exc
            ),
            "git": (
                _git_provenance()
            ),
        }

        _write_report(
            output_path,
            failure,
        )

        print(
            json.dumps(
                failure,
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )

        raise SystemExit(
            1
        ) from exc

    _write_report(
        output_path,
        report,
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    print(
        "Stage 14 report "
        "written to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()