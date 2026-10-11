import hashlib
import json
from pathlib import Path

import pytest

from src.core.services.transcription.text_post_processing_service import (
    TextPostProcessingService,
)
from src.infrastructure.scripts.final_verification import (
    FinalVerificationError,
    _read_json_log_entries,
    _resolve_api_base_url,
    _sha256_file,
    _validate_completed_result,
    _verify_logs,
)


class _ConfigStub:
    def __init__(
        self,
        values: dict[
            str,
            object,
        ],
    ):
        self._values = (
            values
        )

    def get_required(
        self,
        key: str,
    ) -> object:
        return self._values[
            key
        ]


def _completed_result(
    *,
    llm_enabled: bool,
) -> dict[
    str,
    object,
]:
    normalized = (
        "سلام دنیا"
    )

    processed = (
        "سلام، دنیا."
        if llm_enabled
        else normalized
    )

    return {
        "id": (
            "11111111-1111-1111-"
            "1111-111111111111"
        ),
        "status": (
            "completed"
        ),
        "asr_engine": (
            "faster-whisper"
        ),
        "asr_model": (
            "large-v3"
        ),
        "raw_text": (
            "سلام دنيا"
        ),
        "normalized_text": (
            normalized
        ),
        "processed_text": (
            processed
        ),
        "post_processing_model": (
            "test-llm"
            if llm_enabled
            else None
        ),
        "post_processing_version": (
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
        "error_code": None,
        "error_message": None,
        "segments": [
            {
                "sequence": 0,
                "start_time": 0.0,
                "end_time": 1.0,
                "raw_text": (
                    "سلام دنيا"
                ),
                "normalized_text": (
                    normalized
                ),
                "processed_text": None,
                "speaker_label": None,
            },
        ],
    }


def _write_json_log(
    path: Path,
    rows: list[
        dict[
            str,
            object,
        ]
    ],
) -> None:
    path.write_text(
        "\n".join(
            json.dumps(
                row
            )
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def test_resolve_api_base_url_converts_wildcard_host_to_loopback(
) -> None:
    config = _ConfigStub(
        {
            "server.host": (
                "0.0.0.0"
            ),
            "server.port": 8000,
        }
    )

    assert (
        _resolve_api_base_url(
            config,
            None,
        )
        == (
            "http://127.0.0.1:"
            "8000"
        )
    )

    assert (
        _resolve_api_base_url(
            config,
            (
                "http://"
                "localhost:9999/"
            ),
        )
        == (
            "http://localhost:"
            "9999"
        )
    )


def test_sha256_file_matches_hashlib(
    tmp_path: Path,
) -> None:
    sample = (
        tmp_path
        / "sample.bin"
    )

    payload = (
        b"stage-14-"
        b"final-verification"
    )

    sample.write_bytes(
        payload
    )

    assert (
        _sha256_file(
            sample
        )
        == hashlib.sha256(
            payload
        ).hexdigest()
    )


def test_validate_completed_result_accepts_llm_disabled(
) -> None:
    result = (
        _completed_result(
            llm_enabled=False
        )
    )

    comparison = (
        _validate_completed_result(
            result,
            llm_enabled=False,
        )
    )

    assert (
        comparison[
            "normalized_equals_processed"
        ]
        is True
    )

    assert (
        comparison[
            "segment_count"
        ]
        == 1
    )


def test_validate_completed_result_accepts_llm_enabled(
) -> None:
    result = (
        _completed_result(
            llm_enabled=True
        )
    )

    comparison = (
        _validate_completed_result(
            result,
            llm_enabled=True,
        )
    )

    assert (
        comparison[
            "normalized_equals_processed"
        ]
        is False
    )

    assert (
        comparison[
            "processed_text"
        ]
        == "سلام، دنیا."
    )


def test_validate_completed_result_rejects_missing_asr_provenance(
) -> None:
    result = (
        _completed_result(
            llm_enabled=False
        )
    )

    result[
        "asr_model"
    ] = None

    with pytest.raises(
        FinalVerificationError,
        match="asr_model",
    ):
        _validate_completed_result(
            result,
            llm_enabled=False,
        )


def test_read_json_log_entries_ignores_non_json_and_other_ids(
    tmp_path: Path,
) -> None:
    target_id = (
        "11111111-1111-1111-"
        "1111-111111111111"
    )

    log_path = (
        tmp_path
        / "api.log"
    )

    log_path.write_text(
        (
            "INFO: uvicorn line\n"
            + json.dumps(
                {
                    "event": (
                        "api_request_completed"
                    ),
                    "transcription_id": (
                        target_id
                    ),
                }
            )
            + "\n"
            + json.dumps(
                {
                    "event": (
                        "api_request_completed"
                    ),
                    "transcription_id": (
                        "22222222-2222-2222-"
                        "2222-222222222222"
                    ),
                }
            )
            + "\n"
        ),
        encoding="utf-8",
    )

    entries = (
        _read_json_log_entries(
            log_path,
            target_id,
        )
    )

    assert len(
        entries
    ) == 1

    assert (
        entries[
            0
        ][
            "event"
        ]
        == (
            "api_request_completed"
        )
    )


def test_verify_logs_requires_main_and_retry_evidence(
    tmp_path: Path,
) -> None:
    main_id = (
        "11111111-1111-1111-"
        "1111-111111111111"
    )

    retry_id = (
        "22222222-2222-2222-"
        "2222-222222222222"
    )

    api_log = (
        tmp_path
        / "api.log"
    )

    worker_log = (
        tmp_path
        / "worker.log"
    )

    _write_json_log(
        api_log,
        [
            {
                "event": (
                    "api_request_completed"
                ),
                "transcription_id": (
                    main_id
                ),
            },
            {
                "event": (
                    "queue_publish_completed"
                ),
                "transcription_id": (
                    main_id
                ),
            },
            {
                "event": (
                    "queue_publish_completed"
                ),
                "transcription_id": (
                    retry_id
                ),
            },
            {
                "event": (
                    "queue_publish_completed"
                ),
                "transcription_id": (
                    retry_id
                ),
            },
        ],
    )

    _write_json_log(
        worker_log,
        [
            {
                "event": (
                    "worker_job_started"
                ),
                "transcription_id": (
                    main_id
                ),
            },
            {
                "event": (
                    "worker_job_finished"
                ),
                "transcription_id": (
                    main_id
                ),
            },
            {
                "event": (
                    "worker_job_started"
                ),
                "transcription_id": (
                    retry_id
                ),
            },
            {
                "event": (
                    "worker_job_finished"
                ),
                "transcription_id": (
                    retry_id
                ),
            },
            {
                "event": (
                    "worker_job_started"
                ),
                "transcription_id": (
                    retry_id
                ),
            },
            {
                "event": (
                    "worker_job_finished"
                ),
                "transcription_id": (
                    retry_id
                ),
            },
        ],
    )

    result = (
        _verify_logs(
            api_log,
            worker_log,
            main_id,
            retry_id,
        )
    )

    assert (
        result[
            "status"
        ]
        == "verified"
    )

    assert all(
        result[
            "checks"
        ].values()
    )


def test_verify_logs_rejects_missing_second_retry_execution(
    tmp_path: Path,
) -> None:
    main_id = (
        "11111111-1111-1111-"
        "1111-111111111111"
    )

    retry_id = (
        "22222222-2222-2222-"
        "2222-222222222222"
    )

    api_log = (
        tmp_path
        / "api.log"
    )

    worker_log = (
        tmp_path
        / "worker.log"
    )

    _write_json_log(
        api_log,
        [
            {
                "event": (
                    "api_request_completed"
                ),
                "transcription_id": (
                    main_id
                ),
            },
            {
                "event": (
                    "queue_publish_completed"
                ),
                "transcription_id": (
                    main_id
                ),
            },
            {
                "event": (
                    "queue_publish_completed"
                ),
                "transcription_id": (
                    retry_id
                ),
            },
        ],
    )

    _write_json_log(
        worker_log,
        [
            {
                "event": (
                    "worker_job_started"
                ),
                "transcription_id": (
                    main_id
                ),
            },
            {
                "event": (
                    "worker_job_finished"
                ),
                "transcription_id": (
                    main_id
                ),
            },
            {
                "event": (
                    "worker_job_started"
                ),
                "transcription_id": (
                    retry_id
                ),
            },
            {
                "event": (
                    "worker_job_finished"
                ),
                "transcription_id": (
                    retry_id
                ),
            },
        ],
    )

    with pytest.raises(
        FinalVerificationError,
        match=(
            "retry_queue_"
            "published_twice"
        ),
    ):
        _verify_logs(
            api_log,
            worker_log,
            main_id,
            retry_id,
        )