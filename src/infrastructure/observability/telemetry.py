from __future__ import annotations

import json
import logging
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from enum import Enum
from threading import Lock
from typing import Any, Iterator

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
    multiprocess,
    start_http_server,
)

_LOGGER_NAME = "speech_to_text"

_TRANSCRIPTION_ID_CONTEXT: ContextVar[str | None] = ContextVar(
    "speech_to_text_transcription_id",
    default=None,
)

_TRANSCRIPTION_ID_PATTERN = re.compile(
    r"/transcription/(?P<transcription_id>"
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})(?:/|$)"
)


_API_REQUEST_DURATION_SECONDS = Histogram(
    "speech_to_text_api_request_duration_seconds",
    "HTTP request latency in seconds.",
    labelnames=(
        "method",
        "route",
        "status_code",
    ),
)

_QUEUE_PUBLISH_DURATION_SECONDS = Histogram(
    "speech_to_text_queue_publish_duration_seconds",
    "RabbitMQ task publish latency in seconds.",
    labelnames=("outcome",),
)

_QUEUE_WAIT_SECONDS = Histogram(
    "speech_to_text_queue_wait_seconds",
    "Time from queue publish timestamp to Worker pickup.",
)

_WORKER_PROCESSING_DURATION_SECONDS = Histogram(
    "speech_to_text_worker_processing_duration_seconds",
    "End-to-end Worker processing latency in seconds.",
    labelnames=("outcome",),
)

_WORKER_PROCESSING_RTF = Histogram(
    "speech_to_text_worker_processing_realtime_factor",
    "Worker processing latency divided by audio duration.",
    buckets=(
        0.05,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        3.0,
        5.0,
        10.0,
    ),
)

_WORKER_JOBS_TOTAL = Counter(
    "speech_to_text_worker_jobs_total",
    "Worker job results grouped by outcome.",
    labelnames=("outcome",),
)

_TRANSCRIPTION_FAILURES_TOTAL = Counter(
    "speech_to_text_transcription_failures_total",
    "Operational transcription failures grouped by stage and error code.",
    labelnames=(
        "stage",
        "error_code",
    ),
)

_WORKER_RETRIES_TOTAL = Counter(
    "speech_to_text_worker_retries_total",
    "Celery Worker retries scheduled for transcription jobs.",
)


_METRICS_SERVER_LOCK = Lock()
_METRICS_SERVER_STARTED = False

_STANDARD_LOG_RECORD_ATTRIBUTES = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "message",
    "module",
    "msecs",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class JsonLogFormatter(logging.Formatter):
    def format(
        self,
        record: logging.LogRecord,
    ) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(
                UTC
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in (
            record.__dict__.items()
        ):
            if (
                key
                in _STANDARD_LOG_RECORD_ATTRIBUTES
                or key.startswith("_")
                or value is None
            ):
                continue

            payload[key] = (
                self._json_safe_value(
                    value
                )
            )

        if record.exc_info is not None:
            payload["exception"] = (
                self.formatException(
                    record.exc_info
                )
            )

        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _json_safe_value(
        value: Any,
    ) -> Any:
        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        ):
            return value

        if isinstance(
            value,
            (
                list,
                tuple,
                set,
            ),
        ):
            return [
                JsonLogFormatter
                ._json_safe_value(
                    item
                )
                for item
                in value
            ]

        if isinstance(
            value,
            dict,
        ):
            return {
                str(key): (
                    JsonLogFormatter
                    ._json_safe_value(
                        item
                    )
                )
                for key, item
                in value.items()
            }

        return str(
            value
        )


def configure_structured_logging(
    log_level: str = "INFO",
) -> None:
    logger = logging.getLogger(
        _LOGGER_NAME
    )

    resolved_level = getattr(
        logging,
        str(log_level)
        .strip()
        .upper(),
        logging.INFO,
    )

    logger.setLevel(
        resolved_level
    )
    logger.propagate = False

    for handler in logger.handlers:
        if getattr(
            handler,
            "_speech_to_text_json_handler",
            False,
        ):
            handler.setLevel(
                resolved_level
            )
            return

    handler = logging.StreamHandler()

    handler.setLevel(
        resolved_level
    )

    handler.setFormatter(
        JsonLogFormatter()
    )

    setattr(
        handler,
        "_speech_to_text_json_handler",
        True,
    )

    logger.addHandler(
        handler
    )


def get_logger(
    component: str,
) -> logging.Logger:
    normalized_component = (
        str(component)
        .strip()
        .strip(".")
    )

    if not normalized_component:
        return logging.getLogger(
            _LOGGER_NAME
        )

    return logging.getLogger(
        f"{_LOGGER_NAME}."
        f"{normalized_component}"
    )


@contextmanager
def transcription_context(
    transcription_id: str | None,
) -> Iterator[None]:
    normalized_id = (
        str(
            transcription_id
        ).strip()
        if transcription_id
        is not None
        else None
    )

    token = (
        _TRANSCRIPTION_ID_CONTEXT
        .set(
            normalized_id
            or None
        )
    )

    try:
        yield
    finally:
        _TRANSCRIPTION_ID_CONTEXT.reset(
            token
        )


def current_transcription_id(
) -> str | None:
    return (
        _TRANSCRIPTION_ID_CONTEXT
        .get()
    )


def extract_transcription_id_from_path(
    path: str,
) -> str | None:
    match = (
        _TRANSCRIPTION_ID_PATTERN
        .search(
            str(path)
        )
    )

    if match is None:
        return None

    return match.group(
        "transcription_id"
    ).lower()


def log_event(
    logger: logging.Logger,
    *,
    level: int,
    event: str,
    message: str,
    **fields: Any,
) -> None:
    extra: dict[str, Any] = {
        "event": event,
    }

    transcription_id = (
        current_transcription_id()
    )

    if transcription_id is not None:
        extra[
            "transcription_id"
        ] = transcription_id

    extra.update(
        {
            key: value
            for key, value
            in fields.items()
            if value is not None
        }
    )

    logger.log(
        level,
        message,
        extra=extra,
    )


def observe_api_request(
    *,
    method: str,
    route: str,
    status_code: int,
    duration_seconds: float,
) -> None:
    (
        _API_REQUEST_DURATION_SECONDS
        .labels(
            method=(
                str(method).upper()
            ),
            route=str(route),
            status_code=str(
                status_code
            ),
        )
        .observe(
            max(
                0.0,
                float(
                    duration_seconds
                ),
            )
        )
    )


def observe_queue_publish(
    *,
    outcome: str,
    duration_seconds: float,
) -> None:
    (
        _QUEUE_PUBLISH_DURATION_SECONDS
        .labels(
            outcome=str(
                outcome
            )
        )
        .observe(
            max(
                0.0,
                float(
                    duration_seconds
                ),
            )
        )
    )


def observe_queue_wait_from_enqueued_at(
    enqueued_at: str | None,
) -> float | None:
    if not enqueued_at:
        return None

    try:
        queued_datetime = (
            datetime.fromisoformat(
                str(enqueued_at)
            )
        )
    except ValueError:
        return None

    if (
        queued_datetime.tzinfo
        is None
    ):
        queued_datetime = (
            queued_datetime.replace(
                tzinfo=UTC
            )
        )

    queue_wait_seconds = max(
        0.0,
        (
            datetime.now(UTC)
            - queued_datetime
            .astimezone(UTC)
        ).total_seconds(),
    )

    _QUEUE_WAIT_SECONDS.observe(
        queue_wait_seconds
    )

    return queue_wait_seconds


def observe_worker_result(
    transcription: Any,
    *,
    processing_duration_seconds: float,
) -> tuple[
    str,
    float | None,
]:
    outcome = (
        _get_status_value(
            getattr(
                transcription,
                "status",
                None,
            )
        )
        or "unknown"
    )

    (
        _WORKER_PROCESSING_DURATION_SECONDS
        .labels(
            outcome=outcome
        )
        .observe(
            max(
                0.0,
                float(
                    processing_duration_seconds
                ),
            )
        )
    )

    _WORKER_JOBS_TOTAL.labels(
        outcome=outcome
    ).inc()

    audio_duration_seconds = (
        _get_positive_float(
            getattr(
                transcription,
                "duration_seconds",
                None,
            )
        )
    )

    rtf = (
        _observe_worker_processing_rtf(
            processing_duration_seconds=(
                processing_duration_seconds
            ),
            audio_duration_seconds=(
                audio_duration_seconds
            ),
        )
    )

    if outcome == "failed":
        error_code = str(
            getattr(
                transcription,
                "error_code",
                None,
            )
            or "unknown_failure"
        )

        record_transcription_failure(
            stage=(
                _infer_failure_stage(
                    error_code
                )
            ),
            error_code=error_code,
        )

    return (
        outcome,
        rtf,
    )


def observe_worker_exception(
    *,
    processing_duration_seconds: float,
) -> None:
    (
        _WORKER_PROCESSING_DURATION_SECONDS
        .labels(
            outcome="exception"
        )
        .observe(
            max(
                0.0,
                float(
                    processing_duration_seconds
                ),
            )
        )
    )


def _observe_worker_processing_rtf(
    *,
    processing_duration_seconds: float,
    audio_duration_seconds: float | None,
) -> float | None:
    if (
        audio_duration_seconds
        is None
        or audio_duration_seconds
        <= 0
    ):
        return None

    rtf = (
        max(
            0.0,
            float(
                processing_duration_seconds
            ),
        )
        / audio_duration_seconds
    )

    _WORKER_PROCESSING_RTF.observe(
        rtf
    )

    return rtf


def record_transcription_failure(
    *,
    stage: str,
    error_code: str,
) -> None:
    (
        _TRANSCRIPTION_FAILURES_TOTAL
        .labels(
            stage=str(
                stage
            ),
            error_code=str(
                error_code
            ),
        )
        .inc()
    )


def record_worker_retry() -> None:
    _WORKER_RETRIES_TOTAL.inc()


def _infer_failure_stage(
    error_code: str,
) -> str:
    if error_code in {
        "audio_duration_exceeded",
        "audio_validation_failed",
        "invalid_media",
    }:
        return "validating"

    if error_code.startswith(
        "asr_"
    ):
        return "transcribing"

    if (
        error_code.startswith(
            "post_processing"
        )
        or error_code.startswith(
            "llm_"
        )
    ):
        return "post_processing"

    if error_code == "storage_error":
        return "storage"

    if (
        error_code
        == "processing_retry_exhausted"
    ):
        return "worker"

    return "unknown"


def _get_status_value(
    status: Any,
) -> str | None:
    if status is None:
        return None

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


def _get_positive_float(
    raw_value: Any,
) -> float | None:
    if raw_value is None:
        return None

    try:
        value = float(
            raw_value
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    if value <= 0:
        return None

    return value


def render_prometheus_metrics(
) -> bytes:
    return generate_latest(
        _get_exposition_registry()
    )


def prometheus_content_type(
) -> str:
    return CONTENT_TYPE_LATEST


def start_metrics_http_server(
    *,
    host: str,
    port: int,
) -> None:
    global _METRICS_SERVER_STARTED

    with _METRICS_SERVER_LOCK:
        if _METRICS_SERVER_STARTED:
            return

        start_http_server(
            port=int(
                port
            ),
            addr=str(
                host
            ),
            registry=(
                _get_exposition_registry()
            ),
        )

        _METRICS_SERVER_STARTED = True


def _get_exposition_registry():
    if not os.getenv(
        "PROMETHEUS_MULTIPROC_DIR"
    ):
        return REGISTRY

    registry = (
        CollectorRegistry()
    )

    multiprocess.MultiProcessCollector(
        registry
    )

    return registry