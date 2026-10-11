from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit

from sqlalchemy.engine import make_url

from src.infrastructure.utils.config_reader import (
    ConfigReader,
)


_SUPPORTED_ROLES = {
    "migrate",
    "api",
    "worker",
}


def _required_string(
    config_reader: ConfigReader,
    key: str,
) -> str:
    value = str(
        config_reader.get_required(key)
    ).strip()

    if not value:
        raise RuntimeError(
            f"Configuration key '{key}' "
            "must not be empty."
        )

    return value


def _read_bool(
    config_reader: ConfigReader,
    key: str,
) -> bool:
    value = config_reader.get_required(
        key
    )

    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()

    if normalized in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True

    if normalized in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False

    raise RuntimeError(
        f"Configuration key '{key}' "
        "must be a boolean."
    )


def _validate_database(
    config_reader: ConfigReader,
) -> None:
    database_url = _required_string(
        config_reader,
        "database.url",
    )

    try:
        parsed = make_url(
            database_url
        )
    except Exception as exc:
        raise RuntimeError(
            "Configuration key 'database.url' "
            "is not a valid SQLAlchemy URL."
        ) from exc

    if parsed.drivername != "postgresql+asyncpg":
        raise RuntimeError(
            "Production database.url must use "
            "the 'postgresql+asyncpg' driver."
        )

    if not parsed.host:
        raise RuntimeError(
            "Configuration key 'database.url' "
            "must contain a PostgreSQL host."
        )


def _validate_network_url(
    value: str,
    *,
    key: str,
    schemes: Iterable[str],
) -> None:
    parsed = urlsplit(
        value
    )

    allowed_schemes = set(
        schemes
    )

    if parsed.scheme not in allowed_schemes:
        joined_schemes = ", ".join(
            sorted(allowed_schemes)
        )

        raise RuntimeError(
            f"Configuration key '{key}' must use "
            f"one of these URL schemes: "
            f"{joined_schemes}."
        )

    if not parsed.hostname:
        raise RuntimeError(
            f"Configuration key '{key}' "
            "must contain a host."
        )


def _validate_queue(
    config_reader: ConfigReader,
) -> None:
    broker_url = _required_string(
        config_reader,
        "queue.broker_url",
    )

    _validate_network_url(
        broker_url,
        key="queue.broker_url",
        schemes=(
            "amqp",
            "amqps",
        ),
    )

    _required_string(
        config_reader,
        "queue.transcription_queue",
    )


def _validate_storage(
    config_reader: ConfigReader,
) -> None:
    _required_string(
        config_reader,
        "storage.bucket_name",
    )

    endpoint_url = str(
        config_reader.get(
            "storage.endpoint_url",
            "",
        )
        or ""
    ).strip()

    if endpoint_url:
        _validate_network_url(
            endpoint_url,
            key="storage.endpoint_url",
            schemes=(
                "http",
                "https",
            ),
        )


def _normalize_device_indices(
    raw_value: object,
) -> list[int]:
    if (
        isinstance(raw_value, int)
        and not isinstance(raw_value, bool)
    ):
        values = [
            raw_value
        ]

    elif isinstance(raw_value, list):
        values = []

        for item in raw_value:
            if isinstance(item, bool):
                raise RuntimeError(
                    "asr.faster_whisper.device_index "
                    "must contain only integers."
                )

            try:
                values.append(
                    int(item)
                )
            except (
                TypeError,
                ValueError,
            ) as exc:
                raise RuntimeError(
                    "asr.faster_whisper.device_index "
                    "must contain only integers."
                ) from exc

    else:
        raise RuntimeError(
            "asr.faster_whisper.device_index "
            "must be an integer or a list of integers."
        )

    if not values:
        raise RuntimeError(
            "asr.faster_whisper.device_index "
            "must not be empty."
        )

    if any(
        value < 0
        for value in values
    ):
        raise RuntimeError(
            "asr.faster_whisper.device_index "
            "must not contain negative values."
        )

    return values


def _validate_media_runtime(
    config_reader: ConfigReader,
) -> None:
    ffprobe_path = _required_string(
        config_reader,
        "media.ffprobe_path",
    )

    executable = shutil.which(
        ffprobe_path
    )

    if executable is None:
        candidate = Path(
            ffprobe_path
        )

        if not (
            candidate.is_file()
            and candidate.exists()
        ):
            raise RuntimeError(
                "ffprobe is not available. "
                "Install FFmpeg in the Worker image "
                "or configure media.ffprobe_path."
            )


def _validate_asr_runtime(
    config_reader: ConfigReader,
) -> None:
    _required_string(
        config_reader,
        "asr.faster_whisper.model",
    )

    device = _required_string(
        config_reader,
        "asr.faster_whisper.device",
    ).lower()

    compute_type = _required_string(
        config_reader,
        "asr.faster_whisper.compute_type",
    ).lower()

    device_indices = _normalize_device_indices(
        config_reader.get_required(
            "asr.faster_whisper.device_index"
        )
    )

    if device not in {
        "cpu",
        "cuda",
    }:
        raise RuntimeError(
            "asr.faster_whisper.device must be "
            "either 'cpu' or 'cuda'."
        )

    try:
        import ctranslate2
        import faster_whisper
    except Exception as exc:
        raise RuntimeError(
            "faster-whisper/CTranslate2 could not "
            "be imported in the Worker runtime."
        ) from exc

    if device == "cpu":
        try:
            supported_compute_types = set(
                ctranslate2.get_supported_compute_types(
                    "cpu"
                )
            )
        except Exception as exc:
            raise RuntimeError(
                "CTranslate2 CPU runtime validation failed."
            ) from exc

        if (
            compute_type not in {
                "auto",
                "default",
            }
            and compute_type
            not in supported_compute_types
        ):
            raise RuntimeError(
                "Configured ASR compute_type "
                f"'{compute_type}' is not supported "
                "by the CTranslate2 CPU runtime."
            )

        return

    try:
        cuda_device_count = int(
            ctranslate2.get_cuda_device_count()
        )
    except Exception as exc:
        raise RuntimeError(
            "CTranslate2 could not initialize the "
            "CUDA runtime. Check the NVIDIA driver, "
            "container GPU access, cuBLAS and cuDNN."
        ) from exc

    if cuda_device_count <= 0:
        raise RuntimeError(
            "No CUDA device is visible to the Worker. "
            "Start the container with NVIDIA GPU access."
        )

    for device_index in device_indices:
        if device_index >= cuda_device_count:
            raise RuntimeError(
                "Configured ASR device_index "
                f"'{device_index}' is not available. "
                f"Visible CUDA devices: {cuda_device_count}."
            )

        try:
            supported_compute_types = set(
                ctranslate2.get_supported_compute_types(
                    "cuda",
                    device_index=device_index,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                "CTranslate2 could not query supported "
                "CUDA compute types."
            ) from exc

        if (
            compute_type not in {
                "auto",
                "default",
            }
            and compute_type
            not in supported_compute_types
        ):
            raise RuntimeError(
                "Configured ASR compute_type "
                f"'{compute_type}' is not supported "
                f"on CUDA device {device_index}. "
                "Supported values: "
                + ", ".join(
                    sorted(supported_compute_types)
                )
                + "."
            )

    _ = faster_whisper


def _validate_llm_runtime(
    config_reader: ConfigReader,
) -> None:
    enabled = _read_bool(
        config_reader,
        "llm.enabled",
    )

    if not enabled:
        return

    base_url = _required_string(
        config_reader,
        "llm.base_url",
    )

    _validate_network_url(
        base_url,
        key="llm.base_url",
        schemes=(
            "http",
            "https",
        ),
    )

    _required_string(
        config_reader,
        "llm.model",
    )


def validate_runtime_configuration(
    config_reader: ConfigReader,
    *,
    role: str,
) -> None:
    normalized_role = role.strip().lower()

    if normalized_role not in _SUPPORTED_ROLES:
        raise RuntimeError(
            f"Unsupported runtime role '{role}'."
        )

    _validate_database(
        config_reader
    )

    if normalized_role == "migrate":
        return

    _validate_queue(
        config_reader
    )

    _validate_storage(
        config_reader
    )

    if normalized_role == "api":
        return

    _validate_media_runtime(
        config_reader
    )

    _validate_asr_runtime(
        config_reader
    )

    _validate_llm_runtime(
        config_reader
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Speech-to-Text runtime "
            "configuration and host capabilities."
        )
    )

    parser.add_argument(
        "--role",
        required=True,
        choices=sorted(
            _SUPPORTED_ROLES
        ),
        help=(
            "Runtime role to validate."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config_reader = ConfigReader()

    validate_runtime_configuration(
        config_reader,
        role=args.role,
    )

    print(
        "Runtime validation passed: "
        f"role={args.role}, "
        "environment="
        f"{config_reader.get_environment()}"
    )


if __name__ == "__main__":
    main()