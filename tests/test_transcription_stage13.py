from pathlib import Path

import pytest
import yaml

from src.infrastructure.scripts.validate_runtime import (
    validate_runtime_configuration,
)
from src.infrastructure.utils.config_reader import (
    ConfigReader,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


def _set_production_environment(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "env_type",
        raising=False,
    )
    monkeypatch.setenv(
        "ENV_TYPE",
        "production",
    )

    values = {
        "SPEECH_TO_TEXT_DATABASE_URL": (
            "postgresql+asyncpg://"
            "speech_to_text:test@postgres:5432/"
            "speech_to_text"
        ),
        "SPEECH_TO_TEXT_RABBITMQ_URL": (
            "amqp://speech_to_text:test@rabbitmq:5672//"
        ),
        "SPEECH_TO_TEXT_S3_BUCKET_NAME": (
            "speech-to-text-test"
        ),
        "SPEECH_TO_TEXT_ASR_MODEL": (
            "large-v3"
        ),
        "SPEECH_TO_TEXT_LLM_BASE_URL": (
            "http://shared-vllm:8000/v1"
        ),
        "SPEECH_TO_TEXT_LLM_MODEL": (
            "speech-post-processing-test"
        ),
    }

    for key, value in values.items():
        monkeypatch.setenv(
            key,
            value,
        )

    for optional_key in (
        "SPEECH_TO_TEXT_S3_ENDPOINT_URL",
        "SPEECH_TO_TEXT_S3_REGION_NAME",
        "SPEECH_TO_TEXT_S3_ACCESS_KEY_ID",
        "SPEECH_TO_TEXT_S3_SECRET_ACCESS_KEY",
        "SPEECH_TO_TEXT_LLM_API_KEY",
    ):
        monkeypatch.delenv(
            optional_key,
            raising=False,
        )


def test_production_config_resolves_environment_values(
    monkeypatch,
) -> None:
    _set_production_environment(
        monkeypatch
    )

    config_reader = ConfigReader()

    assert (
        config_reader.get_database_url()
        == (
            "postgresql+asyncpg://"
            "speech_to_text:test@postgres:5432/"
            "speech_to_text"
        )
    )

    assert (
        config_reader.get_required(
            "queue.broker_url"
        )
        == (
            "amqp://speech_to_text:test@rabbitmq:5672//"
        )
    )

    assert (
        config_reader.get_required(
            "storage.region_name"
        )
        == "us-east-1"
    )

    assert (
        config_reader.get_required(
            "storage.access_key_id"
        )
        == ""
    )

    assert (
        config_reader.get_required(
            "llm.api_key"
        )
        == ""
    )


def test_missing_required_production_environment_value_fails_fast(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "env_type",
        raising=False,
    )
    monkeypatch.setenv(
        "ENV_TYPE",
        "production",
    )
    monkeypatch.delenv(
        "SPEECH_TO_TEXT_DATABASE_URL",
        raising=False,
    )

    config_reader = ConfigReader()

    with pytest.raises(
        RuntimeError,
        match=(
            "SPEECH_TO_TEXT_DATABASE_URL"
        ),
    ):
        config_reader.get_required(
            "database.url"
        )


def test_api_runtime_validation_accepts_external_production_services(
    monkeypatch,
) -> None:
    _set_production_environment(
        monkeypatch
    )

    config_reader = ConfigReader()

    validate_runtime_configuration(
        config_reader,
        role="api",
    )


def test_dockerfile_defines_separate_api_and_gpu_worker_targets(
) -> None:
    dockerfile = (
        PROJECT_ROOT
        / "Dockerfile"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "FROM runtime-base AS api"
        in dockerfile
    )
    assert (
        "FROM runtime-base AS worker"
        in dockerfile
    )
    assert (
        "nvidia-cublas-cu12"
        not in dockerfile
    )
    assert (
        "requirements.worker.txt"
        in dockerfile
    )
    assert (
        'CMD ["api"]'
        in dockerfile
    )
    assert (
        'CMD ["worker"]'
        in dockerfile
    )


def test_worker_requirements_pin_cuda12_and_cudnn9_families(
) -> None:
    requirements = (
        PROJECT_ROOT
        / "requirements.worker.txt"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "nvidia-cublas-cu12>=12,<13"
        in requirements
    )
    assert (
        "nvidia-cudnn-cu12>=9,<10"
        in requirements
    )


def test_production_compose_uses_migration_gate_external_services_and_gpu(
) -> None:
    compose_path = (
        PROJECT_ROOT
        / "docker-compose.production.yml"
    )

    compose = yaml.safe_load(
        compose_path.read_text(
            encoding="utf-8"
        )
    )

    services = compose[
        "services"
    ]

    assert set(
        services
    ) == {
        "migrate",
        "api",
        "worker",
    }

    assert (
        services[
            "migrate"
        ][
            "command"
        ]
        == [
            "migrate"
        ]
    )

    assert (
        services[
            "api"
        ][
            "depends_on"
        ][
            "migrate"
        ][
            "condition"
        ]
        == "service_completed_successfully"
    )

    worker_device = (
        services[
            "worker"
        ][
            "deploy"
        ][
            "resources"
        ][
            "reservations"
        ][
            "devices"
        ][0]
    )

    assert (
        worker_device[
            "driver"
        ]
        == "nvidia"
    )
    assert (
        worker_device[
            "count"
        ]
        == 1
    )
    assert (
        worker_device[
            "capabilities"
        ]
        == [
            "gpu"
        ]
    )

    assert (
        compose[
            "networks"
        ][
            "speech_to_text_backend"
        ][
            "external"
        ]
        is True
    )

    assert (
        "postgres"
        not in services
    )
    assert (
        "rabbitmq"
        not in services
    )
    assert (
        "minio"
        not in services
    )
    assert (
        "vllm"
        not in services
    )