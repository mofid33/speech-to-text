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

PostgreSQL, RabbitMQ, S3-compatible Object Storage, and vLLM are external production resources. They are intentionally not created by docker-compose.production.yml.

Required host capabilities

The production host must provide Docker Engine, Docker Compose v2, an NVIDIA driver compatible with CUDA 12, and NVIDIA Container Toolkit. The Worker container must be able to start with Docker GPU access.

The Worker image installs FFmpeg/ffprobe, cuBLAS for CUDA 12, cuDNN 9, faster-whisper, and CTranslate2. Startup validation fails before Celery starts when CUDA is unavailable or the configured CTranslate2 compute type is unsupported.

Runtime configuration and secrets

ENV_TYPE=production selects src/host/res/appsettings.yaml.

Production values are not committed to Git. A Vault agent or another approved secret-delivery mechanism must render a shell-compatible environment file on the Docker host. docker-compose.production.yml mounts that file read-only into the containers and src/host/entrypoint.sh loads it before Python starts.

The rendered file must use shell assignment syntax, one variable per line. Values containing special characters must be shell-quoted by the secret-rendering system.

The required runtime variables are:

Variable	Purpose
SPEECH_TO_TEXT_DATABASE_URL	SQLAlchemy async PostgreSQL URL using postgresql+asyncpg
SPEECH_TO_TEXT_RABBITMQ_URL	RabbitMQ broker URL using amqp or amqps
SPEECH_TO_TEXT_S3_BUCKET_NAME	Existing production object-storage bucket
SPEECH_TO_TEXT_ASR_MODEL	ASR model selected by the benchmark; not hard-coded in business logic
SPEECH_TO_TEXT_LLM_BASE_URL	Shared OpenAI-compatible vLLM base URL, normally ending in /v1
SPEECH_TO_TEXT_LLM_MODEL	Model name exposed by the shared vLLM endpoint

The optional runtime variables are:

Variable	Default/behavior
SPEECH_TO_TEXT_S3_ENDPOINT_URL	Empty; lets boto3 use its standard S3 endpoint behavior
SPEECH_TO_TEXT_S3_REGION_NAME	us-east-1
SPEECH_TO_TEXT_S3_ACCESS_KEY_ID	Empty; boto3 credential chain may be used
SPEECH_TO_TEXT_S3_SECRET_ACCESS_KEY	Empty; boto3 credential chain may be used
SPEECH_TO_TEXT_LLM_API_KEY	Empty; supported for an unauthenticated internal vLLM endpoint

The Docker host also needs two deployment-only values before running Compose: SPEECH_TO_TEXT_VAULT_ENV_FILE, which points to the Vault-rendered file on the host, and SPEECH_TO_TEXT_DOCKER_NETWORK, which names an existing Docker network that can route to the external production resources.

Build

From the repository root:

docker compose -f docker-compose.production.yml build

The Compose file builds two targets from one Dockerfile:

api     -> Python runtime, FastAPI, migrations
worker  -> Python runtime + FFmpeg + CUDA 12 cuBLAS + cuDNN 9

Set SPEECH_TO_TEXT_IMAGE_TAG before the build when immutable versioned image tags are required by the deployment process.

GPU preflight

Verify the host first:

nvidia-smi

Then verify Docker GPU access independently of the application image using the NVIDIA Container Toolkit procedure approved on the server.

After the Worker image is built, the application-specific runtime check can be executed through Compose:

docker compose -f docker-compose.production.yml run --rm worker \
  python -m src.infrastructure.scripts.validate_runtime --role worker

A successful check confirms that ffprobe is available, faster-whisper/CTranslate2 import correctly, at least one configured CUDA device is visible, and the configured compute_type is supported.

Migration and startup

The production Compose file contains a one-shot migrate service. API and Worker start only after that service exits successfully.

Start the deployment with:

docker compose -f docker-compose.production.yml up -d api worker

The dependency gate performs:

python -m src.infrastructure.scripts.run_migrations --upgrade

Only committed Alembic migrations are applied. Production startup never runs --autogenerate or --sync.

Health and readiness

API liveness:

curl http://127.0.0.1:8000/api/v1/health_check/

API readiness, including PostgreSQL connectivity:

curl http://127.0.0.1:8000/api/v1/health_check/ready

API Prometheus metrics:

curl http://127.0.0.1:8000/metrics

Worker metrics are exposed on port 9101 inside the production Docker network. The Worker image health check reads that endpoint. Prometheus can scrape worker:9101/metrics from the same backend network.

Inspect container state with:

docker compose -f docker-compose.production.yml ps

Inspect logs with:

docker compose -f docker-compose.production.yml logs --tail=200 api
docker compose -f docker-compose.production.yml logs --tail=200 worker
External dependency checks

The API needs reachable PostgreSQL, RabbitMQ, and S3-compatible Object Storage. The Worker needs the same resources plus the shared vLLM endpoint and an NVIDIA GPU.

Do not run a separate vLLM container as part of this service. SPEECH_TO_TEXT_LLM_BASE_URL must point to the shared production vLLM service.

Do not enable storage.auto_create_bucket in production. The bucket must already exist and access must be provisioned before deployment.

ASR model handling

Stage 13 does not choose the winning ASR model. Set SPEECH_TO_TEXT_ASR_MODEL only to the model selected from the benchmark results.

The Worker stores downloaded model data under /var/lib/speech-to-text/models, backed by the speech_to_text_model_cache Docker volume. When the selected model is provided as a local CTranslate2 directory instead of a Hugging Face/model alias, mount or bake that directory into the Worker and set SPEECH_TO_TEXT_ASR_MODEL to its in-container path.

Rollout verification

After API and Worker are healthy, verify one real transcription through the existing API flow and confirm the status progresses through the expected pipeline to completed. Confirm that the database contains the metadata and text fields, the audio remains in object storage, the Worker consumes from the transcription queue, and logs/metrics contain the transcription identifier.

Full end-to-end acceptance with final raw/normalized/processed text comparison belongs to Stage 14.

Rollback

Application rollback is performed by deploying the previous immutable API and Worker image tag and restarting the services.

Do not automatically downgrade the database. If a committed migration must be reversed, review its Alembic downgrade path and production data impact separately before executing it.

Security rules

Never commit the Vault-rendered environment file, database credentials, RabbitMQ credentials, S3 credentials, or LLM API keys. Keep the secret file readable only by the deployment account and Docker runtime. Rotate credentials through Vault or the approved secret-management process, then recreate the affected containers so the new environment is loaded.
