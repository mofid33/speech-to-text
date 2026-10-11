from src.infrastructure.utils.config_reader import (
    ConfigReader,
)


def _create_config_reader(
    monkeypatch,
    environment: str,
) -> ConfigReader:
    monkeypatch.delenv(
        "env_type",
        raising=False,
    )

    monkeypatch.setenv(
        "ENV_TYPE",
        environment,
    )

    return ConfigReader()


def _set_production_runtime_environment(
    monkeypatch,
) -> None:
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


def _assert_shared_runtime_configuration(
    config_reader: ConfigReader,
) -> None:
    assert int(
        config_reader.get_required(
            "transcription.upload.max_duration_seconds"
        )
    ) == 14400

    assert str(
        config_reader.get_required(
            "media.ffprobe_path"
        )
    ).strip()

    assert float(
        config_reader.get_required(
            "media.ffprobe_timeout_seconds"
        )
    ) > 0

    assert str(
        config_reader.get_required(
            "queue.broker_url"
        )
    ).strip()

    assert str(
        config_reader.get_required(
            "queue.transcription_queue"
        )
    ).strip()

    assert str(
        config_reader.get_required(
            "llm.base_url"
        )
    ).strip()

    assert str(
        config_reader.get_required(
            "llm.model"
        )
    ).strip()

    assert float(
        config_reader.get_required(
            "llm.timeout_seconds"
        )
    ) > 0

    assert int(
        config_reader.get_required(
            "llm.max_retries"
        )
    ) >= 0

    assert float(
        config_reader.get_required(
            "llm.retry_backoff_seconds"
        )
    ) >= 0

    assert int(
        config_reader.get_required(
            "llm.max_tokens"
        )
    ) > 0

    assert int(
        config_reader.get_required(
            "llm.max_chars_per_request"
        )
    ) > 0


def test_development_runtime_configuration(
    monkeypatch,
) -> None:
    config_reader = (
        _create_config_reader(
            monkeypatch,
            "development",
        )
    )

    _assert_shared_runtime_configuration(
        config_reader
    )

    assert (
        config_reader.get_required(
            "llm.enabled"
        )
        is False
    )


def test_production_runtime_configuration(
    monkeypatch,
) -> None:
    _set_production_runtime_environment(
        monkeypatch
    )

    config_reader = (
        _create_config_reader(
            monkeypatch,
            "production",
        )
    )

    _assert_shared_runtime_configuration(
        config_reader
    )

    assert (
        config_reader.get_required(
            "llm.enabled"
        )
        is True
    )