import asyncio
import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from src.core.services.transcription.text_post_processing_service import (
    TextPostProcessingProviderError,
    TextPostProcessingResult,
    TextPostProcessingService,
    TextPostProcessingValidationError,
)
from src.core.services.transcription.transcription_processing_service import (
    TranscriptionProcessingService,
)
from src.infrastructure.providers.llm.openai_compatible_llm_provider import (
    LlmChatCompletionResult,
    OpenAiCompatibleLlmProvider,
)


class FakePostProcessingConfigReader:
    def __init__(
        self,
        max_chars_per_request: int = 12000,
    ):
        self.max_chars_per_request = (
            max_chars_per_request
        )

    def get_required(
        self,
        key: str,
    ) -> object:
        values = {
            "llm.max_chars_per_request": (
                self.max_chars_per_request
            ),
        }

        return values[
            key
        ]


class FakeEnabledLlmProvider:
    def __init__(
        self,
        *,
        output_text: str,
    ):
        self.output_text = output_text
        self.calls = 0

    @property
    def is_enabled(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return "test-model"

    async def complete_chat_async(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LlmChatCompletionResult:
        self.calls += 1

        return LlmChatCompletionResult(
            content=json.dumps(
                {
                    "processed_text": (
                        self.output_text
                    )
                },
                ensure_ascii=False,
            ),
            model="test-model",
            request_id="test-request",
        )


class FakeDisabledLlmProvider:
    def __init__(self):
        self.calls = 0

    @property
    def is_enabled(self) -> bool:
        return False

    @property
    def model_name(self) -> str:
        return "unused-model"

    async def complete_chat_async(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LlmChatCompletionResult:
        self.calls += 1

        raise AssertionError(
            "Disabled LLM must not be called"
        )


class ProviderConfigReader:
    def get(
        self,
        key: str,
        default: object = None,
    ) -> object:
        values = {
            "llm.reasoning_effort": "none",
        }

        return values.get(
            key,
            default,
        )

    def get_required(
        self,
        key: str,
    ) -> object:
        values = {
            "llm.enabled": True,
            "llm.base_url": (
                "http://llm.test/v1"
            ),
            "llm.api_key": "secret",
            "llm.model": "served-model",
            "llm.timeout_seconds": 5,
            "llm.max_retries": 1,
            "llm.retry_backoff_seconds": 0,
            "llm.temperature": 0.0,
            "llm.max_tokens": 512,
        }

        return values[
            key
        ]


class FakeTranscriptionRepository:
    def __init__(
        self,
        transcription: Any,
    ):
        self.transcription = transcription
        self.update_count = 0

    async def get_by_id_async(
        self,
        transcription_id: str,
    ) -> Any:
        if (
            str(
                self.transcription.id
            )
            != transcription_id
        ):
            return None

        return self.transcription

    async def update_async(
        self,
        transcription: Any,
    ) -> Any:
        self.transcription = transcription
        self.update_count += 1

        return transcription


class FakeSegmentRepository:
    async def get_by_transcription_id_async(
        self,
        transcription_id: str,
    ) -> list[Any]:
        return []

    async def insert_many_async(
        self,
        segments: list[Any],
    ) -> list[Any]:
        return segments

    async def update_many_async(
        self,
        segments: list[Any],
    ) -> list[Any]:
        return segments


class ProcessingConfigReader:
    def get_required(
        self,
        key: str,
    ) -> object:
        if (
            key
            == "transcription.upload.max_duration_seconds"
        ):
            return 14400

        raise KeyError(
            key
        )


class FakeSuccessfulTextPostProcessingService:
    async def process_text_async(
        self,
        normalized_text: str,
    ) -> TextPostProcessingResult:
        return TextPostProcessingResult(
            processed_text=(
                "سلام دنیا. "
                "نسخه FastAPI 0.115 است."
            ),
            model="shared-test-model",
            version=(
                "persian-transcript-"
                "postprocess-v1"
            ),
            llm_used=True,
        )


class FakeFailedTextPostProcessingService:
    async def process_text_async(
        self,
        normalized_text: str,
    ) -> TextPostProcessingResult:
        raise TextPostProcessingProviderError(
            "synthetic LLM failure"
        )


def test_disabled_llm_returns_normalized_text_without_request() -> None:
    async def run_test() -> None:
        provider = (
            FakeDisabledLlmProvider()
        )

        service = TextPostProcessingService(
            llm_provider=provider,
            config_reader=(
                FakePostProcessingConfigReader()
            ),
        )

        source = (
            "سلام دنیا FastAPI 0.115"
        )

        result = (
            await service.process_text_async(
                source
            )
        )

        assert (
            result.processed_text
            == source
        )

        assert result.model is None

        assert (
            result.version
            == "llm-disabled-v1"
        )

        assert result.llm_used is False
        assert provider.calls == 0

    asyncio.run(
        run_test()
    )


def test_post_processing_accepts_formatting_only_changes() -> None:
    async def run_test() -> None:
        source = (
            "سلام دنیا نسخه "
            "FastAPI 0.115 است"
        )

        provider = FakeEnabledLlmProvider(
            output_text=(
                "سلام دنیا؛ نسخه "
                "FastAPI 0.115 است."
            )
        )

        service = TextPostProcessingService(
            llm_provider=provider,
            config_reader=(
                FakePostProcessingConfigReader()
            ),
        )

        result = (
            await service.process_text_async(
                source
            )
        )

        assert (
            result.processed_text
            == (
                "سلام دنیا؛ نسخه "
                "FastAPI 0.115 است."
            )
        )

        assert (
            result.model
            == "test-model"
        )

        assert result.llm_used is True
        assert provider.calls == 1

    asyncio.run(
        run_test()
    )


def test_post_processing_rejects_changed_number() -> None:
    async def run_test() -> None:
        provider = FakeEnabledLlmProvider(
            output_text=(
                "نسخه FastAPI "
                "0.116 است."
            )
        )

        service = TextPostProcessingService(
            llm_provider=provider,
            config_reader=(
                FakePostProcessingConfigReader()
            ),
        )

        with pytest.raises(
            TextPostProcessingValidationError
        ):
            await service.process_text_async(
                "نسخه FastAPI 0.115 است"
            )

    asyncio.run(
        run_test()
    )


def test_openai_provider_retries_transient_error_and_uses_json_response_format() -> None:
    async def run_test() -> None:
        requests: list[
            httpx.Request
        ] = []

        def handler(
            request: httpx.Request,
        ) -> httpx.Response:
            requests.append(
                request
            )

            if len(
                requests
            ) == 1:
                return httpx.Response(
                    status_code=503,
                    json={
                        "error": "temporary"
                    },
                )

            request_payload = json.loads(
                request.content.decode(
                    "utf-8"
                )
            )

            assert (
                request_payload[
                    "model"
                ]
                == "served-model"
            )

            assert (
                request_payload[
                    "response_format"
                ]
                == {
                    "type": "json_object"
                }
            )

            assert (
                request_payload[
                    "reasoning_effort"
                ]
                == "none"
            )

            assert (
                request.headers[
                    "authorization"
                ]
                == "Bearer secret"
            )

            return httpx.Response(
                status_code=200,
                headers={
                    "x-request-id": (
                        "request-123"
                    )
                },
                json={
                    "id": "chatcmpl-test",
                    "model": "served-model",
                    "choices": [
                        {
                            "message": {
                                "role": (
                                    "assistant"
                                ),
                                "content": (
                                    '{"processed_text":'
                                    '"سلام دنیا."}'
                                ),
                            }
                        }
                    ],
                },
            )

        provider = (
            OpenAiCompatibleLlmProvider(
                config_reader=(
                    ProviderConfigReader()
                ),
                transport=(
                    httpx.MockTransport(
                        handler
                    )
                ),
            )
        )

        result = (
            await provider.complete_chat_async(
                system_prompt="system",
                user_prompt="user",
            )
        )

        assert len(
            requests
        ) == 2

        assert (
            result.model
            == "served-model"
        )

        assert (
            result.request_id
            == "request-123"
        )

        assert (
            result.content
            == (
                '{"processed_text":'
                '"سلام دنیا."}'
            )
        )

    asyncio.run(
        run_test()
    )


def test_worker_persists_processed_text_and_completes() -> None:
    async def run_test() -> None:
        transcription = SimpleNamespace(
            id=(
                "bbbbbbbb-bbbb-bbbb-"
                "bbbb-bbbbbbbbbbbb"
            ),
            status="post_processing",
            raw_text=(
                "سلام دنيا نسخه "
                "FastAPI 0.115 است"
            ),
            normalized_text=(
                "سلام دنیا نسخه "
                "FastAPI 0.115 است"
            ),
            processed_text=None,
            post_processing_model=None,
            post_processing_version=None,
            completed_datetime=None,
            error_code=None,
            error_message=None,
        )

        repository = (
            FakeTranscriptionRepository(
                transcription
            )
        )

        service = (
            TranscriptionProcessingService(
                transcription_repository=(
                    repository
                ),
                transcription_segment_repository=(
                    FakeSegmentRepository()
                ),
                celery_queue_provider=object(),
                storage_provider=object(),
                ffmpeg_provider=object(),
                faster_whisper_provider=object(),
                persian_text_normalization_service=object(),
                text_post_processing_service=(
                    FakeSuccessfulTextPostProcessingService()
                ),
                config_reader=(
                    ProcessingConfigReader()
                ),
            )
        )

        original_raw = (
            transcription.raw_text
        )

        original_normalized = (
            transcription.normalized_text
        )

        result = (
            await service.process_job_async(
                str(
                    transcription.id
                )
            )
        )

        assert result.status == "completed"

        assert (
            result.raw_text
            == original_raw
        )

        assert (
            result.normalized_text
            == original_normalized
        )

        assert (
            result.processed_text
            == (
                "سلام دنیا. "
                "نسخه FastAPI 0.115 است."
            )
        )

        assert (
            result.post_processing_model
            == "shared-test-model"
        )

        assert (
            result.post_processing_version
            == (
                "persian-transcript-"
                "postprocess-v1"
            )
        )

        assert (
            result.completed_datetime
            is not None
        )

        assert repository.update_count == 1

    asyncio.run(
        run_test()
    )


def test_llm_failure_preserves_raw_and_normalized_text() -> None:
    async def run_test() -> None:
        transcription = SimpleNamespace(
            id=(
                "cccccccc-cccc-cccc-"
                "cccc-cccccccccccc"
            ),
            status="post_processing",
            raw_text="متن خام",
            normalized_text="متن نرمال",
            processed_text=None,
            post_processing_model=None,
            post_processing_version=None,
            completed_datetime=None,
            error_code=None,
            error_message=None,
        )

        repository = (
            FakeTranscriptionRepository(
                transcription
            )
        )

        service = (
            TranscriptionProcessingService(
                transcription_repository=(
                    repository
                ),
                transcription_segment_repository=(
                    FakeSegmentRepository()
                ),
                celery_queue_provider=object(),
                storage_provider=object(),
                ffmpeg_provider=object(),
                faster_whisper_provider=object(),
                persian_text_normalization_service=object(),
                text_post_processing_service=(
                    FakeFailedTextPostProcessingService()
                ),
                config_reader=(
                    ProcessingConfigReader()
                ),
            )
        )

        result = (
            await service.process_job_async(
                str(
                    transcription.id
                )
            )
        )

        assert result.status == "failed"

        assert (
            result.error_code
            == "llm_post_processing_failed"
        )

        assert (
            result.raw_text
            == "متن خام"
        )

        assert (
            result.normalized_text
            == "متن نرمال"
        )

        assert result.processed_text is None

    asyncio.run(
        run_test()
    )