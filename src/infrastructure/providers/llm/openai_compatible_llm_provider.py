import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from src.infrastructure.di.inject import inject
from src.infrastructure.utils.config_reader import ConfigReader


class LlmProviderError(Exception):
    pass


class LlmConfigurationError(LlmProviderError):
    pass


class LlmRequestError(LlmProviderError):
    pass


class LlmResponseError(LlmProviderError):
    pass


@dataclass(
    frozen=True,
    slots=True,
)
class LlmChatCompletionResult:
    content: str
    model: str
    request_id: str | None


@inject
class OpenAiCompatibleLlmProvider:
    def __init__(
        self,
        config_reader: ConfigReader,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._enabled = self._read_bool(
            config_reader=config_reader,
            key="llm.enabled",
        )

        self._base_url = str(
            config_reader.get_required(
                "llm.base_url"
            )
        ).strip().rstrip("/")

        self._api_key = str(
            config_reader.get_required(
                "llm.api_key"
            )
            or ""
        ).strip()

        self._model = str(
            config_reader.get_required(
                "llm.model"
            )
        ).strip()

        self._timeout_seconds = (
            self._read_positive_float(
                config_reader=config_reader,
                key="llm.timeout_seconds",
            )
        )

        self._max_retries = (
            self._read_non_negative_int(
                config_reader=config_reader,
                key="llm.max_retries",
            )
        )

        self._retry_backoff_seconds = (
            self._read_non_negative_float(
                config_reader=config_reader,
                key="llm.retry_backoff_seconds",
            )
        )

        self._temperature = (
            self._read_temperature(
                config_reader=config_reader,
                key="llm.temperature",
            )
        )

        self._max_tokens = (
            self._read_positive_int(
                config_reader=config_reader,
                key="llm.max_tokens",
            )
        )

        self._reasoning_effort = (
            self._read_optional_string(
                config_reader=config_reader,
                key="llm.reasoning_effort",
            )
        )

        self._transport = transport

        if self._enabled:
            self._validate_enabled_configuration()

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def model_name(self) -> str:
        return self._model

    async def complete_chat_async(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LlmChatCompletionResult:
        if not self._enabled:
            raise LlmConfigurationError(
                "LLM provider is disabled"
            )

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
            "response_format": {
                "type": "json_object",
            },
        }

        if self._reasoning_effort is not None:
            payload["reasoning_effort"] = (
                self._reasoning_effort
            )

        headers = {
            "Content-Type": "application/json",
        }

        if self._api_key:
            headers["Authorization"] = (
                f"Bearer {self._api_key}"
            )

        endpoint = (
            f"{self._base_url}/chat/completions"
        )

        timeout = httpx.Timeout(
            self._timeout_seconds
        )

        async with httpx.AsyncClient(
            timeout=timeout,
            transport=self._transport,
            trust_env=False,
        ) as client:
            for attempt in range(
                self._max_retries + 1
            ):
                try:
                    response = await client.post(
                        endpoint,
                        headers=headers,
                        json=payload,
                    )

                except (
                    httpx.TimeoutException,
                    httpx.TransportError,
                ) as exc:
                    if attempt >= self._max_retries:
                        raise LlmRequestError(
                            "LLM endpoint could not be reached"
                        ) from exc

                    await self._sleep_before_retry_async(
                        attempt
                    )
                    continue

                if response.status_code >= 400:
                    if (
                        self._is_retryable_status(
                            response.status_code
                        )
                        and attempt
                        < self._max_retries
                    ):
                        await self._sleep_before_retry_async(
                            attempt
                        )
                        continue

                    raise LlmRequestError(
                        "LLM endpoint returned HTTP "
                        f"{response.status_code}"
                    )

                return self._parse_response(
                    response
                )

        raise LlmRequestError(
            "LLM request could not be completed"
        )

    def _parse_response(
        self,
        response: httpx.Response,
    ) -> LlmChatCompletionResult:
        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise LlmResponseError(
                "LLM endpoint returned invalid JSON"
            ) from exc

        if not isinstance(
            payload,
            dict,
        ):
            raise LlmResponseError(
                "LLM response must be a JSON object"
            )

        choices = payload.get(
            "choices"
        )

        if (
            not isinstance(choices, list)
            or not choices
        ):
            raise LlmResponseError(
                "LLM response does not contain choices"
            )

        first_choice = choices[0]

        if not isinstance(
            first_choice,
            dict,
        ):
            raise LlmResponseError(
                "LLM response choice is invalid"
            )

        message = first_choice.get(
            "message"
        )

        if not isinstance(
            message,
            dict,
        ):
            raise LlmResponseError(
                "LLM response does not contain a message"
            )

        content = message.get(
            "content"
        )

        if (
            not isinstance(content, str)
            or not content.strip()
        ):
            raise LlmResponseError(
                "LLM response content is empty"
            )

        response_model = payload.get(
            "model"
        )

        model = (
            str(response_model).strip()
            if response_model is not None
            else self._model
        )

        if not model:
            model = self._model

        request_id = response.headers.get(
            "x-request-id"
        )

        if request_id is None:
            payload_request_id = payload.get(
                "id"
            )

            if payload_request_id is not None:
                request_id = str(
                    payload_request_id
                )

        return LlmChatCompletionResult(
            content=content,
            model=model,
            request_id=request_id,
        )

    async def _sleep_before_retry_async(
        self,
        attempt: int,
    ) -> None:
        delay_seconds = (
            self._retry_backoff_seconds
            * (
                2
                ** attempt
            )
        )

        if delay_seconds <= 0:
            return

        await asyncio.sleep(
            delay_seconds
        )

    def _validate_enabled_configuration(
        self,
    ) -> None:
        if not self._base_url:
            raise LlmConfigurationError(
                "Configuration key "
                "'llm.base_url' cannot be empty "
                "when LLM is enabled"
            )

        if not self._base_url.startswith(
            (
                "http://",
                "https://",
            )
        ):
            raise LlmConfigurationError(
                "Configuration key "
                "'llm.base_url' must be an HTTP "
                "or HTTPS URL"
            )

        if not self._model:
            raise LlmConfigurationError(
                "Configuration key "
                "'llm.model' cannot be empty "
                "when LLM is enabled"
            )

    @staticmethod
    def _is_retryable_status(
        status_code: int,
    ) -> bool:
        return (
            status_code
            in {
                408,
                429,
            }
            or status_code >= 500
        )

    @staticmethod
    def _read_bool(
        *,
        config_reader: ConfigReader,
        key: str,
    ) -> bool:
        value = config_reader.get_required(
            key
        )

        if isinstance(
            value,
            bool,
        ):
            return value

        if isinstance(
            value,
            str,
        ):
            normalized_value = (
                value.strip().lower()
            )

            if normalized_value in {
                "true",
                "1",
                "yes",
                "on",
            }:
                return True

            if normalized_value in {
                "false",
                "0",
                "no",
                "off",
            }:
                return False

        raise ValueError(
            f"Configuration key '{key}' "
            "must be a boolean"
        )

    @staticmethod
    def _read_positive_int(
        *,
        config_reader: ConfigReader,
        key: str,
    ) -> int:
        raw_value = (
            config_reader.get_required(
                key
            )
        )

        if isinstance(
            raw_value,
            bool,
        ):
            raise ValueError(
                f"Configuration key '{key}' "
                "must be a positive integer"
            )

        try:
            value = int(
                raw_value
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Configuration key '{key}' "
                "must be a positive integer"
            ) from exc

        if value <= 0:
            raise ValueError(
                f"Configuration key '{key}' "
                "must be a positive integer"
            )

        return value

    @staticmethod
    def _read_non_negative_int(
        *,
        config_reader: ConfigReader,
        key: str,
    ) -> int:
        raw_value = (
            config_reader.get_required(
                key
            )
        )

        if isinstance(
            raw_value,
            bool,
        ):
            raise ValueError(
                f"Configuration key '{key}' "
                "must be a non-negative integer"
            )

        try:
            value = int(
                raw_value
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Configuration key '{key}' "
                "must be a non-negative integer"
            ) from exc

        if value < 0:
            raise ValueError(
                f"Configuration key '{key}' "
                "must be a non-negative integer"
            )

        return value

    @staticmethod
    def _read_positive_float(
        *,
        config_reader: ConfigReader,
        key: str,
    ) -> float:
        raw_value = (
            config_reader.get_required(
                key
            )
        )

        if isinstance(
            raw_value,
            bool,
        ):
            raise ValueError(
                f"Configuration key '{key}' "
                "must be a positive number"
            )

        try:
            value = float(
                raw_value
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Configuration key '{key}' "
                "must be a positive number"
            ) from exc

        if value <= 0:
            raise ValueError(
                f"Configuration key '{key}' "
                "must be a positive number"
            )

        return value

    @staticmethod
    def _read_non_negative_float(
        *,
        config_reader: ConfigReader,
        key: str,
    ) -> float:
        raw_value = (
            config_reader.get_required(
                key
            )
        )

        if isinstance(
            raw_value,
            bool,
        ):
            raise ValueError(
                f"Configuration key '{key}' "
                "must be a non-negative number"
            )

        try:
            value = float(
                raw_value
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Configuration key '{key}' "
                "must be a non-negative number"
            ) from exc

        if value < 0:
            raise ValueError(
                f"Configuration key '{key}' "
                "must be a non-negative number"
            )

        return value

    @staticmethod
    def _read_temperature(
        *,
        config_reader: ConfigReader,
        key: str,
    ) -> float:
        raw_value = (
            config_reader.get_required(
                key
            )
        )

        if isinstance(
            raw_value,
            bool,
        ):
            raise ValueError(
                f"Configuration key '{key}' "
                "must be a number between 0 and 2"
            )

        try:
            value = float(
                raw_value
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Configuration key '{key}' "
                "must be a number between 0 and 2"
            ) from exc

        if not 0 <= value <= 2:
            raise ValueError(
                f"Configuration key '{key}' "
                "must be a number between 0 and 2"
            )

        return value

    @staticmethod
    def _read_optional_string(
        *,
        config_reader: ConfigReader,
        key: str,
    ) -> str | None:
        raw_value = config_reader.get(
            key
        )

        if raw_value is None:
            return None

        if not isinstance(
            raw_value,
            str,
        ):
            raise ValueError(
                f"Configuration key '{key}' "
                "must be a string"
            )

        value = raw_value.strip()

        if not value:
            return None

        return value