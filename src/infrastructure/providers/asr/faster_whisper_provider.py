from __future__ import annotations

import asyncio
import gc
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from src.infrastructure.di.inject import inject
from src.infrastructure.utils.config_reader import ConfigReader


class AsrProviderError(RuntimeError):
    """Base error raised by the ASR provider."""


class AsrConfigurationError(AsrProviderError):
    """Raised when ASR configuration is missing or invalid."""


class AsrModelLoadError(AsrProviderError):
    """Raised when a Faster Whisper model cannot be loaded."""


class AsrTranscriptionError(AsrProviderError):
    """Raised when audio transcription fails."""


@dataclass(frozen=True, slots=True)
class AsrSegmentResult:
    sequence: int
    start_time: float
    end_time: float
    text: str
    avg_logprob: float | None
    no_speech_probability: float | None


@dataclass(frozen=True, slots=True)
class AsrTranscriptionResult:
    engine: str
    model: str
    text: str
    detected_language: str
    language_probability: float
    duration_seconds: float
    duration_after_vad_seconds: float
    segments: tuple[AsrSegmentResult, ...]


@inject
class FasterWhisperProvider:
    __di_singleton__ = True

    ENGINE_NAME = "faster-whisper"

    def __init__(self, config_reader: ConfigReader):
        self._configured_model = self._optional_string(
            config_reader.get("asr.faster_whisper.model")
        )

        self._device = self._required_string(
            config_reader,
            "asr.faster_whisper.device",
        )

        self._compute_type = self._required_string(
            config_reader,
            "asr.faster_whisper.compute_type",
        )

        self._device_index = self._read_device_index(
            config_reader.get_required(
                "asr.faster_whisper.device_index"
            )
        )

        self._cpu_threads = self._read_non_negative_int(
            config_reader,
            "asr.faster_whisper.cpu_threads",
        )

        self._num_workers = self._read_positive_int(
            config_reader,
            "asr.faster_whisper.num_workers",
        )

        self._language = self._required_string(
            config_reader,
            "asr.faster_whisper.language",
        )

        self._beam_size = self._read_positive_int(
            config_reader,
            "asr.faster_whisper.beam_size",
        )

        self._vad_filter = self._read_bool(
            config_reader,
            "asr.faster_whisper.vad_filter",
        )

        self._word_timestamps = self._read_bool(
            config_reader,
            "asr.faster_whisper.word_timestamps",
        )

        self._condition_on_previous_text = self._read_bool(
            config_reader,
            "asr.faster_whisper.condition_on_previous_text",
        )

        self._local_files_only = self._read_bool(
            config_reader,
            "asr.faster_whisper.local_files_only",
        )

        self._download_root = self._optional_string(
            config_reader.get(
                "asr.faster_whisper.download_root"
            )
        )

        self._models: dict[str, Any] = {}
        self._model_locks: dict[str, Lock] = {}
        self._model_registry_lock = Lock()

    async def prepare_model_async(
        self,
        model_reference: str | None = None,
    ) -> str:
        resolved_model = self._resolve_model_reference(
            model_reference
        )

        await asyncio.to_thread(
            self._get_or_create_model,
            resolved_model,
        )

        return resolved_model

    async def transcribe_async(
        self,
        audio_path: str | Path,
        model_reference: str | None = None,
    ) -> AsrTranscriptionResult:
        resolved_model = self._resolve_model_reference(
            model_reference
        )

        normalized_audio_path = (
            Path(audio_path)
            .expanduser()
            .resolve()
        )

        if not normalized_audio_path.is_file():
            raise AsrTranscriptionError(
                f"Audio file '{normalized_audio_path}' was not found"
            )

        return await asyncio.to_thread(
            self._transcribe_sync,
            normalized_audio_path,
            resolved_model,
        )

    async def unload_model_async(
        self,
        model_reference: str,
    ) -> None:
        normalized_model = self._normalize_model_reference(
            model_reference
        )

        await asyncio.to_thread(
            self._unload_model_sync,
            normalized_model,
        )

    async def unload_all_models_async(
        self,
    ) -> None:
        await asyncio.to_thread(
            self._unload_all_models_sync
        )

    def _transcribe_sync(
        self,
        audio_path: Path,
        model_reference: str,
    ) -> AsrTranscriptionResult:
        model = self._get_or_create_model(
            model_reference
        )

        inference_lock = self._get_model_lock(
            model_reference
        )

        try:
            with inference_lock:
                segments_generator, info = (
                    model.transcribe(
                        str(audio_path),
                        language=self._language,
                        task="transcribe",
                        beam_size=self._beam_size,
                        vad_filter=self._vad_filter,
                        word_timestamps=self._word_timestamps,
                        condition_on_previous_text=(
                            self._condition_on_previous_text
                        ),
                    )
                )

                provider_segments = []
                raw_text_parts = []

                for sequence, segment in enumerate(
                    segments_generator
                ):
                    segment_text = str(
                        segment.text
                    )

                    raw_text_parts.append(
                        segment_text
                    )

                    provider_segments.append(
                        AsrSegmentResult(
                            sequence=sequence,
                            start_time=max(
                                0.0,
                                float(segment.start),
                            ),
                            end_time=max(
                                0.0,
                                float(segment.end),
                            ),
                            text=segment_text.strip(),
                            avg_logprob=(
                                self._optional_float(
                                    getattr(
                                        segment,
                                        "avg_logprob",
                                        None,
                                    )
                                )
                            ),
                            no_speech_probability=(
                                self._optional_float(
                                    getattr(
                                        segment,
                                        "no_speech_prob",
                                        None,
                                    )
                                )
                            ),
                        )
                    )

            return AsrTranscriptionResult(
                engine=self.ENGINE_NAME,
                model=model_reference,
                text=(
                    "".join(
                        raw_text_parts
                    ).strip()
                ),
                detected_language=str(
                    info.language
                ),
                language_probability=float(
                    info.language_probability
                ),
                duration_seconds=float(
                    info.duration
                ),
                duration_after_vad_seconds=float(
                    info.duration_after_vad
                ),
                segments=tuple(
                    provider_segments
                ),
            )

        except AsrProviderError:
            raise

        except Exception as exc:
            raise AsrTranscriptionError(
                f"Failed to transcribe "
                f"'{audio_path}' with model "
                f"'{model_reference}'"
            ) from exc

    def _get_or_create_model(
        self,
        model_reference: str,
    ) -> Any:
        with self._model_registry_lock:
            existing_model = self._models.get(
                model_reference
            )

            if existing_model is not None:
                return existing_model

            try:
                from faster_whisper import (
                    WhisperModel,
                )

                model = WhisperModel(
                    model_reference,
                    device=self._device,
                    device_index=self._device_index,
                    compute_type=self._compute_type,
                    cpu_threads=self._cpu_threads,
                    num_workers=self._num_workers,
                    download_root=self._download_root,
                    local_files_only=(
                        self._local_files_only
                    ),
                )

            except Exception as exc:
                raise AsrModelLoadError(
                    f"Could not load Faster Whisper "
                    f"model '{model_reference}' "
                    f"on device '{self._device}' "
                    f"with compute_type "
                    f"'{self._compute_type}'"
                ) from exc

            self._models[
                model_reference
            ] = model

            self._model_locks.setdefault(
                model_reference,
                Lock(),
            )

            return model

    def _get_model_lock(
        self,
        model_reference: str,
    ) -> Lock:
        with self._model_registry_lock:
            return self._model_locks.setdefault(
                model_reference,
                Lock(),
            )

    def _unload_model_sync(
        self,
        model_reference: str,
    ) -> None:
        with self._model_registry_lock:
            self._models.pop(
                model_reference,
                None,
            )

            self._model_locks.pop(
                model_reference,
                None,
            )

        gc.collect()

    def _unload_all_models_sync(
        self,
    ) -> None:
        with self._model_registry_lock:
            self._models.clear()
            self._model_locks.clear()

        gc.collect()

    def _resolve_model_reference(
        self,
        model_reference: str | None,
    ) -> str:
        resolved_model = (
            self._optional_string(
                model_reference
            )
            or self._configured_model
        )

        if resolved_model is None:
            raise AsrConfigurationError(
                "No ASR model is selected. Set "
                "'asr.faster_whisper.model' after "
                "the benchmark or pass "
                "model_reference explicitly."
            )

        return self._normalize_model_reference(
            resolved_model
        )

    @staticmethod
    def _normalize_model_reference(
        model_reference: str,
    ) -> str:
        normalized = model_reference.strip()

        if not normalized:
            raise AsrConfigurationError(
                "ASR model reference must not be empty"
            )

        return normalized

    @staticmethod
    def _required_string(
        config_reader: ConfigReader,
        key: str,
    ) -> str:
        value = str(
            config_reader.get_required(
                key
            )
        ).strip()

        if not value:
            raise AsrConfigurationError(
                f"Configuration key "
                f"'{key}' must not be empty"
            )

        return value

    @staticmethod
    def _optional_string(
        value: object,
    ) -> str | None:
        if value is None:
            return None

        normalized = str(
            value
        ).strip()

        return normalized or None

    @staticmethod
    def _read_positive_int(
        config_reader: ConfigReader,
        key: str,
    ) -> int:
        value = int(
            config_reader.get_required(
                key
            )
        )

        if value <= 0:
            raise AsrConfigurationError(
                f"Configuration key "
                f"'{key}' must be greater than zero"
            )

        return value

    @staticmethod
    def _read_non_negative_int(
        config_reader: ConfigReader,
        key: str,
    ) -> int:
        value = int(
            config_reader.get_required(
                key
            )
        )

        if value < 0:
            raise AsrConfigurationError(
                f"Configuration key "
                f"'{key}' must be zero or greater"
            )

        return value

    @staticmethod
    def _read_bool(
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

        normalized = str(
            value
        ).strip().lower()

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

        raise AsrConfigurationError(
            f"Configuration key "
            f"'{key}' must be boolean"
        )

    @staticmethod
    def _read_device_index(
        value: object,
    ) -> int | list[int]:
        if isinstance(value, int):
            if value < 0:
                raise AsrConfigurationError(
                    "asr.faster_whisper."
                    "device_index must not "
                    "be negative"
                )

            return value

        if isinstance(value, list):
            if not value:
                raise AsrConfigurationError(
                    "asr.faster_whisper."
                    "device_index list must "
                    "not be empty"
                )

            normalized = [
                int(item)
                for item in value
            ]

            if any(
                item < 0
                for item in normalized
            ):
                raise AsrConfigurationError(
                    "asr.faster_whisper."
                    "device_index values must "
                    "not be negative"
                )

            return normalized

        raise AsrConfigurationError(
            "asr.faster_whisper."
            "device_index must be an integer "
            "or a list of integers"
        )

    @staticmethod
    def _optional_float(
        value: object,
    ) -> float | None:
        if value is None:
            return None

        return float(value)