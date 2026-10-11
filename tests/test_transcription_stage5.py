import asyncio
import math
import struct
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.core.services.transcription.transcription_processing_service import (
    TranscriptionProcessingService,
)
from src.infrastructure.providers.media.ffmpeg_provider import (
    AudioMediaInfo,
    FfmpegProvider,
    InvalidMediaError,
    MediaProbeError,
)
from src.infrastructure.providers.storage.s3_storage_provider import (
    StorageProviderError,
)


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


class FakeQueueProvider:
    async def enqueue_transcription_async(
        self,
        transcription_id: str,
    ) -> None:
        return None


class FakeStorageProvider:
    def __init__(
        self,
        should_fail: bool = False,
    ):
        self.should_fail = should_fail
        self.downloaded_keys: list[str] = []

    async def download_file_async(
        self,
        object_key: str,
        destination_path: str | Path,
    ) -> Path:
        if self.should_fail:
            raise StorageProviderError(
                "storage unavailable"
            )

        self.downloaded_keys.append(
            object_key
        )

        path = Path(
            destination_path
        )

        path.write_bytes(
            b"fake-audio"
        )

        return path


class FakeFfmpegProvider:
    def __init__(
        self,
        *,
        media_info: AudioMediaInfo | None = None,
        exception: Exception | None = None,
    ):
        self.media_info = (
            media_info
            or AudioMediaInfo(
                duration_seconds=30.0,
                codec_name="mp3",
                sample_rate_hz=44100,
                channels=2,
            )
        )

        self.exception = exception
        self.probed_paths: list[Path] = []

    async def probe_audio_async(
        self,
        file_path: str | Path,
    ) -> AudioMediaInfo:
        self.probed_paths.append(
            Path(
                file_path
            )
        )

        if self.exception is not None:
            raise self.exception

        return self.media_info


class FakeConfigReader:
    def __init__(
        self,
        max_duration_seconds: int = 14400,
    ):
        self.max_duration_seconds = (
            max_duration_seconds
        )

    def get_required(
        self,
        key: str,
    ) -> object:
        if (
            key
            == "transcription.upload.max_duration_seconds"
        ):
            return self.max_duration_seconds

        raise KeyError(
            key
        )


class FfmpegConfigReader:
    def get_required(
        self,
        key: str,
    ) -> object:
        values = {
            "media.ffprobe_path": "ffprobe",
            "media.ffprobe_timeout_seconds": 10,
        }

        return values[
            key
        ]


def create_transcription() -> SimpleNamespace:
    return SimpleNamespace(
        id=(
            "33333333-3333-3333-"
            "3333-333333333333"
        ),
        status="queued",
        storage_key=(
            "transcriptions/"
            "33333333-3333-3333-"
            "3333-333333333333/"
            "source.wav"
        ),
        duration_seconds=None,
        raw_text=None,
        error_code=None,
        error_message=None,
    )


def create_service(
    *,
    transcription: Any,
    storage_provider: Any | None = None,
    ffmpeg_provider: Any | None = None,
    max_duration_seconds: int = 14400,
) -> tuple[
    TranscriptionProcessingService,
    FakeTranscriptionRepository,
]:
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
                object()
            ),
            celery_queue_provider=(
                FakeQueueProvider()
            ),
            storage_provider=(
                storage_provider
                or FakeStorageProvider()
            ),
            ffmpeg_provider=(
                ffmpeg_provider
                or FakeFfmpegProvider()
            ),
            faster_whisper_provider=(
                object()
            ),
            persian_text_normalization_service=(
                object()
            ),
            text_post_processing_service=(
                object()
            ),
            config_reader=(
                FakeConfigReader(
                    max_duration_seconds=(
                        max_duration_seconds
                    )
                )
            ),
        )
    )

    return (
        service,
        repository,
    )


def test_valid_audio_moves_to_transcribing_and_saves_duration() -> None:
    async def run_test() -> None:
        transcription = (
            create_transcription()
        )

        storage = FakeStorageProvider()

        ffmpeg = FakeFfmpegProvider(
            media_info=AudioMediaInfo(
                duration_seconds=125.4,
                codec_name="aac",
                sample_rate_hz=48000,
                channels=2,
            )
        )

        service, repository = (
            create_service(
                transcription=transcription,
                storage_provider=storage,
                ffmpeg_provider=ffmpeg,
            )
        )

        result = await service.process_async(
            str(
                transcription.id
            )
        )

        assert (
            result.status
            == "transcribing"
        )

        assert (
            result.duration_seconds
            == 125.4
        )

        assert result.error_code is None
        assert result.error_message is None

        assert repository.update_count == 2

        assert (
            storage.downloaded_keys
            == [
                transcription.storage_key
            ]
        )

        assert len(
            ffmpeg.probed_paths
        ) == 1

    asyncio.run(
        run_test()
    )


def test_invalid_media_is_failed_before_asr() -> None:
    async def run_test() -> None:
        transcription = (
            create_transcription()
        )

        service, repository = (
            create_service(
                transcription=transcription,
                ffmpeg_provider=(
                    FakeFfmpegProvider(
                        exception=(
                            InvalidMediaError(
                                "Uploaded file "
                                "does not contain "
                                "an audio stream"
                            )
                        )
                    )
                ),
            )
        )

        result = await service.process_async(
            str(
                transcription.id
            )
        )

        assert result.status == "failed"
        assert result.error_code == "invalid_media"

        assert (
            result.error_message
            == (
                "Uploaded file is not a valid "
                "supported audio file"
            )
        )

        assert repository.update_count == 2

    asyncio.run(
        run_test()
    )


def test_duration_limit_is_enforced_from_configuration() -> None:
    async def run_test() -> None:
        transcription = (
            create_transcription()
        )

        service, repository = (
            create_service(
                transcription=transcription,
                max_duration_seconds=60,
                ffmpeg_provider=(
                    FakeFfmpegProvider(
                        media_info=AudioMediaInfo(
                            duration_seconds=60.01,
                            codec_name="flac",
                            sample_rate_hz=16000,
                            channels=1,
                        )
                    )
                ),
            )
        )

        result = await service.process_async(
            str(
                transcription.id
            )
        )

        assert result.status == "failed"

        assert (
            result.duration_seconds
            == 60.01
        )

        assert (
            result.error_code
            == "audio_duration_exceeded"
        )

        assert (
            "60 seconds"
            in result.error_message
        )

        assert repository.update_count == 2

    asyncio.run(
        run_test()
    )


def test_storage_download_failure_stores_error_code() -> None:
    async def run_test() -> None:
        transcription = (
            create_transcription()
        )

        service, repository = (
            create_service(
                transcription=transcription,
                storage_provider=(
                    FakeStorageProvider(
                        should_fail=True
                    )
                ),
            )
        )

        result = await service.process_async(
            str(
                transcription.id
            )
        )

        assert result.status == "failed"

        assert (
            result.error_code
            == "storage_error"
        )

        assert repository.update_count == 2

    asyncio.run(
        run_test()
    )


def test_ffprobe_operational_failure_stores_validation_error() -> None:
    async def run_test() -> None:
        transcription = (
            create_transcription()
        )

        service, repository = (
            create_service(
                transcription=transcription,
                ffmpeg_provider=(
                    FakeFfmpegProvider(
                        exception=(
                            MediaProbeError(
                                "ffprobe could "
                                "not be started"
                            )
                        )
                    )
                ),
            )
        )

        result = await service.process_async(
            str(
                transcription.id
            )
        )

        assert result.status == "failed"

        assert (
            result.error_code
            == "audio_validation_failed"
        )

        assert repository.update_count == 2

    asyncio.run(
        run_test()
    )


def test_ffprobe_reads_duration_codec_sample_rate_and_channels(
    tmp_path: Path,
) -> None:
    async def run_test() -> None:
        audio_path = (
            tmp_path
            / "sample.wav"
        )

        sample_rate = 16000
        duration_seconds = 0.25

        frame_count = int(
            sample_rate
            * duration_seconds
        )

        with wave.open(
            str(
                audio_path
            ),
            "wb",
        ) as wav_file:
            wav_file.setnchannels(
                1
            )

            wav_file.setsampwidth(
                2
            )

            wav_file.setframerate(
                sample_rate
            )

            frames = bytearray()

            for index in range(
                frame_count
            ):
                sample = int(
                    2000
                    * math.sin(
                        2
                        * math.pi
                        * 440
                        * index
                        / sample_rate
                    )
                )

                frames.extend(
                    struct.pack(
                        "<h",
                        sample,
                    )
                )

            wav_file.writeframes(
                bytes(
                    frames
                )
            )

        provider = FfmpegProvider(
            config_reader=(
                FfmpegConfigReader()
            )
        )

        info = await provider.probe_audio_async(
            audio_path
        )

        assert (
            info.codec_name
            == "pcm_s16le"
        )

        assert (
            info.sample_rate_hz
            == 16000
        )

        assert info.channels == 1

        assert (
            info.duration_seconds
            == pytest.approx(
                duration_seconds,
                abs=0.01,
            )
        )

    asyncio.run(
        run_test()
    )


def test_ffprobe_rejects_corrupt_file(
    tmp_path: Path,
) -> None:
    async def run_test() -> None:
        corrupt_path = (
            tmp_path
            / "corrupt.wav"
        )

        corrupt_path.write_bytes(
            b"not-a-real-wave-file"
        )

        provider = FfmpegProvider(
            config_reader=(
                FfmpegConfigReader()
            )
        )

        with pytest.raises(
            InvalidMediaError
        ):
            await (
                provider
                .probe_audio_async(
                    corrupt_path
                )
            )

    asyncio.run(
        run_test()
    )